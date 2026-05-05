#!/usr/bin/env python3
"""
run_limit_entry_study.py
─────────────────────────
Limit entry study: compare limit-order execution vs market-on-open baseline.

Signal generation is IDENTICAL to baseline (no lookahead).
Only the entry execution logic changes.

Market baseline:
  signal bar closes → enter at NEXT bar open (market order)

Limit variants:
  signal bar closes → place limit order for NEXT bar
  if not filled within N bars → skip trade entirely
  never enter at market after limit expires

Conservative handling (no future data):
  - Limit price is computed from SIGNAL bar data only (close, ATR)
  - Fill check: limit fills if bar's LOW ≤ limit (for LONG), HIGH ≥ limit (for SHORT)
  - Worst-case: if fill bar also hits SL in same bar → SL exit (not limit fill)
  - Partial bar ambiguity: assume fill at limit price, then check SL/TP

Limit price variants (relative to estimated next-open = signal bar close):
  For LONG:  limit = close × (1 - offset%)  or  close - k × ATR
  For SHORT: limit = close × (1 + offset%)  or  close + k × ATR

Usage
─────
    python run_limit_entry_study.py
    python run_limit_entry_study.py --tf 8h 12h --strategy swing_breakout
    python run_limit_entry_study.py --max-wait 1 2 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.features.feature_engine import FeatureEngine
from src.backtest.engine import BacktestEngine, SimConfig
from src.backtest.metrics import compute_metrics, yearly_breakdown
from src.ml.ml_sizer_v2 import MLSizerV2, SizerConfig
from src.strategies.structure import SwingBreakoutStrategy
from src.strategies.trend import DonchianBreakoutStrategy

logger = get_logger(__name__, log_file=Path("outputs/logs/limit_entry_study.log"))

BASELINE = {
    "swing_breakout/8h":  {"pf": 1.948, "sharpe": 1.772, "mdd": -0.0272, "trades": 246},
    "swing_breakout/12h": {"pf": 1.911, "sharpe": 1.687, "mdd": -0.0199, "trades": 209},
}

STRATEGY_MAP = {
    "swing_breakout":    SwingBreakoutStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
}

ML_ELIGIBLE = {"6h", "8h", "12h"}


# ── Limit price computation ───────────────────────────────────────────────────

def compute_limit_price(
    signal_bar: pd.Series,
    side:       int,           # 1=long, -1=short
    mode:       str,           # "pct_01", "pct_02", "atr_025", "atr_050"
) -> float:
    """
    Compute limit price from SIGNAL bar data only (no lookahead).

    For LONG: limit is BELOW signal bar close (buy dip into next bar)
    For SHORT: limit is ABOVE signal bar close (sell rally into next bar)

    All offsets relative to signal bar close — the best proxy for next open.
    """
    close = float(signal_bar["close"])
    atr   = float(signal_bar.get("atr_14", signal_bar.get("atr_7", close * 0.015)))
    if np.isnan(atr) or atr <= 0:
        atr = close * 0.015

    offsets = {
        "pct_010": close * 0.001,   # 0.10%
        "pct_020": close * 0.002,   # 0.20%
        "atr_025": 0.25 * atr,      # 0.25× ATR
        "atr_050": 0.50 * atr,      # 0.50× ATR
    }
    offset = offsets.get(mode, close * 0.001)

    if side == 1:    # LONG: buy cheaper → limit below close
        return close - offset
    else:            # SHORT: sell higher → limit above close
        return close + offset


# ── Core simulation ───────────────────────────────────────────────────────────

def simulate_limit(
    df:          pd.DataFrame,
    sim_cfg:     SimConfig,
    size_mults:  pd.Series,
    limit_mode:  str,          # "pct_010", "pct_020", "atr_025", "atr_050"
    max_wait:    int = 1,      # max bars to wait for fill (1, 2, or 3)
) -> Tuple[dict, pd.DataFrame]:
    """
    Simulate with limit entries.

    Returns:
        metrics dict, trades DataFrame
    """
    from src.backtest.engine import BacktestResult

    df   = df.sort_values("timestamp").reset_index(drop=True)
    mults= size_mults.reset_index(drop=True).to_numpy(np.float64)
    n    = len(df)

    opens   = df["open"].to_numpy(np.float64)
    highs   = df["high"].to_numpy(np.float64)
    lows    = df["low"].to_numpy(np.float64)
    closes  = df["close"].to_numpy(np.float64)
    signals = df["signal"].fillna(0).to_numpy(np.int64)
    sl_arr  = df["sl_price"].to_numpy(np.float64) if "sl_price" in df.columns else np.full(n, np.nan)
    tp_arr  = df["tp_price"].to_numpy(np.float64) if "tp_price" in df.columns else np.full(n, np.nan)
    atr_col = next((c for c in ["atr_14","atr_7","atr_21"] if c in df.columns), None)
    atrs    = df[atr_col].to_numpy(np.float64) if atr_col else (highs-lows)

    dir_int = {"both":0,"long":1,"short":-1}.get(sim_cfg.direction, 0)
    equity  = np.full(n, sim_cfg.initial_capital, np.float64)
    capital = float(sim_cfg.initial_capital)

    rows:           List[dict] = []
    pnl_bar         = np.zeros(n)

    # State: pending limit order
    in_pos           = False
    pending_limit    = False
    limit_price      = 0.0
    limit_side       = 0
    limit_bars_left  = 0
    limit_sl         = 0.0
    limit_tp         = 0.0
    limit_mult       = 1.0

    # Open position state
    side       = pos_size = entry_price = sl = tp = 0.0
    entry_bar  = 0
    risk_amt   = 0.0

    # Stats
    signals_generated = 0
    signals_filled    = 0
    signals_skipped   = 0

    for i in range(1, n):
        o,h,l,c = opens[i],highs[i],lows[i],closes[i]
        equity[i] = equity[i-1]

        atr = atrs[i]
        if np.isnan(atr) or atr <= 0: atr = max(h-l, c*0.005)

        # ── Check pending limit order ──────────────────────────────────────────
        if pending_limit and not in_pos:
            filled = False

            # Fill condition: conservative
            # LONG:  bar LOW must touch or go below limit price
            # SHORT: bar HIGH must touch or go above limit price
            if limit_side == 1  and l <= limit_price:
                filled = True
                fill_p = limit_price
            elif limit_side == -1 and h >= limit_price:
                filled = True
                fill_p = limit_price

            if filled:
                # Worst-case: if same bar would also hit SL → SL exit immediately
                sl_hit = (limit_side==1  and l <= limit_sl) or \
                         (limit_side==-1 and h >= limit_sl)

                if sl_hit:
                    # Conservative: fill at limit, immediately stopped at SL
                    # Net result: entry + SL exit in same bar
                    fill_p_adj = fill_p * (1 + limit_side * sim_cfg.slippage)
                    rsk_pct    = abs(fill_p_adj - limit_sl) / fill_p_adj
                    sz         = min((capital * sim_cfg.risk_per_trade * limit_mult) / rsk_pct,
                                     capital * sim_cfg.leverage) if rsk_pct > 0 else 0
                    sl_exit    = limit_sl * (1 + (-limit_side) * sim_cfg.slippage)
                    net        = limit_side * (sl_exit - fill_p_adj) / fill_p_adj * sz - sz * sim_cfg.fees * 2
                    capital   += net
                    equity[i]  = capital
                    pnl_bar[i] += net
                    rows.append({
                        "entry_time": df["timestamp"].iloc[i],
                        "exit_time":  df["timestamp"].iloc[i],
                        "side": limit_side, "entry_price": fill_p_adj,
                        "exit_price": sl_exit, "size": sz, "pnl": net,
                        "pnl_pct": net/sz if sz>0 else 0,
                        "r_multiple": net/(abs(fill_p_adj-limit_sl)/fill_p_adj*sz) if sz>0 else 0,
                        "bars_held": 0, "exit_reason": "sl_same_bar",
                        "limit_mode": limit_mode, "limit_price": limit_price,
                    })
                    signals_filled  += 1
                    pending_limit    = False
                else:
                    # Clean fill — open position
                    fill_p_adj = fill_p * (1 + limit_side * sim_cfg.slippage)
                    rsk_pct    = abs(fill_p_adj - limit_sl) / fill_p_adj
                    if rsk_pct <= 0:
                        pending_limit = False
                        signals_skipped += 1
                        limit_bars_left  = 0
                    else:
                        sz         = min((capital * sim_cfg.risk_per_trade * limit_mult) / rsk_pct,
                                         capital * sim_cfg.leverage)
                        capital   -= sz * sim_cfg.fees
                        equity[i]  = capital
                        pnl_bar[i] -= sz * sim_cfg.fees

                        in_pos       = True
                        side         = limit_side
                        entry_price  = fill_p_adj
                        entry_bar    = i
                        sl           = limit_sl
                        tp           = limit_tp
                        pos_size     = sz
                        risk_amt     = abs(fill_p_adj - limit_sl) / fill_p_adj * sz
                        pending_limit = False
                        signals_filled += 1

            else:
                # Not filled — decrement wait counter
                limit_bars_left -= 1
                if limit_bars_left <= 0:
                    pending_limit   = False
                    signals_skipped += 1

        # ── Manage open position ───────────────────────────────────────────────
        if in_pos:
            bh = i - entry_bar
            ep = -1.0; er = -1

            if side==1  and l <= sl: ep,er = sl,0
            elif side==-1 and h >= sl: ep,er = sl,0

            if ep<0:
                if side==1  and h >= tp: ep,er = tp,1
                elif side==-1 and l <= tp: ep,er = tp,1

            if ep<0:
                sig = int(signals[i-1])
                if (side==1 and sig==-1) or (side==-1 and sig==1): ep,er = o,2

            if ep >= 0:
                fill    = ep * (1 + (-side) * sim_cfg.slippage)
                fee     = pos_size * sim_cfg.fees
                net     = side * (fill-entry_price)/entry_price * pos_size - fee
                capital += net
                equity[i] = capital
                pnl_bar[i] += net
                rows.append({
                    "entry_time": df["timestamp"].iloc[entry_bar],
                    "exit_time":  df["timestamp"].iloc[i],
                    "side": side, "entry_price": entry_price,
                    "exit_price": fill, "size": pos_size, "pnl": net,
                    "pnl_pct": net/pos_size if pos_size>0 else 0,
                    "r_multiple": net/risk_amt if risk_amt>0 else 0,
                    "bars_held": bh,
                    "exit_reason": {0:"sl",1:"tp",2:"signal",3:"end"}.get(er,"?"),
                    "limit_mode": limit_mode, "limit_price": entry_price,
                })
                in_pos = False; side = 0

        # ── Generate new limit order from signal ───────────────────────────────
        if not in_pos and not pending_limit:
            sig = int(signals[i-1])
            if sig == 0: continue
            if dir_int == 1  and sig != 1:  continue
            if dir_int == -1 and sig != -1: continue

            signals_generated += 1

            sd     = sig
            mult   = max(0.5, min(2.0, float(mults[i-1])))

            # SL/TP from signal bar
            raw_sl = float(sl_arr[i-1])
            raw_tp = float(tp_arr[i-1])
            sl_p   = raw_sl if not np.isnan(raw_sl) and raw_sl>0 else o*(1-sd*sim_cfg.default_sl_pct)
            tp_p   = raw_tp if not np.isnan(raw_tp) and raw_tp>0 else o*(1+sd*sim_cfg.default_tp_pct)

            # Compute limit price from signal bar (index i-1 = last closed bar)
            lim_p = compute_limit_price(df.iloc[i-1], sd, limit_mode)

            # Validate: limit must be between SL and current bar open
            # (otherwise it's trivially filled at open = same as market)
            if sd == 1  and (lim_p <= sl_p or lim_p >= o):
                # Limit too low (below SL) or above open (fills instantly = market)
                if lim_p >= o:
                    lim_p = o * 0.9995  # minimal discount
                if lim_p <= sl_p:
                    signals_skipped += 1
                    signals_generated -= 1
                    continue

            if sd == -1 and (lim_p >= sl_p or lim_p <= o):
                if lim_p <= o:
                    lim_p = o * 1.0005
                if lim_p >= sl_p:
                    signals_skipped += 1
                    signals_generated -= 1
                    continue

            pending_limit   = True
            limit_price     = lim_p
            limit_side      = sd
            limit_sl        = sl_p
            limit_tp        = tp_p
            limit_mult      = mult
            limit_bars_left = max_wait

    # Close remaining
    if in_pos:
        fill = closes[n-1] * (1-side*sim_cfg.slippage)
        net  = side*(fill-entry_price)/entry_price*pos_size - pos_size*sim_cfg.fees
        capital += net; equity[n-1] = capital
        rows.append({
            "entry_time": df["timestamp"].iloc[entry_bar],
            "exit_time":  df["timestamp"].iloc[n-1],
            "side": side, "entry_price": entry_price, "exit_price": fill,
            "size": pos_size, "pnl": net,
            "pnl_pct": net/pos_size if pos_size>0 else 0,
            "r_multiple": net/risk_amt if risk_amt>0 else 0,
            "bars_held": n-1-entry_bar, "exit_reason": "end",
            "limit_mode": limit_mode, "limit_price": entry_price,
        })

    eq   = pd.Series(equity, index=df.index)
    dd   = (eq - eq.cummax()) / eq.cummax().replace(0, np.nan)
    tdf  = pd.DataFrame(rows) if rows else pd.DataFrame()
    ts   = df["timestamp"]
    tot  = (ts.iloc[-1]-ts.iloc[0]).total_seconds()
    bpy  = 365.25*86400/(tot/max(len(ts)-1,1)) if tot>0 else 365.0
    met  = compute_metrics(tdf, eq, bpy, sim_cfg.initial_capital)

    fill_rate = signals_filled / max(signals_generated, 1)

    stats = {
        "signals_generated": signals_generated,
        "signals_filled":    signals_filled,
        "signals_skipped":   signals_skipped,
        "fill_rate":         fill_rate,
        "metrics":           met,
        "equity":            eq,
        "drawdown":          dd,
        "trades":            tdf,
    }
    return stats


def simulate_market(
    df:         pd.DataFrame,
    sim_cfg:    SimConfig,
    size_mults: pd.Series,
) -> dict:
    """Market-on-open baseline (matches backtest methodology)."""
    res = BacktestEngine(sim_cfg).run(df, "swing_breakout", "", {})
    eq  = res.equity
    dd  = res.drawdown
    n   = int(res.metrics.get("trade_count", 0))
    return {
        "signals_generated": n,
        "signals_filled":    n,
        "signals_skipped":   0,
        "fill_rate":         1.0,
        "metrics":           res.metrics,
        "equity":            eq,
        "drawdown":          dd,
        "trades":            res.trades,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_params(csv, strategy, tf):
    if not Path(csv).exists(): return None
    df   = pd.read_csv(csv)
    mask = (df["strategy"]==strategy)&(df["timeframe"]==tf)
    sub  = df[mask]
    if sub.empty: return None
    best   = sub.sort_values("robustness",ascending=False).iloc[0]
    p_cols = [c for c in best.index if c.startswith("p_") and not pd.isna(best[c])]
    out    = {}
    for c in p_cols:
        v=best[c]; k=c[2:]
        out[k] = int(v) if isinstance(v,float) and v==int(v) else v
    return out


def _m(name, stats):
    m = stats["metrics"]
    return {
        "version":           name,
        "trades":            m.get("trade_count",   0),
        "pf":                m.get("profit_factor", np.nan),
        "sharpe":            m.get("sharpe",        np.nan),
        "sortino":           m.get("sortino",       np.nan),
        "mdd":               m.get("max_drawdown",  np.nan),
        "total_return":      m.get("total_return",  np.nan),
        "win_rate":          m.get("win_rate",      np.nan),
        "fill_rate":         stats["fill_rate"],
        "signals_generated": stats["signals_generated"],
        "signals_filled":    stats["signals_filled"],
        "signals_skipped":   stats["signals_skipped"],
    }


def _print_table(rows, label, bm_key=None):
    df = pd.DataFrame(rows)
    b  = BASELINE.get(bm_key, {})
    print(f"\n{'─'*88}")
    print(f"  {label}")
    if b: print(f"  Baseline: PF={b['pf']:.3f}  Sh={b['sharpe']:.3f}  trades={b['trades']}")
    print(f"{'─'*88}")
    print(f"  {'Version':<22} {'Trades':>7} {'Fill%':>7} {'PF':>8} {'Sharpe':>8} "
          f"{'MDD':>9} {'Return':>9} {'WR':>7}")
    print(f"  {'─'*80}")
    mkt = df[df["version"]=="MARKET"]
    mkt_sh = mkt.iloc[0]["sharpe"] if not mkt.empty else 0

    for _, r in df.iterrows():
        d_sh   = f"({r['sharpe']-mkt_sh:+.3f})" if r["version"]!="MARKET" else ""
        beats  = " ★" if b and r["sharpe"]>b.get("sharpe",0) and r["fill_rate"]>0.5 else ""
        worse  = " ⚠" if r["sharpe"] < mkt_sh - 0.1 else ""
        print(f"  {r['version']:<22} {int(r['trades'] or 0):>7} "
              f"{r['fill_rate']:>7.1%} "
              f"{r['pf']:>7.3f}  {r['sharpe']:>7.3f}{d_sh:<9} "
              f"{r['mdd']:>8.2%} {r['total_return']:>8.1%} "
              f"{r['win_rate']:>7.1%}{beats}{worse}")
    print(f"{'─'*88}")


def _plot(results, df_oos, label, out_dir):
    colors = {
        "MARKET":        "#1D9E75",
        "pct_010_w1":    "#378ADD",
        "pct_020_w1":    "#7F77DD",
        "atr_025_w1":    "#D85A30",
        "atr_050_w1":    "#BA7517",
        "pct_010_w2":    "#5DCAA5",
        "atr_025_w2":    "#f7c94b",
    }
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor="#0f1117",
                              gridspec_kw={"height_ratios":[3,1.5]})
    ts = df_oos["timestamp"].reset_index(drop=True)

    for ax in axes:
        ax.set_facecolor("#161925")
        ax.tick_params(colors="#aaa", labelsize=8)
        for s in ax.spines.values(): s.set_color("#2a2d3e")

    for name, stats in results.items():
        eq  = stats["equity"].reset_index(drop=True)
        col = colors.get(name, "gray")
        lw  = 2.2 if name == "MARKET" else 1.0
        al  = 0.95 if name == "MARKET" else 0.55
        axes[0].plot(ts[:len(eq)], eq, label=name, color=col, linewidth=lw, alpha=al)

    axes[0].set_title(f"Limit Entry Study — {label}", color="#e0e0e0")
    axes[0].legend(facecolor="#1e2235", labelcolor="#e0e0e0", fontsize=7, ncol=4)
    axes[0].grid(True, color="#2a2d3e", alpha=0.4)

    for name, stats in results.items():
        dd  = stats["drawdown"].reset_index(drop=True)*100
        col = colors.get(name, "gray")
        lw  = 2.0 if name == "MARKET" else 0.7
        axes[1].plot(ts[:len(dd)], dd, color=col, linewidth=lw,
                     alpha=0.9 if name=="MARKET" else 0.5, label=name)
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].set_title("Drawdown %", color="#e0e0e0", fontsize=10)
    axes[1].grid(True, color="#2a2d3e", alpha=0.4)

    plt.tight_layout()
    clean = label.replace("/","_")
    path  = out_dir / f"limit_study_{clean}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    logger.info(f"  Plot → {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Limit entry study")
    p.add_argument("--config",       default="config/config.yaml")
    p.add_argument("--tf",           nargs="+", default=["8h","12h"])
    p.add_argument("--strategy",     nargs="+", default=["swing_breakout"])
    p.add_argument("--direction",    default="both")
    p.add_argument("--results-csv",  default="outputs/rankings/all_results.csv")
    p.add_argument("--max-wait",     nargs="+", type=int, default=[1,2,3])
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--limit-modes",  nargs="+",
                   default=["pct_010","pct_020","atr_025","atr_050"])
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(Path(args.config))
    fe   = FeatureEngine(cfg)

    out_plots = cfg.output_dir/"plots";    out_plots.mkdir(exist_ok=True)
    out_ranks = cfg.output_dir/"rankings"; out_ranks.mkdir(exist_ok=True)
    results_csv = Path(args.results_csv)

    sim_cfg = SimConfig(
        fees=cfg.fees, slippage=cfg.slippage,
        leverage=cfg.leverage, risk_per_trade=cfg.risk_per_trade,
        direction=args.direction,
    )
    sizer_cfg = SizerConfig(
        n_estimators=args.n_estimators,
        min_train_trades=150, min_val_trades=30, min_mult_std=0.05,
    )

    logger.info("=== Limit Entry Study ===")
    logger.info(f"Strategies: {args.strategy}  TFs: {args.tf}")
    logger.info(f"Limit modes: {args.limit_modes}")
    logger.info(f"Max wait bars: {args.max_wait}")

    all_rows: List[dict] = []

    for strat_name in args.strategy:
        if strat_name not in STRATEGY_MAP: continue
        strategy = STRATEGY_MAP[strat_name]()

        for tf in args.tf:
            label  = f"{strat_name}/{tf}"
            bm_key = f"{strat_name}/{tf}"
            logger.info(f"\n{'='*55}\n{label}")

            try:
                df = fe.load(tf)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            except Exception as exc:
                logger.error(f"  Load: {exc}"); continue

            params = _load_params(results_csv, strat_name, tf)
            if params is None:
                logger.warning(f"  No params"); continue

            n     = len(df)
            is_n  = int(n * cfg.validation.is_ratio)
            df_is = df.iloc[:is_n]
            df_oos= df.iloc[is_n:]

            df_oos_sig = strategy.generate_signals(df_oos.copy(), params)
            df_oos_sig = df_oos_sig.reset_index(drop=True)

            # ML size mults (same as baseline)
            size_mults = pd.Series(1.0, index=range(len(df_oos)))
            if tf in ML_ELIGIBLE:
                sizer = MLSizerV2(cfg, sizer_cfg)
                ok    = sizer.fit(strategy, params, df_is, args.direction, label=label)
                if ok:
                    ml_m = sizer.predict(df_oos, strategy, params).reset_index(drop=True)
                    if tf == "8h":
                        # ATR mults for 8h
                        from run_atr_regime_study import compute_atr_regime, atr_size_mults
                        regime_full = compute_atr_regime(df.copy(), 200, 33.0, 66.0)
                        regime_oos  = regime_full.iloc[is_n:].reset_index(drop=True)
                        atr_m       = atr_size_mults(df_oos, regime_oos, 1.3, 0.7)
                        size_mults  = (ml_m * atr_m.reset_index(drop=True)).clip(0.4, 2.0)
                    else:
                        size_mults = ml_m
                    logger.info(f"  ML AUC={sizer.auc:.3f}")

            results: Dict[str, dict] = {}
            section: List[dict] = []

            # ── MARKET baseline ────────────────────────────────────────────────
            logger.info("  Running MARKET baseline...")
            mkt_stats = simulate_market(df_oos_sig.copy(), sim_cfg, size_mults)
            # Re-run with size mults using dynamic engine
            from src.ml.dynamic_engine import run_dynamic
            res_mkt   = run_dynamic(df_oos_sig.copy(), strat_name, tf, sim_cfg, size_mults, params)
            mkt_stats["metrics"]  = res_mkt.metrics
            mkt_stats["equity"]   = res_mkt.equity
            mkt_stats["drawdown"] = res_mkt.drawdown
            mkt_stats["trades"]   = res_mkt.trades
            mkt_stats["signals_generated"] = int(res_mkt.metrics.get("trade_count",0))
            mkt_stats["signals_filled"]    = mkt_stats["signals_generated"]
            mkt_stats["fill_rate"]         = 1.0

            results["MARKET"] = mkt_stats
            section.append(_m("MARKET", mkt_stats))
            logger.info(f"  MARKET: trades={int(res_mkt.metrics.get('trade_count',0))}  "
                        f"PF={res_mkt.metrics.get('profit_factor',0):.3f}  "
                        f"Sh={res_mkt.metrics.get('sharpe',0):.3f}")

            # ── LIMIT variants ─────────────────────────────────────────────────
            for mode in args.limit_modes:
                for wait in args.max_wait:
                    ver_name = f"{mode}_w{wait}"
                    logger.info(f"  Running {ver_name}...")

                    stats = simulate_limit(
                        df          = df_oos_sig.copy(),
                        sim_cfg     = sim_cfg,
                        size_mults  = size_mults,
                        limit_mode  = mode,
                        max_wait    = wait,
                    )
                    results[ver_name] = stats
                    section.append(_m(ver_name, stats))

                    m = stats["metrics"]
                    logger.info(
                        f"    fill={stats['fill_rate']:.1%}  "
                        f"trades={int(m.get('trade_count',0))}  "
                        f"PF={m.get('profit_factor',np.nan):.3f}  "
                        f"Sh={m.get('sharpe',np.nan):.3f}  "
                        f"skipped={stats['signals_skipped']}"
                    )

            _print_table(section, label, bm_key)

            # Fill rate summary
            print(f"\n  Fill rate summary (what % of signals got filled):")
            for r in section:
                if r["version"] != "MARKET":
                    skipped_pct = r["signals_skipped"] / max(r["signals_generated"],1)
                    print(f"    {r['version']:<20}: fill={r['fill_rate']:.1%}  "
                          f"skipped={skipped_pct:.1%}  "
                          f"trades={int(r['trades'])}")

            # Yearly breakdown for key versions
            print(f"\n  Yearly Returns (%):")
            yr = {}
            key_vers = ["MARKET"] + [f"{args.limit_modes[0]}_w{w}" for w in args.max_wait]
            for name in key_vers:
                stats = results.get(name)
                if stats and not stats.get("trades", pd.DataFrame()).empty:
                    tr = stats["trades"]
                    if "exit_time" in tr.columns and "pnl" in tr.columns:
                        tr2 = tr.copy()
                        tr2["exit_time"] = pd.to_datetime(tr2["exit_time"])
                        yr[name] = tr2.groupby(tr2["exit_time"].dt.year)["pnl"].sum().round(2)
            if yr:
                print(pd.DataFrame(yr).fillna(0).to_string())

            for r in section:
                r.update({"strategy":strat_name,"timeframe":tf})
                all_rows.append(r)

            _plot(results, df_oos, label, out_plots)

    # ── Save ───────────────────────────────────────────────────────────────────
    if all_rows:
        sdf  = pd.DataFrame(all_rows)
        path = out_ranks/"limit_entry_results.csv"
        sdf.to_csv(path, index=False)
        logger.info(f"\nResults → {path}")

    # ── Final verdict ──────────────────────────────────────────────────────────
    if all_rows:
        sdf = pd.DataFrame(all_rows)
        print(f"\n{'='*88}")
        print("LIMIT ENTRY STUDY — FINAL VERDICT")
        print(f"{'='*88}")

        for (strat,tf), grp in sdf.groupby(["strategy","timeframe"]):
            mkt = grp[grp["version"]=="MARKET"]
            if mkt.empty: continue
            mkt = mkt.iloc[0]

            limits = grp[grp["version"]!="MARKET"].copy()
            beats  = limits[limits["sharpe"] > mkt["sharpe"]]
            best   = limits.sort_values("sharpe",ascending=False).iloc[0] if not limits.empty else None

            print(f"\n{strat}/{tf}:")
            print(f"  MARKET:  Sh={mkt['sharpe']:.3f}  PF={mkt['pf']:.3f}  trades={int(mkt['trades'])}  fill=100%")
            if best is not None:
                print(f"  Best limit: {best['version']}  "
                      f"Sh={best['sharpe']:.3f}  PF={best['pf']:.3f}  "
                      f"trades={int(best['trades'])}  fill={best['fill_rate']:.1%}")
                d_sh = best["sharpe"] - mkt["sharpe"]
                if len(beats) > 0 and d_sh > 0.05:
                    print(f"  ★ {len(beats)} limit variants beat MARKET by Sh {d_sh:+.3f}")
                    print(f"    But fill rate is {best['fill_rate']:.0%} — {int(mkt['trades']-best['trades'])} fewer trades")
                elif d_sh > 0:
                    print(f"  ~ Marginal improvement ({d_sh:+.3f} Sh) with {best['fill_rate']:.0%} fill rate")
                else:
                    print(f"  ✗ No limit variant beats MARKET on Sharpe")
                    print(f"    Limits reduce trades without improving risk-adjusted return")

        print(f"\nConclusion:")
        # Simple majority
        total_beat = sum(1 for r in all_rows
                         if r["version"]!="MARKET" and r["sharpe"] >
                         next((x["sharpe"] for x in all_rows
                               if x["strategy"]==r["strategy"] and
                               x["timeframe"]==r["timeframe"] and
                               x["version"]=="MARKET"), 99))
        total_lim  = sum(1 for r in all_rows if r["version"]!="MARKET")
        print(f"  {total_beat}/{total_lim} limit variants beat market baseline on Sharpe")
        avg_fill = np.mean([r["fill_rate"] for r in all_rows if r["version"]!="MARKET"])
        print(f"  Average fill rate across all limit variants: {avg_fill:.1%}")
        if total_beat / max(total_lim,1) > 0.5:
            print("  → Limit entries show consistent improvement — worth adopting")
        else:
            print("  → Market entries remain superior for this breakout strategy")
            print("    (Breakout strategies need immediate execution — waiting for pullback")
            print("     either misses the move or catches reversals)")


if __name__ == "__main__":
    main()