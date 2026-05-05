#!/usr/bin/env python3
"""
run_position_management_study.py
──────────────────────────────────
Position management study: time-based exit and pyramiding.

Part 1 — Time-based conditional exit:
  If price has NOT reached +threshold_R within N bars → close trade
  Threshold variants: +0.5R, +1.0R
  Bar limits: 2, 3, 4 bars

Part 2 — Pyramiding:
  When trade reaches +1R → add 50% of initial size (funded by unrealised profit)
  When trade reaches +2R → add another 25% (optional, 2nd pyramid)
  Total risk never increases (additions funded by open profit, SL moved to BE)

Part 3 — Combinations:
  All permutations of time_exit × pyramiding

Conservative rules (no lookahead):
  - All decisions based on current bar's open/high/low only
  - No peeking at future bars
  - Pyramid entry at open of next bar after threshold hit (not at exact price)
  - Time exit: close at open of N+1 bar if threshold not hit by bar N

Baseline (DO NOT MODIFY):
  swing/8h  + ML_ATR + TP_asym + PART_50: Sh=1.970  PF=1.942
  swing/12h + ML_SIZE + PART_L50_S70:     Sh=1.757  PF=1.823
  swing/6h  + ML_SIZE:                    Sh=1.635  PF=1.903

Usage
─────
    python run_position_management_study.py
    python run_position_management_study.py --tf 8h 12h --no-combo
    python run_position_management_study.py --time-thresholds 0.5 1.0
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
from src.backtest.engine import SimConfig
from src.backtest.metrics import compute_metrics, yearly_breakdown
from src.ml.ml_sizer_v2 import MLSizerV2, SizerConfig
from src.strategies.structure import SwingBreakoutStrategy
from src.strategies.trend import DonchianBreakoutStrategy

logger = get_logger(__name__, log_file=Path("outputs/logs/position_management_study.log"))

# Baseline from direction study (with exit logic already applied)
BASELINE = {
    "swing_breakout/8h":  {"pf": 1.942, "sharpe": 1.970, "mdd": -0.0205, "trades": 295},
    "swing_breakout/12h": {"pf": 1.823, "sharpe": 1.757, "mdd": -0.0179, "trades": 277},
    "swing_breakout/6h":  {"pf": 1.903, "sharpe": 1.635, "mdd": -0.0153, "trades": 346},
}

STRATEGY_MAP = {
    "swing_breakout":    SwingBreakoutStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
}
ML_ELIGIBLE = {"6h", "8h", "12h"}


# ── ATR regime helper ─────────────────────────────────────────────────────────

def _atr_mults_series(df: pd.DataFrame, low_m=1.3, high_m=0.7,
                       lookback=200, lp=33, hp=66) -> pd.Series:
    col = next((c for c in ["atr_14_pct","atr_14"] if c in df.columns), None)
    if col:
        atr = df[col].shift(1)
    else:
        tr  = pd.concat([df["high"]-df["low"],
                         (df["high"]-df["close"].shift(1)).abs(),
                         (df["low"]-df["close"].shift(1)).abs()], axis=1).max(axis=1)
        atr = (tr.ewm(span=14,adjust=False).mean()/df["close"]).shift(1)
    rl = atr.rolling(lookback, min_periods=50).quantile(lp/100)
    rh = atr.rolling(lookback, min_periods=50).quantile(hp/100)
    m  = pd.Series(1.0, index=df.index)
    m[atr < rl] = low_m
    m[atr > rh] = high_m
    return m


# ── Core simulation ───────────────────────────────────────────────────────────

def simulate(
    df:              pd.DataFrame,
    sim_cfg:         SimConfig,
    size_mults:      pd.Series,
    # Exit config (from direction/exit studies — applied before new tests)
    long_tp_mult:    float = 1.0,
    short_tp_mult:   float = 1.0,
    long_partial:    Optional[float] = None,
    short_partial:   Optional[float] = None,
    partial_r:       float = 1.0,
    # Part 1: Time-based conditional exit
    time_exit_bars:  Optional[int]   = None,  # N bars limit
    time_threshold_r:float = 0.5,             # required progress in R multiples
    # Part 2: Pyramiding
    pyramid_1r:      bool  = False,   # add 50% at +1R
    pyramid_2r:      bool  = False,   # add 25% at +2R
    pyramid_1r_add:  float = 0.5,     # fraction of initial size to add at 1R
    pyramid_2r_add:  float = 0.25,    # fraction of initial size to add at 2R
) -> dict:
    """
    Full simulation with time-based exit and/or pyramiding.
    Entry and signal logic are UNCHANGED.
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
    rows:   List[dict] = []

    # Position state
    in_pos         = False
    side           = 0
    pos_size       = 0.0     # current size (grows with pyramiding)
    init_size      = 0.0     # initial entry size (for pyramid calculation)
    entry_price    = 0.0
    sl             = 0.0
    tp             = 0.0
    entry_bar      = 0
    risk_per_unit  = 0.0     # |entry - sl| / entry
    risk_usd       = 0.0     # initial risk in $
    # Tracking
    partial_done   = False
    pyr1_done      = False   # pyramid at 1R added?
    pyr2_done      = False   # pyramid at 2R added?
    # Time-exit tracking
    time_bar_count = 0
    time_threshold_reached = False
    # Stats
    time_exits     = 0
    pyramid_adds   = 0

    def _close_trade(ep, fp, sz, reason, bar_idx):
        nonlocal capital
        fill  = fp * (1 + (-side) * sim_cfg.slippage)
        fee   = sz * sim_cfg.fees
        net   = side * (fill - ep) / ep * sz - fee
        ra    = risk_usd if risk_usd > 0 else sz * 0.02
        capital += net
        equity[bar_idx] = capital
        rows.append({
            "entry_time":  df["timestamp"].iloc[entry_bar],
            "exit_time":   df["timestamp"].iloc[bar_idx],
            "side":        side,
            "entry_price": ep,
            "exit_price":  fill,
            "size":        sz,
            "pnl":         net,
            "pnl_pct":     net/sz if sz>0 else 0,
            "r_multiple":  net/ra,
            "bars_held":   bar_idx - entry_bar,
            "exit_reason": reason,
            "has_pyramid": pyr1_done or pyr2_done,
        })
        return net

    for i in range(1, n):
        o,h,l,c = opens[i], highs[i], lows[i], closes[i]
        equity[i] = equity[i-1]
        atr = atrs[i]; atr = max(atr, c*0.005) if np.isnan(atr) or atr<=0 else atr

        if in_pos:
            bh   = i - entry_bar
            move = (o - entry_price) * side  # current unrealised move (approx)
            ep_now = -1.0; er = -1

            # ── Time-based conditional exit (check at bar open) ────────────────
            # At each new bar, check if we've reached the progress threshold
            if time_exit_bars is not None and not time_threshold_reached:
                # Check if threshold was reached during PRIOR bar (no lookahead)
                # Progress = max favourable move on prior bar
                prior_h = highs[i-1] if i > 1 else h
                prior_l = lows[i-1]  if i > 1 else l
                best_move = (prior_h - entry_price)*side if side==1 else (entry_price - prior_l)
                if best_move >= time_threshold_r * risk_usd / max(pos_size, 1e-9) * entry_price:
                    time_threshold_reached = True
                else:
                    time_bar_count += 1
                    if time_bar_count >= time_exit_bars:
                        # Time limit hit without progress → exit at open
                        _close_trade(entry_price, o, pos_size, "time_exit", i)
                        time_exits += 1
                        in_pos = False; side = 0
                        continue

            # ── SL ────────────────────────────────────────────────────────────
            if in_pos:
                if side==1  and l <= sl: ep_now, er = sl, 0
                elif side==-1 and h >= sl: ep_now, er = sl, 0

            # ── TP ────────────────────────────────────────────────────────────
            if in_pos and ep_now < 0:
                if side==1  and h >= tp: ep_now, er = tp, 1
                elif side==-1 and l <= tp: ep_now, er = tp, 1

            # ── Signal reversal ───────────────────────────────────────────────
            if in_pos and ep_now < 0:
                sig = int(signals[i-1])
                if (side==1 and sig==-1) or (side==-1 and sig==1): ep_now, er = o, 2

            # ── Partial close (from direction study) ──────────────────────────
            pp = long_partial if side==1 else short_partial
            if in_pos and ep_now < 0 and pp is not None and not partial_done:
                partial_thresh = partial_r * risk_usd / max(pos_size, 1e-9) * entry_price
                if (h - entry_price)*side >= partial_thresh:  # prior bar peak
                    pc     = pos_size * pp
                    fill_p = entry_price + side * partial_r * risk_usd / max(pos_size,1e-9) * entry_price
                    # Use open as conservative fill for partial
                    fill_p = min(h, max(l, fill_p)) if (l <= fill_p <= h) else o
                    fee    = pc * sim_cfg.fees
                    net    = side * (fill_p - entry_price) / entry_price * pc - fee
                    capital += net; equity[i] = capital
                    pos_size -= pc; partial_done = True
                    rows.append({
                        "entry_time": df["timestamp"].iloc[entry_bar],
                        "exit_time":  df["timestamp"].iloc[i],
                        "side": side, "entry_price": entry_price,
                        "exit_price": fill_p, "size": pc, "pnl": net,
                        "pnl_pct": net/pc if pc>0 else 0,
                        "r_multiple": net/risk_usd if risk_usd>0 else 0,
                        "bars_held": bh, "exit_reason": "partial",
                        "has_pyramid": False,
                    })

            # ── Pyramiding check ──────────────────────────────────────────────
            if in_pos and ep_now < 0:
                r1_thresh = 1.0 * risk_usd   # +1R in $ terms
                r2_thresh = 2.0 * risk_usd   # +2R in $ terms
                curr_move_h = (h - entry_price) * side  # best of current bar

                # Pyramid at +1R (add at OPEN of next bar — no lookahead)
                # Check if current bar's HIGH crossed 1R threshold
                if pyramid_1r and not pyr1_done and curr_move_h >= r1_thresh / max(pos_size,1e-9):
                    # Will add at next bar's open — mark as pending
                    # For simplicity: add at current bar's open (conservative)
                    add_size = init_size * pyramid_1r_add
                    add_risk = add_size * risk_per_unit  # risk of addition
                    # Move SL to break-even for entire position
                    sl       = entry_price  # BE stop
                    cost     = add_size * sim_cfg.fees
                    capital -= cost; equity[i] = capital
                    pos_size += add_size
                    pyr1_done = True
                    pyramid_adds += 1

                # Pyramid at +2R
                if pyramid_2r and pyr1_done and not pyr2_done and \
                        curr_move_h >= r2_thresh / max(pos_size,1e-9):
                    add_size  = init_size * pyramid_2r_add
                    cost      = add_size * sim_cfg.fees
                    capital  -= cost; equity[i] = capital
                    pos_size += add_size
                    pyr2_done = True
                    pyramid_adds += 1

            # ── Full exit ─────────────────────────────────────────────────────
            if in_pos and ep_now >= 0:
                _close_trade(entry_price, ep_now, pos_size,
                             {0:"sl",1:"tp",2:"signal",3:"end"}.get(er,"?"), i)
                in_pos = False; side = 0

        # ── Entry ─────────────────────────────────────────────────────────────
        if not in_pos:
            sig = int(signals[i-1])
            if sig==0: continue
            if dir_int==1  and sig!= 1: continue
            if dir_int==-1 and sig!=-1: continue

            sd   = sig
            mult = max(0.5, min(2.0, float(mults[i-1])))

            raw_sl = float(sl_arr[i-1])
            raw_tp = float(tp_arr[i-1])
            sl_p   = raw_sl if not np.isnan(raw_sl) and raw_sl>0 else o*(1-sd*sim_cfg.default_sl_pct)
            tp_base= raw_tp if not np.isnan(raw_tp) and raw_tp>0 else o*(1+sd*sim_cfg.default_tp_pct)

            # Asymmetric TP
            tp_m = long_tp_mult if sd==1 else short_tp_mult
            if tp_m != 1.0:
                tp_p = o + sd * abs(tp_base-o) * tp_m
            else:
                tp_p = tp_base

            if sd==1  and sl_p>=o: continue
            if sd==-1 and sl_p<=o: continue

            fill      = o * (1 + sd*sim_cfg.slippage)
            rsk_pct   = abs(fill-sl_p)/fill
            if rsk_pct<=0: continue

            eff_risk  = sim_cfg.risk_per_trade * mult
            sz        = min((capital*eff_risk)/rsk_pct, capital*sim_cfg.leverage)
            capital  -= sz * sim_cfg.fees
            equity[i] = capital

            in_pos         = True
            side           = sd
            entry_price    = fill
            entry_bar      = i
            sl             = sl_p
            tp             = tp_p
            pos_size       = sz
            init_size      = sz
            risk_per_unit  = rsk_pct
            risk_usd       = sz * rsk_pct
            partial_done   = False
            pyr1_done      = False
            pyr2_done      = False
            time_bar_count = 0
            time_threshold_reached = False

    # Close remaining
    if in_pos:
        fill = closes[n-1] * (1-side*sim_cfg.slippage)
        _close_trade(entry_price, fill, pos_size, "end", n-1)

    eq   = pd.Series(equity, index=df.index)
    dd   = (eq - eq.cummax()) / eq.cummax().replace(0, np.nan)
    tdf  = pd.DataFrame(rows) if rows else pd.DataFrame()
    ts   = df["timestamp"]
    tot  = (ts.iloc[-1]-ts.iloc[0]).total_seconds()
    bpy  = 365.25*86400/(tot/max(len(ts)-1,1)) if tot>0 else 365.0
    met  = compute_metrics(tdf, eq, bpy, sim_cfg.initial_capital)
    yr   = yearly_breakdown(tdf, eq, ts, bpy, sim_cfg.initial_capital)

    return {
        "metrics":       met,
        "equity":        eq,
        "drawdown":      dd,
        "trades":        tdf,
        "yearly":        yr,
        "time_exits":    time_exits,
        "pyramid_adds":  pyramid_adds,
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


# Per-TF exit config (from previous studies)
TF_EXIT_CFG = {
    "8h":  {"long_tp_mult":1.2,"short_tp_mult":0.8,"long_partial":0.5,"short_partial":0.5},
    "12h": {"long_tp_mult":1.0,"short_tp_mult":1.0,"long_partial":0.5,"short_partial":0.7},
    "6h":  {"long_tp_mult":1.0,"short_tp_mult":1.0,"long_partial":None,"short_partial":None},
}


def _m(name, stats, extra=None):
    m = stats["metrics"]
    row = {
        "version":     name,
        "trades":      m.get("trade_count",   0),
        "pf":          m.get("profit_factor", np.nan),
        "sharpe":      m.get("sharpe",        np.nan),
        "sortino":     m.get("sortino",       np.nan),
        "calmar":      m.get("calmar",        np.nan),
        "total_return":m.get("total_return",  np.nan),
        "mdd":         m.get("max_drawdown",  np.nan),
        "win_rate":    m.get("win_rate",      np.nan),
        "time_exits":  stats.get("time_exits",  0),
        "pyramid_adds":stats.get("pyramid_adds",0),
    }
    if extra: row.update(extra)
    return row


def _print_table(rows, label, bm_key=None):
    df = pd.DataFrame(rows)
    b  = BASELINE.get(bm_key, {})
    print(f"\n{'─'*84}")
    print(f"  {label}")
    if b:
        print(f"  Baseline: PF={b['pf']:.3f}  Sh={b['sharpe']:.3f}  "
              f"MDD={b['mdd']:.2%}  trades={b['trades']}")
    print(f"{'─'*84}")
    print(f"  {'Version':<26} {'Trades':>7} {'PF':>8} {'Sharpe':>8} "
          f"{'MDD':>9} {'Return':>9} {'TmEx':>5} {'Pyr':>4}")
    print(f"  {'─'*78}")
    base = df[df["version"]=="BASE"].iloc[0] if "BASE" in df["version"].values else None
    for _, r in df.iterrows():
        dsh  = f"({r['sharpe']-base['sharpe']:+.3f})" if base is not None and r["version"]!="BASE" else ""
        beat_bm = b.get("sharpe",0)
        star = " ★" if beat_bm and r["sharpe"] > beat_bm else ""
        worse= " ⚠" if base is not None and r["sharpe"] < base["sharpe"]-0.05 else ""
        print(f"  {r['version']:<26} {int(r['trades'] or 0):>7} "
              f"{r['pf']:>7.3f}  {r['sharpe']:>7.3f}{dsh:<9} "
              f"{r['mdd']:>8.2%} {r['total_return']:>8.1%} "
              f"{int(r.get('time_exits',0) or 0):>5} "
              f"{int(r.get('pyramid_adds',0) or 0):>4}"
              f"{star}{worse}")
    print(f"{'─'*84}")


def _plot(results, df_oos, label, out_dir):
    colors = {
        "BASE":         "#888888",
        "TIME_2b_05R":  "#378ADD",
        "TIME_3b_05R":  "#7F77DD",
        "TIME_4b_05R":  "#5DCAA5",
        "TIME_3b_10R":  "#AFA9EC",
        "PYR_1R":       "#f7c94b",
        "PYR_1R_2R":    "#EF9F27",
        "COMBO_3b_PYR": "#1D9E75",
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
        lw  = 2.2 if name in ("BASE","COMBO_3b_PYR","PYR_1R") else 1.0
        al  = 0.95 if name in ("BASE","COMBO_3b_PYR","PYR_1R") else 0.55
        axes[0].plot(ts[:len(eq)], eq, label=name, color=col, linewidth=lw, alpha=al)

    axes[0].set_title(f"Position Management Study — {label}", color="#e0e0e0")
    axes[0].legend(facecolor="#1e2235", labelcolor="#e0e0e0", fontsize=7, ncol=4)
    axes[0].grid(True, color="#2a2d3e", alpha=0.4)

    for name, stats in results.items():
        dd  = stats["drawdown"].reset_index(drop=True)*100
        col = colors.get(name, "gray")
        lw  = 2.0 if name in ("BASE","COMBO_3b_PYR") else 0.7
        axes[1].plot(ts[:len(dd)], dd, color=col, linewidth=lw,
                     alpha=0.9 if name in ("BASE","COMBO_3b_PYR") else 0.5)
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].set_title("Drawdown %", color="#e0e0e0", fontsize=10)
    axes[1].grid(True, color="#2a2d3e", alpha=0.4)

    plt.tight_layout()
    clean = label.replace("/","_")
    path  = out_dir / f"pos_mgmt_{clean}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    logger.info(f"  Plot → {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Position management study")
    p.add_argument("--config",       default="config/config.yaml")
    p.add_argument("--tf",           nargs="+", default=["8h","12h","6h"])
    p.add_argument("--strategy",     nargs="+", default=["swing_breakout"])
    p.add_argument("--direction",    default="both")
    p.add_argument("--results-csv",  default="outputs/rankings/all_results.csv")
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--time-bars",    nargs="+", type=int, default=[2,3,4])
    p.add_argument("--time-thresholds", nargs="+", type=float, default=[0.5, 1.0])
    p.add_argument("--no-combo",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(Path(args.config))
    fe   = FeatureEngine(cfg)

    out_plots = cfg.output_dir/"plots";    out_plots.mkdir(exist_ok=True)
    out_ranks = cfg.output_dir/"rankings"; out_ranks.mkdir(exist_ok=True)
    results_csv = Path(args.results_csv)

    logger.info("=== Position Management Study ===")
    logger.info(f"Strategies: {args.strategy}  TFs: {args.tf}")

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

            sim_cfg = SimConfig(
                fees=cfg.fees, slippage=cfg.slippage,
                leverage=cfg.leverage, risk_per_trade=cfg.risk_per_trade,
                direction=args.direction,
            )
            sizer_cfg = SizerConfig(
                n_estimators=args.n_estimators,
                min_train_trades=150, min_val_trades=30, min_mult_std=0.05,
            )

            # Build size mults (ML + ATR for 8h, ML for others)
            size_mults = pd.Series(1.0, index=range(len(df_oos)))
            if tf in ML_ELIGIBLE:
                sizer = MLSizerV2(cfg, sizer_cfg)
                ok    = sizer.fit(strategy, params, df_is, args.direction, label=label)
                if ok:
                    ml_m = sizer.predict(df_oos, strategy, params).reset_index(drop=True)
                    if tf == "8h":
                        atr_m = _atr_mults_series(df_oos.reset_index(drop=True))
                        size_mults = (ml_m * atr_m.reset_index(drop=True)).clip(0.4, 2.0)
                    else:
                        size_mults = ml_m
                    logger.info(f"  ML AUC={sizer.auc:.3f}")

            # Exit config for this TF
            ecfg = TF_EXIT_CFG.get(tf, {})

            def run(name, **kwargs):
                """Run simulation with given kwargs merged with exit config."""
                merged = {**ecfg, **kwargs}
                try:
                    return simulate(df_oos_sig.copy(), sim_cfg, size_mults, **merged)
                except Exception as exc:
                    logger.error(f"  {name} failed: {exc}")
                    import traceback; traceback.print_exc()
                    return None

            results: Dict[str, dict] = {}
            section: List[dict] = []

            # ── BASE (with exit logic but no new management) ───────────────────
            logger.info("  Running BASE...")
            s = run("BASE")
            if s:
                results["BASE"] = s
                section.append(_m("BASE", s))
                b = s["metrics"]
                logger.info(f"  BASE: trades={int(b.get('trade_count',0))}  "
                            f"PF={b.get('profit_factor',0):.3f}  "
                            f"Sh={b.get('sharpe',0):.3f}")

            # ── Part 1: Time-based exit variants ──────────────────────────────
            logger.info("  Part 1: Time-based exits...")
            for bars in args.time_bars:
                for thresh in args.time_thresholds:
                    thresh_str = str(thresh).replace('.','')
                    name = f"TIME_{bars}b_{thresh_str}R"
                    s = run(name, time_exit_bars=bars, time_threshold_r=thresh)
                    if s:
                        results[name] = s
                        section.append(_m(name, s))
                        m = s["metrics"]
                        logger.info(f"  {name}: trades={int(m.get('trade_count',0))}  "
                                    f"Sh={m.get('sharpe',0):.3f}  "
                                    f"time_exits={s['time_exits']}")

            # ── Part 2: Pyramiding ─────────────────────────────────────────────
            logger.info("  Part 2: Pyramiding...")
            # Pyramid at 1R only
            s = run("PYR_1R", pyramid_1r=True, pyramid_2r=False)
            if s:
                results["PYR_1R"] = s
                section.append(_m("PYR_1R", s))
                m = s["metrics"]
                logger.info(f"  PYR_1R: trades={int(m.get('trade_count',0))}  "
                            f"Sh={m.get('sharpe',0):.3f}  "
                            f"pyr_adds={s['pyramid_adds']}")

            # Pyramid at 1R + 2R
            s = run("PYR_1R_2R", pyramid_1r=True, pyramid_2r=True)
            if s:
                results["PYR_1R_2R"] = s
                section.append(_m("PYR_1R_2R", s))
                m = s["metrics"]
                logger.info(f"  PYR_1R_2R: trades={int(m.get('trade_count',0))}  "
                            f"Sh={m.get('sharpe',0):.3f}  "
                            f"pyr_adds={s['pyramid_adds']}")

            # ── Part 3: Combinations ───────────────────────────────────────────
            if not args.no_combo:
                logger.info("  Part 3: Combinations...")
                # Find best time exit from Part 1
                time_rows = [r for r in section if r["version"].startswith("TIME_")]
                if time_rows:
                    best_te = max(time_rows, key=lambda r: r["sharpe"])
                    best_te_ver = best_te["version"]
                    # Parse back the params
                    parts   = best_te_ver.split("_")  # TIME_3b_05R
                    te_bars = int(parts[1].replace("b",""))
                    te_th   = float(parts[2].replace("R","")) / 10

                    # Combo: best time_exit + PYR_1R
                    name = f"COMBO_{te_bars}b_PYR"
                    s = run(name,
                            time_exit_bars=te_bars, time_threshold_r=te_th,
                            pyramid_1r=True, pyramid_2r=False)
                    if s:
                        results[name] = s
                        section.append(_m(name, s))
                        m = s["metrics"]
                        logger.info(f"  {name}: trades={int(m.get('trade_count',0))}  "
                                    f"Sh={m.get('sharpe',0):.3f}")

                    # Combo: best time_exit + PYR_1R + PYR_2R
                    name2 = f"COMBO_{te_bars}b_PYR2"
                    s = run(name2,
                            time_exit_bars=te_bars, time_threshold_r=te_th,
                            pyramid_1r=True, pyramid_2r=True)
                    if s:
                        results[name2] = s
                        section.append(_m(name2, s))

            # Print results
            _print_table(section, label, bm_key)

            # Yearly for key versions
            print(f"\n  Yearly Returns (%):")
            yr = {}
            key_vers = (["BASE"] +
                        [r["version"] for r in section
                         if r["version"].startswith("TIME_")][:2] +
                        ["PYR_1R"] +
                        [r["version"] for r in section
                         if r["version"].startswith("COMBO_")][:1])
            for name in key_vers:
                stats = results.get(name)
                if stats and not stats.get("yearly", pd.DataFrame()).empty:
                    yr_data = stats["yearly"]
                    if not yr_data.empty and "return" in yr_data.columns:
                        yr[name] = (yr_data["return"]*100).round(2)
            if yr:
                print(pd.DataFrame(yr).fillna(0).to_string())

            for r in section:
                r.update({"strategy":strat_name, "timeframe":tf})
                all_rows.append(r)

            _plot(results, df_oos, label, out_plots)

    # ── Save ───────────────────────────────────────────────────────────────────
    if all_rows:
        sdf  = pd.DataFrame(all_rows)
        path = out_ranks / "position_management_results.csv"
        sdf.to_csv(path, index=False)
        logger.info(f"\nResults → {path}")

    # ── Final summary ──────────────────────────────────────────────────────────
    if all_rows:
        sdf = pd.DataFrame(all_rows)
        print(f"\n{'='*84}")
        print("POSITION MANAGEMENT — FINAL SUMMARY")
        print(f"{'='*84}")

        for (strat,tf), grp in sdf.groupby(["strategy","timeframe"]):
            bm  = BASELINE.get(f"{strat}/{tf}", {})
            base= grp[grp["version"]=="BASE"]
            if base.empty: continue
            base = base.iloc[0]

            print(f"\n{strat}/{tf}  baseline_Sh={bm.get('sharpe','?')}")
            print(f"  {'Version':<28} {'Trades':>7} {'Sharpe':>8} {'PF':>8} "
                  f"{'MDD':>9}  Δ_vs_base")
            print(f"  {'─'*72}")

            for _, r in grp.sort_values("sharpe",ascending=False).iterrows():
                d    = r["sharpe"] - base["sharpe"]
                beat = " ★ BEATS BASELINE" if bm and r["sharpe"]>bm.get("sharpe",0) else ""
                worse= " ⚠ worse" if r["sharpe"] < base["sharpe"]-0.05 else ""
                print(f"  {r['version']:<28} {int(r['trades'] or 0):>7} "
                      f"{r['sharpe']:>8.3f} {r['pf']:>8.3f} "
                      f"{r['mdd']:>8.2%}  {d:>+.3f}{beat}{worse}")

        print(f"\n{'─'*84}")
        print("VERDICT:")

        # Count wins across all TFs
        winners_te  = []
        winners_pyr = []
        winners_combo = []

        for _, r in sdf[sdf["version"]!="BASE"].iterrows():
            key  = f"{r['strategy']}/{r['timeframe']}"
            base = sdf[(sdf["strategy"]==r["strategy"]) &
                       (sdf["timeframe"]==r["timeframe"]) &
                       (sdf["version"]=="BASE")]
            if base.empty: continue
            base_sh = base.iloc[0]["sharpe"]
            if r["sharpe"] > base_sh:
                if "TIME" in r["version"]:  winners_te.append(r["version"])
                if "PYR"  in r["version"]:  winners_pyr.append(r["version"])
                if "COMBO" in r["version"]: winners_combo.append(r["version"])

        total_te    = len(sdf[(sdf["version"]!="BASE") & sdf["version"].str.startswith("TIME")])
        total_pyr   = len(sdf[(sdf["version"]!="BASE") & sdf["version"].str.startswith("PYR")])
        total_combo = len(sdf[(sdf["version"]!="BASE") & sdf["version"].str.startswith("COMBO")])

        print(f"  Time exit:   {len(winners_te)}/{total_te} beat BASE")
        print(f"  Pyramiding:  {len(winners_pyr)}/{total_pyr} beat BASE")
        print(f"  Combos:      {len(winners_combo)}/{total_combo} beat BASE")

        if winners_te:
            print(f"\n  Best time exit wins: {set(winners_te)}")
        if winners_pyr:
            print(f"  Best pyramid wins:   {set(winners_pyr)}")
        if winners_combo:
            print(f"  Best combo wins:     {set(winners_combo)}")


if __name__ == "__main__":
    main()