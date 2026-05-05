#!/usr/bin/env python3
"""
run_exit_study.py
──────────────────
Exit logic and position management study.

Tests improvements to exit/management without touching entry logic.

Versions tested:
    BASE         — original SL/TP, no modification
    BE_{x}R      — break-even at x×R (0.5R, 1R, 1.5R)
    PARTIAL_{x}  — partial close x% at 1R, rest to TP/SL
    TRAIL_{x}    — trailing stop x×ATR
    TIME_{n}     — time-based exit after n bars if TP not hit
    DYN_TP_{k}   — dynamic TP = k×ATR (k=2,3,4,5)
    COMBO        — best combination (BE_1R + TRAIL_2ATR)

Baseline (DO NOT MODIFY entry logic):
    swing/8h  + ML_ATR:  PF=2.015  Sharpe=1.813  MDD=-2.71%
    swing/12h + ML_SIZE: PF=1.911  Sharpe=1.687  MDD=-1.99%

Usage
─────
    python run_exit_study.py
    python run_exit_study.py --tf 8h 12h --strategy swing_breakout
    python run_exit_study.py --no-partial --no-trail  # only BE and time
    python run_exit_study.py --be-levels 0.5 1.0 1.5
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

logger = get_logger(__name__, log_file=Path("outputs/logs/exit_study.log"))

BASELINE = {
    "swing_breakout/8h":  {"pf": 2.015, "sharpe": 1.813, "mdd": -0.0271, "trades": 246},
    "swing_breakout/12h": {"pf": 1.911, "sharpe": 1.687, "mdd": -0.0199, "trades": 209},
}

STRATEGY_MAP = {
    "swing_breakout":    SwingBreakoutStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
}

ML_ELIGIBLE = {"6h", "8h", "12h"}


# ── ATR regime (for ML_ATR on 8h) ────────────────────────────────────────────

def _atr_mults(df: pd.DataFrame, low_mult=1.3, high_mult=0.7,
               lookback=200, low_pct=33, high_pct=66) -> pd.Series:
    atr_col = next((c for c in ["atr_14_pct","atr_14"] if c in df.columns), None)
    if atr_col:
        atr = df[atr_col].shift(1)
    else:
        tr  = pd.concat([df["high"]-df["low"],
                         (df["high"]-df["close"].shift(1)).abs(),
                         (df["low"] -df["close"].shift(1)).abs()], axis=1).max(axis=1)
        atr = (tr.ewm(span=14,adjust=False).mean()/df["close"]).shift(1)
    rl = atr.rolling(lookback, min_periods=50).quantile(low_pct/100)
    rh = atr.rolling(lookback, min_periods=50).quantile(high_pct/100)
    mult = pd.Series(1.0, index=df.index)
    mult[atr < rl] = low_mult
    mult[atr > rh] = high_mult
    return mult


def _load_params(csv: Path, strategy: str, tf: str) -> Optional[dict]:
    if not csv.exists(): return None
    df   = pd.read_csv(csv)
    mask = (df["strategy"]==strategy)&(df["timeframe"]==tf)
    sub  = df[mask]
    if sub.empty: return None
    best   = sub.sort_values("robustness", ascending=False).iloc[0]
    p_cols = [c for c in best.index if c.startswith("p_") and not pd.isna(best[c])]
    out    = {}
    for c in p_cols:
        v=best[c]; k=c[2:]
        out[k] = int(v) if isinstance(v,float) and v==int(v) else v
    return out


# ── Core simulation with exit logic ──────────────────────────────────────────

_EXIT = {0:"sl",1:"tp",2:"be",3:"trail",4:"partial",5:"time",6:"signal",7:"end"}


def simulate(
    df:           pd.DataFrame,
    sim_cfg:      SimConfig,
    size_mults:   pd.Series,
    # Exit config
    be_r:         Optional[float] = None,   # break-even at X×R
    partial_r:    float = 1.0,              # partial close trigger (in R)
    partial_pct:  Optional[float] = None,   # fraction to close (e.g. 0.5)
    trail_atr:    Optional[float] = None,   # trailing stop × ATR
    max_bars:     Optional[int]   = None,   # time-based exit
    dyn_tp_k:     Optional[float] = None,   # dynamic TP = k × ATR (overrides signal TP)
) -> "BacktestResult":
    """
    Full bar-by-bar simulation with configurable exit logic.
    Entry logic is unchanged — uses signal/sl_price/tp_price from df.
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

    dir_map = {"both":0,"long":1,"short":-1}
    dir_int = dir_map.get(sim_cfg.direction, 0)

    equity  = np.full(n, sim_cfg.initial_capital, np.float64)
    capital = float(sim_cfg.initial_capital)
    rows: List[dict] = []

    in_pos = False
    side   = pos_size = entry_price = sl_price = tp_price = 0.0
    entry_bar = 0
    trail_sl  = 0.0
    be_hit    = False
    partial_done = False
    partial_size = 0.0
    risk_amt  = 0.0

    for i in range(1, n):
        o,h,l,c = opens[i],highs[i],lows[i],closes[i]
        equity[i] = equity[i-1]

        atr = atrs[i]
        if np.isnan(atr) or atr<=0: atr = max(h-l, c*0.005)

        if in_pos:
            bars_held = i - entry_bar
            ep = -1.0; er = -1

            # ── Trailing stop update ───────────────────────────────────────────
            if trail_atr is not None:
                dist = trail_atr * atr
                if side == 1:
                    new_trail = o - dist
                    if new_trail > trail_sl: trail_sl = new_trail
                else:
                    new_trail = o + dist
                    if new_trail < trail_sl: trail_sl = new_trail

            # ── Check exits ───────────────────────────────────────────────────
            # 1. Stop loss
            if side==1  and l <= sl_price: ep,er = sl_price,0
            elif side==-1 and h >= sl_price: ep,er = sl_price,0

            # 2. Trailing stop
            if ep<0 and trail_atr is not None:
                if side==1  and l <= trail_sl: ep,er = max(trail_sl, sl_price),3
                elif side==-1 and h >= trail_sl: ep,er = min(trail_sl, sl_price),3

            # 3. Take profit
            if ep<0:
                if side==1  and h >= tp_price: ep,er = tp_price,1
                elif side==-1 and l <= tp_price: ep,er = tp_price,1

            # 4. Time-based exit
            if ep<0 and max_bars is not None and bars_held >= max_bars:
                ep,er = o,5

            # 5. Signal reversal — only exit on opposite signal
            if ep<0:
                sig = int(signals[i-1])
                if (side==1 and sig==-1) or (side==-1 and sig==1):
                    ep,er = o,6

            # ── Partial close check (before full exit) ────────────────────────
            if ep<0 and partial_pct is not None and not partial_done:
                move = (o - entry_price) * side
                if move >= partial_r * risk_amt / pos_size:
                    partial_close = pos_size * partial_pct
                    fill     = o * (1.0 + (-side)*sim_cfg.slippage)
                    fee      = partial_close * sim_cfg.fees
                    raw      = side * (fill - entry_price)/entry_price * partial_close
                    net      = raw - fee
                    capital += net
                    equity[i]= capital
                    pos_size -= partial_close
                    partial_done = True
                    partial_size = partial_close
                    rows.append(_trade_row(df, entry_bar, i, side, entry_price,
                                           fill, partial_close, net, risk_amt, bars_held, "partial"))

            # ── Break-even update ─────────────────────────────────────────────
            if ep<0 and be_r is not None and not be_hit:
                move = (o - entry_price) * side
                if move >= be_r * risk_amt / pos_size:
                    sl_price = entry_price
                    be_hit   = True

            # ── Full exit ─────────────────────────────────────────────────────
            if ep >= 0:
                fill    = ep * (1.0 + (-side)*sim_cfg.slippage)
                fee     = pos_size * sim_cfg.fees
                raw     = side * (fill - entry_price)/entry_price * pos_size
                net     = raw - fee
                capital += net
                equity[i]= capital
                rows.append(_trade_row(df, entry_bar, i, side, entry_price,
                                       fill, pos_size, net, risk_amt, bars_held,
                                       _EXIT.get(er,"?")))
                in_pos = False; side = 0

        # ── Entry ─────────────────────────────────────────────────────────────
        if not in_pos:
            sig = int(signals[i-1])
            if sig==0: continue
            if dir_int==1  and sig!= 1: continue
            if dir_int==-1 and sig!=-1: continue

            mult = float(mults[i-1])
            mult = max(0.5, min(2.0, mult))

            sd   = sig
            raw_sl = float(sl_arr[i-1])
            sl = raw_sl if not np.isnan(raw_sl) and raw_sl>0 else o*(1-sd*sim_cfg.default_sl_pct)
            raw_tp = float(tp_arr[i-1])

            # Dynamic TP override
            if dyn_tp_k is not None:
                tp = o + sd * dyn_tp_k * atr
            else:
                tp = raw_tp if not np.isnan(raw_tp) and raw_tp>0 else o*(1+sd*sim_cfg.default_tp_pct)

            if sd==1  and sl>=o: continue
            if sd==-1 and sl<=o: continue

            fill     = o*(1+sd*sim_cfg.slippage)
            risk_pct = abs(fill-sl)/fill
            if risk_pct<=0: continue

            eff_risk  = sim_cfg.risk_per_trade * mult
            size      = min((capital*eff_risk)/risk_pct, capital*sim_cfg.leverage)
            capital  -= size * sim_cfg.fees
            equity[i] = capital

            in_pos  = True
            side    = sd
            entry_price = fill
            entry_bar   = i
            sl_price    = sl
            tp_price    = tp
            pos_size    = size
            risk_amt    = abs(fill-sl)/fill * size
            trail_sl    = sl if trail_atr else 0.0
            be_hit      = False
            partial_done= False
            partial_size= 0.0

    # Close remaining
    if in_pos:
        fill = closes[n-1]*(1-side*sim_cfg.slippage)
        net  = side*(fill-entry_price)/entry_price*pos_size - pos_size*sim_cfg.fees
        capital += net; equity[n-1] = capital
        rows.append(_trade_row(df,entry_bar,n-1,side,entry_price,fill,pos_size,net,risk_amt,n-1-entry_bar,"end"))

    eq  = pd.Series(equity, index=df.index)
    dd  = (eq - eq.cummax()) / eq.cummax().replace(0, np.nan)
    tdf = pd.DataFrame(rows) if rows else pd.DataFrame()
    ts  = df["timestamp"]
    tot = (ts.iloc[-1]-ts.iloc[0]).total_seconds()
    bpy = 365.25*86400/(tot/max(len(ts)-1,1)) if tot>0 else 365.0

    met = compute_metrics(tdf, eq, bpy, sim_cfg.initial_capital)
    yr  = yearly_breakdown(tdf, eq, ts, bpy, sim_cfg.initial_capital)
    return BacktestResult(metrics=met, trades=tdf, equity=eq, drawdown=dd,
                          yearly=yr, timeframe="", strategy_name="", params={})


def _trade_row(df, entry_bar, exit_bar, side, ep, fp, sz, net, ra, bh, reason) -> dict:
    ra2 = ra if ra > 0 else sz * 0.02
    return {
        "entry_time":  df["timestamp"].iloc[entry_bar],
        "exit_time":   df["timestamp"].iloc[exit_bar],
        "side":        side,
        "entry_price": ep,
        "exit_price":  fp,
        "size":        sz,
        "pnl":         net,
        "pnl_pct":     net/sz if sz>0 else 0,
        "r_multiple":  net/ra2 if ra2>0 else 0,
        "bars_held":   bh,
        "exit_reason": reason,
    }


# ── Parse args ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exit logic study")
    p.add_argument("--config",       default="config/config.yaml")
    p.add_argument("--tf",           nargs="+", default=["8h","12h"])
    p.add_argument("--strategy",     nargs="+", default=["swing_breakout"])
    p.add_argument("--direction",    default="both")
    p.add_argument("--results-csv",  default="outputs/rankings/all_results.csv")
    p.add_argument("--n-estimators", type=int, default=400)
    # Break-even
    p.add_argument("--be-levels",    nargs="+", type=float, default=[0.5, 1.0, 1.5])
    # Partial
    p.add_argument("--no-partial",   action="store_true")
    p.add_argument("--partial-splits", nargs="+", type=float, default=[0.3, 0.5],
                   help="Fraction to close at 1R")
    # Trailing
    p.add_argument("--no-trail",     action="store_true")
    p.add_argument("--trail-levels", nargs="+", type=float, default=[1.5, 2.0, 3.0])
    # Time
    p.add_argument("--no-time",      action="store_true")
    # Dynamic TP
    p.add_argument("--no-dyn-tp",    action="store_true")
    p.add_argument("--dyn-tp-k",     nargs="+", type=float, default=[2.0, 3.0, 4.0, 5.0])
    return p.parse_args()


def _m(name: str, res) -> dict:
    m = res.metrics
    return {
        "version":     name,
        "trades":      m.get("trade_count",   0),
        "pf":          m.get("profit_factor", np.nan),
        "sharpe":      m.get("sharpe",        np.nan),
        "sortino":     m.get("sortino",       np.nan),
        "calmar":      m.get("calmar",        np.nan),
        "total_return":m.get("total_return",  np.nan),
        "mdd":         m.get("max_drawdown",  np.nan),
        "win_rate":    m.get("win_rate",      np.nan),
        "expectancy":  m.get("expectancy",    np.nan),
    }


def _print_table(rows: List[dict], label: str, bm_key: str) -> None:
    df = pd.DataFrame(rows)
    b  = BASELINE.get(bm_key, {})
    print(f"\n{'─'*80}")
    print(f"  {label}")
    if b:
        print(f"  Baseline: PF={b['pf']:.3f}  Sh={b['sharpe']:.3f}  "
              f"MDD={b['mdd']:.2%}  trades={b['trades']}")
    print(f"{'─'*80}")
    print(f"{'Version':<22} {'Trades':>7} {'PF':>8} {'Sharpe':>8} "
          f"{'Sortino':>8} {'MDD':>9} {'Return':>9} {'WR':>7}")
    print(f"{'─'*80}")
    base = df[df["version"]=="BASE"].iloc[0] if "BASE" in df["version"].values else None
    for _, r in df.iterrows():
        dsh  = f"({r['sharpe']-base['sharpe']:+.3f})" if base is not None and r["version"]!="BASE" else ""
        beat = " ★" if b and r["sharpe"]>b.get("sharpe",0) and r["trades"]>=b.get("trades",0)*0.85 else ""
        print(f"{r['version']:<22} {int(r['trades'] or 0):>7} "
              f"{r['pf']:>7.3f}  {r['sharpe']:>7.3f}{dsh:<9} "
              f"{r['sortino']:>7.3f}  {r['mdd']:>8.2%} {r['total_return']:>8.1%} "
              f"{r['win_rate']:>7.1%}{beat}")
    print(f"{'─'*80}")


def _plot(results: Dict[str, object], df_oos: pd.DataFrame,
          label: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor="#0f1117",
                              gridspec_kw={"height_ratios":[3,1.5]})
    ts = df_oos["timestamp"].reset_index(drop=True)
    cmap = plt.cm.get_cmap("tab10", max(len(results), 2))
    colors = {"BASE":"#888"}

    for ax in axes:
        ax.set_facecolor("#161925")
        ax.tick_params(colors="#aaa", labelsize=8)
        for s in ax.spines.values(): s.set_color("#2a2d3e")

    for idx, (name, res) in enumerate(results.items()):
        if not hasattr(res,"equity") or res.equity.empty: continue
        eq = res.equity.reset_index(drop=True)
        col= colors.get(name, cmap(idx/max(len(results),1)))
        lw = 2.0 if name in ("BASE","COMBO") else 1.0
        alpha = 0.9 if name in ("BASE","COMBO") else 0.65
        axes[0].plot(ts[:len(eq)], eq, label=name, color=col, linewidth=lw, alpha=alpha)

    axes[0].set_title(f"Exit Study — {label}", color="#e0e0e0")
    axes[0].legend(facecolor="#1e2235", labelcolor="#e0e0e0", fontsize=7, ncol=3)
    axes[0].grid(True, color="#2a2d3e", alpha=0.4)

    for idx, (name, res) in enumerate(results.items()):
        if not hasattr(res,"drawdown") or res.drawdown.empty: continue
        dd  = res.drawdown.reset_index(drop=True)*100
        col = colors.get(name, cmap(idx/max(len(results),1)))
        lw  = 2.0 if name in ("BASE","COMBO") else 0.8
        axes[1].plot(ts[:len(dd)], dd, color=col, linewidth=lw, alpha=0.8, label=name)
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].set_title("Drawdown %", color="#e0e0e0", fontsize=10)
    axes[1].grid(True, color="#2a2d3e", alpha=0.4)

    plt.tight_layout()
    clean = label.replace("/","_")
    path  = out_dir / f"exit_study_{clean}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    logger.info(f"  Plot → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
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

    logger.info("=== Exit Study ===")
    logger.info(f"Strategies: {args.strategy}  TFs: {args.tf}")

    all_rows:  List[dict] = []
    best_rows: List[dict] = []

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

            # Generate signals (entry logic unchanged)
            df_oos_sig = strategy.generate_signals(df_oos.copy(), params)
            df_oos_sig = df_oos_sig.reset_index(drop=True)

            # Size multipliers (ML + ATR for 8h, ML only for 12h)
            size_mults = pd.Series(1.0, index=range(len(df_oos)))

            if tf in ML_ELIGIBLE:
                sizer = MLSizerV2(cfg, sizer_cfg)
                ok    = sizer.fit(strategy, params, df_is, args.direction, label=label)
                if ok:
                    ml_m = sizer.predict(df_oos, strategy, params).reset_index(drop=True)
                    if tf == "8h":  # Add ATR regime
                        atr_m = _atr_mults(df_oos.reset_index(drop=True)).reset_index(drop=True)
                        size_mults = (ml_m * atr_m).clip(0.4, 2.0)
                        logger.info(f"  Using ML_ATR sizing (AUC={sizer.auc:.3f})")
                    else:
                        size_mults = ml_m
                        logger.info(f"  Using ML_SIZE sizing (AUC={sizer.auc:.3f})")

            # Helper to run one config
            def run_cfg(name, **kwargs) -> Optional[dict]:
                try:
                    res = simulate(df_oos_sig.copy(), sim_cfg, size_mults, **kwargs)
                    return res
                except Exception as exc:
                    logger.error(f"  {name} failed: {exc}")
                    return None

            results: Dict[str, object] = {}
            section_rows: List[dict] = []

            # ── BASE ──────────────────────────────────────────────────────────
            res_base = run_cfg("BASE")
            if res_base:
                results["BASE"] = res_base
                section_rows.append(_m("BASE", res_base))

            # ── A: Break-even ──────────────────────────────────────────────────
            logger.info("  Testing break-even levels...")
            for be in args.be_levels:
                name = f"BE_{be}R"
                res  = run_cfg(name, be_r=be)
                if res:
                    results[name] = res
                    section_rows.append(_m(name, res))

            # ── B: Partial close ───────────────────────────────────────────────
            if not args.no_partial:
                logger.info("  Testing partial close...")
                for pct in args.partial_splits:
                    name = f"PART_{int(pct*100)}"
                    res  = run_cfg(name, partial_pct=pct, partial_r=1.0)
                    if res:
                        results[name] = res
                        section_rows.append(_m(name, res))

            # ── C: Trailing stop ───────────────────────────────────────────────
            if not args.no_trail:
                logger.info("  Testing trailing stops...")
                for tl in args.trail_levels:
                    name = f"TRAIL_{tl}ATR"
                    res  = run_cfg(name, trail_atr=tl)
                    if res:
                        results[name] = res
                        section_rows.append(_m(name, res))

            # ── D: Time-based exit ─────────────────────────────────────────────
            if not args.no_time:
                logger.info("  Testing time-based exits...")
                time_map = {"8h":[10,15,20], "12h":[8,10,15]}.get(tf, [10,15])
                for nb in time_map:
                    name = f"TIME_{nb}b"
                    res  = run_cfg(name, max_bars=nb)
                    if res:
                        results[name] = res
                        section_rows.append(_m(name, res))

            # ── E: Dynamic TP ──────────────────────────────────────────────────
            if not args.no_dyn_tp:
                logger.info("  Testing dynamic TP (k×ATR)...")
                for k in args.dyn_tp_k:
                    name = f"DYN_TP_{k}x"
                    res  = run_cfg(name, dyn_tp_k=k)
                    if res:
                        results[name] = res
                        section_rows.append(_m(name, res))

            # ── F: Combo (best from above) ─────────────────────────────────────
            logger.info("  Testing combo (BE_1R + TRAIL_2ATR)...")
            res_combo = run_cfg("COMBO", be_r=1.0, trail_atr=2.0)
            if res_combo:
                results["COMBO"] = res_combo
                section_rows.append(_m("COMBO", res_combo))

            logger.info("  Testing combo2 (PART_50 + TRAIL_2ATR)...")
            res_c2 = run_cfg("COMBO2", partial_pct=0.5, partial_r=1.0, trail_atr=2.0)
            if res_c2:
                results["COMBO2"] = res_c2
                section_rows.append(_m("COMBO2", res_c2))

            # ── Print results ──────────────────────────────────────────────────
            _print_table(section_rows, label, bm_key)

            # Yearly for key versions
            print(f"\n  Yearly Returns (%) — key versions:")
            yr_data = {}
            key_vers = ["BASE","BE_1.0R","TRAIL_2.0ATR","DYN_TP_3.0x","COMBO","COMBO2"]
            for name in key_vers:
                res = results.get(name)
                if res and not res.yearly.empty and "return" in res.yearly.columns:
                    yr_data[name] = (res.yearly["return"]*100).round(2)
            if yr_data:
                print(pd.DataFrame(yr_data).to_string())

            # Best version
            best_df = pd.DataFrame(section_rows)
            best_df = best_df[best_df["trades"] >= (BASELINE.get(bm_key,{}).get("trades",0)*0.85)]
            if not best_df.empty and res_base:
                top = best_df.sort_values("sharpe", ascending=False).iloc[0]
                b_sh = res_base.metrics.get("sharpe",0)
                d_sh = top["sharpe"] - b_sh
                logger.info(f"  Best: {top['version']}  Sh={top['sharpe']:.3f}  ({d_sh:+.3f} vs BASE)")

            # Save
            for r in section_rows:
                r.update({"strategy":strat_name,"timeframe":tf})
                all_rows.append(r)

            _plot(results, df_oos, label, out_plots)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    if all_rows:
        sdf  = pd.DataFrame(all_rows)
        path = out_ranks/"exit_study_results.csv"
        sdf.to_csv(path, index=False)
        logger.info(f"\nResults → {path}")

    # ── Final summary ──────────────────────────────────────────────────────────
    if all_rows:
        sdf = pd.DataFrame(all_rows)
        print(f"\n{'='*80}")
        print("EXIT STUDY — FINAL SUMMARY vs BASELINE")
        print(f"{'='*80}")

        for (strat,tf), grp in sdf.groupby(["strategy","timeframe"]):
            bm  = BASELINE.get(f"{strat}/{tf}", {})
            base= grp[grp["version"]=="BASE"]
            if base.empty: continue
            base = base.iloc[0]

            print(f"\n{strat}/{tf}  baseline_Sh={bm.get('sharpe','?')}  baseline_PF={bm.get('pf','?')}")
            print(f"  {'Version':<22} {'Trades':>7} {'PF':>8} {'Sharpe':>8} {'MDD':>9}  verdict")
            print(f"  {'─'*65}")

            grp_valid = grp[grp["trades"]>=bm.get("trades",0)*0.85].copy()
            for _, r in grp_valid.sort_values("sharpe",ascending=False).iterrows():
                d   = r["sharpe"] - base["sharpe"]
                vs_bm = " ★ BEATS BASELINE" if bm and r["sharpe"]>bm.get("sharpe",0) else ""
                worse = " ⚠ worse than BASE" if r["sharpe"] < base["sharpe"]-0.05 else ""
                print(f"  {r['version']:<22} {int(r['trades'] or 0):>7} "
                      f"{r['pf']:>7.3f}  {r['sharpe']:>7.3f}  {r['mdd']:>8.2%} "
                      f" Δ={d:>+.3f}{vs_bm}{worse}")

        # Section verdicts
        print(f"\n{'─'*80}")
        print("SECTION VERDICTS:")
        sections = {
            "Break-even": [r for r in all_rows if r["version"].startswith("BE_")],
            "Partial":    [r for r in all_rows if r["version"].startswith("PART_")],
            "Trailing":   [r for r in all_rows if r["version"].startswith("TRAIL_")],
            "Time exit":  [r for r in all_rows if r["version"].startswith("TIME_")],
            "Dynamic TP": [r for r in all_rows if r["version"].startswith("DYN_TP_")],
            "Combo":      [r for r in all_rows if r["version"].startswith("COMBO")],
        }
        base_sh_map = {}
        for r in all_rows:
            if r["version"]=="BASE":
                key = f"{r['strategy']}/{r['timeframe']}"
                base_sh_map[key] = r["sharpe"]

        for sec_name, sec_rows in sections.items():
            if not sec_rows: continue
            beats = sum(1 for r in sec_rows
                        if r["sharpe"] > base_sh_map.get(f"{r['strategy']}/{r['timeframe']}",99))
            total = len(sec_rows)
            best  = max(sec_rows, key=lambda r: r["sharpe"])
            print(f"  {sec_name:<14}: {beats}/{total} beat BASE  "
                  f"best={best['version']} Sh={best['sharpe']:.3f}")


if __name__ == "__main__":
    main()