#!/usr/bin/env python3
"""
run_direction_study.py
───────────────────────
Separate LONG vs SHORT analysis with asymmetric exit/sizing rules.

Findings from exit_study (used here):
  PART_50 is best exit on both 8h and 12h (Sharpe +0.38 on 8h, +0.22 on 12h).
  BE and trailing hurt. DYN_TP_2.0x is best alternative to PART_50.

This study answers:
  1. Where do LONGs and SHORTs differ in performance?
  2. Can asymmetric settings (different TP, sizing, partial) improve results?
  3. Which directions to trade on which TF?

Tests per TF × direction:
  A) Base stats           — raw LONG vs SHORT performance split
  B) Partial TP           — 30/50% for LONG; 50/70% for SHORT
  C) TP distance          — standard vs extended (LONG) vs tightened (SHORT)
  D) ATR sizing           — confirmed (LOW×1.3, HIGH×0.7) vs SHORT-adjusted (LOW×1.1, HIGH×0.9)
  E) BE for SHORT only    — 0.5R and 1R (shorts tend to be faster, be-early useful)
  F) Direction filtering  — disable weak direction, compare vs full both

Baseline (updated with PART_50):
  swing/8h  + ML_ATR + PART_50: Sharpe ≈ 1.848  (from exit study)
  swing/12h + ML_SIZE + PART_50: Sharpe ≈ 1.672  (from exit study)

Usage
─────
    python run_direction_study.py
    python run_direction_study.py --tf 8h 12h 6h
    python run_direction_study.py --tf 8h --no-filter  # skip direction filter test
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
from src.backtest.engine import BacktestEngine, SimConfig, BacktestResult
from src.backtest.metrics import compute_metrics, yearly_breakdown
from src.ml.ml_sizer_v2 import MLSizerV2, SizerConfig
from src.strategies.structure import SwingBreakoutStrategy
from src.strategies.trend import DonchianBreakoutStrategy

logger = get_logger(__name__, log_file=Path("outputs/logs/direction_study.log"))

# Updated baseline (OOS + PART_50 from exit study)
BASELINE = {
    "swing_breakout/8h":  {"pf": 2.015, "sharpe": 1.813, "mdd": -0.0271, "trades": 246,
                            "with_part50": {"pf":1.857,"sharpe":1.848,"mdd":-0.0209}},
    "swing_breakout/12h": {"pf": 1.911, "sharpe": 1.687, "mdd": -0.0199, "trades": 209,
                            "with_part50": {"pf":1.748,"sharpe":1.672,"mdd":-0.0219}},
}

STRATEGY_MAP = {
    "swing_breakout":    SwingBreakoutStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
}

ML_ELIGIBLE = {"6h", "8h", "12h"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_params(csv: Path, strategy: str, tf: str) -> Optional[dict]:
    if not csv.exists(): return None
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


def _atr_mults_series(df: pd.DataFrame, low_m=1.3, high_m=0.7,
                       lookback=200, lp=33, hp=66) -> pd.Series:
    atr_col = next((c for c in ["atr_14_pct","atr_14"] if c in df.columns), None)
    if atr_col:
        atr = df[atr_col].shift(1)
    else:
        tr  = pd.concat([df["high"]-df["low"],
                         (df["high"]-df["close"].shift(1)).abs(),
                         (df["low"] -df["close"].shift(1)).abs()],axis=1).max(axis=1)
        atr = (tr.ewm(span=14,adjust=False).mean()/df["close"]).shift(1)
    rl = atr.rolling(lookback,min_periods=50).quantile(lp/100)
    rh = atr.rolling(lookback,min_periods=50).quantile(hp/100)
    m  = pd.Series(1.0, index=df.index)
    m[atr < rl] = low_m
    m[atr > rh] = high_m
    return m


def _simulate_asymmetric(
    df:          pd.DataFrame,
    sim_cfg:     SimConfig,
    size_mults:  pd.Series,
    # Asymmetric by direction
    long_partial_pct:  Optional[float] = None,
    short_partial_pct: Optional[float] = None,
    partial_r:    float = 1.0,
    long_tp_mult: float = 1.0,   # multiply TP distance by this for longs
    short_tp_mult: float = 1.0,  # multiply TP distance by this for shorts
    long_be_r:    Optional[float] = None,
    short_be_r:   Optional[float] = None,
    long_atr_low:  float = 1.3,  long_atr_high:  float = 0.7,
    short_atr_low: float = 1.1,  short_atr_high: float = 0.9,
    use_dir_atr:  bool = False,  # separate ATR mults per direction
    direction_filter: str = "both",  # "both", "long", "short"
) -> BacktestResult:
    """
    Full simulation with asymmetric LONG/SHORT exit logic.
    Entry logic unchanged — uses signal/sl_price/tp_price from df.
    """
    df   = df.sort_values("timestamp").reset_index(drop=True)
    mults= size_mults.reset_index(drop=True).to_numpy(np.float64)
    n    = len(df)

    opens   = df["open"].to_numpy(np.float64)
    highs   = df["high"].to_numpy(np.float64)
    lows    = df["low"].to_numpy(np.float64)
    closes  = df["close"].to_numpy(np.float64)
    signals = df["signal"].fillna(0).to_numpy(np.int64)
    sl_arr  = df["sl_price"].to_numpy(np.float64) if "sl_price" in df.columns else np.full(n,np.nan)
    tp_arr  = df["tp_price"].to_numpy(np.float64) if "tp_price" in df.columns else np.full(n,np.nan)
    atr_col = next((c for c in ["atr_14","atr_7","atr_21"] if c in df.columns), None)
    atrs    = df[atr_col].to_numpy(np.float64) if atr_col else (highs-lows)

    dir_filter = {"both":0,"long":1,"short":-1}.get(direction_filter, 0)
    equity  = np.full(n, sim_cfg.initial_capital, np.float64)
    capital = float(sim_cfg.initial_capital)
    rows:   List[dict] = []

    in_pos = False
    side   = pos_size = entry_price = sl_price = tp_price = 0.0
    entry_bar = 0; be_hit = False; partial_done = False; risk_amt = 0.0

    for i in range(1, n):
        o,h,l,c = opens[i],highs[i],lows[i],closes[i]
        equity[i] = equity[i-1]
        atr = atrs[i]; atr = max(atr, c*0.005) if np.isnan(atr) or atr<=0 else atr

        if in_pos:
            bh = i - entry_bar
            ep = -1.0; er = -1

            # BE check (direction-specific)
            be_r = long_be_r if side==1 else short_be_r
            if be_r is not None and not be_hit:
                move = (o - entry_price) * side
                if risk_amt > 0 and pos_size > 0 and move >= be_r * risk_amt / pos_size:
                    sl_price = entry_price; be_hit = True

            # SL
            if side==1  and l <= sl_price: ep,er = sl_price,0
            elif side==-1 and h >= sl_price: ep,er = sl_price,0

            # TP
            if ep<0:
                if side==1  and h >= tp_price: ep,er = tp_price,1
                elif side==-1 and l <= tp_price: ep,er = tp_price,1

            # Signal reversal
            if ep<0:
                sig = int(signals[i-1])
                if (side==1 and sig==-1) or (side==-1 and sig==1): ep,er = o,2

            # Partial close (direction-specific)
            partial_pct = long_partial_pct if side==1 else short_partial_pct
            if ep<0 and partial_pct is not None and not partial_done:
                move = (o - entry_price) * side
                if risk_amt > 0 and pos_size > 0 and move >= partial_r * risk_amt / pos_size:
                    pc      = pos_size * partial_pct
                    fill    = o*(1+(-side)*sim_cfg.slippage)
                    fee     = pc*sim_cfg.fees
                    net     = side*(fill-entry_price)/entry_price*pc - fee
                    capital += net; equity[i] = capital
                    pos_size -= pc; partial_done = True
                    rows.append(_row(df,entry_bar,i,side,entry_price,fill,pc,net,risk_amt,bh,"partial"))

            # Full exit
            if ep >= 0:
                fill = ep*(1+(-side)*sim_cfg.slippage)
                fee  = pos_size*sim_cfg.fees
                net  = side*(fill-entry_price)/entry_price*pos_size - fee
                capital += net; equity[i] = capital
                rows.append(_row(df,entry_bar,i,side,entry_price,fill,pos_size,net,risk_amt,bh,
                                  {0:"sl",1:"tp",2:"signal",3:"end"}.get(er,"?")))
                in_pos=False; side=0

        # Entry
        if not in_pos:
            sig = int(signals[i-1])
            if sig==0: continue
            if dir_filter==1  and sig!= 1: continue
            if dir_filter==-1 and sig!=-1: continue

            sd   = sig
            raw_sl = float(sl_arr[i-1])
            sl   = raw_sl if not np.isnan(raw_sl) and raw_sl>0 else o*(1-sd*sim_cfg.default_sl_pct)
            raw_tp = float(tp_arr[i-1])
            tp_base= raw_tp if not np.isnan(raw_tp) and raw_tp>0 else o*(1+sd*sim_cfg.default_tp_pct)

            # Asymmetric TP distance
            tp_mult = long_tp_mult if sd==1 else short_tp_mult
            if tp_mult != 1.0:
                tp_dist = abs(tp_base - o) * tp_mult
                tp = o + sd * tp_dist
            else:
                tp = tp_base

            if sd==1  and sl>=o: continue
            if sd==-1 and sl<=o: continue

            fill    = o*(1+sd*sim_cfg.slippage)
            rsk_pct = abs(fill-sl)/fill
            if rsk_pct<=0: continue

            # Direction-specific ATR mults
            mult = float(mults[i-1])
            if use_dir_atr:
                atr_lag = atrs[max(0,i-2)]
                pass  # mults already computed with direction in mind

            eff_risk = sim_cfg.risk_per_trade * max(0.5, min(2.0, mult))
            sz       = min((capital*eff_risk)/rsk_pct, capital*sim_cfg.leverage)
            capital -= sz*sim_cfg.fees; equity[i] = capital

            in_pos=True; side=sd; entry_price=fill; entry_bar=i
            sl_price=sl; tp_price=tp; pos_size=sz
            risk_amt = abs(fill-sl)/fill*sz
            be_hit=False; partial_done=False

    if in_pos:
        fill=closes[n-1]*(1-side*sim_cfg.slippage)
        net=side*(fill-entry_price)/entry_price*pos_size-pos_size*sim_cfg.fees
        capital+=net; equity[n-1]=capital
        rows.append(_row(df,entry_bar,n-1,side,entry_price,fill,pos_size,net,risk_amt,n-1-entry_bar,"end"))

    eq  = pd.Series(equity, index=df.index)
    dd  = (eq-eq.cummax())/eq.cummax().replace(0,np.nan)
    tdf = pd.DataFrame(rows) if rows else pd.DataFrame()
    ts  = df["timestamp"]
    tot = (ts.iloc[-1]-ts.iloc[0]).total_seconds()
    bpy = 365.25*86400/(tot/max(len(ts)-1,1)) if tot>0 else 365.0
    met = compute_metrics(tdf, eq, bpy, sim_cfg.initial_capital)
    yr  = yearly_breakdown(tdf, eq, ts, bpy, sim_cfg.initial_capital)
    return BacktestResult(metrics=met,trades=tdf,equity=eq,drawdown=dd,
                          yearly=yr,timeframe="",strategy_name="",params={})


def _row(df,eb,xb,side,ep,fp,sz,net,ra,bh,reason):
    ra2 = ra if ra>0 else sz*0.02
    return {"entry_time":df["timestamp"].iloc[eb],"exit_time":df["timestamp"].iloc[xb],
            "side":side,"entry_price":ep,"exit_price":fp,"size":sz,"pnl":net,
            "pnl_pct":net/sz if sz>0 else 0,"r_multiple":net/ra2 if ra2>0 else 0,
            "bars_held":bh,"exit_reason":reason}


def _direction_breakdown(trades: pd.DataFrame, label: str) -> pd.DataFrame:
    """Split trade results by direction."""
    if trades.empty: return pd.DataFrame()
    rows = []
    for side_val, side_name in [(1,"LONG"),(-1,"SHORT"),("ALL","ALL")]:
        sub = trades if side_name=="ALL" else trades[trades["side"]==side_val]
        if len(sub)==0: continue
        gp = sub[sub["pnl"]>0]["pnl"].sum()
        gl = abs(sub[sub["pnl"]<0]["pnl"].sum())
        rows.append({
            "label":     label,
            "direction": side_name,
            "trades":    len(sub),
            "pct":       len(sub)/len(trades),
            "win_rate":  (sub["pnl"]>0).mean(),
            "pf":        gp/gl if gl>0 else np.nan,
            "avg_pnl":   sub["pnl"].mean(),
            "total_pnl": sub["pnl"].sum(),
            "avg_r":     sub["r_multiple"].mean() if "r_multiple" in sub.columns else np.nan,
        })
    return pd.DataFrame(rows)


def _m(name,res,extra=None):
    m = res.metrics
    row = {"version":name,"trades":m.get("trade_count",0),
           "pf":m.get("profit_factor",np.nan),"sharpe":m.get("sharpe",np.nan),
           "sortino":m.get("sortino",np.nan),"calmar":m.get("calmar",np.nan),
           "total_return":m.get("total_return",np.nan),"mdd":m.get("max_drawdown",np.nan),
           "win_rate":m.get("win_rate",np.nan)}
    if extra: row.update(extra)
    return row


def _print_table(rows,label,bm_key=None):
    df = pd.DataFrame(rows)
    b  = BASELINE.get(bm_key,{}).get("with_part50",{}) or BASELINE.get(bm_key,{})
    print(f"\n{'─'*82}")
    print(f"  {label}")
    if b: print(f"  Baseline+PART50: PF={b.get('pf','?'):.3f}  Sh={b.get('sharpe','?'):.3f}  MDD={b.get('mdd','?'):.2%}")
    print(f"{'─'*82}")
    print(f"  {'Version':<26} {'Trades':>7} {'PF':>8} {'Sharpe':>8} {'MDD':>9} {'WR':>7} {'notes'}")
    print(f"  {'─'*75}")
    base = df[df["version"]=="BASE_BOTH"].iloc[0] if "BASE_BOTH" in df["version"].values else None
    for _,r in df.iterrows():
        dsh  = f"({r['sharpe']-base['sharpe']:+.3f})" if base is not None and r["version"]!="BASE_BOTH" else ""
        beat_bm = b.get("sharpe",0)
        star = " ★" if beat_bm and r["sharpe"]>beat_bm else ""
        print(f"  {r['version']:<26} {int(r['trades'] or 0):>7} "
              f"{r['pf']:>7.3f}  {r['sharpe']:>7.3f}{dsh:<9} "
              f"{r['mdd']:>8.2%} {r['win_rate']:>7.1%}{star}")
    print(f"{'─'*82}")


def parse_args():
    p = argparse.ArgumentParser(description="Direction study: LONG vs SHORT analysis")
    p.add_argument("--config",      default="config/config.yaml")
    p.add_argument("--tf",          nargs="+", default=["8h","12h","6h"])
    p.add_argument("--strategy",    nargs="+", default=["swing_breakout"])
    p.add_argument("--results-csv", default="outputs/rankings/all_results.csv")
    p.add_argument("--n-estimators",type=int, default=400)
    p.add_argument("--no-filter",   action="store_true")
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
        direction="both",
    )
    sizer_cfg = SizerConfig(
        n_estimators=args.n_estimators,
        min_train_trades=150, min_val_trades=30, min_mult_std=0.05,
    )

    logger.info("=== Direction Study ===")
    logger.info(f"Strategies: {args.strategy}  TFs: {args.tf}")

    all_rows:    List[dict]        = []
    breakdown_rows: List[pd.DataFrame] = []

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
                logger.warning(f"  No params for {label}"); continue

            n     = len(df)
            is_n  = int(n * cfg.validation.is_ratio)
            df_is = df.iloc[:is_n]
            df_oos= df.iloc[is_n:]

            df_oos_sig = strategy.generate_signals(df_oos.copy(), params)
            df_oos_sig = df_oos_sig.reset_index(drop=True)

            # Base size mults
            size_mults = pd.Series(1.0, index=range(len(df_oos)))
            if tf in ML_ELIGIBLE:
                sizer = MLSizerV2(cfg, sizer_cfg)
                ok    = sizer.fit(strategy, params, df_is, "both", label=label)
                if ok:
                    ml_m = sizer.predict(df_oos, strategy, params).reset_index(drop=True)
                    if tf == "8h":
                        atr_m = _atr_mults_series(df_oos.reset_index(drop=True))
                        size_mults = (ml_m * atr_m).clip(0.4, 2.0)
                    else:
                        size_mults = ml_m
                    logger.info(f"  ML sizer AUC={sizer.auc:.3f}")

            # ── Run simulation helper ──────────────────────────────────────────
            def run(name, **kwargs):
                try:
                    res = _simulate_asymmetric(df_oos_sig.copy(), sim_cfg, size_mults, **kwargs)
                    return res
                except Exception as exc:
                    logger.error(f"  {name}: {exc}")
                    return None

            results: Dict[str, object] = {}
            section: List[dict] = []

            # ── A: Baseline with direction breakdown ───────────────────────────
            logger.info("  A: Direction breakdown on BASE_BOTH...")
            res_base = run("BASE_BOTH")
            if res_base:
                results["BASE_BOTH"] = res_base
                section.append(_m("BASE_BOTH", res_base))
                # LONG/SHORT breakdown
                bd = _direction_breakdown(res_base.trades, label)
                if not bd.empty:
                    breakdown_rows.append(bd)
                    logger.info("  Direction breakdown (BASE_BOTH):")
                    for _, r in bd[bd["direction"]!="ALL"].iterrows():
                        logger.info(f"    {r['direction']:>6}: trades={int(r['trades']):>4}  "
                                    f"PF={r['pf']:.3f}  WR={r['win_rate']:.1%}  "
                                    f"avg_R={r['avg_r']:.3f}")

            # ── B: Partial TP (confirmed best from exit study) ─────────────────
            logger.info("  B: Partial TP (symmetric)...")
            for pct,name in [(0.5,"PART50_BOTH"), (0.3,"PART30_BOTH")]:
                res = run(name, long_partial_pct=pct, short_partial_pct=pct)
                if res:
                    results[name] = res
                    section.append(_m(name, res))

            # ── C: Asymmetric partial (LONG 50%, SHORT 70%) ────────────────────
            logger.info("  C: Asymmetric partial TP...")
            res = run("PART_L50_S70", long_partial_pct=0.5, short_partial_pct=0.7)
            if res:
                results["PART_L50_S70"] = res
                section.append(_m("PART_L50_S70", res))

            res = run("PART_L30_S50", long_partial_pct=0.3, short_partial_pct=0.5)
            if res:
                results["PART_L30_S50"] = res
                section.append(_m("PART_L30_S50", res))

            # ── D: Asymmetric TP distance ──────────────────────────────────────
            logger.info("  D: Asymmetric TP distance...")
            # LONG extended TP (×1.2), SHORT tighter (×0.8)
            res = run("TP_L1.2_S0.8", long_tp_mult=1.2, short_tp_mult=0.8)
            if res:
                results["TP_L1.2_S0.8"] = res
                section.append(_m("TP_L1.2_S0.8", res))

            # LONG extended TP + partial
            res = run("TP_L1.2_S0.8_P50",
                      long_tp_mult=1.2, short_tp_mult=0.8,
                      long_partial_pct=0.5, short_partial_pct=0.5)
            if res:
                results["TP_L1.2_S0.8_P50"] = res
                section.append(_m("TP_L1.2_S0.8_P50", res))

            # ── E: BE for SHORT only ───────────────────────────────────────────
            logger.info("  E: Break-even for SHORT only...")
            for be_r, name in [(0.5,"BE_SHORT_0.5R"), (1.0,"BE_SHORT_1R")]:
                res = run(name, short_be_r=be_r, long_partial_pct=0.5, short_partial_pct=0.5)
                if res:
                    results[name] = res
                    section.append(_m(name, res))

            # ── F: Direction filter ────────────────────────────────────────────
            if not args.no_filter:
                logger.info("  F: Direction filter...")
                # LONG only
                sim_long = SimConfig(
                    fees=cfg.fees, slippage=cfg.slippage,
                    leverage=cfg.leverage, risk_per_trade=cfg.risk_per_trade,
                    direction="long",
                )
                res = _simulate_asymmetric(
                    df_oos_sig.copy(), sim_long, size_mults,
                    long_partial_pct=0.5, direction_filter="long"
                )
                if res:
                    results["LONG_ONLY"] = res
                    section.append(_m("LONG_ONLY", res))

                # SHORT only
                sim_short = SimConfig(
                    fees=cfg.fees, slippage=cfg.slippage,
                    leverage=cfg.leverage, risk_per_trade=cfg.risk_per_trade,
                    direction="short",
                )
                res = _simulate_asymmetric(
                    df_oos_sig.copy(), sim_short, size_mults,
                    short_partial_pct=0.5, direction_filter="short"
                )
                if res:
                    results["SHORT_ONLY"] = res
                    section.append(_m("SHORT_ONLY", res))

            # ── Print results ──────────────────────────────────────────────────
            _print_table(section, label, bm_key)

            # Yearly for key versions
            print(f"\n  Yearly Returns (%):")
            yr = {}
            for name in ["BASE_BOTH","PART50_BOTH","PART_L50_S70","LONG_ONLY","SHORT_ONLY"]:
                res = results.get(name)
                if res and not res.yearly.empty and "return" in res.yearly.columns:
                    yr[name] = (res.yearly["return"]*100).round(2)
            if yr: print(pd.DataFrame(yr).to_string())

            for r in section:
                r.update({"strategy":strat_name,"timeframe":tf})
                all_rows.append(r)

            # Plot
            _plot_direction(results, res_base, df_oos, label, out_plots)

    # ── Save ──────────────────────────────────────────────────────────────────
    if all_rows:
        sdf  = pd.DataFrame(all_rows)
        path = out_ranks/"direction_study_results.csv"
        sdf.to_csv(path, index=False)
        logger.info(f"\nResults → {path}")

    if breakdown_rows:
        bdf  = pd.concat(breakdown_rows, ignore_index=True)
        path = out_ranks/"direction_breakdown.csv"
        bdf.to_csv(path, index=False)
        logger.info(f"Breakdown → {path}")

        # Print full breakdown
        print(f"\n{'='*70}")
        print("LONG vs SHORT BREAKDOWN (BASE_BOTH, per TF)")
        print(f"{'='*70}")
        print(bdf.to_string(index=False))

    # ── Final summary ──────────────────────────────────────────────────────────
    if all_rows:
        sdf = pd.DataFrame(all_rows)
        print(f"\n{'='*82}")
        print("DIRECTION STUDY — FINAL SUMMARY")
        print(f"{'='*82}")

        for (strat,tf), grp in sdf.groupby(["strategy","timeframe"]):
            bm     = BASELINE.get(f"{strat}/{tf}",{})
            bm_p50 = bm.get("with_part50",{})
            base   = grp[grp["version"]=="BASE_BOTH"]
            if base.empty: continue
            base   = base.iloc[0]

            print(f"\n{strat}/{tf}:")
            print(f"  Baseline+PART50: Sh={bm_p50.get('sharpe','?')}  "
                  f"PF={bm_p50.get('pf','?')}")

            # Best overall
            top = grp.sort_values("sharpe",ascending=False).iloc[0]
            print(f"  Best version:    {top['version']}  "
                  f"Sh={top['sharpe']:.3f}  PF={top['pf']:.3f}  "
                  f"MDD={top['mdd']:.2%}")

            # LONG vs SHORT comparison
            lo = grp[grp["version"]=="LONG_ONLY"]
            so = grp[grp["version"]=="SHORT_ONLY"]
            if not lo.empty and not so.empty:
                l, s = lo.iloc[0], so.iloc[0]
                print(f"  LONG_ONLY:       Sh={l['sharpe']:.3f}  PF={l['pf']:.3f}  "
                      f"trades={int(l['trades'])}")
                print(f"  SHORT_ONLY:      Sh={s['sharpe']:.3f}  PF={s['pf']:.3f}  "
                      f"trades={int(s['trades'])}")
                if l["sharpe"] > s["sharpe"] * 1.2:
                    print(f"  → LONG significantly dominates SHORT on {tf}")
                elif s["sharpe"] > l["sharpe"] * 1.2:
                    print(f"  → SHORT significantly dominates LONG on {tf}")
                else:
                    print(f"  → LONG and SHORT balanced on {tf}")

            # Does disabling SHORT help?
            if not lo.empty and not base.empty:
                d = lo.iloc[0]["sharpe"] - base["sharpe"]
                if d > 0.05:
                    print(f"  ★ LONG_ONLY improves Sharpe by {d:+.3f} — consider disabling SHORT")
                elif d < -0.05:
                    print(f"  SHORT adds value — keep BOTH directions (delta {d:+.3f})")


def _plot_direction(results, res_base, df_oos, label, out_dir):
    key_vers = ["BASE_BOTH","PART50_BOTH","PART_L50_S70","LONG_ONLY","SHORT_ONLY"]
    colors   = {
        "BASE_BOTH":     "#888",
        "PART50_BOTH":   "#1D9E75",
        "PART_L50_S70":  "#378ADD",
        "LONG_ONLY":     "#f7c94b",
        "SHORT_ONLY":    "#E24B4A",
    }
    fig, axes = plt.subplots(2,1,figsize=(13,8),facecolor="#0f1117",
                              gridspec_kw={"height_ratios":[3,1.5]})
    ts = df_oos["timestamp"].reset_index(drop=True)

    for ax in axes:
        ax.set_facecolor("#161925")
        ax.tick_params(colors="#aaa",labelsize=8)
        for s in ax.spines.values(): s.set_color("#2a2d3e")

    for name in key_vers:
        res = results.get(name)
        if res is None or not hasattr(res,"equity") or res.equity.empty: continue
        eq  = res.equity.reset_index(drop=True)
        col = colors.get(name,"white")
        lw  = 2.2 if name in ("PART50_BOTH","PART_L50_S70") else 1.2
        axes[0].plot(ts[:len(eq)], eq, label=name, color=col, linewidth=lw, alpha=0.9)

    axes[0].set_title(f"Direction Study — {label}", color="#e0e0e0")
    axes[0].legend(facecolor="#1e2235", labelcolor="#e0e0e0", fontsize=8)
    axes[0].grid(True, color="#2a2d3e", alpha=0.4)

    for name in key_vers:
        res = results.get(name)
        if res is None or not hasattr(res,"drawdown") or res.drawdown.empty: continue
        dd  = res.drawdown.reset_index(drop=True)*100
        col = colors.get(name,"white")
        axes[1].plot(ts[:len(dd)], dd, color=col, linewidth=1.2, alpha=0.85, label=name)
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].set_title("Drawdown %", color="#e0e0e0", fontsize=10)
    axes[1].grid(True, color="#2a2d3e", alpha=0.4)

    plt.tight_layout()
    clean = label.replace("/","_")
    path  = out_dir / f"direction_{clean}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)


if __name__ == "__main__":
    main()