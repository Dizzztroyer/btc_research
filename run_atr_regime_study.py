#!/usr/bin/env python3
"""
run_atr_regime_study.py
────────────────────────
Volatility / ATR regime study for BTC trading strategies.

Splits market into three ATR percentile regimes:
    LOW    : ATR < 33rd percentile  → low volatility, calm market
    MEDIUM : ATR 33rd–66th pct      → normal conditions
    HIGH   : ATR > 66th percentile  → high volatility, big moves

Tests:
    BASE          — original strategy, fixed size
    ML_SIZE       — ML position sizing (benchmark)
    ATR_SIZE      — size modifier based on ATR regime:
                    LOW ×0.7, MEDIUM ×1.0, HIGH ×1.2
    ML_ATR        — ML_SIZE × ATR regime modifier
    ATR_FILTER_H  — trade only in HIGH ATR regime (breakouts need volatility)
    ATR_FILTER_M  — trade only in MEDIUM+HIGH ATR regime

Baseline (DO NOT MODIFY):
    swing_breakout / 8h  + ML_SIZE  : PF=1.948  Sharpe=1.772
    swing_breakout / 12h + ML_VOL   : PF=2.018  Sharpe=1.789

Usage
─────
    python run_atr_regime_study.py
    python run_atr_regime_study.py --tf 8h 12h --strategy swing_breakout
    python run_atr_regime_study.py --low-mult 0.5 --high-mult 1.4
    python run_atr_regime_study.py --no-filter  # skip ATR filter tests
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
from src.ml.ml_sizer_v2 import MLSizerV2, SizerConfig
from src.ml.dynamic_engine import run_dynamic
from src.strategies.trend import DonchianBreakoutStrategy
from src.strategies.structure import SwingBreakoutStrategy

logger = get_logger(__name__, log_file=Path("outputs/logs/atr_regime_study.log"))

BASELINE = {
    "swing_breakout/8h+ML":  {"pf": 1.948, "sharpe": 1.772, "mdd": -0.0272, "trades": 246},
    "swing_breakout/12h+ML": {"pf": 2.018, "sharpe": 1.789, "mdd": -0.0186, "trades": 209},
}
STRATEGY_MAP = {
    "swing_breakout":    SwingBreakoutStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
}
ML_ELIGIBLE = {"6h", "8h", "12h"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ATR Regime Study")
    p.add_argument("--config",       default="config/config.yaml")
    p.add_argument("--tf",           nargs="+", default=["8h","12h","6h"])
    p.add_argument("--strategy",     nargs="+",
                   default=["swing_breakout","donchian_breakout"])
    p.add_argument("--direction",    default="both")
    p.add_argument("--results-csv",  default="outputs/rankings/all_results.csv")
    p.add_argument("--low-pct",      type=float, default=33.0)
    p.add_argument("--high-pct",     type=float, default=66.0)
    p.add_argument("--lookback",     type=int,   default=200,
                   help="Rolling window for ATR percentile")
    p.add_argument("--low-mult",     type=float, default=0.7)
    p.add_argument("--high-mult",    type=float, default=1.2)
    p.add_argument("--no-filter",    action="store_true")
    p.add_argument("--n-estimators", type=int,   default=400)
    return p.parse_args()


def compute_atr_regime(
    df:        pd.DataFrame,
    lookback:  int,
    low_pct:   float,
    high_pct:  float,
    atr_col:   str = "atr_14_pct",
) -> pd.Series:
    """
    Classify each bar into ATR regime using rolling percentile.
    Uses PRIOR bar ATR (shift 1) — no lookahead.
    Returns: Series of strings ['low','medium','high']
    """
    if atr_col not in df.columns:
        # Fallback: compute ATR from OHLC
        atr_col = "_atr_raw"
        tr  = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"]  - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df[atr_col] = tr.ewm(span=14, adjust=False).mean() / df["close"]

    atr      = df[atr_col].shift(1)
    roll_low = atr.rolling(lookback, min_periods=50).quantile(low_pct  / 100)
    roll_hgh = atr.rolling(lookback, min_periods=50).quantile(high_pct / 100)

    regime = pd.Series("medium", index=df.index)
    regime[atr < roll_low] = "low"
    regime[atr > roll_hgh] = "high"
    return regime


def atr_size_mults(
    df:        pd.DataFrame,
    regime:    pd.Series,
    low_mult:  float,
    high_mult: float,
) -> pd.Series:
    # Reset both to positional index to avoid alignment issues
    reg = regime.reset_index(drop=True)
    mults = pd.Series(1.0, index=range(len(reg)))
    mults[reg == "low"]  = low_mult
    mults[reg == "high"] = high_mult
    return mults


def atr_filter_signals(
    df_sig: pd.DataFrame,
    regime: pd.Series,
    keep:   List[str],
) -> pd.DataFrame:
    df_out  = df_sig.copy().reset_index(drop=True)
    reg     = regime.reset_index(drop=True)
    bad     = ~reg.isin(keep)
    sig_bad = (df_out["signal"] != 0) & bad
    df_out.loc[sig_bad, "signal"]   = 0
    df_out.loc[sig_bad, "sl_price"] = np.nan
    df_out.loc[sig_bad, "tp_price"] = np.nan
    pct = sig_bad.sum() / max((df_sig["signal"] != 0).sum(), 1)
    logger.info(f"  ATR filter ({keep}): removed {sig_bad.sum()} signals ({pct:.1%})")
    return df_out


def regime_breakdown(trades_df: pd.DataFrame, regime: pd.Series,
                     df_full: pd.DataFrame) -> pd.DataFrame:
    """Analyse trade performance by ATR regime."""
    if trades_df.empty or "entry_time" not in trades_df.columns:
        return pd.DataFrame()
    trades = trades_df.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    ts = pd.to_datetime(df_full["timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)
    ts_to_regime = dict(zip(ts, regime))
    trades["regime"] = trades["entry_time"].map(ts_to_regime).fillna("unknown")

    rows = []
    for reg in ["low","medium","high","ALL"]:
        sub = trades if reg == "ALL" else trades[trades["regime"] == reg]
        if len(sub) == 0:
            continue
        gp = sub[sub["pnl"]>0]["pnl"].sum()
        gl = abs(sub[sub["pnl"]<0]["pnl"].sum())
        rows.append({
            "regime":    reg,
            "trades":    len(sub),
            "win_rate":  (sub["pnl"]>0).mean(),
            "pf":        gp/gl if gl>0 else np.nan,
            "avg_pnl":   sub["pnl"].mean(),
            "total_pnl": sub["pnl"].sum(),
        })
    return pd.DataFrame(rows)


def _load_params(csv: Path, strategy: str, tf: str) -> Optional[dict]:
    if not csv.exists(): return None
    df   = pd.read_csv(csv)
    mask = (df["strategy"]==strategy) & (df["timeframe"]==tf)
    sub  = df[mask]
    if sub.empty: return None
    best   = sub.sort_values("robustness", ascending=False).iloc[0]
    p_cols = [c for c in best.index if c.startswith("p_") and not pd.isna(best[c])]
    out    = {}
    for c in p_cols:
        v = best[c]; k = c[2:]
        out[k] = int(v) if isinstance(v, float) and v == int(v) else v
    return out


def _m(name: str, res, extra: dict = None) -> dict:
    m = res.metrics
    row = {
        "version":      name,
        "trades":       m.get("trade_count",   0),
        "pf":           m.get("profit_factor", np.nan),
        "sharpe":       m.get("sharpe",        np.nan),
        "sortino":      m.get("sortino",       np.nan),
        "calmar":       m.get("calmar",        np.nan),
        "total_return": m.get("total_return",  np.nan),
        "mdd":          m.get("max_drawdown",  np.nan),
        "win_rate":     m.get("win_rate",      np.nan),
    }
    if extra: row.update(extra)
    return row


def _print_table(rows: List[dict], label: str, bm_key: str = None) -> None:
    df = pd.DataFrame(rows)
    b  = BASELINE.get(bm_key, {})
    print(f"\n{'─'*75}")
    print(f"  {label}")
    if b: print(f"  Baseline: PF={b['pf']:.3f}  Sh={b['sharpe']:.3f}  trades={b['trades']}")
    print(f"{'─'*75}")
    print(f"{'Version':<22} {'Trades':>7} {'PF':>8} {'Sharpe':>8} {'MDD':>9} {'Return':>9}")
    print(f"{'─'*75}")
    base = df[df["version"]=="BASE"].iloc[0] if "BASE" in df["version"].values else None
    for _, r in df.iterrows():
        dsh  = f"({r['sharpe']-base['sharpe']:+.3f})" if base is not None and r["version"]!="BASE" else ""
        beat = " ★" if b and r["sharpe"]>b.get("sharpe",0) and r["trades"]>=b.get("trades",0)*0.9 else ""
        print(f"{r['version']:<22} {int(r['trades'] or 0):>7} "
              f"{r['pf']:>7.3f}  {r['sharpe']:>7.3f}{dsh:<9} "
              f"{r['mdd']:>8.2%} {r['total_return']:>8.1%}{beat}")
    print(f"{'─'*75}")


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

    logger.info("=== ATR Regime Study ===")
    logger.info(f"Regimes: LOW<{args.low_pct}pct  MEDIUM  HIGH>{args.high_pct}pct")
    logger.info(f"Sizing: LOW×{args.low_mult}  MED×1.0  HIGH×{args.high_mult}")

    all_rows:     List[dict] = []
    breakdown_rows: List[pd.DataFrame] = []

    for strat_name in args.strategy:
        if strat_name not in STRATEGY_MAP: continue
        strategy = STRATEGY_MAP[strat_name]()

        for tf in args.tf:
            label  = f"{strat_name}/{tf}"
            bm_key = f"{strat_name}/{tf}+ML"
            logger.info(f"\n{'='*55}\n{label}")

            try:
                df = fe.load(tf)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            except Exception as exc:
                logger.error(f"  Feature load: {exc}"); continue

            params = _load_params(results_csv, strat_name, tf)
            if params is None:
                logger.warning(f"  No params"); continue

            n     = len(df)
            is_n  = int(n * cfg.validation.is_ratio)
            df_is = df.iloc[:is_n]
            df_oos= df.iloc[is_n:]

            # Compute ATR regime on full df, apply to OOS
            regime_full = compute_atr_regime(
                df.copy(), args.lookback, args.low_pct, args.high_pct
            )
            regime_oos  = regime_full.iloc[is_n:].reset_index(drop=True)

            reg_counts = regime_oos.value_counts(normalize=True)
            logger.info(f"  OOS regime distribution: "
                        f"low={reg_counts.get('low',0):.1%}  "
                        f"med={reg_counts.get('medium',0):.1%}  "
                        f"high={reg_counts.get('high',0):.1%}")

            df_oos_sig = strategy.generate_signals(df_oos.copy(), params)
            df_oos_sig = df_oos_sig.reset_index(drop=True)
            results: Dict[str, object] = {}

            # BASE
            res_base = BacktestEngine(sim_cfg).run(df_oos_sig.copy(), strat_name, tf, params)
            results["BASE"] = res_base

            # Regime breakdown on BASE trades
            bd = regime_breakdown(res_base.trades, regime_oos, df_oos)
            if not bd.empty:
                bd["label"] = label
                breakdown_rows.append(bd)
                logger.info("  ATR regime breakdown (BASE):")
                for _, r in bd[bd["regime"]!="ALL"].iterrows():
                    logger.info(f"    {r['regime']:>8}: trades={int(r['trades']):>4}  "
                                f"PF={r['pf']:.3f}  WR={r['win_rate']:.1%}")

            # ATR_SIZE — no skipping, just resize
            atr_mults = atr_size_mults(df_oos, regime_oos, args.low_mult, args.high_mult)
            results["ATR_SIZE"] = run_dynamic(
                df_oos_sig.copy(), strat_name, tf, sim_cfg,
                atr_mults.reset_index(drop=True), params
            )

            # ATR_FILTER_H — only high regime
            if not args.no_filter:
                df_filt_h = atr_filter_signals(df_oos_sig.copy(), regime_oos, ["high"])
                results["ATR_FILT_HIGH"] = BacktestEngine(sim_cfg).run(
                    df_filt_h, strat_name, tf, params
                )
                # MEDIUM+HIGH
                df_filt_mh = atr_filter_signals(df_oos_sig.copy(), regime_oos, ["medium","high"])
                results["ATR_FILT_MH"] = BacktestEngine(sim_cfg).run(
                    df_filt_mh, strat_name, tf, params
                )

            # ML_SIZE + ML_ATR
            ml_mults   = pd.Series(1.0, index=range(len(df_oos)))
            ml_trained = False

            if tf in ML_ELIGIBLE:
                sizer = MLSizerV2(cfg, sizer_cfg)
                ok    = sizer.fit(strategy, params, df_is, args.direction, label=label)
                if ok:
                    ml_mults   = sizer.predict(df_oos, strategy, params).reset_index(drop=True)
                    ml_trained = True
                    results["ML_SIZE"] = run_dynamic(
                        df_oos_sig.copy(), strat_name, tf, sim_cfg, ml_mults, params
                    )
                    # ML × ATR
                    combined = (ml_mults * atr_mults.reset_index(drop=True)).clip(0.4, 2.0)
                    results["ML_ATR"] = run_dynamic(
                        df_oos_sig.copy(), strat_name, tf, sim_cfg, combined, params
                    )

            rows = [_m(k, v, {"strategy":strat_name,"timeframe":tf}) for k,v in results.items()]
            _print_table(rows, label, bm_key)

            # Yearly
            print(f"\n  Yearly Returns (%):")
            yr = {}
            for name in ["BASE","ATR_SIZE","ML_SIZE","ML_ATR"]:
                res = results.get(name)
                if res and not res.yearly.empty and "return" in res.yearly.columns:
                    yr[name] = (res.yearly["return"]*100).round(2)
            if yr: print(pd.DataFrame(yr).to_string())

            for r in rows:
                all_rows.append(r)

    # Save
    if all_rows:
        sdf  = pd.DataFrame(all_rows)
        path = out_ranks / "atr_regime_results.csv"
        sdf.to_csv(path, index=False)
        logger.info(f"\nResults → {path}")

    if breakdown_rows:
        bdf  = pd.concat(breakdown_rows, ignore_index=True)
        path = out_ranks / "atr_regime_breakdown.csv"
        bdf.to_csv(path, index=False)
        logger.info(f"Breakdown → {path}")

    # Summary
    if all_rows:
        sdf = pd.DataFrame(all_rows)
        print(f"\n{'='*75}")
        print("ATR REGIME STUDY — FINAL vs BASELINE")
        print(f"{'='*75}")
        for (st,tf), grp in sdf.groupby(["strategy","timeframe"]):
            bm = BASELINE.get(f"{st}/{tf}+ML", {})
            base = grp[grp["version"]=="BASE"]
            if base.empty: continue
            print(f"\n{st}/{tf}  baseline_Sh={bm.get('sharpe','?')}")
            for _, r in grp.sort_values("sharpe",ascending=False).iterrows():
                d = r["sharpe"] - base.iloc[0]["sharpe"]
                beat = " ★ BEATS BASELINE" if bm and r["sharpe"]>bm.get("sharpe",0) else ""
                print(f"  {r['version']:<22}: PF={r['pf']:.3f}  "
                      f"Sh={r['sharpe']:.3f}  trades={int(r['trades'])}  "
                      f"Δ={d:+.3f}{beat}")


if __name__ == "__main__":
    main()