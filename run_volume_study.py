#!/usr/bin/env python3
"""
run_volume_study.py
────────────────────
Volume-based strategy enhancement study.

Tests four versions:
    BASE         — original strategy, fixed size
    ML_SIZE      — ML position sizing (rank-normalized XGBoost)
    VOL_CONFIRM  — entry only if vol_zscore > threshold (size unchanged)
    ML_VOL       — ML sizing × volume size modifier

Volume size modifier:
    vol_zscore > high_thresh  → size × 1.2  (strong volume = larger bet)
    vol_zscore < low_thresh   → size × 0.8  (weak volume = smaller bet)
    otherwise                 → size × 1.0  (neutral)

Key principle: VOL_CONFIRM may reduce trade count (acceptable if PF improves
significantly). ML_VOL never reduces trade count below BASE.

Comparison vs benchmark:
    swing_breakout / 8h  + ML: PF=1.849, Sharpe=1.762
    swing_breakout / 12h + ML: PF=1.890, Sharpe=1.736

Usage
─────
    python run_volume_study.py
    python run_volume_study.py --tf 8h 12h --strategy swing_breakout
    python run_volume_study.py --vol-thresh 1.0 1.5 2.0  # sweep thresholds
    python run_volume_study.py --no-confirm  # skip VOL_CONFIRM, only ML_VOL
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

logger = get_logger(__name__, log_file=Path("outputs/logs/volume_study.log"))

BENCHMARK = {
    "swing_breakout/8h+ML":  {"pf": 1.849, "sharpe": 1.762, "mdd": -0.1321, "trades": 246},
    "swing_breakout/12h+ML": {"pf": 1.890, "sharpe": 1.736, "mdd": -0.0851, "trades": 209},
}

STRATEGY_MAP = {
    "swing_breakout":    SwingBreakoutStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
}

ML_ELIGIBLE = {"6h", "8h", "12h"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Volume confirmation study")
    p.add_argument("--config",      default="config/config.yaml")
    p.add_argument("--tf",          nargs="+", default=["8h", "12h"])
    p.add_argument("--strategy",    nargs="+", default=["swing_breakout"])
    p.add_argument("--direction",   default="both")
    p.add_argument("--results-csv", default="outputs/rankings/all_results.csv")
    p.add_argument("--vol-thresh",  nargs="+", type=float, default=[1.0],
                   help="vol_zscore threshold(s) for VOL_CONFIRM entry filter")
    p.add_argument("--vol-high",    type=float, default=1.5,
                   help="vol_zscore above this → size ×1.2 (ML_VOL)")
    p.add_argument("--vol-low",     type=float, default=-0.5,
                   help="vol_zscore below this → size ×0.8 (ML_VOL)")
    p.add_argument("--size-boost",  type=float, default=1.2,
                   help="Size multiplier on high volume")
    p.add_argument("--size-reduce", type=float, default=0.8,
                   help="Size multiplier on low volume")
    p.add_argument("--no-confirm",  action="store_true",
                   help="Skip VOL_CONFIRM (binary filter), run only ML_VOL")
    p.add_argument("--n-estimators",type=int, default=500)
    return p.parse_args()


def _load_params(csv: Path, strategy: str, tf: str) -> Optional[dict]:
    if not csv.exists():
        return None
    df   = pd.read_csv(csv)
    mask = (df["strategy"] == strategy) & (df["timeframe"] == tf)
    sub  = df[mask]
    if sub.empty:
        return None
    best   = sub.sort_values("robustness", ascending=False).iloc[0]
    p_cols = [c for c in best.index if c.startswith("p_") and not pd.isna(best[c])]
    out    = {}
    for c in p_cols:
        v = best[c]; k = c[2:]
        out[k] = int(v) if isinstance(v, float) and v == int(v) else v
    return out


def _metrics(name: str, res) -> dict:
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
    }


def _vol_confirm_signals(
    df_sig:    pd.DataFrame,
    vol_col:   str,
    threshold: float,
) -> pd.DataFrame:
    """
    Zero out signals where vol_zscore < threshold.
    Returns modified copy of df_sig.
    Trades are filtered — acceptable ONLY if PF improvement justifies it.
    """
    df_out = df_sig.copy()
    if vol_col not in df_out.columns:
        logger.warning(f"  {vol_col} not in DataFrame — VOL_CONFIRM skipped")
        return df_out

    # Use prior bar volume (no lookahead)
    vol = df_out[vol_col].shift(1)
    low_vol_mask = (df_out["signal"] != 0) & (vol < threshold)
    df_out.loc[low_vol_mask, "signal"]   = 0
    df_out.loc[low_vol_mask, "sl_price"] = np.nan
    df_out.loc[low_vol_mask, "tp_price"] = np.nan

    n_filtered = low_vol_mask.sum()
    n_signals  = (df_sig["signal"] != 0).sum()
    logger.info(
        f"  VOL_CONFIRM (thresh={threshold}): filtered {n_filtered}/{n_signals} signals "
        f"({n_filtered/n_signals:.1%})"
    )
    return df_out


def _vol_size_mults(
    df:          pd.DataFrame,
    vol_col:     str,
    high_thresh: float,
    low_thresh:  float,
    size_boost:  float,
    size_reduce: float,
) -> pd.Series:
    """
    Volume-based size multiplier (no trade skipping).
    Uses prior bar volume.
    """
    mults = pd.Series(1.0, index=df.index)
    if vol_col not in df.columns:
        logger.warning(f"  {vol_col} not found — volume sizing skipped")
        return mults

    vol = df[vol_col].shift(1)
    mults[vol > high_thresh] = size_boost
    mults[vol < low_thresh]  = size_reduce
    high_pct = (vol > high_thresh).mean()
    low_pct  = (vol < low_thresh).mean()
    logger.info(
        f"  Volume mults: high={high_pct:.1%} (×{size_boost}) "
        f"low={low_pct:.1%} (×{size_reduce}) "
        f"neutral={1-high_pct-low_pct:.1%} (×1.0)"
    )
    return mults


def _print_results(rows: List[dict], label: str, benchmark_key: str) -> None:
    df = pd.DataFrame(rows)
    b  = BENCHMARK.get(benchmark_key, {})

    print(f"\n{'─'*78}")
    print(f"  {label}")
    if b:
        print(f"  Benchmark ({benchmark_key}): PF={b['pf']:.3f}  Sharpe={b['sharpe']:.3f}  Trades={b['trades']}")
    print(f"{'─'*78}")
    print(f"{'Version':<18} {'Trades':>7} {'PF':>8} {'Sharpe':>8} {'Sortino':>8} "
          f"{'Return':>9} {'MDD':>9}")
    print(f"{'─'*78}")
    base = df[df["version"]=="BASE"].iloc[0] if "BASE" in df["version"].values else None
    for _, r in df.iterrows():
        dpf = f"({r['pf']-base['pf']:+.3f})" if base is not None and r["version"]!="BASE" else ""
        dsh = f"({r['sharpe']-base['sharpe']:+.3f})" if base is not None and r["version"]!="BASE" else ""
        beat = " ★" if b and r["sharpe"] > b.get("sharpe", 0) else ""
        print(
            f"{r['version']:<18} "
            f"{int(r['trades'] or 0):>7} "
            f"{r['pf']:>7.3f}{dpf:<9} "
            f"{r['sharpe']:>7.3f}{dsh:<9} "
            f"{r['total_return']:>8.1%} "
            f"{r['mdd']:>8.2%}{beat}"
        )
    print(f"{'─'*78}")


def _plot_comparison(
    results: Dict[str, object],
    df_oos:  pd.DataFrame,
    label:   str,
    out_dir: Path,
) -> None:
    colors = {
        "BASE":        "#888888",
        "ML_SIZE":     "#1D9E75",
        "VOL_CONFIRM": "#378ADD",
        "ML_VOL":      "#f7c94b",
    }
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor="#0f1117",
                              gridspec_kw={"height_ratios": [3, 1.5]})
    ts = df_oos["timestamp"].reset_index(drop=True)

    for ax in axes:
        ax.set_facecolor("#161925")
        ax.tick_params(colors="#aaa", labelsize=9)
        for s in ax.spines.values():
            s.set_color("#2a2d3e")

    for name, res in results.items():
        if res.equity.empty: continue
        eq = res.equity.reset_index(drop=True)
        lw = 2.0 if name in ("ML_VOL","ML_SIZE") else 1.0
        axes[0].plot(ts[:len(eq)], eq, label=name,
                     color=colors.get(name,"white"), linewidth=lw, alpha=0.9)
    axes[0].set_title(f"Equity — {label}", color="#e0e0e0")
    axes[0].set_ylabel("Value ($)", color="#aaa")
    axes[0].legend(facecolor="#1e2235", labelcolor="#e0e0e0", fontsize=9)
    axes[0].grid(True, color="#2a2d3e", alpha=0.4)

    for name, res in results.items():
        if res.drawdown.empty: continue
        dd = res.drawdown.reset_index(drop=True) * 100
        axes[1].plot(ts[:len(dd)], dd, label=name,
                     color=colors.get(name,"white"),
                     linewidth=1.5 if name in ("ML_VOL","ML_SIZE") else 0.8, alpha=0.85)
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].set_title("Drawdown (%)", color="#e0e0e0", fontsize=10)
    axes[1].legend(facecolor="#1e2235", labelcolor="#e0e0e0", fontsize=8)
    axes[1].grid(True, color="#2a2d3e", alpha=0.4)

    plt.tight_layout()
    clean = label.replace("/","_")
    path  = out_dir / f"volume_study_{clean}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    logger.info(f"  Plot → {path}")


def main() -> None:
    args = parse_args()
    cfg  = load_config(Path(args.config))
    fe   = FeatureEngine(cfg)

    out_plots = cfg.output_dir / "plots";    out_plots.mkdir(exist_ok=True)
    out_ranks = cfg.output_dir / "rankings"; out_ranks.mkdir(exist_ok=True)
    results_csv = Path(args.results_csv)

    sim_cfg = SimConfig(
        fees=cfg.fees, slippage=cfg.slippage,
        leverage=cfg.leverage, risk_per_trade=cfg.risk_per_trade,
        direction=args.direction,
    )
    sizer_cfg = SizerConfig(
        n_estimators     = args.n_estimators,
        min_train_trades = 150,
        min_val_trades   = 30,
        min_mult_std     = 0.05,
    )

    logger.info("=== Volume Confirmation Study ===")
    logger.info(f"Strategies : {args.strategy}")
    logger.info(f"Timeframes : {args.tf}")
    logger.info(f"Vol thresh : {args.vol_thresh}")
    logger.info(f"Vol high/low: {args.vol_high} / {args.vol_low}")
    logger.info(f"Size boost/reduce: ×{args.size_boost} / ×{args.size_reduce}")

    all_rows: List[dict] = []

    for strat_name in args.strategy:
        if strat_name not in STRATEGY_MAP:
            logger.warning(f"Unknown: {strat_name}"); continue
        strategy = STRATEGY_MAP[strat_name]()

        for tf in args.tf:
            label = f"{strat_name}/{tf}"
            bm_key = f"{strat_name}/{tf}+ML"
            logger.info(f"\n{'='*55}\n{label}")

            try:
                df = fe.load(tf)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            except Exception as exc:
                logger.error(f"Feature load: {exc}"); continue

            params = _load_params(results_csv, strat_name, tf)
            if params is None:
                logger.warning(f"No params for {label}"); continue

            n     = len(df)
            is_n  = int(n * cfg.validation.is_ratio)
            df_is = df.iloc[:is_n]
            df_oos= df.iloc[is_n:]
            df_oos_sig = strategy.generate_signals(df_oos.copy(), params)

            results: Dict[str, object] = {}

            # ── BASE ──────────────────────────────────────────────────────────
            logger.info("  Running BASE...")
            results["BASE"] = BacktestEngine(sim_cfg).run(
                df_oos_sig.copy(), strat_name, tf, params
            )

            # ── ML_SIZE ───────────────────────────────────────────────────────
            ml_mults = pd.Series(1.0, index=df_oos.index)
            ml_trained = False

            if tf in ML_ELIGIBLE:
                logger.info("  Training ML sizer...")
                sizer = MLSizerV2(cfg, sizer_cfg)
                ok    = sizer.fit(strategy, params, df_is, args.direction, label=label)
                if ok:
                    ml_mults   = sizer.predict(df_oos, strategy, params)
                    ml_trained = True
                    logger.info("  Running ML_SIZE...")
                    results["ML_SIZE"] = run_dynamic(
                        df_oos_sig.copy(), strat_name, tf, sim_cfg,
                        ml_mults.reset_index(drop=True), params
                    )

            # ── VOL_CONFIRM ───────────────────────────────────────────────────
            if not args.no_confirm:
                vol_col = "vol_zscore_20"
                for vt in args.vol_thresh:
                    ver_name = f"VOL_CONF_{vt}"
                    logger.info(f"  Running {ver_name}...")
                    df_vol = _vol_confirm_signals(df_oos_sig.copy(), vol_col, vt)
                    results[ver_name] = BacktestEngine(sim_cfg).run(
                        df_vol, strat_name, tf, params
                    )

            # ── ML_VOL ────────────────────────────────────────────────────────
            logger.info("  Running ML_VOL (ML × volume size)...")
            vol_col  = "vol_zscore_20"
            vol_adj  = _vol_size_mults(
                df_oos, vol_col,
                args.vol_high, args.vol_low,
                args.size_boost, args.size_reduce,
            )

            if ml_trained:
                # Combined: ML score × volume adjustment
                combined = (ml_mults * vol_adj).clip(0.5, 2.0)
            else:
                # No ML → just volume adjustment
                combined = vol_adj.clip(0.5, 2.0)

            results["ML_VOL"] = run_dynamic(
                df_oos_sig.copy(), strat_name, tf, sim_cfg,
                combined.reset_index(drop=True), params
            )

            # Print
            rows = [_metrics(k, v) for k, v in results.items()]
            _print_results(rows, label, bm_key)

            # Yearly breakdown
            print("\n  Yearly Returns (%):")
            yearly = {}
            for name, res in results.items():
                if not res.yearly.empty and "return" in res.yearly.columns:
                    yearly[name] = (res.yearly["return"]*100).round(2)
            if yearly:
                print(pd.DataFrame(yearly).to_string())

            # Save
            for r in rows:
                r.update({"strategy": strat_name, "timeframe": tf})
                all_rows.append(r)

            _plot_comparison(results, df_oos, label, out_plots)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    if all_rows:
        out_df = pd.DataFrame(all_rows)
        path   = out_ranks / "volume_study_results.csv"
        out_df.to_csv(path, index=False)
        logger.info(f"\nResults → {path}")

        # Final comparison vs benchmark
        print(f"\n{'='*78}")
        print("VOLUME STUDY — FINAL COMPARISON VS BENCHMARK")
        print(f"{'='*78}")
        for bm_name, bm in BENCHMARK.items():
            strat, tf_bm = bm_name.rsplit("/",1)
            tf_bm = tf_bm.split("+")[0]
            sub   = out_df[(out_df["strategy"]==strat)&(out_df["timeframe"]==tf_bm)]
            print(f"\n{strat}/{tf_bm}  benchmark: PF={bm['pf']:.3f}  Sharpe={bm['sharpe']:.3f}  Trades={bm['trades']}")
            for _, r in sub.iterrows():
                beats = "★ BEATS BENCHMARK" if r["sharpe"] > bm["sharpe"] and r["trades"] >= bm["trades"]*0.9 else ""
                print(f"  {r['version']:<18}: PF={r['pf']:.3f}  Sh={r['sharpe']:.3f}  Trades={int(r['trades'])}  {beats}")


if __name__ == "__main__":
    main()