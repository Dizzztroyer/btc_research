#!/usr/bin/env python3
"""
run_wf_validation.py
─────────────────────
Walk-forward validation for ML_ATR vs ML_SIZE vs BASE.

Tests each version across N sequential non-overlapping OOS windows.
For each window: train ML on IS (all data before window), test on OOS window.

Validated configurations:
    BASE    — fixed sizing, no ML, no ATR
    ML_SIZE — XGBoost rank-normalized position sizer
    ML_ATR  — ML_SIZE × ATR regime modifier (LOW×1.3, MED×1.0, HIGH×0.7)

ATR regime (inverted — confirmed correct direction):
    LOW  ATR (<33rd pct) → ×1.3  (calm market, breakouts work better)
    MED  ATR (33–66th)   → ×1.0  (neutral)
    HIGH ATR (>66th pct) → ×0.7  (chaotic, reduce exposure)

Usage
─────
    python run_wf_validation.py
    python run_wf_validation.py --tf 8h 12h --windows 5
    python run_wf_validation.py --low-mult 1.3 --high-mult 0.7
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
from src.backtest.metrics import compute_metrics
from src.ml.ml_sizer_v2 import MLSizerV2, SizerConfig
from src.ml.dynamic_engine import run_dynamic
from src.strategies.structure import SwingBreakoutStrategy
from src.strategies.trend import DonchianBreakoutStrategy

logger = get_logger(__name__, log_file=Path("outputs/logs/wf_validation.log"))

# ── Fixed config ───────────────────────────────────────────────────────────────
STRATEGY_MAP = {
    "swing_breakout":    SwingBreakoutStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
}

# Confirmed benchmark numbers (OOS backtest)
BASELINE_OOS = {
    "swing_breakout/8h/ML_SIZE":  {"pf": 1.948, "sharpe": 1.772},
    "swing_breakout/8h/ML_ATR":   {"pf": 2.015, "sharpe": 1.813},
    "swing_breakout/12h/ML_SIZE": {"pf": 1.911, "sharpe": 1.687},
    "swing_breakout/12h/ML_VOL":  {"pf": 2.018, "sharpe": 1.789},
}


# ── ATR regime helpers ────────────────────────────────────────────────────────

def compute_atr_regime(
    df: pd.DataFrame,
    lookback: int = 200,
    low_pct:  float = 33.0,
    high_pct: float = 66.0,
) -> pd.Series:
    """Classify bars into low/medium/high ATR regime. No lookahead (shift 1)."""
    atr_col = next((c for c in ["atr_14_pct","atr_14","atr_7_pct"] if c in df.columns), None)
    if atr_col is None:
        tr  = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"]  - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean() / df["close"]
    else:
        atr = df[atr_col]

    atr_lag   = atr.shift(1)
    roll_low  = atr_lag.rolling(lookback, min_periods=50).quantile(low_pct  / 100)
    roll_high = atr_lag.rolling(lookback, min_periods=50).quantile(high_pct / 100)

    regime = pd.Series("medium", index=df.index)
    regime[atr_lag < roll_low]  = "low"
    regime[atr_lag > roll_high] = "high"
    return regime


def atr_multipliers(
    regime:     pd.Series,
    low_mult:   float = 1.3,
    high_mult:  float = 0.7,
) -> pd.Series:
    """Map regime labels to size multipliers."""
    mults = pd.Series(1.0, index=range(len(regime)))
    reg   = regime.reset_index(drop=True)
    mults[reg == "low"]  = low_mult
    mults[reg == "high"] = high_mult
    return mults


# ── Load params ───────────────────────────────────────────────────────────────

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


# ── Per-window backtest ────────────────────────────────────────────────────────

def run_window(
    df_is:      pd.DataFrame,
    df_oos:     pd.DataFrame,
    strategy,
    params:     dict,
    sim_cfg:    SimConfig,
    sizer_cfg:  SizerConfig,
    low_mult:   float,
    high_mult:  float,
    direction:  str = "both",
    atr_lookback: int = 200,
) -> Dict[str, dict]:
    """
    Train on df_is, evaluate BASE / ML_SIZE / ML_ATR on df_oos.
    Returns dict of version → metrics dict.
    """
    engine = BacktestEngine(sim_cfg)
    strat_name = strategy.name
    tf_label   = ""

    df_oos_sig = strategy.generate_signals(df_oos.copy(), params)
    df_oos_sig = df_oos_sig.reset_index(drop=True)

    results = {}

    # ── BASE ──────────────────────────────────────────────────────────────────
    res_base = engine.run(df_oos_sig.copy(), strat_name, tf_label, params)
    results["BASE"] = res_base.metrics

    # ── ML_SIZE ───────────────────────────────────────────────────────────────
    sizer = MLSizerV2(
        cfg      = sim_cfg,   # reuse sim as proxy — sizer reads cfg.fees etc
        sizer_cfg= sizer_cfg,
    )
    # Pass the real cfg via a small wrapper
    sizer.cfg = _CfgProxy(sim_cfg)

    ml_ok   = sizer.fit(strategy, params, df_is, direction, label="wf")
    ml_mults = pd.Series(1.0, index=range(len(df_oos)))

    if ml_ok:
        ml_mults = sizer.predict(df_oos, strategy, params).reset_index(drop=True)
        res_ml   = run_dynamic(
            df_oos_sig.copy(), strat_name, tf_label,
            sim_cfg, ml_mults, params
        )
        results["ML_SIZE"] = res_ml.metrics
    else:
        results["ML_SIZE"] = res_base.metrics.copy()
        results["ML_SIZE"]["_ml_disabled"] = True

    # ── ATR regime ────────────────────────────────────────────────────────────
    # Compute regime on full window (IS+OOS) but only use OOS portion
    df_full   = pd.concat([df_is, df_oos], ignore_index=True)
    regime_full = compute_atr_regime(df_full, atr_lookback)
    regime_oos  = regime_full.iloc[len(df_is):].reset_index(drop=True)
    atr_mults   = atr_multipliers(regime_oos, low_mult, high_mult)

    reg_dist = regime_oos.value_counts(normalize=True).to_dict()

    # ── ML_ATR ────────────────────────────────────────────────────────────────
    if ml_ok:
        combined = (ml_mults * atr_mults).clip(0.4, 2.0)
        res_mlatr = run_dynamic(
            df_oos_sig.copy(), strat_name, tf_label,
            sim_cfg, combined, params
        )
        results["ML_ATR"] = res_mlatr.metrics
    else:
        results["ML_ATR"] = results["ML_SIZE"].copy()

    results["_meta"] = {
        "regime_low":    reg_dist.get("low",    0),
        "regime_medium": reg_dist.get("medium", 0),
        "regime_high":   reg_dist.get("high",   0),
        "ml_enabled":    ml_ok,
    }

    return results


class _CfgProxy:
    """Minimal proxy so MLSizerV2 can access cfg attributes from SimConfig."""
    def __init__(self, sim: SimConfig):
        self.fees          = sim.fees
        self.slippage      = sim.slippage
        self.leverage      = sim.leverage
        self.risk_per_trade= sim.risk_per_trade

    class _val:
        is_ratio  = 0.75
        oos_ratio = 0.25

    validation = _val()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward validation: ML_ATR vs ML_SIZE vs BASE")
    p.add_argument("--config",       default="config/config.yaml")
    p.add_argument("--tf",           nargs="+", default=["8h", "12h"])
    p.add_argument("--strategy",     nargs="+", default=["swing_breakout"])
    p.add_argument("--windows",      type=int,  default=5)
    p.add_argument("--direction",    default="both")
    p.add_argument("--results-csv",  default="outputs/rankings/all_results.csv")
    p.add_argument("--low-mult",     type=float, default=1.3)
    p.add_argument("--high-mult",    type=float, default=0.7)
    p.add_argument("--atr-lookback", type=int,   default=200)
    p.add_argument("--atr-low-pct",  type=float, default=33.0)
    p.add_argument("--atr-high-pct", type=float, default=66.0)
    p.add_argument("--n-estimators", type=int,   default=400)
    p.add_argument("--min-is-trades",type=int,   default=100)
    return p.parse_args()


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_wf(wf_df: pd.DataFrame, label: str, out_dir: Path) -> None:
    versions = ["BASE", "ML_SIZE", "ML_ATR"]
    colors   = {"BASE": "#888", "ML_SIZE": "#378ADD", "ML_ATR": "#1D9E75"}
    metrics  = ["pf", "sharpe", "mdd", "trade_count"]
    titles   = ["Profit Factor", "Sharpe Ratio", "Max Drawdown %", "Trade Count"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#0f1117")
    axes = axes.flatten()

    for ax in axes:
        ax.set_facecolor("#161925")
        ax.tick_params(colors="#aaa", labelsize=9)
        for s in ax.spines.values():
            s.set_color("#2a2d3e")

    for ax, metric, title in zip(axes, metrics, titles):
        x = wf_df["window"].unique()
        w = 0.25
        offsets = {"BASE": -w, "ML_SIZE": 0, "ML_ATR": w}

        for ver in versions:
            sub = wf_df[wf_df["version"] == ver].sort_values("window")
            if sub.empty:
                continue
            vals = sub[metric].values
            if metric == "mdd":
                vals = vals * 100  # to percent
            ax.bar(sub["window"] + offsets[ver], vals, w * 0.9,
                   label=ver, color=colors[ver], alpha=0.85)

        if metric in ("pf",):
            ax.axhline(1, color="gray", linewidth=0.7, linestyle="--")
        if metric in ("sharpe",):
            ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")

        ax.set_title(title, color="#e0e0e0", fontsize=10)
        ax.set_xlabel("WF Window", color="#aaa")
        ax.legend(facecolor="#1e2235", labelcolor="#e0e0e0", fontsize=8)
        ax.grid(True, color="#2a2d3e", alpha=0.4, axis="y")

    plt.suptitle(f"Walk-Forward Validation — {label}", color="#e0e0e0", fontsize=12)
    plt.tight_layout()
    clean = label.replace("/", "_")
    path  = out_dir / f"wf_validation_{clean}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    logger.info(f"  Plot → {path}")


def _plot_cumulative(wf_df: pd.DataFrame, label: str, out_dir: Path) -> None:
    """Cumulative Sharpe and PF across windows."""
    versions = ["BASE", "ML_SIZE", "ML_ATR"]
    colors   = {"BASE": "#888", "ML_SIZE": "#378ADD", "ML_ATR": "#1D9E75"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0f1117")
    for ax in axes:
        ax.set_facecolor("#161925")
        ax.tick_params(colors="#aaa")
        for s in ax.spines.values():
            s.set_color("#2a2d3e")

    for ver in versions:
        sub = wf_df[wf_df["version"] == ver].sort_values("window")
        if sub.empty:
            continue
        windows = sub["window"].values
        axes[0].plot(windows, sub["sharpe"].values,  "o-", label=ver,
                     color=colors[ver], linewidth=1.8, markersize=5)
        axes[1].plot(windows, sub["pf"].values,      "o-", label=ver,
                     color=colors[ver], linewidth=1.8, markersize=5)

    axes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[0].set_title("Sharpe per WF Window", color="#e0e0e0")
    axes[0].legend(facecolor="#1e2235", labelcolor="#e0e0e0")
    axes[0].grid(True, color="#2a2d3e", alpha=0.4)

    axes[1].axhline(1, color="gray", linewidth=0.5, linestyle="--")
    axes[1].set_title("PF per WF Window", color="#e0e0e0")
    axes[1].legend(facecolor="#1e2235", labelcolor="#e0e0e0")
    axes[1].grid(True, color="#2a2d3e", alpha=0.4)

    plt.suptitle(f"WF Stability — {label}", color="#e0e0e0", fontsize=12)
    plt.tight_layout()
    clean = label.replace("/", "_")
    path  = out_dir / f"wf_stability_{clean}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

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
        min_train_trades = args.min_is_trades,
        min_val_trades   = max(15, args.min_is_trades // 5),
        min_mult_std     = 0.05,
        is_train_ratio   = 0.75,
    )

    logger.info("=== Walk-Forward Validation ===")
    logger.info(f"Strategies : {args.strategy}")
    logger.info(f"Timeframes : {args.tf}")
    logger.info(f"WF windows : {args.windows}")
    logger.info(f"ATR mults  : LOW×{args.low_mult}  MED×1.0  HIGH×{args.high_mult}")

    all_rows:     List[dict] = []
    summary_rows: List[dict] = []

    for strat_name in args.strategy:
        if strat_name not in STRATEGY_MAP:
            logger.warning(f"Unknown: {strat_name}"); continue
        strategy = STRATEGY_MAP[strat_name]()

        for tf in args.tf:
            label = f"{strat_name}/{tf}"
            logger.info(f"\n{'='*55}\n{label}")

            # Load features
            try:
                df = fe.load(tf)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            except Exception as exc:
                logger.error(f"  Feature load: {exc}"); continue

            # Load best params
            params = _load_params(results_csv, strat_name, tf)
            if params is None:
                logger.warning(f"  No params for {label}"); continue
            logger.info(f"  Params: {params}")

            n = len(df)

            # ── WF split design ────────────────────────────────────────────────
            # Use last 40% as OOS pool, split into windows
            # IS: all data before each window (expanding)
            is_pool_end = int(n * cfg.validation.is_ratio)
            oos_total   = n - is_pool_end
            win_size    = max(50, oos_total // args.windows)

            logger.info(
                f"  Total bars={n} | IS pool={is_pool_end} | "
                f"OOS pool={oos_total} | Window size={win_size}"
            )

            window_rows: List[dict] = []

            for w in range(args.windows):
                oos_start = is_pool_end + w * win_size
                oos_end   = oos_start + win_size if w < args.windows - 1 else n

                if oos_start >= n or oos_end > n:
                    break

                df_is  = df.iloc[:oos_start]
                df_oos = df.iloc[oos_start:oos_end]

                is_bars  = len(df_is)
                oos_bars = len(df_oos)

                logger.info(
                    f"  Window {w+1}/{args.windows} | "
                    f"IS {df_is['timestamp'].iloc[0].date()}→{df_is['timestamp'].iloc[-1].date()} "
                    f"({is_bars} bars) | "
                    f"OOS {df_oos['timestamp'].iloc[0].date()}→{df_oos['timestamp'].iloc[-1].date()} "
                    f"({oos_bars} bars)"
                )

                if is_bars < 300 or oos_bars < 20:
                    logger.warning(f"  Window {w+1}: too small, skipping")
                    continue

                try:
                    window_results = run_window(
                        df_is=df_is, df_oos=df_oos,
                        strategy=strategy, params=params,
                        sim_cfg=sim_cfg, sizer_cfg=sizer_cfg,
                        low_mult=args.low_mult, high_mult=args.high_mult,
                        direction=args.direction,
                        atr_lookback=args.atr_lookback,
                    )
                except Exception as exc:
                    logger.error(f"  Window {w+1} failed: {exc}")
                    import traceback; traceback.print_exc()
                    continue

                meta = window_results.pop("_meta", {})
                ml_ok = meta.get("ml_enabled", False)

                for version, metrics in window_results.items():
                    row = {
                        "strategy":       strat_name,
                        "timeframe":      tf,
                        "window":         w + 1,
                        "version":        version,
                        "is_bars":        is_bars,
                        "oos_bars":       oos_bars,
                        "oos_start_date": str(df_oos["timestamp"].iloc[0].date()),
                        "oos_end_date":   str(df_oos["timestamp"].iloc[-1].date()),
                        "ml_enabled":     ml_ok,
                        "regime_low":     meta.get("regime_low",  0),
                        "regime_med":     meta.get("regime_medium",0),
                        "regime_high":    meta.get("regime_high", 0),
                        "trade_count":    metrics.get("trade_count",   0),
                        "pf":             metrics.get("profit_factor", np.nan),
                        "sharpe":         metrics.get("sharpe",        np.nan),
                        "sortino":        metrics.get("sortino",       np.nan),
                        "calmar":         metrics.get("calmar",        np.nan),
                        "total_return":   metrics.get("total_return",  np.nan),
                        "mdd":            metrics.get("max_drawdown",  np.nan),
                        "win_rate":       metrics.get("win_rate",      np.nan),
                    }
                    window_rows.append(row)
                    all_rows.append(row)

                # Per-window log
                for ver in ["BASE", "ML_SIZE", "ML_ATR"]:
                    m = window_results.get(ver, {})
                    logger.info(
                        f"  W{w+1} {ver:<10}: "
                        f"trades={int(m.get('trade_count',0)):>4}  "
                        f"PF={m.get('profit_factor',float('nan')):>6.3f}  "
                        f"Sh={m.get('sharpe',float('nan')):>6.3f}  "
                        f"MDD={m.get('max_drawdown',float('nan')):>7.2%}"
                    )

            if not window_rows:
                logger.warning(f"  No valid windows for {label}")
                continue

            wf_df = pd.DataFrame(window_rows)

            # ── Aggregate stats ────────────────────────────────────────────────
            print(f"\n{'─'*70}")
            print(f"  {label} — Walk-Forward Summary ({args.windows} windows)")
            print(f"{'─'*70}")

            agg_data = {}
            for ver in ["BASE", "ML_SIZE", "ML_ATR"]:
                sub = wf_df[wf_df["version"] == ver]
                if sub.empty:
                    continue
                agg = {
                    "avg_pf":      sub["pf"].mean(),
                    "std_pf":      sub["pf"].std(),
                    "avg_sharpe":  sub["sharpe"].mean(),
                    "std_sharpe":  sub["sharpe"].std(),
                    "avg_mdd":     sub["mdd"].mean(),
                    "avg_trades":  sub["trade_count"].mean(),
                    "pct_positive_sh": (sub["sharpe"] > 0).mean(),
                    "pct_positive_pf": (sub["pf"] > 1.0).mean(),
                }
                agg_data[ver] = agg
                summary_rows.append({
                    "strategy": strat_name, "timeframe": tf, "version": ver,
                    **agg
                })

            print(f"\n  {'Version':<12} {'avg_PF':>8} {'std_PF':>8} {'avg_Sh':>8} "
                  f"{'std_Sh':>8} {'avg_MDD':>8} {'%Sh>0':>7} {'%PF>1':>7}")
            print(f"  {'─'*70}")
            for ver in ["BASE","ML_SIZE","ML_ATR"]:
                if ver not in agg_data: continue
                a  = agg_data[ver]
                bm = BASELINE_OOS.get(f"{strat_name}/{tf}/{ver}", {})
                note = f"  (OOS was {bm['sharpe']:.3f})" if bm else ""
                print(f"  {ver:<12} "
                      f"{a['avg_pf']:>8.3f} {a['std_pf']:>8.3f} "
                      f"{a['avg_sharpe']:>8.3f} {a['std_sharpe']:>8.3f} "
                      f"{a['avg_mdd']:>8.2%} "
                      f"{a['pct_positive_sh']:>7.0%} "
                      f"{a['pct_positive_pf']:>7.0%}{note}")

            # ML_ATR vs ML_SIZE delta
            if "ML_ATR" in agg_data and "ML_SIZE" in agg_data:
                d_pf = agg_data["ML_ATR"]["avg_pf"]    - agg_data["ML_SIZE"]["avg_pf"]
                d_sh = agg_data["ML_ATR"]["avg_sharpe"] - agg_data["ML_SIZE"]["avg_sharpe"]
                print(f"\n  ML_ATR vs ML_SIZE: avg_PF {d_pf:+.3f}  avg_Sh {d_sh:+.3f}")

            # Per-window table
            print(f"\n  Per-window detail:")
            print(f"  {'W':>3} {'version':<12} {'trades':>7} {'PF':>7} {'Sharpe':>8} "
                  f"{'MDD':>8} {'low%':>6} {'high%':>6}")
            print(f"  {'─'*65}")
            for _, r in wf_df.sort_values(["window","version"]).iterrows():
                print(f"  {int(r['window']):>3} {r['version']:<12} "
                      f"{int(r['trade_count'] or 0):>7} {r['pf']:>7.3f} "
                      f"{r['sharpe']:>8.3f} {r['mdd']:>8.2%} "
                      f"{r['regime_low']:>6.0%} {r['regime_high']:>6.0%}")

            # Plots
            _plot_wf(wf_df, label, out_plots)
            _plot_cumulative(wf_df, label, out_plots)

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    if all_rows:
        rdf  = pd.DataFrame(all_rows)
        path = out_ranks / "wf_validation_detail.csv"
        rdf.to_csv(path, index=False)
        logger.info(f"\nDetail → {path}")

    if summary_rows:
        sdf  = pd.DataFrame(summary_rows)
        path = out_ranks / "wf_validation_summary.csv"
        sdf.to_csv(path, index=False)
        logger.info(f"Summary → {path}")

    # ── Final recommendation ───────────────────────────────────────────────────
    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        print(f"\n{'='*70}")
        print("WALK-FORWARD VALIDATION — FINAL RECOMMENDATION")
        print(f"{'='*70}")

        for (strat, tf), grp in sdf.groupby(["strategy","timeframe"]):
            print(f"\n{strat}/{tf}:")
            base_sh  = grp[grp["version"]=="BASE"]["avg_sharpe"].values[0] \
                       if "BASE" in grp["version"].values else 0
            ml_sh    = grp[grp["version"]=="ML_SIZE"]["avg_sharpe"].values[0] \
                       if "ML_SIZE" in grp["version"].values else 0
            mlatr_sh = grp[grp["version"]=="ML_ATR"]["avg_sharpe"].values[0] \
                       if "ML_ATR" in grp["version"].values else None

            for ver in ["BASE","ML_SIZE","ML_ATR"]:
                sub = grp[grp["version"]==ver]
                if sub.empty: continue
                r = sub.iloc[0]
                print(f"  {ver:<12}: avg_Sh={r['avg_sharpe']:.3f} ± {r['std_sharpe']:.3f}  "
                      f"avg_PF={r['avg_pf']:.3f} ± {r['std_pf']:.3f}  "
                      f"%positive={r['pct_positive_sh']:.0%}")

            if mlatr_sh is not None:
                d = mlatr_sh - ml_sh
                if mlatr_sh > ml_sh and mlatr_sh > 0:
                    verdict = "✓ ACCEPT — ML_ATR is new baseline"
                elif mlatr_sh > ml_sh * 0.95:
                    verdict = "~ NEUTRAL — ML_ATR is comparable, keep ML_SIZE"
                else:
                    verdict = "✗ REJECT — ML_ATR underperforms in WF"
                print(f"\n  ML_ATR vs ML_SIZE: avg_Sh {d:+.3f}")
                print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()