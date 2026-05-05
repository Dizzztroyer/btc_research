#!/usr/bin/env python3
"""
run_enhanced_backtest.py  v2
─────────────────────────────
Enhanced backtest: ML position sizing only.
No binary filtering. No regime skipping. Every signal trades.

Compares:
    BASE     — fixed position size (risk_per_trade constant)
    ML_SIZE  — dynamic position size via XGBoost score

Only runs on timeframes with enough trades for reliable ML training (8h, 12h).
For 1d: runs BASE only (not enough IS trades for ML).

Diagnostics logged:
    - AUC
    - Number of IS training trades
    - std of size multipliers
    - min/max/mean of multipliers
    - auto-disable if std < 0.05

Usage
─────
    python run_enhanced_backtest.py
    python run_enhanced_backtest.py --tf 8h 12h
    python run_enhanced_backtest.py --strategy donchian_breakout --tf 12h
    python run_enhanced_backtest.py --size-min 0.6 --size-max 1.4  # tighter range
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
from src.ml.ml_sizer_v2 import MLSizerV2, SizerConfig, score_to_mult
from src.ml.dynamic_engine import run_dynamic
from src.strategies.trend import DonchianBreakoutStrategy
from src.strategies.structure import SwingBreakoutStrategy

logger = get_logger(__name__, log_file=Path("outputs/logs/enhanced_backtest.log"))

STRATEGY_MAP = {
    "donchian_breakout": DonchianBreakoutStrategy,
    "swing_breakout":    SwingBreakoutStrategy,
}

# Timeframes with enough trades for reliable ML training
ML_ELIGIBLE_TF = {"8h", "12h", "6h"}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enhanced backtest v2: ML position sizing")
    p.add_argument("--config",      default="config/config.yaml")
    p.add_argument("--tf",          nargs="+", default=["8h", "12h"])
    p.add_argument("--strategy",    nargs="+",
                   default=["donchian_breakout", "swing_breakout"])
    p.add_argument("--direction",   default="both")
    p.add_argument("--results-csv", default="outputs/rankings/all_results.csv")
    p.add_argument("--size-min",    type=float, default=0.5)
    p.add_argument("--size-max",    type=float, default=1.5)
    p.add_argument("--n-estimators",type=int,   default=500)
    return p.parse_args()


# ── Load best params ──────────────────────────────────────────────────────────

def _load_params(csv: Path, strategy: str, tf: str) -> Optional[dict]:
    if not csv.exists():
        logger.error(f"all_results.csv not found: {csv}")
        return None
    df   = pd.read_csv(csv)
    mask = (df["strategy"] == strategy) & (df["timeframe"] == tf)
    sub  = df[mask]
    if sub.empty:
        logger.warning(f"No results for {strategy}/{tf}")
        return None
    best   = sub.sort_values("robustness", ascending=False).iloc[0]
    p_cols = [c for c in best.index if c.startswith("p_") and not pd.isna(best[c])]
    out    = {}
    for c in p_cols:
        v = best[c]; k = c[2:]
        out[k] = int(v) if isinstance(v, float) and v == int(v) else v
    return out


# ── Metrics display ───────────────────────────────────────────────────────────

def _show(name: str, r) -> dict:
    m = r.metrics
    return {
        "version":    name,
        "trades":     m.get("trade_count",   0),
        "pf":         m.get("profit_factor", np.nan),
        "sharpe":     m.get("sharpe",        np.nan),
        "sortino":    m.get("sortino",       np.nan),
        "calmar":     m.get("calmar",        np.nan),
        "return":     m.get("total_return",  np.nan),
        "mdd":        m.get("max_drawdown",  np.nan),
        "win_rate":   m.get("win_rate",      np.nan),
        "expectancy": m.get("expectancy",    np.nan),
    }


def _print_table(rows: List[dict], label: str) -> None:
    df = pd.DataFrame(rows)
    print(f"\n{'─'*72}")
    print(f"  {label}")
    print(f"{'─'*72}")
    print(f"{'Version':<14} {'Trades':>7} {'PF':>8} {'Sharpe':>8} "
          f"{'Sortino':>8} {'Return':>9} {'MDD':>8} {'WinRate':>8}")
    print(f"{'─'*72}")

    base = df[df["version"]=="BASE"].iloc[0] if "BASE" in df["version"].values else None

    for _, r in df.iterrows():
        pf_d  = f"({r['pf']-base['pf']:+.3f})"     if base is not None and r["version"]!="BASE" else ""
        sh_d  = f"({r['sharpe']-base['sharpe']:+.3f})" if base is not None and r["version"]!="BASE" else ""
        print(
            f"{r['version']:<14} "
            f"{int(r['trades'] or 0):>7} "
            f"{r['pf']:>7.3f} {pf_d:<9} "
            f"{r['sharpe']:>7.3f} {sh_d:<9} "
            f"{r['return']:>8.1%} "
            f"{r['mdd']:>8.2%} "
            f"{r['win_rate']:>8.1%}"
        )
    print(f"{'─'*72}")


def _print_yearly(results: Dict[str, object]) -> None:
    yearly_data = {}
    for name, res in results.items():
        if not res.yearly.empty and "return" in res.yearly.columns:
            yearly_data[name] = (res.yearly["return"] * 100).round(2)
    if not yearly_data:
        return
    ydf = pd.DataFrame(yearly_data)
    print(f"\n  Yearly Returns (%):")
    print(ydf.to_string())


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot(
    results: Dict[str, object],
    df_full: pd.DataFrame,
    mults_series: Optional[pd.Series],
    label: str,
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), facecolor="#0f1117",
                              gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    colors = {"BASE": "#888888", "ML_SIZE": "#1D9E75"}

    for ax in axes:
        ax.set_facecolor("#161925")
        ax.tick_params(colors="#aaa", labelsize=9)
        for s in ax.spines.values():
            s.set_color("#2a2d3e")

    ts = df_full["timestamp"].reset_index(drop=True)

    # Equity
    ax = axes[0]
    for name, res in results.items():
        if res.equity.empty: continue
        eq = res.equity.reset_index(drop=True)
        lw = 2.0 if name == "ML_SIZE" else 1.2
        ax.plot(ts[:len(eq)], eq, label=name, color=colors.get(name,"white"),
                linewidth=lw, alpha=0.95)
    ax.set_title(f"Equity — {label}", color="#e0e0e0", fontsize=12)
    ax.set_ylabel("Value ($)", color="#aaa")
    ax.legend(facecolor="#1e2235", labelcolor="#e0e0e0")
    ax.grid(True, color="#2a2d3e", alpha=0.4)

    # Drawdown
    ax = axes[1]
    for name, res in results.items():
        if res.drawdown.empty: continue
        dd = res.drawdown.reset_index(drop=True) * 100
        ax.plot(ts[:len(dd)], dd, label=name, color=colors.get(name,"white"),
                linewidth=1.5 if name=="ML_SIZE" else 1.0, alpha=0.9)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title("Drawdown (%)", color="#e0e0e0", fontsize=10)
    ax.set_ylabel("DD %", color="#aaa")
    ax.legend(facecolor="#1e2235", labelcolor="#e0e0e0", fontsize=8)
    ax.grid(True, color="#2a2d3e", alpha=0.4)

    # Size multiplier distribution
    ax = axes[2]
    if mults_series is not None:
        sig_mask = mults_series != 1.0
        if sig_mask.any():
            sm = mults_series[sig_mask]
            ax.hist(sm.values, bins=30, color="#1D9E75", alpha=0.75, edgecolor="#0f1117")
            ax.axvline(1.0, color="#f7c94b", linewidth=1.5, linestyle="--", label="base=1.0")
            ax.axvline(sm.mean(), color="white", linewidth=1.0, linestyle=":", label=f"mean={sm.mean():.3f}")
            ax.set_title(f"Size Multiplier Distribution (std={sm.std():.3f})",
                         color="#e0e0e0", fontsize=10)
            ax.set_xlabel("Multiplier", color="#aaa")
            ax.set_ylabel("Count", color="#aaa")
            ax.legend(facecolor="#1e2235", labelcolor="#e0e0e0", fontsize=8)
    ax.grid(True, color="#2a2d3e", alpha=0.4)

    plt.tight_layout()
    clean = label.replace("/","_").replace(" ","_")
    path  = out_dir / f"enhanced_{clean}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    logger.info(f"  Plot → {path}")
    return path


def _plot_importance(imp: pd.DataFrame, label: str, out_dir: Path) -> None:
    if imp.empty:
        return
    top = imp.head(20)
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0f1117")
    ax.set_facecolor("#161925")
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="#1D9E75", alpha=0.85)
    ax.set_xlabel("Importance", color="#aaa")
    ax.set_title(f"Feature Importances — {label}", color="#e0e0e0")
    ax.tick_params(colors="#aaa")
    for s in ax.spines.values():
        s.set_color("#2a2d3e")
    plt.tight_layout()
    clean = label.replace("/","_")
    path  = out_dir / f"ml_importance_{clean}.png"
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
        n_estimators=args.n_estimators,
        min_train_trades=150,
        min_val_trades=30,
        min_mult_std=0.05,
    )

    summary_rows: List[dict] = []
    diag_rows:    List[dict] = []

    logger.info("=== Enhanced Backtest v2 (ML position sizing only) ===")
    logger.info(f"Strategies : {args.strategy}")
    logger.info(f"Timeframes : {args.tf}")
    logger.info(f"ML eligible TFs: {ML_ELIGIBLE_TF}")
    logger.info(f"Size range : [{args.size_min}, {args.size_max}]")
    logger.info(f"Min IS trades : {sizer_cfg.min_train_trades}")

    for strat_name in args.strategy:
        if strat_name not in STRATEGY_MAP:
            logger.warning(f"Unknown: {strat_name}")
            continue
        strategy = STRATEGY_MAP[strat_name]()

        for tf in args.tf:
            label = f"{strat_name}/{tf}"
            logger.info(f"\n{'='*55}")
            logger.info(f"Processing: {label}")

            # Load features
            try:
                df = fe.load(tf)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            except Exception as exc:
                logger.error(f"Feature load: {exc}")
                continue

            # Load params
            params = _load_params(results_csv, strat_name, tf)
            if params is None:
                continue

            # IS/OOS split
            n    = len(df)
            is_n = int(n * cfg.validation.is_ratio)
            df_is  = df.iloc[:is_n]
            df_oos = df.iloc[is_n:]

            logger.info(
                f"  IS: {len(df_is)} bars | OOS: {len(df_oos)} bars | "
                f"dates {df_oos['timestamp'].iloc[0].date()} → {df_oos['timestamp'].iloc[-1].date()}"
            )

            # Generate signals on OOS
            df_oos_sig = strategy.generate_signals(df_oos.copy(), params)

            results: Dict[str, object] = {}

            # ── BASE ──────────────────────────────────────────────────────────
            logger.info("  Running BASE...")
            results["BASE"] = BacktestEngine(sim_cfg).run(
                df_oos_sig.copy(), strat_name, tf, params
            )

            # ── ML SIZE ───────────────────────────────────────────────────────
            mults_oos = None
            ml_eligible = tf in ML_ELIGIBLE_TF

            if not ml_eligible:
                logger.info(
                    f"  ML skipped: {tf} not in ML_ELIGIBLE_TF {ML_ELIGIBLE_TF}. "
                    f"(Too few trades for reliable ML training.)"
                )
            else:
                logger.info("  Training ML sizer on IS data...")
                sizer = MLSizerV2(cfg, sizer_cfg)
                ok    = sizer.fit(strategy, params, df_is, args.direction, label=label)

                d = sizer.diagnostics.copy()
                d["strategy"] = strat_name
                d["timeframe"] = tf
                d["ml_enabled"] = ok
                diag_rows.append(d)

                if ok:
                    logger.info("  Predicting OOS size multipliers...")
                    mults_oos = sizer.predict(df_oos, strategy, params)
                    # Reset index to match df_oos_sig
                    mults_oos = mults_oos.reset_index(drop=True)

                    # Verify variance
                    sig_mask = df_oos_sig["signal"] != 0
                    if sig_mask.any():
                        sig_mults = mults_oos[sig_mask.values]
                        std_val   = sig_mults.std()
                        logger.info(
                            f"  OOS mult distribution: "
                            f"mean={sig_mults.mean():.3f} "
                            f"std={std_val:.3f} "
                            f"[{sig_mults.min():.3f}, {sig_mults.max():.3f}]"
                        )
                        if std_val < sizer_cfg.min_mult_std:
                            logger.warning(
                                f"  OOS std={std_val:.4f} < {sizer_cfg.min_mult_std} — "
                                f"running BASE only (sizer not useful)"
                            )
                            mults_oos = None

                    if mults_oos is not None:
                        logger.info("  Running ML_SIZE...")
                        results["ML_SIZE"] = run_dynamic(
                            df=df_oos_sig.copy(),
                            strategy_name=strat_name,
                            timeframe=tf,
                            sim_cfg=sim_cfg,
                            size_mults=mults_oos,
                            params=params,
                        )

                        # Log feature importances
                        _plot_importance(sizer.importances, label, out_plots)
                else:
                    logger.info(
                        f"  ML sizer disabled "
                        f"({d.get('reason_disabled','unknown')}). Running BASE only."
                    )

            # ── Print results ─────────────────────────────────────────────────
            rows = [_show(k, v) for k, v in results.items()]
            _print_table(rows, label)
            _print_yearly(results)

            # ── Plot ──────────────────────────────────────────────────────────
            _plot(results, df_oos, mults_oos, label, out_plots)

            # ── Save to summary ────────────────────────────────────────────────
            for name, res in results.items():
                m = res.metrics
                summary_rows.append({
                    "strategy":    strat_name,
                    "timeframe":   tf,
                    "version":     name,
                    "ml_eligible": ml_eligible,
                    "trades":      m.get("trade_count",   0),
                    "pf":          m.get("profit_factor", np.nan),
                    "sharpe":      m.get("sharpe",        np.nan),
                    "sortino":     m.get("sortino",       np.nan),
                    "calmar":      m.get("calmar",        np.nan),
                    "total_return":m.get("total_return",  np.nan),
                    "mdd":         m.get("max_drawdown",  np.nan),
                    "win_rate":    m.get("win_rate",      np.nan),
                    "expectancy":  m.get("expectancy",    np.nan),
                })

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    if summary_rows:
        sdf  = pd.DataFrame(summary_rows)
        path = out_ranks / "enhanced_backtest_summary.csv"
        sdf.to_csv(path, index=False)
        logger.info(f"\nSummary → {path}")

    if diag_rows:
        ddf  = pd.DataFrame(diag_rows)
        path = out_ranks / "ml_sizer_diagnostics.csv"
        ddf.to_csv(path, index=False)
        logger.info(f"Diagnostics → {path}")

    # ── Final summary ──────────────────────────────────────────────────────────
    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        print(f"\n{'='*72}")
        print("FINAL COMPARISON — BASE vs ML_SIZE")
        print(f"{'='*72}")

        for (strat, tf), grp in sdf.groupby(["strategy","timeframe"]):
            base_row = grp[grp["version"]=="BASE"]
            ml_row   = grp[grp["version"]=="ML_SIZE"]
            if base_row.empty:
                continue
            b = base_row.iloc[0]
            print(f"\n{strat}/{tf}:")
            print(f"  {'':12} {'Trades':>7} {'PF':>8} {'Sharpe':>8} {'MDD':>9} {'Return':>9}")
            print(f"  {'BASE':12} {int(b['trades']):>7} {b['pf']:>8.3f} "
                  f"{b['sharpe']:>8.3f} {b['mdd']:>8.2%} {b['total_return']:>8.2%}")
            if not ml_row.empty:
                ml = ml_row.iloc[0]
                dpf = ml['pf']-b['pf']
                dsh = ml['sharpe']-b['sharpe']
                dmdd= ml['mdd']-b['mdd']
                drt = ml['total_return']-b['total_return']
                print(f"  {'ML_SIZE':12} {int(ml['trades']):>7} {ml['pf']:>8.3f} "
                      f"{ml['sharpe']:>8.3f} {ml['mdd']:>8.2%} {ml['total_return']:>8.2%}")
                print(f"  {'delta':12} {'0':>7} {dpf:>+8.3f} "
                      f"{dsh:>+8.3f} {dmdd:>+8.2%} {drt:>+8.2%}")
            else:
                print(f"  ML_SIZE: not trained (insufficient trades or low variance)")


if __name__ == "__main__":
    main()