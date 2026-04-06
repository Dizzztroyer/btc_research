#!/usr/bin/env python3
"""
run_research.py
───────────────
Main research orchestrator.

Workflow
────────
1. Load configuration
2. For each enabled strategy family × each locally stored timeframe:
        a. Load feature DataFrame from disk
        b. Run parameter optimisation (IS/OOS split)
        c. Run walk-forward validation on best params
        d. Run stress test on best params
        e. Compute year-by-year breakdown
        f. Collect results
3. Rank all results by robustness score
4. Generate all reports (CSV, HTML, plots)

Usage
─────
    python run_research.py
    python run_research.py --tf 4h 1d             # restrict timeframes
    python run_research.py --strategy trend       # restrict strategy families
    python run_research.py --tf 1h --strategy trend mean_reversion
    python run_research.py --no-wf               # skip walk-forward (faster)
    python run_research.py --direction long       # long-only
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config_loader import load_config, Config
from src.utils.logger import get_logger
from src.features.feature_engine import FeatureEngine
from src.backtest.engine import BacktestEngine, SimConfig
from src.backtest.metrics import yearly_breakdown
from src.strategies import STRATEGY_FAMILIES, STRATEGY_REGISTRY
from src.strategies.base import BaseStrategy
from src.research.optimizer import ParameterOptimizer
from src.research.walk_forward import WalkForwardEngine
from src.research.reporter import Reporter

logger = get_logger(__name__, log_file=Path("outputs/logs/research.log"))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BTC Strategy Research Platform")
    p.add_argument("--config",    default="config/config.yaml")
    p.add_argument("--tf",        nargs="+", default=None, metavar="TIMEFRAME",
                   help="Restrict to specific timeframes")
    p.add_argument("--strategy",  nargs="+", default=None, metavar="FAMILY",
                   help="Restrict to specific strategy families")
    p.add_argument("--direction", default="both", choices=["long", "short", "both"])
    p.add_argument("--no-wf",     action="store_true", help="Skip walk-forward validation")
    p.add_argument("--no-stress", action="store_true", help="Skip stress tests")
    p.add_argument("--top-n",     type=int, default=None, help="Override top_n for reports")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _available_timeframes(cfg: Config, requested: Optional[List[str]]) -> List[str]:
    """Return locally available feature timeframes, optionally filtered."""
    paths = sorted(cfg.features_dir.glob("*_features.parquet"))
    avail = [p.stem.replace("_features", "") for p in paths]
    if not avail:
        # Fall back to raw dir
        paths = sorted(cfg.raw_dir.glob("*.parquet"))
        avail = [p.stem for p in paths]
    if requested:
        filtered = [t for t in requested if t in avail]
        missing  = [t for t in requested if t not in avail]
        if missing:
            logger.warning(f"Requested timeframes not found locally: {missing}")
        return filtered
    return avail


def _load_features(cfg: Config, timeframe: str) -> pd.DataFrame:
    """Load feature parquet for one timeframe, building if necessary."""
    engine = FeatureEngine(cfg)
    df     = engine.load(timeframe)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # Apply date range filter from config
    if cfg.start_date:
        start = pd.Timestamp(cfg.start_date, tz="UTC")
        df    = df[df["timestamp"] >= start]
    if cfg.end_date:
        end = pd.Timestamp(cfg.end_date, tz="UTC")
        df  = df[df["timestamp"] <= end]
    return df.reset_index(drop=True)


def _stress_test(
    strategy:  BaseStrategy,
    df:        pd.DataFrame,
    params:    dict,
    cfg:       Config,
    direction: str,
) -> dict:
    """Run backtest with 2× fees and 2× slippage; return metrics."""
    sim = SimConfig(
        fees           = cfg.fees    * cfg.validation.stress_fee_mult,
        slippage       = cfg.slippage * cfg.validation.stress_slip_mult,
        leverage       = cfg.leverage,
        risk_per_trade = cfg.risk_per_trade,
        direction      = direction,
    )
    engine = BacktestEngine(sim)
    # Use the OOS/test slice for stress
    n      = len(df)
    test_start = int(n * (cfg.validation.is_ratio + cfg.validation.oos_ratio))
    df_test    = df.iloc[test_start:] if test_start < n else df
    if len(df_test) < 30:
        df_test = df  # fall back to full

    try:
        df_sig  = strategy.generate_signals(df_test.copy(), params)
        result  = engine.run(df_sig, strategy.name, "", params)
        return result.metrics
    except Exception:
        return {}


def _extract_best_params(opt_df: pd.DataFrame) -> dict:
    """Extract parameter dict from top row of optimiser output."""
    if opt_df.empty:
        return {}
    row    = opt_df.iloc[0]
    p_cols = [c for c in row.index if c.startswith("p_")]
    return {c[2:]: row[c] for c in p_cols}


def _make_sim_config(cfg: Config, direction: str) -> SimConfig:
    return SimConfig(
        fees           = cfg.fees,
        slippage       = cfg.slippage,
        leverage       = cfg.leverage,
        risk_per_trade = cfg.risk_per_trade,
        direction      = direction,
    )


# ── Main research loop ─────────────────────────────────────────────────────────

def run_research(
    cfg:        Config,
    timeframes: List[str],
    families:   List[str],
    direction:  str,
    skip_wf:    bool,
    skip_stress:bool,
    top_n:      int,
) -> None:
    """
    Full research loop:
    strategies × timeframes → optimise → WF → stress → report
    """
    optimizer = ParameterOptimizer(cfg)
    wf_engine = WalkForwardEngine(cfg)
    reporter  = Reporter(cfg)

    all_opt_results:  List[pd.DataFrame] = []
    wf_summary_rows:  List[dict] = []
    stress_rows:      List[dict] = []
    equity_plots:     List[Path] = []
    rejected_rows:    List[dict] = []

    best_yearly       = pd.DataFrame()
    best_result_seen  = None     # (robustness_score, equity, drawdown, timestamps, name)

    total = sum(
        len(STRATEGY_FAMILIES[f])
        for f in families
        if f in STRATEGY_FAMILIES
    ) * len(timeframes)

    logger.info(f"Research plan: {len(families)} families × {len(timeframes)} timeframes")
    logger.info(f"Total runs: ~{total} (strategy × timeframe)")

    run_count = 0

    for family_name in families:
        if family_name not in STRATEGY_FAMILIES:
            logger.warning(f"Unknown strategy family: {family_name} — skipping")
            continue

        strategy_classes = STRATEGY_FAMILIES[family_name]

        for StratClass in strategy_classes:
            strategy = StratClass()
            logger.info(f"\n{'='*60}")
            logger.info(f"Strategy: {strategy.name} (family: {family_name})")
            logger.info(f"{'='*60}")

            for timeframe in timeframes:
                run_count += 1
                logger.info(f"\n[{run_count}/{total}] {strategy.name} / {timeframe}")
                t0 = time.time()

                # ── Load features ──────────────────────────────────────────────
                try:
                    df = _load_features(cfg, timeframe)
                except Exception as exc:
                    logger.error(f"  Cannot load features for {timeframe}: {exc}")
                    continue

                if len(df) < 200:
                    logger.warning(f"  Skipping {timeframe}: only {len(df)} rows")
                    continue

                # ── Parameter optimisation ────────────────────────────────────
                try:
                    opt_df = optimizer.optimize(
                        strategy  = strategy,
                        df        = df,
                        timeframe = timeframe,
                        direction = direction,
                    )
                except Exception as exc:
                    logger.error(f"  Optimisation failed: {exc}\n{traceback.format_exc()}")
                    continue

                if opt_df.empty:
                    logger.warning(f"  No valid results — skipping")
                    rejected_rows.append({
                        "strategy": strategy.name, "timeframe": timeframe,
                        "reason": "empty optimiser output",
                    })
                    continue

                all_opt_results.append(opt_df)

                best_params = _extract_best_params(opt_df)
                best_row    = opt_df.iloc[0]

                # ── Reject filters ─────────────────────────────────────────────
                oos_pf     = best_row.get("oos_pf",     0) or 0
                oos_trades = best_row.get("oos_trades",  0) or 0
                robustness = best_row.get("robustness",  -9999) or -9999

                if oos_trades < cfg.validation.min_trades:
                    logger.warning(f"  Rejected: OOS trades={oos_trades} < {cfg.validation.min_trades}")
                    rejected_rows.append({
                        "strategy": strategy.name, "timeframe": timeframe,
                        "reason": f"OOS trades={oos_trades} < min",
                        **{f"p_{k}": v for k, v in best_params.items()},
                    })
                    continue

                if oos_pf < cfg.validation.min_profit_factor:
                    logger.warning(f"  Rejected: OOS PF={oos_pf:.2f} < {cfg.validation.min_profit_factor}")
                    rejected_rows.append({
                        "strategy": strategy.name, "timeframe": timeframe,
                        "reason": f"OOS PF={oos_pf:.2f} < min",
                    })
                    continue

                logger.info(
                    f"  ✓ Accepted | OOS PF={oos_pf:.2f} | "
                    f"OOS Sharpe={best_row.get('oos_sharpe', float('nan')):.2f} | "
                    f"Robustness={robustness:.2f}"
                )

                # ── Walk-forward validation ────────────────────────────────────
                if not skip_wf:
                    try:
                        wf_result = wf_engine.run(
                            strategy  = strategy,
                            df        = df,
                            timeframe = timeframe,
                            direction = direction,
                        )
                        wf_m = wf_result.combined_metrics
                        wf_summary_rows.append({
                            "strategy":    strategy.name,
                            "timeframe":   timeframe,
                            "wf_sharpe":   wf_m.get("sharpe",        np.nan),
                            "wf_pf":       wf_m.get("profit_factor", np.nan),
                            "wf_return":   wf_m.get("total_return",  np.nan),
                            "wf_mdd":      wf_m.get("max_drawdown",  np.nan),
                            "wf_trades":   wf_m.get("trade_count",   0),
                            "n_windows":   len(wf_result.windows),
                        })

                        # Use WF equity for best-result tracking
                        if (
                            not wf_result.combined_equity.empty and
                            (best_result_seen is None or robustness > best_result_seen[0])
                        ):
                            best_result_seen = (
                                robustness,
                                wf_result.combined_equity,
                                wf_result.combined_equity,  # placeholder for drawdown
                                None,
                                f"{strategy.name}_{timeframe}_wf",
                            )
                    except Exception as exc:
                        logger.warning(f"  Walk-forward failed: {exc}")

                # ── Stress test ────────────────────────────────────────────────
                if not skip_stress:
                    try:
                        stress_m = _stress_test(strategy, df, best_params, cfg, direction)
                        stress_rows.append({
                            "strategy":     strategy.name,
                            "timeframe":    timeframe,
                            "stress_sharpe":stress_m.get("sharpe",        np.nan),
                            "stress_pf":    stress_m.get("profit_factor", np.nan),
                            "stress_return":stress_m.get("total_return",  np.nan),
                            "stress_mdd":   stress_m.get("max_drawdown",  np.nan),
                            "stress_trades":stress_m.get("trade_count",   0),
                        })
                    except Exception as exc:
                        logger.warning(f"  Stress test failed: {exc}")

                # ── Full-sample backtest for equity plot ───────────────────────
                if cfg.reporting.plot_equity_curves:
                    try:
                        sim = _make_sim_config(cfg, direction)
                        eng = BacktestEngine(sim)
                        df_sig = strategy.generate_signals(df.copy(), best_params)
                        res    = eng.run(df_sig, strategy.name, timeframe, best_params)

                        if res.ok:
                            plot_name = f"{strategy.name}_{timeframe}"
                            ts = df_sig["timestamp"] if "timestamp" in df_sig.columns else None
                            p  = reporter.plot_equity_curve(
                                equity    = res.equity,
                                drawdown  = res.drawdown,
                                name      = plot_name,
                                timestamps = ts,
                            )
                            equity_plots.append(p)

                            # Track best for yearly breakdown
                            if best_result_seen is None or robustness > best_result_seen[0]:
                                best_result_seen = (
                                    robustness, res.equity, res.drawdown, ts, plot_name
                                )
                                # Compute yearly breakdown
                                best_yearly = yearly_breakdown(
                                    trades          = res.trades,
                                    equity          = res.equity,
                                    timestamps      = df_sig["timestamp"],
                                    bars_per_year   = _estimate_bpy(df),
                                    initial_capital = cfg.risk_per_trade,
                                )
                    except Exception as exc:
                        logger.warning(f"  Equity plot failed: {exc}")

                elapsed = time.time() - t0
                logger.info(f"  Completed in {elapsed:.1f}s")

    # ── Combine all optimiser results ──────────────────────────────────────────
    if not all_opt_results:
        logger.error("No valid results produced. Exiting.")
        return

    combined = pd.concat(all_opt_results, ignore_index=True)
    combined = combined.sort_values("robustness", ascending=False).reset_index(drop=True)
    rejected_df = pd.DataFrame(rejected_rows)

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    reporter.save_rankings(combined, "all_results.csv")

    if wf_summary_rows:
        reporter.save_wf_summary(wf_summary_rows, "wf_summary.csv")

    if stress_rows:
        reporter.save_stress_summary(stress_rows)

    if not best_yearly.empty:
        reporter.save_yearly_breakdown(best_yearly, "best_strategy")

    # ── Heatmaps ───────────────────────────────────────────────────────────────
    heatmap_plots: List[Path] = []
    if cfg.reporting.plot_heatmaps and "strategy" in combined.columns:
        for metric in ["oos_sharpe", "oos_pf", "oos_return"]:
            try:
                p = reporter.plot_tf_strategy_heatmap(combined, metric=metric)
                if p and p.exists():
                    heatmap_plots.append(p)
            except Exception as exc:
                logger.warning(f"Heatmap failed ({metric}): {exc}")

    # ── HTML report ────────────────────────────────────────────────────────────
    if cfg.reporting.html_report:
        try:
            reporter.generate_html_report(
                all_results   = combined,
                wf_rows       = wf_summary_rows,
                best_yearly   = best_yearly,
                rejected      = rejected_df,
                equity_plots  = equity_plots,
                heatmap_plots = heatmap_plots,
                top_n         = top_n,
            )
        except Exception as exc:
            logger.error(f"HTML report failed: {exc}\n{traceback.format_exc()}")

    # ── Final console summary ──────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("RESEARCH COMPLETE")
    logger.info("="*60)
    logger.info(f"Total result rows : {len(combined)}")
    logger.info(f"Rejected          : {len(rejected_rows)}")
    logger.info(f"WF runs completed : {len(wf_summary_rows)}")
    logger.info("")
    logger.info("TOP 10 by Robustness:")
    display_cols = ["strategy", "timeframe", "oos_sharpe", "oos_pf",
                    "oos_return", "oos_drawdown", "oos_trades", "robustness"]
    cols = [c for c in display_cols if c in combined.columns]
    with pd.option_context("display.max_columns", None, "display.width", 120):
        logger.info("\n" + combined[cols].head(10).to_string(index=False))

    logger.info(f"\nOutputs saved to: {cfg.output_dir}")


def _estimate_bpy(df: pd.DataFrame) -> float:
    if len(df) < 2 or "timestamp" not in df.columns:
        return 365.0
    ts    = pd.to_datetime(df["timestamp"])
    total = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
    n     = len(ts) - 1
    if total <= 0:
        return 365.0
    return 365.25 * 86_400 / (total / n)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = load_config(Path(args.config))

    # ── Resolve timeframes ─────────────────────────────────────────────────────
    timeframes = _available_timeframes(cfg, args.tf)
    if not timeframes:
        logger.error(
            "No locally stored timeframes found.\n"
            "Run: python download_all_timeframes.py\n"
            "     python build_features.py"
        )
        sys.exit(1)

    logger.info(f"Timeframes to research: {timeframes}")

    # ── Resolve strategy families ──────────────────────────────────────────────
    if args.strategy:
        families = args.strategy
    else:
        families = cfg.enabled_strategies

    unknown = [f for f in families if f not in STRATEGY_FAMILIES]
    if unknown:
        logger.warning(f"Unknown strategy families: {unknown}")
    families = [f for f in families if f in STRATEGY_FAMILIES]

    if not families:
        logger.error("No valid strategy families selected.")
        sys.exit(1)

    logger.info(f"Strategy families: {families}")
    logger.info(f"Direction        : {args.direction}")
    logger.info(f"Walk-forward     : {'disabled' if args.no_wf else 'enabled'}")
    logger.info(f"Stress test      : {'disabled' if args.no_stress else 'enabled'}")

    top_n = args.top_n or cfg.reporting.top_n

    run_research(
        cfg         = cfg,
        timeframes  = timeframes,
        families    = families,
        direction   = args.direction,
        skip_wf     = args.no_wf,
        skip_stress = args.no_stress,
        top_n       = top_n,
    )


if __name__ == "__main__":
    main()
