#!/usr/bin/env python3
"""
run_research.py

Full research run:
  1. Load feature files for all available timeframes
  2. Run all enabled strategy families
  3. Optimize parameters
  4. Walk-forward validation
  5. Robustness evaluation
  6. Generate reports

Usage:
    python run_research.py
    python run_research.py --config config/config.yaml
    python run_research.py --timeframes 1h 4h --strategies trend_following
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from src.utils.config_loader import load_config, Config
from src.utils.logger import get_logger
from src.features.feature_builder import load_features
from src.backtest.engine import SimParams
from src.strategies import ALL_STRATEGIES, STRATEGY_FAMILIES
from src.research.robustness import evaluate_robustness
from src.research.walk_forward import run_walk_forward

# -- Param ranges per strategy --
PARAM_RANGES: Dict[str, Dict[str, List]] = {
    "EMA_TrendPullback": {
        "trend_ema": [100, 200],
        "fast_ema": [13, 21, 34],
        "slow_ema": [34, 55, 89],
        "atr_length": [14],
        "sl_atr_mult": [1.0, 1.5, 2.0],
        "tp_atr_mult": [2.0, 3.0, 4.0],
        "min_adx": [15, 20, 25],
    },
    "Donchian_Breakout": {
        "dc_length": [10, 20, 40, 55],
        "atr_length": [14],
        "sl_atr_mult": [1.0, 1.5, 2.0],
        "tp_atr_mult": [2.0, 3.0, 4.0],
        "min_adx": [0, 20],
    },
    "EMA_Crossover": {
        "fast_ema": [8, 13, 21],
        "slow_ema": [34, 55, 89],
        "atr_length": [14],
        "sl_atr_mult": [1.5, 2.0, 2.5],
        "tp_atr_mult": [2.0, 3.0, 4.0],
    },
    "Triple_EMA_Trend": {
        "fast_ema": [8, 13],
        "medium_ema": [21, 34],
        "slow_ema": [55, 89],
        "atr_length": [14],
        "sl_atr_mult": [1.0, 1.5, 2.0],
        "tp_atr_mult": [2.0, 3.0],
    },
    "RSI_Reversion": {
        "rsi_length": [7, 14, 21],
        "oversold": [25, 30, 35],
        "overbought": [65, 70, 75],
        "trend_ema": [0, 50, 200],
        "atr_length": [14],
        "sl_atr_mult": [1.5, 2.0],
        "tp_atr_mult": [1.5, 2.0, 3.0],
    },
    "Bollinger_Reversion": {
        "bb_length": [20],
        "bb_std": [2.0, 2.5],
        "rsi_length": [14],
        "oversold_rsi": [35, 40, 45],
        "overbought_rsi": [55, 60, 65],
        "atr_length": [14],
        "sl_atr_mult": [1.5, 2.0],
        "tp_atr_mult": [1.5, 2.0],
    },
    "EMA_Deviation_Reversion": {
        "ema_length": [34, 50, 89],
        "dev_threshold": [0.02, 0.03, 0.05],
        "atr_length": [14],
        "sl_atr_mult": [1.5, 2.0],
        "tp_atr_mult": [1.0, 1.5, 2.0],
    },
    "Squeeze_Breakout": {
        "bb_length": [20],
        "bb_std": [2.0],
        "dc_length": [20, 40],
        "squeeze_percentile": [15, 20, 25],
        "squeeze_window": [100],
        "atr_length": [14],
        "sl_atr_mult": [1.5, 2.0],
        "tp_atr_mult": [2.0, 3.0],
    },
    "Consolidation_Breakout": {
        "lookback": [5, 10, 15],
        "range_atr_mult": [0.75, 1.0, 1.5],
        "atr_length": [14],
        "sl_atr_mult": [1.5, 2.0],
        "tp_atr_mult": [2.0, 3.0],
    },
    "Swing_Breakout": {
        "swing_left": [3, 5],
        "swing_right": [3, 5],
        "atr_length": [14],
        "sl_atr_mult": [1.5, 2.0],
        "tp_atr_mult": [2.0, 3.0],
    },
    "BOS_Structure": {
        "swing_left": [3, 5],
        "swing_right": [3, 5],
        "atr_length": [14],
        "sl_atr_mult": [1.5, 2.0],
        "tp_atr_mult": [2.0, 3.0],
    },
    "Regime_Switch": {
        "trend_fast_ema": [21],
        "trend_slow_ema": [55],
        "rsi_length": [14],
        "oversold": [30],
        "overbought": [70],
        "atr_length": [14],
        "sl_atr_mult": [1.5, 2.0],
        "tp_atr_mult": [2.0, 3.0],
        "skip_high_vol": [True, False],
    },
    "Ensemble_Voting": {
        "min_agreement": [2, 3],
        "atr_length": [14],
        "sl_atr_mult": [1.5, 2.0],
        "tp_atr_mult": [2.0, 3.0],
    },
}


def get_available_timeframes(cfg: Config) -> List[str]:
    feat_dir = cfg.feature_dir()
    if not feat_dir.exists():
        return []
    files = sorted(feat_dir.glob("*_features.parquet"))
    return [f.stem.replace("_features", "") for f in files]


def build_sim_params(cfg: Config) -> SimParams:
    return SimParams(
        initial_capital=cfg.simulation.initial_capital,
        fee_rate=cfg.simulation.fee_rate,
        slippage_rate=cfg.simulation.slippage_rate,
        leverage=cfg.simulation.leverage,
        risk_per_trade=cfg.simulation.risk_per_trade,
        direction=cfg.simulation.direction,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BTC research platform.")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--timeframes", nargs="+", default=None)
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Strategy family names to run (e.g. trend_following mean_reversion)")
    parser.add_argument("--no-wf", action="store_true", help="Skip walk-forward (faster)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_dir = cfg.outputs_dir() / "logs"
    logger = get_logger("run_research", log_dir=log_dir)

    logger.info("=" * 70)
    logger.info("BTC Research Platform — Full Research Run")
    logger.info("=" * 70)

    sim_params = build_sim_params(cfg)

    # Determine timeframes
    available_tfs = get_available_timeframes(cfg)
    if not available_tfs:
        logger.error("No feature files found. Run build_features.py first.")
        return

    timeframes = args.timeframes if args.timeframes else available_tfs
    timeframes = [tf for tf in timeframes if tf in available_tfs]
    if not timeframes:
        logger.error(f"No valid timeframes. Available: {available_tfs}")
        return
    logger.info(f"Timeframes: {timeframes}")

    # Determine strategies
    if args.strategies:
        families = args.strategies
    else:
        families = [k for k, v in cfg.strategies.items() if v]

    strategy_names = []
    for family in families:
        if family in STRATEGY_FAMILIES:
            strategy_names.extend(STRATEGY_FAMILIES[family])
        elif family in ALL_STRATEGIES:
            strategy_names.append(family)
    strategy_names = list(dict.fromkeys(strategy_names))  # deduplicate, preserve order
    logger.info(f"Strategies: {strategy_names}")

    # Output dirs
    reports_dir = cfg.outputs_dir() / "reports"
    rankings_dir = cfg.outputs_dir() / "rankings"
    plots_dir = cfg.outputs_dir() / "plots"
    for d in [reports_dir, rankings_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    wf_results_list = []

    for tf in timeframes:
        logger.info(f"\n{'='*50}")
        logger.info(f"Loading features: {tf}")
        df = load_features(cfg.feature_dir().parent, cfg.data_symbol, tf)

        if df.empty or len(df) < 200:
            logger.warning(f"[{tf}] Insufficient data ({len(df)} rows). Skipping.")
            continue

        atr_col = "atr_14"

        for strat_name in strategy_names:
            if strat_name not in ALL_STRATEGIES:
                logger.warning(f"Unknown strategy: {strat_name}")
                continue
            if strat_name not in PARAM_RANGES:
                logger.warning(f"No param ranges defined for: {strat_name}. Skipping.")
                continue

            strategy_class = ALL_STRATEGIES[strat_name]
            param_ranges = PARAM_RANGES[strat_name]

            logger.info(f"  [{tf}] Robustness evaluation: {strat_name}")
            try:
                rob = evaluate_robustness(
                    strategy_class=strategy_class,
                    param_ranges=param_ranges,
                    df=df,
                    sim_params=sim_params,
                    robustness_cfg=cfg.robustness,
                    opt_cfg=cfg.optimization,
                    timeframe=tf,
                    atr_col=atr_col,
                )

                result_row = {
                    "strategy": strat_name,
                    "timeframe": tf,
                    "is_robust": rob.is_robust,
                    "robustness_score": rob.robustness_score,
                    "rejection_reasons": "; ".join(rob.rejection_reasons),
                    # IS metrics
                    "is_sharpe": rob.is_metrics.sharpe_ratio,
                    "is_pf": rob.is_metrics.profit_factor,
                    "is_return": rob.is_metrics.total_return,
                    "is_maxdd": rob.is_metrics.max_drawdown,
                    "is_trades": rob.is_metrics.trade_count,
                    # Val metrics
                    "val_sharpe": rob.val_metrics.sharpe_ratio,
                    "val_pf": rob.val_metrics.profit_factor,
                    "val_return": rob.val_metrics.total_return,
                    "val_maxdd": rob.val_metrics.max_drawdown,
                    "val_trades": rob.val_metrics.trade_count,
                    # Test metrics
                    "test_sharpe": rob.test_metrics.sharpe_ratio,
                    "test_pf": rob.test_metrics.profit_factor,
                    "test_return": rob.test_metrics.total_return,
                    "test_maxdd": rob.test_metrics.max_drawdown,
                    "test_trades": rob.test_metrics.trade_count,
                    # Stress
                    "stress_sharpe": rob.stress_metrics.sharpe_ratio,
                    "stress_pf": rob.stress_metrics.profit_factor,
                    # Neighbor
                    "neighbor_pass_rate": rob.neighbor_pass_rate,
                }
                result_row.update({f"param_{k}": v for k, v in rob.params.items()})
                all_results.append(result_row)

            except Exception as e:
                logger.error(f"  [{tf}|{strat_name}] Robustness eval failed: {e}", exc_info=True)

            # Walk-forward (optional)
            if not args.no_wf and len(df) > 500:
                logger.info(f"  [{tf}] Walk-forward: {strat_name}")
                try:
                    wf = run_walk_forward(
                        strategy_class=strategy_class,
                        param_ranges=param_ranges,
                        df=df,
                        sim_params=sim_params,
                        n_windows=cfg.optimization.walk_forward.n_windows,
                        train_ratio=cfg.optimization.walk_forward.train_ratio,
                        anchored=cfg.optimization.walk_forward.anchored,
                        method=cfg.optimization.method,
                        n_random=min(100, cfg.optimization.n_random_samples),
                        min_trades=cfg.optimization.min_trades,
                        timeframe=tf,
                        atr_col=atr_col,
                    )
                    wf_results_list.append({
                        "strategy": strat_name,
                        "timeframe": tf,
                        "oos_sharpe": wf.combined_oos_metrics.sharpe_ratio,
                        "oos_pf": wf.combined_oos_metrics.profit_factor,
                        "oos_return": wf.combined_oos_metrics.total_return,
                        "oos_maxdd": wf.combined_oos_metrics.max_drawdown,
                        "oos_trades": wf.combined_oos_metrics.trade_count,
                        "n_windows": len(wf.windows),
                    })

                    # Plot OOS equity
                    if cfg.reporting.plot_equity_curves and not wf.combined_oos_equity.empty:
                        from src.backtest.metrics import compute_metrics
                        rolling_max = wf.combined_oos_equity.cummax()
                        dd = (wf.combined_oos_equity - rolling_max) / rolling_max.replace(0, float("nan"))
                        from src.utils.reporter import plot_equity_curve
                        plot_equity_curve(
                            equity_curve=wf.combined_oos_equity,
                            drawdown_curve=dd,
                            title=f"WF OOS — {strat_name} [{tf}]",
                            output_path=plots_dir / f"wf_{strat_name}_{tf}.png",
                        )

                except Exception as e:
                    logger.error(f"  [{tf}|{strat_name}] Walk-forward failed: {e}", exc_info=True)

    # ---- Generate reports ----
    if not all_results:
        logger.warning("No results generated. Check data availability and strategy configs.")
        return

    from src.utils.reporter import save_csv, build_html_report, plot_heatmap, compile_rankings

    all_df = compile_rankings(all_results)

    # Save full results CSV
    save_csv(all_df, rankings_dir / "all_results.csv", "Full results")

    # Robust strategies only
    if "is_robust" in all_df.columns:
        robust_df = all_df[all_df["is_robust"] == True]
        if not robust_df.empty:
            save_csv(robust_df, rankings_dir / "robust_strategies.csv", "Robust strategies")
            build_html_report(
                robust_df.head(cfg.reporting.top_n_strategies),
                title="Robust Strategies — BTC Research",
                output_path=reports_dir / "robust_strategies.html",
            )

    # Walk-forward results
    if wf_results_list:
        wf_df = pd.DataFrame(wf_results_list).sort_values("oos_sharpe", ascending=False)
        save_csv(wf_df, rankings_dir / "walk_forward_results.csv", "Walk-forward OOS")
        build_html_report(wf_df, title="Walk-Forward OOS Results", output_path=reports_dir / "walk_forward.html")

    # Heatmaps: strategy vs timeframe
    if cfg.reporting.plot_heatmaps and "val_sharpe" in all_df.columns:
        try:
            pivot = all_df.pivot_table(
                index="strategy", columns="timeframe", values="val_sharpe", aggfunc="max"
            )
            plot_heatmap(
                pivot,
                title="Validation Sharpe — Strategy vs Timeframe",
                output_path=plots_dir / "heatmap_sharpe.png",
            )
            pf_pivot = all_df.pivot_table(
                index="strategy", columns="timeframe", values="val_pf", aggfunc="max"
            )
            plot_heatmap(
                pf_pivot,
                title="Validation Profit Factor — Strategy vs Timeframe",
                output_path=plots_dir / "heatmap_pf.png",
                cmap="RdYlGn",
            )
        except Exception as e:
            logger.warning(f"Heatmap generation failed: {e}")

    # Full HTML report
    build_html_report(
        all_df.head(cfg.reporting.top_n_strategies),
        title="BTC Research — Full Strategy Rankings",
        output_path=reports_dir / "full_rankings.html",
    )

    logger.info("=" * 70)
    logger.info(f"Research complete. {len(all_results)} strategy/timeframe combinations evaluated.")
    logger.info(f"Reports saved to: {cfg.outputs_dir()}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()