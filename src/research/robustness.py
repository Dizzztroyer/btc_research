"""
Robustness tests and validation framework.

Implements:
- IS/OOS/Test three-way split evaluation
- Stress testing with elevated fees/slippage
- Neighbor stability check
- Robustness scoring and rejection logic
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest, SimParams
from src.backtest.metrics import PerformanceMetrics
from src.research.optimizer import run_optimization, OptResult, sensitivity_analysis
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RobustnessReport:
    strategy_name: str
    timeframe: str
    params: Dict[str, Any]
    is_metrics: PerformanceMetrics       # in-sample
    val_metrics: PerformanceMetrics      # validation
    test_metrics: PerformanceMetrics     # held-out test
    stress_metrics: PerformanceMetrics   # stress test
    sensitivity_df: pd.DataFrame
    neighbor_pass_rate: float            # fraction of neighbors that are profitable
    is_robust: bool
    rejection_reasons: List[str]
    robustness_score: float


def evaluate_robustness(
    strategy_class: type,
    param_ranges: Dict[str, List],
    df: pd.DataFrame,
    sim_params: SimParams,
    robustness_cfg,
    opt_cfg,
    timeframe: str = "",
    atr_col: Optional[str] = None,
    trailing_atr_mult: float = 0.0,
) -> RobustnessReport:
    """
    Full robustness evaluation pipeline for one strategy/timeframe.

    Steps:
    1. Split into IS / validation / test
    2. Optimize on IS
    3. Evaluate on validation
    4. Evaluate on test (held out)
    5. Stress test
    6. Sensitivity / neighbor analysis
    7. Score and flag robust vs overfit
    """
    n = len(df)
    is_end = int(n * opt_cfg.in_sample_ratio)
    val_end = is_end + int(n * opt_cfg.validation_ratio)

    is_df = df.iloc[:is_end]
    val_df = df.iloc[is_end:val_end]
    test_df = df.iloc[val_end:]

    logger.info(
        f"[{strategy_class.__name__}|{timeframe}] "
        f"IS={len(is_df)} / Val={len(val_df)} / Test={len(test_df)} bars"
    )

    # Step 1: Optimize on IS
    opt_results = run_optimization(
        strategy_class=strategy_class,
        param_ranges=param_ranges,
        df=is_df,
        sim_params=sim_params,
        method=opt_cfg.method,
        n_random=opt_cfg.n_random_samples,
        min_trades=opt_cfg.min_trades,
        timeframe=timeframe,
        atr_col=atr_col,
        trailing_atr_mult=trailing_atr_mult,
    )

    if not opt_results:
        empty_m = PerformanceMetrics(initial_capital=sim_params.initial_capital)
        return RobustnessReport(
            strategy_name=strategy_class.__name__,
            timeframe=timeframe,
            params={},
            is_metrics=empty_m,
            val_metrics=empty_m,
            test_metrics=empty_m,
            stress_metrics=empty_m,
            sensitivity_df=pd.DataFrame(),
            neighbor_pass_rate=0.0,
            is_robust=False,
            rejection_reasons=["No valid IS parameter sets found"],
            robustness_score=0.0,
        )

    best = opt_results[0]
    best_params = best.params

    def _run_on_slice(slice_df: pd.DataFrame, sp: SimParams) -> PerformanceMetrics:
        if slice_df.empty:
            return PerformanceMetrics(initial_capital=sp.initial_capital)
        strat = strategy_class(best_params)
        sig = strat.generate_signals(slice_df)
        atr_s = slice_df[atr_col] if atr_col and atr_col in slice_df.columns else None
        bt = run_backtest(
            df=slice_df,
            signals=sig.signals,
            stop_losses=sig.stop_losses,
            take_profits=sig.take_profits,
            params=sp,
            atr_series=atr_s,
            trailing_atr_mult=trailing_atr_mult,
        )
        return bt.metrics

    # Step 2: IS metrics
    is_metrics = best.metrics

    # Step 3: Validation metrics
    val_metrics = _run_on_slice(val_df, sim_params)

    # Step 4: Test metrics (held out — only looked at in final evaluation)
    test_metrics = _run_on_slice(test_df, sim_params)

    # Step 5: Stress test (2x fees and slippage)
    stress_sp = SimParams(
        initial_capital=sim_params.initial_capital,
        fee_rate=sim_params.fee_rate * robustness_cfg.stress_fee_multiplier,
        slippage_rate=sim_params.slippage_rate * robustness_cfg.stress_fee_multiplier,
        leverage=sim_params.leverage,
        risk_per_trade=sim_params.risk_per_trade,
        direction=sim_params.direction,
    )
    stress_metrics = _run_on_slice(val_df, stress_sp)

    # Step 6: Sensitivity analysis
    sensitivity_df = sensitivity_analysis(
        strategy_class=strategy_class,
        best_params=best_params,
        param_ranges=param_ranges,
        df=is_df,
        sim_params=sim_params,
        min_trades=10,
        timeframe=timeframe,
        atr_col=atr_col,
        trailing_atr_mult=trailing_atr_mult,
    )

    # Neighbor pass rate: fraction of param neighbors with profit_factor > 1.0
    neighbor_pass_rate = 0.0
    if not sensitivity_df.empty and "profit_factor" in sensitivity_df.columns:
        non_best = sensitivity_df[~sensitivity_df["is_best"]]
        if len(non_best) > 0:
            profitable = (non_best["profit_factor"] > 1.0).sum()
            neighbor_pass_rate = profitable / len(non_best)

    # Step 7: Robustness scoring and rejection
    rejection_reasons: List[str] = []

    if val_metrics.trade_count < robustness_cfg.min_trades:
        rejection_reasons.append(f"Insufficient trades on validation: {val_metrics.trade_count}")
    if val_metrics.profit_factor < robustness_cfg.min_profit_factor:
        rejection_reasons.append(
            f"Validation PF below threshold: {val_metrics.profit_factor:.2f} < {robustness_cfg.min_profit_factor}"
        )
    if val_metrics.max_drawdown > robustness_cfg.max_drawdown:
        rejection_reasons.append(
            f"Validation drawdown too high: {val_metrics.max_drawdown:.2%} > {robustness_cfg.max_drawdown:.2%}"
        )
    if val_metrics.sharpe_ratio < robustness_cfg.min_sharpe:
        rejection_reasons.append(
            f"Validation Sharpe too low: {val_metrics.sharpe_ratio:.2f} < {robustness_cfg.min_sharpe}"
        )
    if neighbor_pass_rate < robustness_cfg.neighbor_pass_rate:
        rejection_reasons.append(
            f"Parameter instability: neighbor pass rate {neighbor_pass_rate:.2%} < {robustness_cfg.neighbor_pass_rate:.2%}"
        )

    is_robust = len(rejection_reasons) == 0

    # Composite robustness score
    robustness_score = (
        val_metrics.sharpe_ratio * 0.3
        + min(val_metrics.profit_factor, 5.0) * 0.2
        + (1 - val_metrics.max_drawdown) * 0.2
        + neighbor_pass_rate * 0.15
        + stress_metrics.sharpe_ratio * 0.15
    ) if is_robust else 0.0

    return RobustnessReport(
        strategy_name=strategy_class.__name__,
        timeframe=timeframe,
        params=best_params,
        is_metrics=is_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        stress_metrics=stress_metrics,
        sensitivity_df=sensitivity_df,
        neighbor_pass_rate=neighbor_pass_rate,
        is_robust=is_robust,
        rejection_reasons=rejection_reasons,
        robustness_score=robustness_score,
    )