"""
Walk-forward validation.

Divides data into rolling train/test windows.
Optimizes on train window, evaluates on test window.
Aggregates out-of-sample performance.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest, SimParams, BacktestResult
from src.backtest.metrics import PerformanceMetrics, compute_metrics, TradeRecord
from src.research.optimizer import run_optimization, OptResult
from src.strategies.base import BaseStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WFWindow:
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: Dict[str, Any]
    train_metrics: PerformanceMetrics
    test_metrics: PerformanceMetrics
    test_trades: List[TradeRecord]
    test_equity: pd.Series


@dataclass
class WalkForwardResult:
    windows: List[WFWindow]
    combined_oos_metrics: PerformanceMetrics
    combined_oos_equity: pd.Series
    combined_oos_trades: List[TradeRecord]
    oos_yearly: pd.DataFrame
    strategy_name: str
    timeframe: str
    final_params: Dict[str, Any]   # params from last window (or most common)


def build_wf_windows(
    df: pd.DataFrame,
    n_windows: int,
    train_ratio: float = 0.7,
    anchored: bool = False,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Divide df into (train, test) window pairs.

    Args:
        df: Full feature DataFrame.
        n_windows: Number of WF windows.
        train_ratio: Fraction of each window used for training.
        anchored: If True, training always starts from df start.

    Returns:
        List of (train_df, test_df) tuples.
    """
    n = len(df)
    window_size = n // n_windows
    pairs = []

    for i in range(n_windows):
        if anchored:
            window_start = 0
        else:
            window_start = i * window_size

        window_end = window_start + window_size
        if i == n_windows - 1:
            window_end = n  # last window gets remainder

        train_end_idx = window_start + int((window_end - window_start) * train_ratio)
        train_df = df.iloc[window_start:train_end_idx]
        test_df = df.iloc[train_end_idx:window_end]

        if len(train_df) > 50 and len(test_df) > 20:
            pairs.append((train_df, test_df))

    return pairs


def run_walk_forward(
    strategy_class: type,
    param_ranges: Dict[str, List],
    df: pd.DataFrame,
    sim_params: SimParams,
    n_windows: int = 5,
    train_ratio: float = 0.7,
    anchored: bool = False,
    method: str = "grid",
    n_random: int = 300,
    min_trades: int = 15,
    timeframe: str = "",
    atr_col: str = None,
    trailing_atr_mult: float = 0.0,
    max_bars_in_trade: int = 0,
) -> WalkForwardResult:
    """
    Full walk-forward optimization and evaluation.

    Args:
        strategy_class: Strategy class to test.
        param_ranges: Dict of parameter search ranges.
        df: Full feature DataFrame.
        sim_params: Simulation parameters.
        n_windows: Number of WF windows.
        train_ratio: Train fraction per window.
        anchored: If True, training grows from fixed start.
        method: "grid" or "random".
        n_random: Random samples if method=random.
        min_trades: Minimum trades per window.
        timeframe: Label.
        atr_col: ATR column for trailing stop.
        trailing_atr_mult: Trailing stop multiplier.
        max_bars_in_trade: Time-based exit.

    Returns:
        WalkForwardResult with per-window and combined OOS metrics.
    """
    windows_data = build_wf_windows(df, n_windows, train_ratio, anchored)
    windows: List[WFWindow] = []
    all_oos_trades: List[TradeRecord] = []
    all_oos_equity_segments: List[pd.Series] = []
    last_best_params: Dict[str, Any] = {}

    for w_id, (train_df, test_df) in enumerate(windows_data):
        logger.info(
            f"  WF window {w_id + 1}/{len(windows_data)}: "
            f"train={train_df.index[0].date()} to {train_df.index[-1].date()}, "
            f"test={test_df.index[0].date()} to {test_df.index[-1].date()}"
        )

        # Optimize on train
        opt_results = run_optimization(
            strategy_class=strategy_class,
            param_ranges=param_ranges,
            df=train_df,
            sim_params=sim_params,
            method=method,
            n_random=n_random,
            min_trades=min_trades,
            timeframe=timeframe,
            atr_col=atr_col,
            trailing_atr_mult=trailing_atr_mult,
            max_bars_in_trade=max_bars_in_trade,
        )

        if not opt_results:
            logger.warning(f"  WF window {w_id + 1}: No valid param sets found on train.")
            continue

        best = opt_results[0]
        last_best_params = best.params

        # Evaluate on test (out-of-sample)
        strategy = strategy_class(best.params)
        sig_result = strategy.generate_signals(test_df)

        atr_series = test_df[atr_col] if atr_col and atr_col in test_df.columns else None

        test_bt = run_backtest(
            df=test_df,
            signals=sig_result.signals,
            stop_losses=sig_result.stop_losses,
            take_profits=sig_result.take_profits,
            params=sim_params,
            atr_series=atr_series,
            trailing_atr_mult=trailing_atr_mult,
            max_bars_in_trade=max_bars_in_trade,
        )

        wf_window = WFWindow(
            window_id=w_id,
            train_start=train_df.index[0],
            train_end=train_df.index[-1],
            test_start=test_df.index[0],
            test_end=test_df.index[-1],
            best_params=best.params,
            train_metrics=best.metrics,
            test_metrics=test_bt.metrics,
            test_trades=test_bt.trades,
            test_equity=test_bt.equity_curve,
        )
        windows.append(wf_window)
        all_oos_trades.extend(test_bt.trades)
        all_oos_equity_segments.append(test_bt.equity_curve)

    # Combine OOS equity (chain segments)
    if all_oos_equity_segments:
        combined_equity = pd.concat(all_oos_equity_segments).sort_index()
        combined_equity = combined_equity[~combined_equity.index.duplicated(keep="first")]
    else:
        combined_equity = pd.Series(dtype=float)

    combined_metrics = compute_metrics(
        trades=all_oos_trades,
        equity_curve=combined_equity,
        initial_capital=sim_params.initial_capital,
        total_bars=len(df),
    )

    from src.backtest.metrics import yearly_breakdown
    oos_yearly = yearly_breakdown(all_oos_trades, combined_equity, sim_params.initial_capital)

    strat_inst = strategy_class({})
    return WalkForwardResult(
        windows=windows,
        combined_oos_metrics=combined_metrics,
        combined_oos_equity=combined_equity,
        combined_oos_trades=all_oos_trades,
        oos_yearly=oos_yearly,
        strategy_name=strat_inst.name,
        timeframe=timeframe,
        final_params=last_best_params,
    )