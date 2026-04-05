"""
Parameter optimization engine.

Supports:
- Grid search
- Random search
- Sensitivity analysis around optimum
"""

from __future__ import annotations

import itertools
import random
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest, SimParams
from src.backtest.metrics import PerformanceMetrics
from src.strategies.base import BaseStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OptResult:
    params: Dict[str, Any]
    metrics: PerformanceMetrics
    strategy_name: str
    timeframe: str

    def score(self) -> float:
        """Primary ranking score: Sharpe * PF / max(drawdown, 0.01)"""
        if self.metrics.trade_count == 0:
            return -999.0
        pf = min(self.metrics.profit_factor, 10.0)  # cap to avoid inf dominating
        return self.metrics.sharpe_ratio * pf / max(self.metrics.max_drawdown, 0.01)

    def to_dict(self) -> Dict[str, Any]:
        d = {"strategy": self.strategy_name, "timeframe": self.timeframe}
        d.update(self.params)
        d.update(self.metrics.to_dict())
        d["opt_score"] = self.score()
        return d


def _make_param_grid(param_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
    """Expand a parameter range dict into a list of all combinations."""
    keys = list(param_ranges.keys())
    values = list(param_ranges.values())
    combos = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combos]


def _random_sample_params(param_ranges: Dict[str, List], n: int) -> List[Dict[str, Any]]:
    """Random sample n parameter combinations from ranges."""
    keys = list(param_ranges.keys())
    values = list(param_ranges.values())
    samples = []
    for _ in range(n):
        combo = {k: random.choice(v) for k, v in zip(keys, values)}
        samples.append(combo)
    return samples


def run_optimization(
    strategy_class: type,
    param_ranges: Dict[str, List],
    df: pd.DataFrame,
    sim_params: SimParams,
    method: str = "grid",
    n_random: int = 500,
    min_trades: int = 30,
    timeframe: str = "",
    atr_col: Optional[str] = None,
    trailing_atr_mult: float = 0.0,
    max_bars_in_trade: int = 0,
) -> List[OptResult]:
    """
    Run parameter optimization for one strategy on one dataset slice.

    Args:
        strategy_class: Uninstantiated strategy class.
        param_ranges: Dict of {param_name: [value1, value2, ...]}.
        df: Feature DataFrame (already sliced to in-sample period).
        sim_params: Simulation parameters.
        method: "grid" or "random".
        n_random: Number of random samples (method=random only).
        min_trades: Minimum trade count to include result.
        timeframe: Label for result tagging.
        atr_col: ATR column name for trailing stop (if used).
        trailing_atr_mult: Trailing stop ATR multiplier.
        max_bars_in_trade: Max bars before forced exit.

    Returns:
        List of OptResult, sorted by score descending.
    """
    if method == "grid":
        param_sets = _make_param_grid(param_ranges)
    else:
        param_sets = _random_sample_params(param_ranges, n_random)

    logger.info(
        f"[{strategy_class.__name__}] Optimizing {len(param_sets)} param sets "
        f"on {len(df)} bars [{timeframe}] ..."
    )

    results: List[OptResult] = []

    for params in param_sets:
        try:
            strategy = strategy_class(params)
            sig_result = strategy.generate_signals(df)

            atr_series = df[atr_col] if atr_col and atr_col in df.columns else None

            bt_result = run_backtest(
                df=df,
                signals=sig_result.signals,
                stop_losses=sig_result.stop_losses,
                take_profits=sig_result.take_profits,
                params=sim_params,
                atr_series=atr_series,
                trailing_atr_mult=trailing_atr_mult,
                max_bars_in_trade=max_bars_in_trade,
            )

            if bt_result.metrics.trade_count < min_trades:
                continue

            opt_result = OptResult(
                params=params,
                metrics=bt_result.metrics,
                strategy_name=strategy.name,
                timeframe=timeframe,
            )
            results.append(opt_result)

        except Exception as e:
            logger.debug(f"Param set {params} failed: {e}")
            continue

    results.sort(key=lambda r: r.score(), reverse=True)
    logger.info(f"  -> {len(results)} valid results, best score: "
                f"{results[0].score():.3f}" if results else "  -> 0 valid results.")
    return results


def sensitivity_analysis(
    strategy_class: type,
    best_params: Dict[str, Any],
    param_ranges: Dict[str, List],
    df: pd.DataFrame,
    sim_params: SimParams,
    min_trades: int = 10,
    timeframe: str = "",
    atr_col: Optional[str] = None,
    trailing_atr_mult: float = 0.0,
) -> pd.DataFrame:
    """
    For each parameter, vary it while holding others at best value.
    Returns a DataFrame showing metric sensitivity per parameter value.
    """
    rows = []
    for param_name, values in param_ranges.items():
        for val in values:
            test_params = dict(best_params)
            test_params[param_name] = val
            try:
                strategy = strategy_class(test_params)
                sig_result = strategy.generate_signals(df)
                atr_series = df[atr_col] if atr_col and atr_col in df.columns else None
                bt = run_backtest(
                    df=df,
                    signals=sig_result.signals,
                    stop_losses=sig_result.stop_losses,
                    take_profits=sig_result.take_profits,
                    params=sim_params,
                    atr_series=atr_series,
                    trailing_atr_mult=trailing_atr_mult,
                )
                if bt.metrics.trade_count >= min_trades:
                    rows.append({
                        "param": param_name,
                        "value": val,
                        "is_best": val == best_params.get(param_name),
                        "sharpe": bt.metrics.sharpe_ratio,
                        "profit_factor": bt.metrics.profit_factor,
                        "max_drawdown": bt.metrics.max_drawdown,
                        "total_return": bt.metrics.total_return,
                        "trade_count": bt.metrics.trade_count,
                    })
            except Exception:
                continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()