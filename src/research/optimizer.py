"""
optimizer.py
────────────
Parameter search engine.

Supports:
- Grid search (full exhaustive sweep over param_grid)
- Random search (random sample from param_grid)

For each parameter set:
    1. Run backtest on IS period
    2. Run backtest on OOS period
    3. Record both metric sets

Sensitivity analysis:
    After finding the best IS parameter set, report how performance varies
    for nearby parameter combinations (robustness check).

Outputs:
    DataFrame of all tested combinations sorted by OOS robustness score
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult, SimConfig
from src.strategies.base import BaseStrategy
from src.utils.config_loader import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OptimResult:
    """Single optimization result row."""
    strategy:   str
    timeframe:  str
    params:     dict
    # IS metrics
    is_return:       float
    is_sharpe:       float
    is_pf:           float
    is_trades:       int
    is_drawdown:     float
    # OOS metrics
    oos_return:      float
    oos_sharpe:      float
    oos_pf:          float
    oos_trades:      int
    oos_drawdown:    float
    # Robustness score (composite)
    robustness_score: float

    def to_dict(self) -> dict:
        d = {
            "strategy":  self.strategy,
            "timeframe": self.timeframe,
            **{f"p_{k}": v for k, v in self.params.items()},
            "is_return":    self.is_return,
            "is_sharpe":    self.is_sharpe,
            "is_pf":        self.is_pf,
            "is_trades":    self.is_trades,
            "is_drawdown":  self.is_drawdown,
            "oos_return":   self.oos_return,
            "oos_sharpe":   self.oos_sharpe,
            "oos_pf":       self.oos_pf,
            "oos_trades":   self.oos_trades,
            "oos_drawdown": self.oos_drawdown,
            "robustness":   self.robustness_score,
        }
        return d


def _robustness_score(
    is_m:  dict,
    oos_m: dict,
    min_trades: int,
) -> float:
    """
    Composite robustness score.

    Penalises:
    - Low OOS profit factor
    - High IS/OOS Sharpe ratio decay
    - Insufficient trade count

    Returns float; higher is better.
    """
    oos_pf     = oos_m.get("profit_factor", 0) or 0
    oos_sharpe = oos_m.get("sharpe", -99)       or -99
    oos_trades = oos_m.get("trade_count", 0)    or 0
    is_sharpe  = is_m.get("sharpe", 0)           or 0
    oos_mdd    = abs(oos_m.get("max_drawdown", 1) or 1)

    if oos_trades < min_trades:
        return -1000.0  # reject

    if oos_pf < 1.0:
        return -500.0   # not profitable in OOS

    # Sharpe decay ratio (penalise if OOS sharpe is much worse than IS)
    if is_sharpe > 0:
        decay = oos_sharpe / is_sharpe
    else:
        decay = 1.0

    score = (
        oos_pf        * 2.0    # primary: OOS profitability
        + oos_sharpe  * 1.5    # secondary: OOS risk-adjusted return
        + decay       * 1.0    # penalty for IS→OOS decay
        - oos_mdd     * 0.5    # penalty for large drawdown
    )
    return score


def _split_df(
    df: pd.DataFrame,
    is_ratio: float,
    oos_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into IS, OOS, TEST portions."""
    n       = len(df)
    is_end  = int(n * is_ratio)
    oos_end = int(n * (is_ratio + oos_ratio))
    return df.iloc[:is_end], df.iloc[is_end:oos_end], df.iloc[oos_end:]


class ParameterOptimizer:
    """
    Optimizes strategy parameters using IS/OOS split.

    Usage
    -----
        opt = ParameterOptimizer(cfg)
        results_df = opt.optimize(strategy, df_features, timeframe)
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.val = cfg.validation
        self.opt = cfg.optimization

    def _make_sim_config(self, direction: str = "both") -> SimConfig:
        return SimConfig(
            fees            = self.cfg.fees,
            slippage        = self.cfg.slippage,
            leverage        = self.cfg.leverage,
            risk_per_trade  = self.cfg.risk_per_trade,
            direction       = direction,
        )

    def optimize(
        self,
        strategy: BaseStrategy,
        df:       pd.DataFrame,
        timeframe: str,
        direction: str = "both",
        random_n:  Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run parameter search.

        Parameters
        ----------
        strategy  : instantiated strategy object
        df        : full feature DataFrame
        timeframe : label string
        direction : "long" | "short" | "both"
        random_n  : if set, randomly sample this many param combinations

        Returns
        -------
        DataFrame sorted by robustness score descending
        """
        df = df.dropna(subset=["close"]).copy()
        if len(df) < 200:
            logger.warning(f"Insufficient data ({len(df)} rows) for {strategy.name}/{timeframe}")
            return pd.DataFrame()

        df_is, df_oos, _ = _split_df(df, self.val.is_ratio, self.val.oos_ratio)

        if len(df_is) < 100 or len(df_oos) < 50:
            logger.warning(f"IS or OOS too small for {strategy.name}/{timeframe}")
            return pd.DataFrame()

        param_grid = strategy.param_grid()
        if not param_grid:
            logger.warning(f"Empty param grid for {strategy.name}")
            return pd.DataFrame()

        # Sampling
        method = self.opt.method
        if random_n is not None or method == "random":
            n_sample = random_n or self.opt.random_n
            if n_sample < len(param_grid):
                param_grid = random.sample(param_grid, n_sample)

        sim_cfg = self._make_sim_config(direction)
        engine  = BacktestEngine(sim_cfg)

        logger.info(
            f"  [{strategy.name}/{timeframe}] "
            f"Testing {len(param_grid)} parameter sets …"
        )

        results: List[dict] = []

        for i, params in enumerate(param_grid):
            try:
                # ── IS backtest ────────────────────────────────────────────────
                df_is_sig = strategy.generate_signals(df_is.copy(), params)
                res_is    = engine.run(df_is_sig, strategy.name, timeframe, params)

                # ── OOS backtest ───────────────────────────────────────────────
                df_oos_sig = strategy.generate_signals(df_oos.copy(), params)
                res_oos    = engine.run(df_oos_sig, strategy.name, timeframe, params)

                score = _robustness_score(
                    res_is.metrics, res_oos.metrics, self.val.min_trades
                )

                row = OptimResult(
                    strategy   = strategy.name,
                    timeframe  = timeframe,
                    params     = params,
                    is_return  = res_is.metrics.get("total_return", 0),
                    is_sharpe  = res_is.metrics.get("sharpe", np.nan),
                    is_pf      = res_is.metrics.get("profit_factor", np.nan),
                    is_trades  = res_is.metrics.get("trade_count", 0),
                    is_drawdown= res_is.metrics.get("max_drawdown", 0),
                    oos_return = res_oos.metrics.get("total_return", 0),
                    oos_sharpe = res_oos.metrics.get("sharpe", np.nan),
                    oos_pf     = res_oos.metrics.get("profit_factor", np.nan),
                    oos_trades = res_oos.metrics.get("trade_count", 0),
                    oos_drawdown=res_oos.metrics.get("max_drawdown", 0),
                    robustness_score=score,
                )
                results.append(row.to_dict())

            except Exception as exc:
                logger.debug(f"    Param set {i} failed: {exc}")
                continue

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values("robustness", ascending=False).reset_index(drop=True)

        logger.info(
            f"  [{strategy.name}/{timeframe}] Done. "
            f"Best robustness={result_df['robustness'].iloc[0]:.3f} | "
            f"OOS PF={result_df['oos_pf'].iloc[0]:.2f} | "
            f"OOS Sharpe={result_df['oos_sharpe'].iloc[0]:.2f}"
        )

        return result_df

    def sensitivity_analysis(
        self,
        strategy:   BaseStrategy,
        df:         pd.DataFrame,
        timeframe:  str,
        best_params: dict,
        top_n:      int = 10,
        direction:  str = "both",
    ) -> pd.DataFrame:
        """
        Run all param combinations and return the top_n nearest to best_params.

        This shows whether performance is stable around the optimum or a spike.
        """
        full_grid  = strategy.param_grid()
        sim_cfg    = self._make_sim_config(direction)
        engine     = BacktestEngine(sim_cfg)
        df_is, df_oos, _ = _split_df(df, self.val.is_ratio, self.val.oos_ratio)

        rows = []
        for params in full_grid:
            # Measure 'distance' from best params (for numeric params)
            dist = sum(
                abs(params.get(k, 0) - best_params.get(k, 0))
                for k in best_params
                if isinstance(best_params.get(k), (int, float))
            )
            try:
                df_sig  = strategy.generate_signals(df_oos.copy(), params)
                res     = engine.run(df_sig, strategy.name, timeframe, params)
                rows.append({
                    "dist_from_best": dist,
                    "oos_pf":     res.metrics.get("profit_factor", np.nan),
                    "oos_sharpe": res.metrics.get("sharpe", np.nan),
                    "oos_return": res.metrics.get("total_return", 0),
                    "trades":     res.metrics.get("trade_count", 0),
                    **{f"p_{k}": v for k, v in params.items()},
                })
            except Exception:
                continue

        if not rows:
            return pd.DataFrame()

        df_sens = pd.DataFrame(rows).sort_values("dist_from_best")
        return df_sens.head(top_n)
