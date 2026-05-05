"""
optimizer.py
────────────
Parameter search engine — parallel version using joblib.

Each parameter combination is evaluated independently,
so they can all run in parallel across CPU cores.

Speedup vs sequential: N_cores × (minus overhead) ≈ 3-7×
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from src.backtest.engine import BacktestEngine, BacktestResult, SimConfig
from src.strategies.base import BaseStrategy
from src.utils.config_loader import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Robustness score ──────────────────────────────────────────────────────────

def _robustness_score(is_m: dict, oos_m: dict, min_trades: int) -> float:
    oos_pf     = oos_m.get("profit_factor", 0) or 0
    oos_sharpe = oos_m.get("sharpe",        -99) or -99
    oos_trades = oos_m.get("trade_count",   0)   or 0
    is_sharpe  = is_m.get("sharpe",         0)   or 0
    oos_mdd    = abs(oos_m.get("max_drawdown", 1) or 1)

    if oos_trades < min_trades:
        return -1000.0
    if oos_pf < 1.0:
        return -500.0

    decay = (oos_sharpe / is_sharpe) if is_sharpe > 0 else 1.0
    return (
        oos_pf        * 2.0
        + oos_sharpe  * 1.5
        + decay       * 1.0
        - oos_mdd     * 0.5
    )


def _split_df(df, is_ratio, oos_ratio):
    n       = len(df)
    is_end  = int(n * is_ratio)
    oos_end = int(n * (is_ratio + oos_ratio))
    return df.iloc[:is_end], df.iloc[is_end:oos_end], df.iloc[oos_end:]


# ── Single-param-set worker (runs in subprocess when parallel) ────────────────

def _evaluate_one(
    params:        dict,
    df_is_dict:    dict,     # passed as plain dict for pickle compatibility
    df_oos_dict:   dict,
    strategy_cls,            # class (not instance — picklable)
    strategy_name: str,
    timeframe:     str,
    sim_kwargs:    dict,
    min_trades:    int,
) -> Optional[dict]:
    """
    Evaluate one parameter set on IS and OOS.
    Returns a result dict, or None on failure.
    Designed to be called via joblib.delayed.
    """
    try:
        df_is  = pd.DataFrame(df_is_dict)
        df_oos = pd.DataFrame(df_oos_dict)

        # Re-parse timestamps (lost during dict conversion)
        for df_ in [df_is, df_oos]:
            if "timestamp" in df_.columns:
                df_["timestamp"] = pd.to_datetime(df_["timestamp"], utc=True)

        strategy = strategy_cls()
        sim_cfg  = SimConfig(**sim_kwargs)
        engine   = BacktestEngine(sim_cfg)

        df_is_sig  = strategy.generate_signals(df_is,  params)
        df_oos_sig = strategy.generate_signals(df_oos, params)

        res_is  = engine.run(df_is_sig,  strategy_name, timeframe, params)
        res_oos = engine.run(df_oos_sig, strategy_name, timeframe, params)

        score = _robustness_score(res_is.metrics, res_oos.metrics, min_trades)

        return {
            "strategy":   strategy_name,
            "timeframe":  timeframe,
            "params":     params,
            **{f"p_{k}": v for k, v in params.items()},
            "is_return":   res_is.metrics.get("total_return", 0),
            "is_sharpe":   res_is.metrics.get("sharpe",       np.nan),
            "is_pf":       res_is.metrics.get("profit_factor",np.nan),
            "is_trades":   res_is.metrics.get("trade_count",  0),
            "is_drawdown": res_is.metrics.get("max_drawdown", 0),
            "oos_return":  res_oos.metrics.get("total_return", 0),
            "oos_sharpe":  res_oos.metrics.get("sharpe",       np.nan),
            "oos_pf":      res_oos.metrics.get("profit_factor",np.nan),
            "oos_trades":  res_oos.metrics.get("trade_count",  0),
            "oos_drawdown":res_oos.metrics.get("max_drawdown", 0),
            "robustness":  score,
        }
    except Exception:
        return None


class ParameterOptimizer:
    """
    Parallel parameter search (joblib) + optional random sampling.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.val = cfg.validation
        self.opt = cfg.optimization

    def _sim_kwargs(self, direction: str) -> dict:
        return dict(
            fees            = self.cfg.fees,
            slippage        = self.cfg.slippage,
            leverage        = self.cfg.leverage,
            risk_per_trade  = self.cfg.risk_per_trade,
            direction       = direction,
        )

    def optimize(
        self,
        strategy:  BaseStrategy,
        df:        pd.DataFrame,
        timeframe: str,
        direction: str = "both",
        random_n:  Optional[int] = None,
        n_jobs:    int = -1,       # -1 = all cores
    ) -> pd.DataFrame:
        """
        Run parallel parameter search.

        Parameters
        ----------
        strategy  : instantiated strategy
        df        : full feature DataFrame
        timeframe : label
        direction : trade direction
        random_n  : if set, randomly sample this many param sets
        n_jobs    : joblib parallel workers (-1 = all cores)
        """
        df = df.dropna(subset=["close"]).copy()
        if len(df) < 200:
            logger.warning(f"Insufficient data for {strategy.name}/{timeframe}")
            return pd.DataFrame()

        df_is, df_oos, _ = _split_df(df, self.val.is_ratio, self.val.oos_ratio)
        if len(df_is) < 100 or len(df_oos) < 50:
            logger.warning(f"IS/OOS too small for {strategy.name}/{timeframe}")
            return pd.DataFrame()

        param_grid = strategy.param_grid()
        if not param_grid:
            return pd.DataFrame()

        # Random sampling
        if random_n is not None and random_n > 0 and random_n < len(param_grid):
            param_grid = random.sample(param_grid, random_n)

        logger.info(
            f"  [{strategy.name}/{timeframe}] "
            f"Testing {len(param_grid)} param sets "
            f"({'parallel' if JOBLIB_AVAILABLE else 'sequential'}) …"
        )

        # Convert to plain dicts for pickle compatibility with joblib
        is_dict  = df_is.reset_index(drop=True).to_dict(orient="list")
        oos_dict = df_oos.reset_index(drop=True).to_dict(orient="list")

        strategy_cls  = type(strategy)
        strategy_name = strategy.name
        sim_kw        = self._sim_kwargs(direction)
        min_trades    = self.val.min_trades

        if JOBLIB_AVAILABLE and n_jobs != 1:
            # Parallel evaluation
            raw_results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(
                delayed(_evaluate_one)(
                    params, is_dict, oos_dict,
                    strategy_cls, strategy_name, timeframe,
                    sim_kw, min_trades,
                )
                for params in param_grid
            )
        else:
            # Sequential fallback
            raw_results = [
                _evaluate_one(
                    params, is_dict, oos_dict,
                    strategy_cls, strategy_name, timeframe,
                    sim_kw, min_trades,
                )
                for params in param_grid
            ]

        results = [r for r in raw_results if r is not None]

        if not results:
            return pd.DataFrame()

        result_df = (
            pd.DataFrame(results)
            .sort_values("robustness", ascending=False)
            .reset_index(drop=True)
        )

        logger.info(
            f"  [{strategy.name}/{timeframe}] Done. "
            f"Best robustness={result_df['robustness'].iloc[0]:.3f} | "
            f"OOS PF={result_df['oos_pf'].iloc[0]:.2f} | "
            f"OOS Sharpe={result_df['oos_sharpe'].iloc[0]:.2f}"
        )

        return result_df

    def sensitivity_analysis(
        self,
        strategy:    BaseStrategy,
        df:          pd.DataFrame,
        timeframe:   str,
        best_params: dict,
        top_n:       int = 10,
        direction:   str = "both",
    ) -> pd.DataFrame:
        full_grid = strategy.param_grid()
        df_is, df_oos, _ = _split_df(df, self.val.is_ratio, self.val.oos_ratio)

        is_dict  = df_is.reset_index(drop=True).to_dict(orient="list")
        oos_dict = df_oos.reset_index(drop=True).to_dict(orient="list")
        strategy_cls = type(strategy)

        raw = Parallel(n_jobs=-1, prefer="processes")(
            delayed(_evaluate_one)(
                params, is_dict, oos_dict,
                strategy_cls, strategy.name, timeframe,
                self._sim_kwargs(direction), self.val.min_trades,
            )
            for params in full_grid
        ) if JOBLIB_AVAILABLE else [
            _evaluate_one(params, is_dict, oos_dict, strategy_cls,
                          strategy.name, timeframe, self._sim_kwargs(direction),
                          self.val.min_trades)
            for params in full_grid
        ]

        rows = []
        for r, params in zip(raw, full_grid):
            if r is None:
                continue
            dist = sum(
                abs(params.get(k, 0) - best_params.get(k, 0))
                for k in best_params
                if isinstance(best_params.get(k), (int, float))
            )
            rows.append({"dist_from_best": dist, **r})

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("dist_from_best").head(top_n)