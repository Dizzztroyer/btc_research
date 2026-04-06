"""
walk_forward.py
───────────────
Walk-forward validation engine.

Algorithm
─────────
1. Split the full dataset into N sequential windows
2. For each window:
        a. IS period  = first is_ratio fraction of window
        b. OOS period = remaining fraction
        c. Run full param grid on IS, pick best by IS Sharpe
        d. Run best params on OOS, record OOS metrics
3. Concatenate all OOS results → walk-forward equity curve
4. Compute aggregate WF metrics

This avoids data-snooping bias: the OOS periods are never seen during optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult, SimConfig
from src.backtest.metrics import compute_metrics
from src.strategies.base import BaseStrategy
from src.utils.config_loader import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WFWindow:
    window_idx:   int
    is_start:     pd.Timestamp
    is_end:       pd.Timestamp
    oos_start:    pd.Timestamp
    oos_end:      pd.Timestamp
    best_params:  dict
    is_metrics:   dict
    oos_metrics:  dict
    oos_trades:   pd.DataFrame
    oos_equity:   pd.Series


@dataclass
class WFResult:
    strategy:         str
    timeframe:        str
    windows:          List[WFWindow]
    combined_metrics: dict
    combined_equity:  pd.Series
    combined_trades:  pd.DataFrame
    yearly:           pd.DataFrame


def _split_windows(
    df: pd.DataFrame,
    n_windows: int,
    is_ratio: float,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Split df into n_windows sequential (IS, OOS) pairs.
    Each window covers 1/n_windows of the data.
    IS takes is_ratio of that window; OOS takes the rest.
    """
    n = len(df)
    window_size = n // n_windows
    if window_size < 100:
        raise ValueError(
            f"Window size too small ({window_size} bars). "
            "Reduce wf_windows or use more data."
        )

    pairs = []
    for i in range(n_windows):
        start = i * window_size
        end   = start + window_size if i < n_windows - 1 else n
        window_df = df.iloc[start:end]
        is_end    = int(len(window_df) * is_ratio)
        pairs.append((window_df.iloc[:is_end], window_df.iloc[is_end:]))

    return pairs


def _best_params_by_metric(
    results: pd.DataFrame,
    metric: str = "is_sharpe",
    min_trades: int = 10,
) -> dict:
    """Pick best parameter set from optimizer output."""
    if results.empty:
        return {}
    filtered = results[results["is_trades"] >= min_trades]
    if filtered.empty:
        filtered = results
    best_row = filtered.sort_values(metric, ascending=False).iloc[0]
    # Extract param columns
    p_cols = [c for c in best_row.index if c.startswith("p_")]
    params = {c[2:]: best_row[c] for c in p_cols}
    return params


class WalkForwardEngine:
    """
    Walk-forward validation for a single strategy on a single timeframe.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.val = cfg.validation

    def _make_sim_config(self, direction: str = "both") -> SimConfig:
        return SimConfig(
            fees           = self.cfg.fees,
            slippage       = self.cfg.slippage,
            leverage       = self.cfg.leverage,
            risk_per_trade = self.cfg.risk_per_trade,
            direction      = direction,
        )

    def run(
        self,
        strategy:  BaseStrategy,
        df:        pd.DataFrame,
        timeframe: str,
        direction: str = "both",
    ) -> WFResult:
        """
        Run walk-forward validation.

        Parameters
        ----------
        strategy  : instantiated strategy
        df        : full feature DataFrame
        timeframe : label
        direction : trade direction

        Returns
        -------
        WFResult with per-window and aggregate metrics
        """
        n_windows = self.val.wf_windows
        is_ratio  = self.val.is_ratio / (self.val.is_ratio + self.val.oos_ratio)

        try:
            pairs = _split_windows(df, n_windows, is_ratio)
        except ValueError as exc:
            logger.warning(f"WF: {exc}")
            return _empty_wf_result(strategy.name, timeframe)

        sim_cfg = self._make_sim_config(direction)
        engine  = BacktestEngine(sim_cfg)
        param_grid = strategy.param_grid()

        windows:      List[WFWindow] = []
        all_trades:   List[pd.DataFrame] = []
        all_eq_parts: List[pd.Series] = []

        for i, (df_is, df_oos) in enumerate(pairs):
            logger.info(
                f"  WF window {i+1}/{n_windows} | "
                f"IS: {df_is['timestamp'].iloc[0].date()} → {df_is['timestamp'].iloc[-1].date()} | "
                f"OOS: {df_oos['timestamp'].iloc[0].date()} → {df_oos['timestamp'].iloc[-1].date()}"
            )

            if len(df_is) < 50 or len(df_oos) < 20:
                logger.warning(f"  Window {i+1} too small — skipping")
                continue

            # ── IS: find best params ───────────────────────────────────────────
            is_results = []
            for params in param_grid:
                try:
                    df_sig = strategy.generate_signals(df_is.copy(), params)
                    res    = engine.run(df_sig, strategy.name, timeframe, params)
                    if res.metrics["trade_count"] >= self.val.min_trades // 2:
                        is_results.append({
                            "params":     params,
                            "is_sharpe":  res.metrics.get("sharpe", -99),
                            "is_pf":      res.metrics.get("profit_factor", 0),
                            "is_trades":  res.metrics["trade_count"],
                            "is_metrics": res.metrics,
                        })
                except Exception:
                    continue

            if not is_results:
                logger.warning(f"  Window {i+1}: no valid IS results")
                continue

            # Pick best IS params by Sharpe
            is_results.sort(key=lambda x: x["is_sharpe"], reverse=True)
            best = is_results[0]
            best_params = best["params"]
            is_metrics  = best["is_metrics"]

            # ── OOS: run with best params ──────────────────────────────────────
            try:
                df_oos_sig = strategy.generate_signals(df_oos.copy(), best_params)
                res_oos    = engine.run(df_oos_sig, strategy.name, timeframe, best_params)
            except Exception as exc:
                logger.warning(f"  Window {i+1} OOS failed: {exc}")
                continue

            oos_metrics = res_oos.metrics

            window = WFWindow(
                window_idx  = i + 1,
                is_start    = df_is["timestamp"].iloc[0],
                is_end      = df_is["timestamp"].iloc[-1],
                oos_start   = df_oos["timestamp"].iloc[0],
                oos_end     = df_oos["timestamp"].iloc[-1],
                best_params = best_params,
                is_metrics  = is_metrics,
                oos_metrics = oos_metrics,
                oos_trades  = res_oos.trades,
                oos_equity  = res_oos.equity,
            )
            windows.append(window)

            if not res_oos.trades.empty:
                all_trades.append(res_oos.trades)

            all_eq_parts.append(res_oos.equity)

        if not windows:
            return _empty_wf_result(strategy.name, timeframe)

        # ── Combine OOS equity curves ──────────────────────────────────────────
        # Chain equity curves: each window starts where the previous ended
        combined_equity = _chain_equity(all_eq_parts, self.cfg.risk_per_trade)
        combined_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

        # ── Aggregate metrics over the chained OOS curve ───────────────────────
        bpy = _estimate_bpy(df)
        agg_metrics = compute_metrics(
            trades          = combined_trades,
            equity          = combined_equity,
            bars_per_year   = bpy,
            initial_capital = self.cfg.risk_per_trade,  # proxy
        )

        # ── Year-by-year breakdown of combined OOS ─────────────────────────────
        from src.backtest.metrics import yearly_breakdown

        if not combined_trades.empty and "entry_time" in combined_trades.columns:
            combined_ts = pd.to_datetime(combined_trades["entry_time"])
        else:
            combined_ts = pd.Series(dtype="datetime64[ns, UTC]")

        yearly = pd.DataFrame()  # simplified; full yearly requires aligned timestamps

        logger.info(
            f"  WF [{strategy.name}/{timeframe}] DONE | "
            f"OOS Sharpe={agg_metrics.get('sharpe', float('nan')):.2f} | "
            f"OOS PF={agg_metrics.get('profit_factor', float('nan')):.2f} | "
            f"OOS MDD={agg_metrics.get('max_drawdown', float('nan')):.2%}"
        )

        return WFResult(
            strategy         = strategy.name,
            timeframe        = timeframe,
            windows          = windows,
            combined_metrics = agg_metrics,
            combined_equity  = combined_equity,
            combined_trades  = combined_trades,
            yearly           = yearly,
        )


def _chain_equity(parts: List[pd.Series], initial: float = 10_000.0) -> pd.Series:
    """Chain multiple equity curve segments end-to-end."""
    if not parts:
        return pd.Series([initial])

    chains = []
    current_capital = initial

    for part in parts:
        if part.empty:
            continue
        part_start = part.iloc[0]
        if part_start == 0:
            scale = 1.0
        else:
            scale = current_capital / part_start
        scaled = part * scale
        chains.append(scaled)
        current_capital = scaled.iloc[-1]

    if not chains:
        return pd.Series([initial])

    return pd.concat(chains, ignore_index=True)


def _estimate_bpy(df: pd.DataFrame) -> float:
    """Estimate bars per year from timestamp column."""
    if len(df) < 2 or "timestamp" not in df.columns:
        return 365.0
    ts    = pd.to_datetime(df["timestamp"])
    total = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
    n     = len(ts) - 1
    if total <= 0:
        return 365.0
    return 365.25 * 86_400 / (total / n)


def _empty_wf_result(strategy: str, timeframe: str) -> WFResult:
    return WFResult(
        strategy         = strategy,
        timeframe        = timeframe,
        windows          = [],
        combined_metrics = {},
        combined_equity  = pd.Series(),
        combined_trades  = pd.DataFrame(),
        yearly           = pd.DataFrame(),
    )
