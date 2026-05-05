"""
engine.py
─────────
BacktestEngine — uses the Numba-compiled loop from fast_engine.py
when numba is available, falls back to pure Python otherwise.

Public API unchanged from v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from src.backtest.fast_engine import (
    NUMBA_AVAILABLE, get_fast_loop, _backtest_loop_python
)
from src.backtest.metrics import compute_metrics, yearly_breakdown
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DIRECTION_MAP = {"both": 0, "long": 1, "short": -1}
_EXIT_REASONS  = {0: "sl", 1: "tp", 2: "trail", 3: "time", 4: "signal", 5: "end_of_data"}


@dataclass
class SimConfig:
    fees:            float = 0.00075
    slippage:        float = 0.0003
    leverage:        float = 1.0
    risk_per_trade:  float = 0.01
    initial_capital: float = 10_000.0
    direction:       str   = "both"
    default_sl_pct:  float = 0.02
    default_tp_pct:  float = 0.04
    trailing_stop:   bool  = False
    trail_atr_mult:  float = 2.0
    max_bars_held:   Optional[int] = None
    partial_tp_pct:  float = 0.0
    partial_tp_level:float = 0.0


@dataclass
class BacktestResult:
    metrics:       dict
    trades:        pd.DataFrame
    equity:        pd.Series
    drawdown:      pd.Series
    yearly:        pd.DataFrame
    timeframe:     str = ""
    strategy_name: str = ""
    params:        dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.metrics.get("trade_count", 0) > 0


def _bars_per_year(timestamps: pd.Series) -> float:
    if len(timestamps) < 2:
        return 365.0
    total = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
    n     = len(timestamps) - 1
    return 365.25 * 86_400 / (total / n) if total > 0 else 365.0


class BacktestEngine:
    """
    Bar-based backtest engine with Numba acceleration.

    First call compiles the Numba loop (~5s), subsequent calls are instant.
    Falls back to pure Python if numba is not installed.
    """

    def __init__(self, sim_cfg: SimConfig) -> None:
        self.cfg = sim_cfg

    def run(
        self,
        df:            pd.DataFrame,
        strategy_name: str = "unnamed",
        timeframe:     str = "",
        params:        Optional[dict] = None,
    ) -> BacktestResult:

        required = {"timestamp", "open", "high", "low", "close", "signal"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        df = df.sort_values("timestamp").reset_index(drop=True)
        n  = len(df)

        if n < 10:
            return self._empty_result(strategy_name, timeframe, params)

        # ── Prepare arrays ─────────────────────────────────────────────────────
        opens    = df["open"].to_numpy(dtype=np.float64)
        highs    = df["high"].to_numpy(dtype=np.float64)
        lows     = df["low"].to_numpy(dtype=np.float64)
        closes   = df["close"].to_numpy(dtype=np.float64)
        signals  = df["signal"].fillna(0).to_numpy(dtype=np.int64)

        sl_arr   = df["sl_price"].to_numpy(dtype=np.float64) \
                   if "sl_price" in df.columns else np.full(n, np.nan)
        tp_arr   = df["tp_price"].to_numpy(dtype=np.float64) \
                   if "tp_price" in df.columns else np.full(n, np.nan)

        # Best available ATR
        atr_col  = next(
            (c for c in ["atr_14", "atr_7", "atr_21"] if c in df.columns), None
        )
        atrs     = df[atr_col].to_numpy(dtype=np.float64) \
                   if atr_col else (highs - lows)

        direction_int = _DIRECTION_MAP.get(self.cfg.direction, 0)
        max_bh        = self.cfg.max_bars_held if self.cfg.max_bars_held else -1

        # ── Run compiled loop ──────────────────────────────────────────────────
        loop_fn = get_fast_loop() if NUMBA_AVAILABLE else None

        loop_args = (
            opens, highs, lows, closes, signals, sl_arr, tp_arr, atrs,
            self.cfg.initial_capital,
            self.cfg.fees, self.cfg.slippage,
            self.cfg.risk_per_trade, self.cfg.leverage,
            self.cfg.default_sl_pct, self.cfg.default_tp_pct,
            self.cfg.trailing_stop, self.cfg.trail_atr_mult,
            max_bh, direction_int,
        )

        if loop_fn is not None:
            results = loop_fn(*loop_args)
        else:
            results = _backtest_loop_python(*loop_args)

        (equity_arr,
         t_entry_bar, t_exit_bar, t_side,
         t_entry_price, t_exit_price,
         t_pnl, t_pnl_pct, t_r_mult,
         t_bars_held, t_exit_reason,
         trade_count) = results

        # ── Build trade DataFrame ──────────────────────────────────────────────
        if trade_count > 0:
            ts_vals = df["timestamp"].values
            trades  = pd.DataFrame({
                "entry_time":  ts_vals[t_entry_bar],
                "exit_time":   ts_vals[t_exit_bar],
                "side":        t_side,
                "entry_price": t_entry_price,
                "exit_price":  t_exit_price,
                "size":        np.zeros(trade_count),  # approx
                "pnl":         t_pnl,
                "pnl_pct":     t_pnl_pct,
                "r_multiple":  t_r_mult,
                "bars_held":   t_bars_held,
                "exit_reason": [_EXIT_REASONS.get(int(r), "unknown") for r in t_exit_reason],
            })
        else:
            trades = pd.DataFrame()

        eq_series  = pd.Series(equity_arr, index=df.index)
        dd_series  = (eq_series - eq_series.cummax()) / eq_series.cummax().replace(0, np.nan)

        bpy     = _bars_per_year(df["timestamp"])
        metrics = compute_metrics(
            trades          = trades,
            equity          = eq_series,
            bars_per_year   = bpy,
            initial_capital = self.cfg.initial_capital,
        )

        yearly = yearly_breakdown(
            trades          = trades,
            equity          = eq_series,
            timestamps      = df["timestamp"],
            bars_per_year   = bpy,
            initial_capital = self.cfg.initial_capital,
        )

        return BacktestResult(
            metrics       = metrics,
            trades        = trades,
            equity        = eq_series,
            drawdown      = dd_series,
            yearly        = yearly,
            timeframe     = timeframe,
            strategy_name = strategy_name,
            params        = params or {},
        )

    def _empty_result(self, strategy_name, timeframe, params) -> BacktestResult:
        from src.backtest.metrics import _empty_metrics
        return BacktestResult(
            metrics       = _empty_metrics(),
            trades        = pd.DataFrame(),
            equity        = pd.Series(),
            drawdown      = pd.Series(),
            yearly        = pd.DataFrame(),
            timeframe     = timeframe,
            strategy_name = strategy_name,
            params        = params or {},
        )