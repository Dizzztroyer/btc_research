"""
base.py
───────
Abstract base class for all strategy families.

Every concrete strategy must implement:
    generate_signals(df: pd.DataFrame, params: dict) → pd.DataFrame

The returned DataFrame must contain all columns from the input df plus:
    signal   : int — 1 = go long, -1 = go short, 0 = flat
    sl_price : float — stop-loss price for this bar's signal
    tp_price : float — take-profit price for this bar's signal
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd


class BaseStrategy(ABC):
    """Abstract base for all strategy families."""

    name: str = "base"

    @abstractmethod
    def param_grid(self) -> List[Dict[str, Any]]:
        """
        Return a list of parameter dicts to sweep.
        Each dict is one parameter combination.
        """
        ...

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Compute entry/exit signals.

        Parameters
        ----------
        df     : feature-enriched OHLCV DataFrame
        params : parameter dict (one combination from param_grid)

        Returns
        -------
        df with added columns: signal (int), sl_price (float), tp_price (float)
        """
        ...

    def _add_signal_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure signal/sl/tp columns exist with neutral defaults."""
        if "signal"   not in df.columns: df["signal"]   = 0
        if "sl_price" not in df.columns: df["sl_price"] = float("nan")
        if "tp_price" not in df.columns: df["tp_price"] = float("nan")
        return df
