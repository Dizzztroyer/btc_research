"""
Breakout / volatility expansion strategy family.

Strategies:
1. SqueezeBreakout     — Low ATR compression then range expansion
2. ConsolidationBreakout — N-bar range tightening then breakout
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
from src.strategies.base import BaseStrategy, StrategyResult


class SqueezeBreakout(BaseStrategy):
    """
    Volatility squeeze breakout.

    Squeeze condition: BB width < its rolling percentile low (compression).
    Breakout: close breaks above/below the Donchian high/low while squeeze fires.

    Params:
        bb_length: (default 20)
        bb_std: (default 2.0)
        dc_length: Donchian length (default 20)
        squeeze_percentile: BB width percentile threshold (default 20)
        squeeze_window: Rolling window for percentile (default 100)
        atr_length: (default 14)
        sl_atr_mult: (default 1.5)
        tp_atr_mult: (default 3.0)
    """

    @property
    def name(self) -> str:
        return "Squeeze_Breakout"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        p = self.params
        bb_len = p.get("bb_length", 20)
        bb_std = p.get("bb_std", 2.0)
        dc_len = p.get("dc_length", 20)
        sq_pct = p.get("squeeze_percentile", 20)
        sq_win = p.get("squeeze_window", 100)
        atr_len = p.get("atr_length", 14)
        sl_mult = p.get("sl_atr_mult", 1.5)
        tp_mult = p.get("tp_atr_mult", 3.0)

        close = df["close"]
        signals = pd.Series(0, index=df.index, dtype=int)
        sl = pd.Series(np.nan, index=df.index)
        tp = pd.Series(np.nan, index=df.index)

        bb_std_str = str(bb_std).replace(".", "")
        bb_width_col = f"bb_width_{bb_len}_{bb_std_str}"
        dc_high_col = f"dc_{dc_len}_high"
        dc_low_col = f"dc_{dc_len}_low"
        atr_col = f"atr_{atr_len}"

        for col in [bb_width_col, dc_high_col, dc_low_col, atr_col]:
            if col not in df.columns:
                return StrategyResult(signals=signals, stop_losses=sl, take_profits=tp,
                                      name=self.name, params=p)

        bb_width = df[bb_width_col]
        width_thresh = bb_width.rolling(sq_win, min_periods=20).quantile(sq_pct / 100)
        in_squeeze = bb_width < width_thresh

        # Breakout on current bar: close > previous Donchian high
        prev_dc_high = df[dc_high_col].shift(1)
        prev_dc_low = df[dc_low_col].shift(1)
        atr = df[atr_col]

        # Only trigger on first bar out of squeeze
        was_in_squeeze = in_squeeze.shift(1).fillna(False)

        long_sig = was_in_squeeze & (close > prev_dc_high)
        short_sig = was_in_squeeze & (close < prev_dc_low)

        signals[long_sig] = 1
        signals[short_sig] = -1

        sl_dist = sl_mult * atr
        tp_dist = tp_mult * atr

        sl[long_sig] = close[long_sig] - sl_dist[long_sig]
        tp[long_sig] = close[long_sig] + tp_dist[long_sig]
        sl[short_sig] = close[short_sig] + sl_dist[short_sig]
        tp[short_sig] = close[short_sig] - tp_dist[short_sig]

        return StrategyResult(signals=signals, stop_losses=sl, take_profits=tp,
                              name=self.name, params=p)


class ConsolidationBreakout(BaseStrategy):
    """
    Tight range consolidation followed by breakout.

    Consolidation: last N bars have range < range_atr_mult * ATR.
    Breakout: close breaks highest high or lowest low of consolidation zone.

    Params:
        lookback: Bars to define consolidation (default 10)
        range_atr_mult: Max range as ATR multiple to qualify (default 1.0)
        atr_length: (default 14)
        sl_atr_mult: (default 1.5)
        tp_atr_mult: (default 3.0)
    """

    @property
    def name(self) -> str:
        return "Consolidation_Breakout"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        p = self.params
        lookback = p.get("lookback", 10)
        range_mult = p.get("range_atr_mult", 1.0)
        atr_len = p.get("atr_length", 14)
        sl_mult = p.get("sl_atr_mult", 1.5)
        tp_mult = p.get("tp_atr_mult", 3.0)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        signals = pd.Series(0, index=df.index, dtype=int)
        sl = pd.Series(np.nan, index=df.index)
        tp = pd.Series(np.nan, index=df.index)

        atr_col = f"atr_{atr_len}"
        if atr_col not in df.columns:
            return StrategyResult(signals=signals, stop_losses=sl, take_profits=tp,
                                  name=self.name, params=p)

        atr = df[atr_col]

        # Rolling consolidation zone
        roll_high = high.shift(1).rolling(lookback).max()
        roll_low = low.shift(1).rolling(lookback).min()
        zone_range = roll_high - roll_low
        is_consolidation = zone_range < range_mult * atr

        long_sig = is_consolidation & (close > roll_high)
        short_sig = is_consolidation & (close < roll_low)

        signals[long_sig] = 1
        signals[short_sig] = -1

        sl_dist = sl_mult * atr
        tp_dist = tp_mult * atr

        sl[long_sig] = close[long_sig] - sl_dist[long_sig]
        tp[long_sig] = close[long_sig] + tp_dist[long_sig]
        sl[short_sig] = close[short_sig] + sl_dist[short_sig]
        tp[short_sig] = close[short_sig] - tp_dist[short_sig]

        return StrategyResult(signals=signals, stop_losses=sl, take_profits=tp,
                              name=self.name, params=p)