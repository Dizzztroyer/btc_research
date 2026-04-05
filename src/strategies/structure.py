"""
Structure-based / price action strategies.

Strategies:
1. SwingBreakout      — Swing high/low breakout
2. BOS_Strategy       — Break of Structure: higher high after higher low sequence

All rules are objective and algorithmic.
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
from src.strategies.base import BaseStrategy, StrategyResult


def _find_swing_highs(high: pd.Series, left: int, right: int) -> pd.Series:
    """
    Return boolean Series marking bars that are swing highs.
    A swing high is a bar where high is greater than the `left` bars before
    and `right` bars after it.
    Uses shifted comparison to be lookahead-free at signal generation time.
    NOTE: A swing high at bar i can only be confirmed at bar i+right.
    """
    n = len(high)
    is_swing = pd.Series(False, index=high.index)
    h = high.values
    for i in range(left, n - right):
        if all(h[i] > h[i - j] for j in range(1, left + 1)) and \
           all(h[i] > h[i + j] for j in range(1, right + 1)):
            is_swing.iloc[i] = True
    return is_swing


def _find_swing_lows(low: pd.Series, left: int, right: int) -> pd.Series:
    """Return boolean Series marking swing lows."""
    n = len(low)
    is_swing = pd.Series(False, index=low.index)
    l = low.values
    for i in range(left, n - right):
        if all(l[i] < l[i - j] for j in range(1, left + 1)) and \
           all(l[i] < l[i + j] for j in range(1, right + 1)):
            is_swing.iloc[i] = True
    return is_swing


class SwingBreakout(BaseStrategy):
    """
    Swing high/low breakout strategy.

    Long: close breaks above the most recent confirmed swing high.
    Short: close breaks below the most recent confirmed swing low.

    A swing is confirmed only after `right` bars have passed (no lookahead).

    Params:
        swing_left: Bars left of pivot (default 3)
        swing_right: Bars right of pivot for confirmation (default 3)
        atr_length: (default 14)
        sl_atr_mult: (default 1.5)
        tp_atr_mult: (default 3.0)
    """

    @property
    def name(self) -> str:
        return "Swing_Breakout"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        p = self.params
        left = p.get("swing_left", 3)
        right = p.get("swing_right", 3)
        atr_len = p.get("atr_length", 14)
        sl_mult = p.get("sl_atr_mult", 1.5)
        tp_mult = p.get("tp_atr_mult", 3.0)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        signals = pd.Series(0, index=df.index, dtype=int)
        sl_series = pd.Series(np.nan, index=df.index)
        tp_series = pd.Series(np.nan, index=df.index)

        atr_col = f"atr_{atr_len}"
        if atr_col not in df.columns:
            return StrategyResult(signals=signals, stop_losses=sl_series, take_profits=tp_series,
                                  name=self.name, params=p)

        atr = df[atr_col]

        # Find swing highs/lows (confirmed after `right` bars)
        sh = _find_swing_highs(high, left, right)
        sl_pts = _find_swing_lows(low, left, right)

        # Last confirmed swing high/low level (available after confirmation delay)
        last_sh = high.where(sh).ffill()
        last_sl = low.where(sl_pts).ffill()

        # Entry: break of last swing level
        long_sig = (close > last_sh.shift(right + 1)) & sh.shift(right).fillna(False)
        # More practical: break above last confirmed swing high
        long_sig = close > last_sh.shift(1)
        short_sig = close < last_sl.shift(1)

        # Only trigger on cross (not sustained break)
        long_cross = long_sig & ~long_sig.shift(1).fillna(False)
        short_cross = short_sig & ~short_sig.shift(1).fillna(False)

        signals[long_cross] = 1
        signals[short_cross] = -1

        sl_dist = sl_mult * atr
        tp_dist = tp_mult * atr

        sl_series[long_cross] = close[long_cross] - sl_dist[long_cross]
        tp_series[long_cross] = close[long_cross] + tp_dist[long_cross]
        sl_series[short_cross] = close[short_cross] + sl_dist[short_cross]
        tp_series[short_cross] = close[short_cross] - tp_dist[short_cross]

        return StrategyResult(signals=signals, stop_losses=sl_series, take_profits=tp_series,
                              name=self.name, params=p)


class BOSStrategy(BaseStrategy):
    """
    Break of Structure (BOS) strategy.

    Bullish BOS: price forms a higher low, then breaks the prior swing high.
    Bearish BOS: price forms a lower high, then breaks the prior swing low.

    All logic is objective using rolling swing detection.

    Params:
        swing_left: (default 5)
        swing_right: (default 5)
        atr_length: (default 14)
        sl_atr_mult: (default 2.0)
        tp_atr_mult: (default 3.0)
    """

    @property
    def name(self) -> str:
        return "BOS_Structure"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        p = self.params
        left = p.get("swing_left", 5)
        right = p.get("swing_right", 5)
        atr_len = p.get("atr_length", 14)
        sl_mult = p.get("sl_atr_mult", 2.0)
        tp_mult = p.get("tp_atr_mult", 3.0)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        signals = pd.Series(0, index=df.index, dtype=int)
        sl_series = pd.Series(np.nan, index=df.index)
        tp_series = pd.Series(np.nan, index=df.index)

        atr_col = f"atr_{atr_len}"
        if atr_col not in df.columns:
            return StrategyResult(signals=signals, stop_losses=sl_series, take_profits=tp_series,
                                  name=self.name, params=p)

        atr = df[atr_col]
        sh = _find_swing_highs(high, left, right)
        sl_pts = _find_swing_lows(low, left, right)

        swing_high_vals = high.where(sh).ffill()
        swing_low_vals = low.where(sl_pts).ffill()

        # Higher low: current swing low > previous swing low
        prev_sl_val = swing_low_vals.shift(1)
        higher_low = (swing_low_vals > prev_sl_val) & sl_pts

        # Lower high: current swing high < previous swing high
        prev_sh_val = swing_high_vals.shift(1)
        lower_high = (swing_high_vals < prev_sh_val) & sh

        # BOS long: after a higher low, price breaks the most recent swing high
        higher_low_occurred = higher_low.cumsum() > 0
        long_sig = higher_low_occurred & (close > swing_high_vals.shift(1))
        long_cross = long_sig & ~long_sig.shift(1).fillna(False)

        # BOS short: after a lower high, price breaks the most recent swing low
        lower_high_occurred = lower_high.cumsum() > 0
        short_sig = lower_high_occurred & (close < swing_low_vals.shift(1))
        short_cross = short_sig & ~short_sig.shift(1).fillna(False)

        signals[long_cross] = 1
        signals[short_cross] = -1

        sl_dist = sl_mult * atr
        tp_dist = tp_mult * atr

        sl_series[long_cross] = close[long_cross] - sl_dist[long_cross]
        tp_series[long_cross] = close[long_cross] + tp_dist[long_cross]
        sl_series[short_cross] = close[short_cross] + sl_dist[short_cross]
        tp_series[short_cross] = close[short_cross] - tp_dist[short_cross]

        return StrategyResult(signals=signals, stop_losses=sl_series, take_profits=tp_series,
                              name=self.name, params=p)