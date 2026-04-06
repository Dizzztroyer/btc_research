"""
structure.py
────────────
Structure-based / price-action strategy family.

All rules are fully algorithmic and objective — no subjectivity.

Strategies
──────────
1. SwingBreakout    — Break of confirmed swing high / swing low
2. LiquiditySweep   — Approximate liquidity sweep: wick beyond prior extreme + reversal
3. BOSStrategy      — Break of Structure: price closes beyond the last significant swing
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


# ── Shared helper: pivot detection ────────────────────────────────────────────

def _swing_highs(high: pd.Series, left: int, right: int) -> pd.Series:
    """
    Return a boolean Series marking confirmed swing highs.
    A swing high at bar i requires:
        high[i] = max of (high[i-left … i+right])
    and the right bars have already closed (no lookahead via shift).
    """
    n = len(high)
    is_sh = pd.Series(False, index=high.index)
    for i in range(left, n - right):
        window = high.iloc[i - left: i + right + 1]
        if high.iloc[i] == window.max():
            is_sh.iloc[i] = True
    return is_sh


def _swing_lows(low: pd.Series, left: int, right: int) -> pd.Series:
    """
    Return a boolean Series marking confirmed swing lows.
    A swing low at bar i: low[i] = min of (low[i-left … i+right])
    """
    n = len(low)
    is_sl = pd.Series(False, index=low.index)
    for i in range(left, n - right):
        window = low.iloc[i - left: i + right + 1]
        if low.iloc[i] == window.min():
            is_sl.iloc[i] = True
    return is_sl


def _rolling_pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    """
    Vectorised swing high detection.
    Returns the most recent confirmed swing high price at each bar.
    Confirmation requires `right` bars to have passed since the pivot.
    """
    window = left + right + 1
    # Local max within the window, shifted so the confirmation lag is respected
    local_max = high.rolling(window).max().shift(right)
    is_pivot  = high.shift(right) == local_max
    # Forward-fill: carry the confirmed pivot level forward
    pivot_level = high.shift(right).where(is_pivot)
    return pivot_level.ffill()


def _rolling_pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
    """Vectorised swing low detection."""
    window    = left + right + 1
    local_min = low.rolling(window).min().shift(right)
    is_pivot  = low.shift(right) == local_min
    pivot_level = low.shift(right).where(is_pivot)
    return pivot_level.ffill()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Swing High / Low Breakout
# ═══════════════════════════════════════════════════════════════════════════════

class SwingBreakoutStrategy(BaseStrategy):
    """
    Break of a confirmed swing high or swing low.

    Entry:
        Long  : close > most recent confirmed swing high
        Short : close < most recent confirmed swing low

    Exit:
        SL = most recent confirmed swing low (long) / swing high (short)
        TP = entry ± atr_tp_mult * ATR
    """

    name = "swing_breakout"

    def param_grid(self) -> List[Dict[str, Any]]:
        lefts   = [3, 5, 8, 10]
        rights  = [2, 3, 5]
        atr_lens= [7, 14]
        tp_mults= [2.0, 3.0, 4.0]
        return [
            {"left": lf, "right": rt, "atr_len": a, "tp_mult": tp}
            for lf, rt, a, tp in itertools.product(lefts, rights, atr_lens, tp_mults)
        ]

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        self._add_signal_columns(df)

        lf   = params["left"]
        rt   = params["right"]
        atr_col = f"atr_{params['atr_len']}"

        c = df["close"]
        atr = df[atr_col] if atr_col in df.columns else (df["high"] - df["low"])

        # Most recent confirmed pivot levels (available without lookahead)
        last_sh = _rolling_pivot_high(df["high"], lf, rt)
        last_sl = _rolling_pivot_low (df["low"],  lf, rt)

        # Signal: breakout of prior confirmed swing level
        long_signal  = c > last_sh.shift(1)
        short_signal = c < last_sl.shift(1)

        # Avoid repeated signals while in the same breakout
        long_signal  = long_signal  & ~long_signal.shift(1).fillna(False)
        short_signal = short_signal & ~short_signal.shift(1).fillna(False)

        df["signal"] = np.where(long_signal, 1, np.where(short_signal, -1, 0))

        tp_dist = params["tp_mult"] * atr

        # SL: opposite structural level
        df["sl_price"] = np.where(
            df["signal"] == 1,  last_sl,
            np.where(df["signal"] == -1, last_sh, np.nan)
        )
        df["tp_price"] = np.where(
            df["signal"] == 1,  c + tp_dist,
            np.where(df["signal"] == -1, c - tp_dist, np.nan)
        )

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Liquidity Sweep (Algorithmic approximation)
# ═══════════════════════════════════════════════════════════════════════════════

class LiquiditySweepStrategy(BaseStrategy):
    """
    Approximate liquidity sweep reversal.

    A liquidity sweep occurs when price briefly exceeds a prior extreme
    (triggering stops) and then reverses sharply.

    Objective rules:
        Bearish sweep → Bullish reversal:
            1. Low of current bar < lowest low of past N bars (wick sweeps lows)
            2. Close of current bar > prior low (close recovers above the prior extreme)
            3. Close is in upper half of the bar's range

        Bullish sweep → Bearish reversal:
            1. High of current bar > highest high of past N bars
            2. Close of current bar < prior high
            3. Close is in lower half of the bar's range

    Exit:
        SL = sweep low/high ± ATR buffer
        TP = entry ± tp_mult * ATR
    """

    name = "liquidity_sweep"

    def param_grid(self) -> List[Dict[str, Any]]:
        lookbacks = [5, 10, 15, 20]
        atr_lens  = [7, 14]
        sl_mults  = [0.5, 1.0, 1.5]
        tp_mults  = [2.0, 3.0, 4.0]
        return [
            {"lookback": lb, "atr_len": a, "sl_mult": sl, "tp_mult": tp}
            for lb, a, sl, tp in itertools.product(lookbacks, atr_lens, sl_mults, tp_mults)
        ]

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        self._add_signal_columns(df)

        lb     = params["lookback"]
        atr_col = f"atr_{params['atr_len']}"

        c = df["close"]
        h = df["high"]
        l = df["low"]
        o = df["open"]

        atr = df[atr_col] if atr_col in df.columns else (h - l)

        prior_low  = l.shift(1).rolling(lb).min()
        prior_high = h.shift(1).rolling(lb).max()

        bar_range = h - l
        close_upper_half = c > (l + bar_range * 0.55)
        close_lower_half = c < (l + bar_range * 0.45)

        # Bullish reversal after sweep of lows
        bull_sweep = (
            (l < prior_low) &           # wick below prior low
            (c > prior_low) &           # close recovers above it
            close_upper_half            # bullish close
        )

        # Bearish reversal after sweep of highs
        bear_sweep = (
            (h > prior_high) &          # wick above prior high
            (c < prior_high) &          # close fails to hold
            close_lower_half            # bearish close
        )

        df["signal"] = np.where(bull_sweep, 1, np.where(bear_sweep, -1, 0))

        sl_dist = params["sl_mult"] * atr
        tp_dist = params["tp_mult"] * atr

        # SL just beyond the sweep extreme
        df["sl_price"] = np.where(
            df["signal"] == 1,  l - sl_dist,     # below sweep low
            np.where(df["signal"] == -1, h + sl_dist, np.nan)
        )
        df["tp_price"] = np.where(
            df["signal"] == 1,  c + tp_dist,
            np.where(df["signal"] == -1, c - tp_dist, np.nan)
        )

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Break of Structure (BOS)
# ═══════════════════════════════════════════════════════════════════════════════

class BOSStrategy(BaseStrategy):
    """
    Break of Structure (BOS).

    Objective definition:
        Track the sequence of swing highs and swing lows.
        Bullish BOS : close breaks above the most recent confirmed swing high
                      (in a higher-high, higher-low sequence)
        Bearish BOS : close breaks below the most recent confirmed swing low

    Trend context filter:
        Only trade bullish BOS when ema_fast > ema_slow (and vice versa)
        to avoid counter-trend BOS trades.

    Exit:
        SL = most recent swing low (bullish BOS)
        TP = entry ± atr_tp_mult * ATR
    """

    name = "bos"

    def param_grid(self) -> List[Dict[str, Any]]:
        lefts    = [3, 5, 8]
        rights   = [2, 3]
        ema_fast = [8, 13, 21]
        ema_slow = [34, 55, 89]
        atr_lens = [7, 14]
        tp_mults = [2.0, 3.0]
        return [
            {"left": lf, "right": rt, "ema_fast": ef, "ema_slow": es,
             "atr_len": a, "tp_mult": tp}
            for lf, rt, ef, es, a, tp in itertools.product(
                lefts, rights, ema_fast, ema_slow, atr_lens, tp_mults
            )
            if ef < es
        ]

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        self._add_signal_columns(df)

        lf   = params["left"]
        rt   = params["right"]
        atr_col   = f"atr_{params['atr_len']}"
        ema_f_col = f"ema_{params['ema_fast']}"
        ema_s_col = f"ema_{params['ema_slow']}"

        c = df["close"]
        atr = df[atr_col] if atr_col in df.columns else (df["high"] - df["low"])

        ema_fast = df[ema_f_col] if ema_f_col in df.columns else \
                   c.ewm(span=params["ema_fast"], adjust=False).mean()
        ema_slow = df[ema_s_col] if ema_s_col in df.columns else \
                   c.ewm(span=params["ema_slow"], adjust=False).mean()

        last_sh = _rolling_pivot_high(df["high"], lf, rt)
        last_sl = _rolling_pivot_low (df["low"],  lf, rt)

        # BOS: close breaks and CLOSES beyond the most recent swing level
        bos_bull = (c > last_sh.shift(1)) & (c.shift(1) <= last_sh.shift(2))
        bos_bear = (c < last_sl.shift(1)) & (c.shift(1) >= last_sl.shift(2))

        # Trend context
        uptrend   = ema_fast > ema_slow
        downtrend = ema_fast < ema_slow

        df["signal"] = np.where(
            bos_bull & uptrend,   1,
            np.where(bos_bear & downtrend, -1, 0)
        )

        tp_dist = params["tp_mult"] * atr

        df["sl_price"] = np.where(
            df["signal"] == 1,  last_sl,
            np.where(df["signal"] == -1, last_sh, np.nan)
        )
        df["tp_price"] = np.where(
            df["signal"] == 1,  c + tp_dist,
            np.where(df["signal"] == -1, c - tp_dist, np.nan)
        )

        return df
