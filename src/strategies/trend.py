"""
trend.py
────────
Trend-following strategy family.

Implemented strategies
──────────────────────
1. EMACross      — EMA fast/slow crossover with ATR-based stops
2. DonchianBreakout — Donchian channel breakout with ATR trailing
3. PullbackTrend  — EMA trend filter + pullback entry on RSI dip
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return column or compute EMA on close if missing."""
    if name in df.columns:
        return df[name]
    # If asked for ema_N and not present, compute on close
    if name.startswith("ema_"):
        n = int(name.split("_")[1])
        return df["close"].ewm(span=n, adjust=False).mean()
    if name.startswith("sma_"):
        n = int(name.split("_")[1])
        return df["close"].rolling(n).mean()
    raise KeyError(f"Column '{name}' not found in DataFrame")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. EMA Crossover
# ═══════════════════════════════════════════════════════════════════════════════

class EMACrossStrategy(BaseStrategy):
    """
    Classic EMA crossover.

    Entry rules:
        Long  : fast EMA crosses above slow EMA AND price > slow EMA
        Short : fast EMA crosses below slow EMA AND price < slow EMA

    Exit (stop/target):
        SL    = entry ± atr_sl_mult * ATR
        TP    = entry ± atr_tp_mult * ATR
    """

    name = "ema_cross"

    def param_grid(self) -> List[Dict[str, Any]]:
        fast_lengths = [5, 8, 13, 21]
        slow_lengths = [21, 34, 55, 89, 144]
        atr_lengths  = [7, 14]
        sl_mults     = [1.5, 2.0, 2.5]
        tp_mults     = [2.0, 3.0, 4.0]

        grid = []
        for f, s, a, sl, tp in itertools.product(
            fast_lengths, slow_lengths, atr_lengths, sl_mults, tp_mults
        ):
            if f >= s:
                continue  # fast must be shorter than slow
            grid.append({"fast": f, "slow": s, "atr_len": a,
                          "sl_mult": sl, "tp_mult": tp})
        return grid

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        self._add_signal_columns(df)

        fast_col = f"ema_{params['fast']}"
        slow_col = f"ema_{params['slow']}"
        atr_col  = f"atr_{params['atr_len']}"

        ema_fast = _col(df, fast_col)
        ema_slow = _col(df, slow_col)
        atr      = _col(df, atr_col) if atr_col in df.columns else (df["high"] - df["low"])

        cross_up = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        cross_dn = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))

        # Trend filter: price on the right side of slow EMA
        trend_up = df["close"] > ema_slow
        trend_dn = df["close"] < ema_slow

        df["signal"] = np.where(
            cross_up & trend_up, 1,
            np.where(cross_dn & trend_dn, -1, 0)
        )

        # SL / TP based on signal bar's ATR
        sl_dist = params["sl_mult"] * atr
        tp_dist = params["tp_mult"] * atr

        df["sl_price"] = np.where(
            df["signal"] == 1,  df["close"] - sl_dist,
            np.where(df["signal"] == -1, df["close"] + sl_dist, np.nan)
        )
        df["tp_price"] = np.where(
            df["signal"] == 1,  df["close"] + tp_dist,
            np.where(df["signal"] == -1, df["close"] - tp_dist, np.nan)
        )

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Donchian Breakout
# ═══════════════════════════════════════════════════════════════════════════════

class DonchianBreakoutStrategy(BaseStrategy):
    """
    Donchian channel breakout.

    Entry rules:
        Long  : close > Donchian high of past N bars
        Short : close < Donchian low  of past N bars

    Exit:
        SL = entry ± atr_sl_mult * ATR (trailing)
        TP = entry ± atr_tp_mult * ATR
    """

    name = "donchian_breakout"

    def param_grid(self) -> List[Dict[str, Any]]:
        lengths  = [10, 20, 40, 55]
        atrs     = [7, 14]
        sl_mults = [1.5, 2.0, 2.5]
        tp_mults = [2.0, 3.0, 4.0]
        return [
            {"length": l, "atr_len": a, "sl_mult": sl, "tp_mult": tp}
            for l, a, sl, tp in itertools.product(lengths, atrs, sl_mults, tp_mults)
        ]

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        self._add_signal_columns(df)

        n       = params["length"]
        atr_col = f"atr_{params['atr_len']}"

        dh_col = f"don_{n}_high"
        dl_col = f"don_{n}_low"

        if dh_col in df.columns:
            don_high = df[dh_col]
            don_low  = df[dl_col]
        else:
            don_high = df["high"].shift(1).rolling(n).max()
            don_low  = df["low"].shift(1).rolling(n).min()

        atr = df[atr_col] if atr_col in df.columns else (df["high"] - df["low"])

        long_entry  = df["close"] > don_high
        short_entry = df["close"] < don_low

        df["signal"] = np.where(long_entry, 1, np.where(short_entry, -1, 0))

        sl_dist = params["sl_mult"] * atr
        tp_dist = params["tp_mult"] * atr

        df["sl_price"] = np.where(
            df["signal"] == 1,  df["close"] - sl_dist,
            np.where(df["signal"] == -1, df["close"] + sl_dist, np.nan)
        )
        df["tp_price"] = np.where(
            df["signal"] == 1,  df["close"] + tp_dist,
            np.where(df["signal"] == -1, df["close"] - tp_dist, np.nan)
        )

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Pullback Trend
# ═══════════════════════════════════════════════════════════════════════════════

class PullbackTrendStrategy(BaseStrategy):
    """
    Trend + pullback entry.

    Rules:
        Trend  : EMA fast > EMA slow (price > both → uptrend)
        Entry  : RSI dips below os_level then recovers above it
        SL     : low of last N bars minus small buffer
        TP     : entry + atr_tp_mult * ATR
    """

    name = "pullback_trend"

    def param_grid(self) -> List[Dict[str, Any]]:
        fast    = [8, 13, 21]
        slow    = [55, 89, 144]
        rsi_len = [7, 9, 14]
        rsi_os  = [30, 35, 40]
        atr_len = [7, 14]
        tp_mult = [2.0, 3.0, 4.0]
        sl_mult = [1.0, 1.5, 2.0]
        return [
            {"fast": f, "slow": s, "rsi_len": r, "rsi_os": os,
             "atr_len": a, "tp_mult": tp, "sl_mult": sl}
            for f, s, r, os, a, tp, sl in itertools.product(
                fast, slow, rsi_len, rsi_os, atr_len, tp_mult, sl_mult
            )
            if f < s
        ]

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        self._add_signal_columns(df)

        ema_fast = _col(df, f"ema_{params['fast']}")
        ema_slow = _col(df, f"ema_{params['slow']}")
        rsi_col  = f"rsi_{params['rsi_len']}"
        atr_col  = f"atr_{params['atr_len']}"

        rsi = df[rsi_col] if rsi_col in df.columns else pd.Series(50, index=df.index)
        atr = df[atr_col] if atr_col in df.columns else (df["high"] - df["low"])

        uptrend = (ema_fast > ema_slow) & (df["close"] > ema_slow)
        # RSI recovery: was below os, now above os
        rsi_was_os  = rsi.shift(1) < params["rsi_os"]
        rsi_recover = rsi >= params["rsi_os"]
        long_signal = uptrend & rsi_was_os & rsi_recover

        # Short side: price below both EMAs, RSI was overbought, now falling
        downtrend   = (ema_fast < ema_slow) & (df["close"] < ema_slow)
        rsi_was_ob  = rsi.shift(1) > (100 - params["rsi_os"])
        rsi_sell    = rsi <= (100 - params["rsi_os"])
        short_signal = downtrend & rsi_was_ob & rsi_sell

        df["signal"] = np.where(long_signal, 1, np.where(short_signal, -1, 0))

        sl_dist = params["sl_mult"] * atr
        tp_dist = params["tp_mult"] * atr

        df["sl_price"] = np.where(
            df["signal"] == 1,  df["close"] - sl_dist,
            np.where(df["signal"] == -1, df["close"] + sl_dist, np.nan)
        )
        df["tp_price"] = np.where(
            df["signal"] == 1,  df["close"] + tp_dist,
            np.where(df["signal"] == -1, df["close"] - tp_dist, np.nan)
        )

        return df


# Avoid linter complaint about emu_slow used before definition — walrus operator is fine in 3.8+
# (The walrus pattern above is intentional and valid Python.)
