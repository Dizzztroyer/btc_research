"""
breakout.py
───────────
Breakout / volatility-expansion strategy family.

Strategies
──────────
1. SqueezeBreakout     — Bollinger/Keltner squeeze release breakout
2. ConsolidationBreakout — Range detection + breakout of that range
3. ATRExpansionBreakout  — Entry on sudden ATR expansion vs recent avg
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Squeeze Breakout
# ═══════════════════════════════════════════════════════════════════════════════

class SqueezeBreakoutStrategy(BaseStrategy):
    """
    Bollinger / Keltner squeeze breakout.

    Logic:
        Squeeze ON  = Bollinger Bands inside Keltner Channels
        Squeeze OFF = BB expands outside KC

        Entry:
            Long  : squeeze just released AND close > midline
            Short : squeeze just released AND close < midline

    Exit:
        SL = low/high of the squeeze range ± buffer
        TP = entry ± atr_tp_mult * ATR
    """

    name = "squeeze_breakout"

    def param_grid(self) -> List[Dict[str, Any]]:
        lengths  = [10, 14, 20]
        bb_stds  = [1.5, 2.0]
        kc_mults = [1.0, 1.5]
        atr_lens = [7, 14]
        tp_mults = [2.0, 3.0, 4.0]
        sl_mults = [1.0, 1.5, 2.0]
        return [
            {"length": l, "bb_std": bs, "kc_mult": km,
             "atr_len": a, "tp_mult": tp, "sl_mult": sl}
            for l, bs, km, a, tp, sl in itertools.product(
                lengths, bb_stds, kc_mults, atr_lens, tp_mults, sl_mults
            )
        ]

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        self._add_signal_columns(df)

        n      = params["length"]
        bs     = params["bb_std"]
        km     = params["kc_mult"]
        atr_col = f"atr_{params['atr_len']}"

        c = df["close"]
        h = df["high"]
        l = df["low"]

        # Bollinger Bands
        bb_mid   = c.rolling(n).mean()
        bb_sigma = c.rolling(n).std(ddof=0)
        bb_upper = bb_mid + bs * bb_sigma
        bb_lower = bb_mid - bs * bb_sigma

        # Keltner Channels
        raw_tr  = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr_kc  = raw_tr.ewm(span=n, adjust=False).mean()
        kc_upper = bb_mid + km * atr_kc
        kc_lower = bb_mid - km * atr_kc

        squeeze_on  = (bb_upper <= kc_upper) & (bb_lower >= kc_lower)
        squeeze_off = ~squeeze_on

        # Release: squeeze was on last bar, now off
        sq_col = f"squeeze_{n}"
        if sq_col in df.columns:
            squeeze_release = (df[sq_col].shift(1) == 1) & (df[sq_col] == 0)
        else:
            squeeze_release = squeeze_on.shift(1).fillna(False) & squeeze_off

        long_signal  = squeeze_release & (c > bb_mid)
        short_signal = squeeze_release & (c < bb_mid)

        df["signal"] = np.where(long_signal, 1, np.where(short_signal, -1, 0))

        atr = df[atr_col] if atr_col in df.columns else raw_tr.ewm(span=14, adjust=False).mean()

        sl_dist = params["sl_mult"] * atr
        tp_dist = params["tp_mult"] * atr

        df["sl_price"] = np.where(
            df["signal"] == 1,  c - sl_dist,
            np.where(df["signal"] == -1, c + sl_dist, np.nan)
        )
        df["tp_price"] = np.where(
            df["signal"] == 1,  c + tp_dist,
            np.where(df["signal"] == -1, c - tp_dist, np.nan)
        )

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Consolidation Breakout
# ═══════════════════════════════════════════════════════════════════════════════

class ConsolidationBreakoutStrategy(BaseStrategy):
    """
    Consolidation + breakout.

    Consolidation detection:
        Rolling ATR / rolling average ATR < compress_ratio
        → price is in a tight range

    Entry:
        Long  : price breaks above rolling high of consolidation window
        Short : price breaks below rolling low  of consolidation window

    Exit:
        SL = opposite side of the range
        TP = entry ± atr_tp_mult * ATR
    """

    name = "consolidation_breakout"

    def param_grid(self) -> List[Dict[str, Any]]:
        lookbacks     = [10, 15, 20, 30]
        compress_rats = [0.5, 0.6, 0.7]
        atr_lens      = [7, 14]
        tp_mults      = [2.0, 3.0, 4.0]
        return [
            {"lookback": lb, "compress_ratio": cr, "atr_len": a, "tp_mult": tp}
            for lb, cr, a, tp in itertools.product(
                lookbacks, compress_rats, atr_lens, tp_mults
            )
        ]

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        self._add_signal_columns(df)

        lb     = params["lookback"]
        cr     = params["compress_ratio"]
        atr_col = f"atr_{params['atr_len']}"

        c = df["close"]
        h = df["high"]
        l = df["low"]

        atr     = df[atr_col] if atr_col in df.columns else (h - l)
        avg_atr = atr.rolling(lb * 3).mean()

        # Compression: recent ATR is small relative to its own history
        is_compressed = (atr / avg_atr.replace(0, np.nan)) < cr

        # Rolling high/low of the consolidation window (exclude current bar)
        roll_high = h.shift(1).rolling(lb).max()
        roll_low  = l.shift(1).rolling(lb).min()

        long_signal  = is_compressed.shift(1) & (c > roll_high)
        short_signal = is_compressed.shift(1) & (c < roll_low)

        df["signal"] = np.where(long_signal, 1, np.where(short_signal, -1, 0))

        tp_dist = params["tp_mult"] * atr

        # SL = opposite side of the range
        df["sl_price"] = np.where(
            df["signal"] == 1,  roll_low,
            np.where(df["signal"] == -1, roll_high, np.nan)
        )
        df["tp_price"] = np.where(
            df["signal"] == 1,  c + tp_dist,
            np.where(df["signal"] == -1, c - tp_dist, np.nan)
        )

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ATR Expansion Breakout
# ═══════════════════════════════════════════════════════════════════════════════

class ATRExpansionBreakoutStrategy(BaseStrategy):
    """
    Volatility expansion breakout.

    Entry:
        Current bar's range (H-L) is expansion_mult × the N-bar avg range
        AND close direction confirms:
            Long  : close near high (body in upper half) + price > EMA
            Short : close near low  (body in lower half) + price < EMA

    Exit:
        SL = entry ∓ sl_mult * ATR
        TP = entry ± tp_mult * ATR
    """

    name = "atr_expansion_breakout"

    def param_grid(self) -> List[Dict[str, Any]]:
        atr_lens   = [7, 10, 14]
        exp_mults  = [1.5, 2.0, 2.5]
        ema_lens   = [21, 55]
        sl_mults   = [1.0, 1.5, 2.0]
        tp_mults   = [2.0, 3.0]
        return [
            {"atr_len": a, "exp_mult": em, "ema_len": el, "sl_mult": sl, "tp_mult": tp}
            for a, em, el, sl, tp in itertools.product(
                atr_lens, exp_mults, ema_lens, sl_mults, tp_mults
            )
        ]

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        self._add_signal_columns(df)

        atr_col = f"atr_{params['atr_len']}"
        ema_col = f"ema_{params['ema_len']}"

        c = df["close"]
        h = df["high"]
        l = df["low"]

        atr = df[atr_col] if atr_col in df.columns else (h - l)
        ema = df[ema_col] if ema_col in df.columns else c.ewm(span=params["ema_len"], adjust=False).mean()

        bar_range = h - l
        avg_range = bar_range.shift(1).rolling(params["atr_len"]).mean()

        expanding = bar_range > params["exp_mult"] * avg_range.replace(0, np.nan)

        # Directional confirmation via body position
        body_mid   = (df["open"] + c) / 2
        upper_half = body_mid > (l + bar_range * 0.5)
        lower_half = body_mid < (l + bar_range * 0.5)

        long_signal  = expanding & upper_half & (c > ema)
        short_signal = expanding & lower_half & (c < ema)

        df["signal"] = np.where(long_signal, 1, np.where(short_signal, -1, 0))

        sl_dist = params["sl_mult"] * atr
        tp_dist = params["tp_mult"] * atr

        df["sl_price"] = np.where(
            df["signal"] == 1,  c - sl_dist,
            np.where(df["signal"] == -1, c + sl_dist, np.nan)
        )
        df["tp_price"] = np.where(
            df["signal"] == 1,  c + tp_dist,
            np.where(df["signal"] == -1, c - tp_dist, np.nan)
        )

        return df
