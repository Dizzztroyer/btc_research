"""
mean_reversion.py
─────────────────
Mean-reversion strategy family.

Strategies
──────────
1. RSIReversion       — Buy oversold RSI, sell overbought RSI
2. BollingerReversion — Trade back to BB midline from extremes
3. EMADeviation       — Enter when price deviates significantly from EMA
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RSI Mean Reversion
# ═══════════════════════════════════════════════════════════════════════════════

class RSIReversionStrategy(BaseStrategy):
    """
    RSI extreme reversion.

    Entry:
        Long  : RSI crosses above os_level (was below, now above)
        Short : RSI crosses below ob_level (was above, now below)

    Filter:
        Optional ADX < adx_threshold to confirm ranging market

    Exit:
        SL = entry ± sl_atr_mult * ATR
        TP = entry ± tp_atr_mult * ATR  (target midpoint / mean)
    """

    name = "rsi_reversion"

    def param_grid(self) -> List[Dict[str, Any]]:
        rsi_len   = [7, 9, 14, 21]
        rsi_os    = [20, 25, 30]
        rsi_ob    = [70, 75, 80]
        atr_len   = [7, 14]
        sl_mult   = [1.0, 1.5, 2.0]
        tp_mult   = [1.5, 2.0, 3.0]
        adx_thresh= [0, 25, 30]   # 0 = no filter
        return [
            {"rsi_len": r, "rsi_os": os, "rsi_ob": ob,
             "atr_len": a, "sl_mult": sl, "tp_mult": tp,
             "adx_thresh": adx}
            for r, os, ob, a, sl, tp, adx in itertools.product(
                rsi_len, rsi_os, rsi_ob, atr_len, sl_mult, tp_mult, adx_thresh
            )
        ]

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        self._add_signal_columns(df)

        rsi_col = f"rsi_{params['rsi_len']}"
        atr_col = f"atr_{params['atr_len']}"
        adx_col = "adx_14"

        rsi = df[rsi_col] if rsi_col in df.columns else pd.Series(50, index=df.index)
        atr = df[atr_col] if atr_col in df.columns else (df["high"] - df["low"])

        long_cross  = (rsi.shift(1) < params["rsi_os"]) & (rsi >= params["rsi_os"])
        short_cross = (rsi.shift(1) > params["rsi_ob"]) & (rsi <= params["rsi_ob"])

        # ADX ranging filter
        if params["adx_thresh"] > 0 and adx_col in df.columns:
            is_ranging  = df[adx_col] < params["adx_thresh"]
            long_cross  = long_cross  & is_ranging
            short_cross = short_cross & is_ranging

        df["signal"] = np.where(long_cross, 1, np.where(short_cross, -1, 0))

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
# 2. Bollinger Band Reversion
# ═══════════════════════════════════════════════════════════════════════════════

class BollingerReversionStrategy(BaseStrategy):
    """
    Bollinger Band mean reversion.

    Entry:
        Long  : close crosses below lower BB (touches or breaches) then closes back inside
        Short : close crosses above upper BB then closes back inside

    Exit:
        SL  : outside the BB extreme that was touched
        TP  : BB midline
    """

    name = "bollinger_reversion"

    def param_grid(self) -> List[Dict[str, Any]]:
        bb_len  = [14, 20, 30]
        bb_std  = [1.5, 2.0, 2.5]
        atr_len = [7, 14]
        sl_mult = [1.0, 1.5]
        return [
            {"bb_len": bl, "bb_std": bs, "atr_len": a, "sl_mult": sl}
            for bl, bs, a, sl in itertools.product(bb_len, bb_std, atr_len, sl_mult)
        ]

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        self._add_signal_columns(df)

        bl  = params["bb_len"]
        bs  = params["bb_std"]
        tag = f"bb{bl}s{str(bs).replace('.','')}"

        upper_col = f"{tag}_upper"
        lower_col = f"{tag}_lower"
        mid_col   = f"{tag}_mid"
        atr_col   = f"atr_{params['atr_len']}"

        if upper_col not in df.columns:
            mid   = df["close"].rolling(bl).mean()
            sigma = df["close"].rolling(bl).std(ddof=0)
            upper = mid + bs * sigma
            lower = mid - bs * sigma
        else:
            upper = df[upper_col]
            lower = df[lower_col]
            mid   = df[mid_col]

        atr = df[atr_col] if atr_col in df.columns else (df["high"] - df["low"])

        # Price was outside band and is now returning inside
        long_signal  = (df["close"].shift(1) < lower.shift(1)) & (df["close"] >= lower)
        short_signal = (df["close"].shift(1) > upper.shift(1)) & (df["close"] <= upper)

        df["signal"] = np.where(long_signal, 1, np.where(short_signal, -1, 0))

        sl_dist = params["sl_mult"] * atr
        # TP = midline
        df["sl_price"] = np.where(
            df["signal"] == 1,  df["close"] - sl_dist,
            np.where(df["signal"] == -1, df["close"] + sl_dist, np.nan)
        )
        df["tp_price"] = np.where(
            df["signal"] == 1,  mid,
            np.where(df["signal"] == -1, mid, np.nan)
        )

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EMA Deviation Reversion
# ═══════════════════════════════════════════════════════════════════════════════

class EMADeviationStrategy(BaseStrategy):
    """
    Enter when price is stretched far from an EMA (measured in ATR units)
    and expect a return toward the mean.

    Entry:
        Long  : (close - EMA) / ATR < -dev_threshold  →  price too far below
        Short : (close - EMA) / ATR >  dev_threshold  →  price too far above

    Exit:
        SL = entry ± sl_mult * ATR
        TP = EMA (the mean)
    """

    name = "ema_deviation"

    def param_grid(self) -> List[Dict[str, Any]]:
        ema_len   = [21, 34, 55, 89]
        atr_len   = [7, 14]
        dev_thresh = [1.5, 2.0, 2.5, 3.0]
        sl_mult   = [1.0, 1.5, 2.0]
        return [
            {"ema_len": e, "atr_len": a, "dev_thresh": d, "sl_mult": sl}
            for e, a, d, sl in itertools.product(ema_len, atr_len, dev_thresh, sl_mult)
        ]

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        self._add_signal_columns(df)

        ema_col = f"ema_{params['ema_len']}"
        atr_col = f"atr_{params['atr_len']}"

        ema = df[ema_col] if ema_col in df.columns else \
              df["close"].ewm(span=params["ema_len"], adjust=False).mean()
        atr = df[atr_col] if atr_col in df.columns else (df["high"] - df["low"])

        deviation = (df["close"] - ema) / atr.replace(0, np.nan)

        long_signal  = deviation < -params["dev_thresh"]
        short_signal = deviation >  params["dev_thresh"]

        df["signal"] = np.where(long_signal, 1, np.where(short_signal, -1, 0))

        sl_dist = params["sl_mult"] * atr

        df["sl_price"] = np.where(
            df["signal"] == 1,  df["close"] - sl_dist,
            np.where(df["signal"] == -1, df["close"] + sl_dist, np.nan)
        )
        # TP = EMA (mean reversion target)
        df["tp_price"] = np.where(
            df["signal"] != 0, ema, np.nan
        )

        return df
