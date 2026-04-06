"""
regime.py
─────────
Regime-based switching strategy.

The regime switcher routes capital between:
    - A trend-following model (active in trending regimes)
    - A mean-reversion model (active in ranging regimes)

Regime is determined by:
    1. ADX level        (ADX > threshold → trending)
    2. EMA alignment    (price > EMA hierarchy → bullish trend)
    3. Volatility level (rolling std relative to its own history)

This is fully algorithmic — regime is determined bar-by-bar from
lagged indicators (no lookahead).
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.strategies.trend import EMACrossStrategy
from src.strategies.mean_reversion import RSIReversionStrategy


class RegimeSwitchStrategy(BaseStrategy):
    """
    Routes signals from a trend model or a mean-reversion model
    depending on the detected market regime.

    Regime determination (all computed on prior bar to avoid lookahead):
        Trending  : ADX > adx_thresh  OR  EMA alignment (fast > slow > price relative)
        Ranging   : ADX ≤ adx_thresh  AND  no clear EMA trend

    In trending regime  → use EMACross signals
    In ranging  regime  → use RSIReversion signals
    """

    name = "regime_switch"

    def __init__(self) -> None:
        self._trend_strat = EMACrossStrategy()
        self._mr_strat    = RSIReversionStrategy()

    def param_grid(self) -> List[Dict[str, Any]]:
        adx_thresholds = [20, 25, 30]
        adx_lengths    = [10, 14]
        # Trend sub-params
        trend_params = [
            {"t_fast": 8,  "t_slow": 34, "t_atr": 14, "t_sl": 2.0, "t_tp": 3.0},
            {"t_fast": 13, "t_slow": 55, "t_atr": 14, "t_sl": 2.0, "t_tp": 3.0},
            {"t_fast": 21, "t_slow": 89, "t_atr": 14, "t_sl": 2.0, "t_tp": 4.0},
        ]
        # MR sub-params
        mr_params = [
            {"mr_rsi_len": 14, "mr_os": 30, "mr_ob": 70, "mr_atr": 14, "mr_sl": 1.5, "mr_tp": 2.0},
            {"mr_rsi_len": 9,  "mr_os": 25, "mr_ob": 75, "mr_atr": 14, "mr_sl": 1.0, "mr_tp": 2.0},
        ]
        grid = []
        for adx_t, adx_l, tp, mp in itertools.product(
            adx_thresholds, adx_lengths, trend_params, mr_params
        ):
            p = {"adx_thresh": adx_t, "adx_len": adx_l}
            p.update(tp)
            p.update(mp)
            grid.append(p)
        return grid

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        self._add_signal_columns(df)

        adx_col = f"adx_{params['adx_len']}"

        # ── Regime classification ──────────────────────────────────────────────
        if adx_col in df.columns:
            adx = df[adx_col]
        else:
            # Compute ADX inline
            h, l, c = df["high"], df["low"], df["close"]
            n       = params["adx_len"]
            raw_tr  = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
            dm_p    = (h - h.shift(1)).clip(lower=0)
            dm_m    = (l.shift(1) - l).clip(lower=0)
            mask    = dm_p >= dm_m
            dm_p    = dm_p.where(mask, 0)
            dm_m    = dm_m.where(~mask, 0)
            atr_s   = raw_tr.ewm(span=n, adjust=False).mean()
            dip     = 100 * dm_p.ewm(span=n, adjust=False).mean() / atr_s.replace(0, np.nan)
            dim     = 100 * dm_m.ewm(span=n, adjust=False).mean() / atr_s.replace(0, np.nan)
            dx      = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
            adx     = dx.ewm(span=n, adjust=False).mean()

        # Use prior bar's ADX (no lookahead)
        is_trending = adx.shift(1) > params["adx_thresh"]
        is_ranging  = ~is_trending

        # ── Generate trend signals ─────────────────────────────────────────────
        trend_p = {
            "fast":    params["t_fast"],
            "slow":    params["t_slow"],
            "atr_len": params["t_atr"],
            "sl_mult": params["t_sl"],
            "tp_mult": params["t_tp"],
        }
        df_trend = self._trend_strat.generate_signals(df, trend_p)

        # ── Generate mean-reversion signals ───────────────────────────────────
        mr_p = {
            "rsi_len":   params["mr_rsi_len"],
            "rsi_os":    params["mr_os"],
            "rsi_ob":    params["mr_ob"],
            "atr_len":   params["mr_atr"],
            "sl_mult":   params["mr_sl"],
            "tp_mult":   params["mr_tp"],
            "adx_thresh": 0,     # no inner ADX filter
        }
        df_mr = self._mr_strat.generate_signals(df, mr_p)

        # ── Combine: use trend signal in trend regime, MR in ranging ──────────
        df["signal"] = np.where(
            is_trending, df_trend["signal"],
            np.where(is_ranging, df_mr["signal"], 0)
        )
        df["sl_price"] = np.where(
            is_trending, df_trend["sl_price"], df_mr["sl_price"]
        )
        df["tp_price"] = np.where(
            is_trending, df_trend["tp_price"], df_mr["tp_price"]
        )

        return df
