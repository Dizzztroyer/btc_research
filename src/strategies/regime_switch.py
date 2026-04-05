"""
Regime-switching strategy.

Uses the regime label column (computed in feature builder) to select
which underlying strategy to run:
  - trending regime → apply trend-following sub-strategy
  - ranging regime  → apply mean-reversion sub-strategy
  - high-vol regime → reduce or skip signals
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
from src.strategies.base import BaseStrategy, StrategyResult
from src.strategies.trend_following import EMACrossover
from src.strategies.mean_reversion import RSIReversion


class RegimeSwitchStrategy(BaseStrategy):
    """
    Regime-adaptive strategy that delegates to different sub-strategies
    depending on the current market regime label.

    Regimes:
      0 = range      → use mean reversion
      1 = trend_up   → use trend following (long only)
      2 = trend_down → use trend following (short only)
      3 = high_vol   → no trades (skip)

    Params:
        trend_fast_ema: Fast EMA for trend-following sub-strategy (default 21)
        trend_slow_ema: Slow EMA for trend-following sub-strategy (default 55)
        rsi_length: RSI for mean-reversion sub-strategy (default 14)
        oversold: RSI oversold level (default 30)
        overbought: RSI overbought level (default 70)
        atr_length: ATR for both sub-strategies (default 14)
        sl_atr_mult: Stop multiplier (default 2.0)
        tp_atr_mult: Target multiplier (default 3.0)
        skip_high_vol: If True, no signals in regime 3 (default True)
    """

    @property
    def name(self) -> str:
        return "Regime_Switch"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        p = self.params
        skip_high_vol = p.get("skip_high_vol", True)

        signals = pd.Series(0, index=df.index, dtype=int)
        sl = pd.Series(np.nan, index=df.index)
        tp = pd.Series(np.nan, index=df.index)

        if "regime" not in df.columns:
            return StrategyResult(signals=signals, stop_losses=sl, take_profits=tp,
                                  name=self.name, params=p)

        regime = df["regime"]

        # Trend sub-strategy (used in regimes 1 and 2)
        trend_params = {
            "fast_ema": p.get("trend_fast_ema", 21),
            "slow_ema": p.get("trend_slow_ema", 55),
            "atr_length": p.get("atr_length", 14),
            "sl_atr_mult": p.get("sl_atr_mult", 2.0),
            "tp_atr_mult": p.get("tp_atr_mult", 3.0),
        }
        trend_strat = EMACrossover(trend_params)
        trend_result = trend_strat.generate_signals(df)

        # Mean-reversion sub-strategy (used in regime 0)
        mr_params = {
            "rsi_length": p.get("rsi_length", 14),
            "oversold": p.get("oversold", 30),
            "overbought": p.get("overbought", 70),
            "atr_length": p.get("atr_length", 14),
            "sl_atr_mult": p.get("sl_atr_mult", 2.0),
            "tp_atr_mult": p.get("tp_atr_mult", 3.0),
        }
        mr_strat = RSIReversion(mr_params)
        mr_result = mr_strat.generate_signals(df)

        # Blend by regime
        for i in range(len(df)):
            r = regime.iloc[i]
            if r == 0:
                # Range: mean reversion
                signals.iloc[i] = mr_result.signals.iloc[i]
                sl.iloc[i] = mr_result.stop_losses.iloc[i]
                tp.iloc[i] = mr_result.take_profits.iloc[i]
            elif r == 1:
                # Trending up: only longs
                sig = trend_result.signals.iloc[i]
                if sig == 1:
                    signals.iloc[i] = 1
                    sl.iloc[i] = trend_result.stop_losses.iloc[i]
                    tp.iloc[i] = trend_result.take_profits.iloc[i]
            elif r == 2:
                # Trending down: only shorts
                sig = trend_result.signals.iloc[i]
                if sig == -1:
                    signals.iloc[i] = -1
                    sl.iloc[i] = trend_result.stop_losses.iloc[i]
                    tp.iloc[i] = trend_result.take_profits.iloc[i]
            elif r == 3 and skip_high_vol:
                signals.iloc[i] = 0

        return StrategyResult(signals=signals, stop_losses=sl, take_profits=tp,
                              name=self.name, params=p)