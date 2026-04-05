"""
Ensemble / portfolio strategy.

Combines signals from multiple strategies using voting or score-weighting.
Signals are only taken when enough sub-strategies agree.
"""

from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from src.strategies.base import BaseStrategy, StrategyResult
from src.strategies.trend_following import EMACrossover, DonchianBreakout
from src.strategies.mean_reversion import RSIReversion, BollingerReversion
from src.strategies.breakout import SqueezeBreakout


class EnsembleVoting(BaseStrategy):
    """
    Majority-vote ensemble across multiple strategy signals.

    Runs several sub-strategies and takes a position only if
    at least `min_agreement` of them agree on direction.

    Params:
        min_agreement: Minimum number of agreeing signals (default 2)
        atr_length: (default 14)
        sl_atr_mult: (default 2.0)
        tp_atr_mult: (default 3.0)
    """

    @property
    def name(self) -> str:
        return "Ensemble_Voting"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        p = self.params
        min_agree = p.get("min_agreement", 2)
        atr_len = p.get("atr_length", 14)
        sl_mult = p.get("sl_atr_mult", 2.0)
        tp_mult = p.get("tp_atr_mult", 3.0)

        signals = pd.Series(0, index=df.index, dtype=int)
        sl = pd.Series(np.nan, index=df.index)
        tp = pd.Series(np.nan, index=df.index)

        # Sub-strategies
        sub_strategies: List[BaseStrategy] = [
            EMACrossover({"fast_ema": 21, "slow_ema": 55, "atr_length": atr_len,
                          "sl_atr_mult": sl_mult, "tp_atr_mult": tp_mult}),
            DonchianBreakout({"dc_length": 20, "atr_length": atr_len,
                              "sl_atr_mult": sl_mult, "tp_atr_mult": tp_mult}),
            RSIReversion({"rsi_length": 14, "oversold": 35, "overbought": 65,
                          "atr_length": atr_len, "sl_atr_mult": sl_mult, "tp_atr_mult": tp_mult}),
            BollingerReversion({"bb_length": 20, "bb_std": 2.0, "rsi_length": 14,
                                "oversold_rsi": 40, "overbought_rsi": 60,
                                "atr_length": atr_len, "sl_atr_mult": sl_mult, "tp_atr_mult": tp_mult}),
            SqueezeBreakout({"bb_length": 20, "bb_std": 2.0, "dc_length": 20,
                             "atr_length": atr_len, "sl_atr_mult": sl_mult, "tp_atr_mult": tp_mult}),
        ]

        results = [s.generate_signals(df) for s in sub_strategies]
        signal_matrix = pd.concat([r.signals for r in results], axis=1)
        signal_matrix.columns = range(len(results))

        long_votes = (signal_matrix == 1).sum(axis=1)
        short_votes = (signal_matrix == -1).sum(axis=1)

        long_agree = long_votes >= min_agree
        short_agree = short_votes >= min_agree

        # Collect average stop/tp from agreeing strategies
        atr_col = f"atr_{atr_len}"
        if atr_col in df.columns:
            atr = df[atr_col]
            sl_dist = sl_mult * atr
            tp_dist = tp_mult * atr

            signals[long_agree] = 1
            sl[long_agree] = df["close"][long_agree] - sl_dist[long_agree]
            tp[long_agree] = df["close"][long_agree] + tp_dist[long_agree]

            signals[short_agree] = -1
            sl[short_agree] = df["close"][short_agree] + sl_dist[short_agree]
            tp[short_agree] = df["close"][short_agree] - tp_dist[short_agree]

        return StrategyResult(signals=signals, stop_losses=sl, take_profits=tp,
                              name=self.name, params=p)