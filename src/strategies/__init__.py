from .trend_following import EMATrendPullback, DonchianBreakout, EMACrossover, TripleEMATrend
from .mean_reversion import RSIReversion, BollingerReversion, EMADeviationReversion
from .breakout import SqueezeBreakout, ConsolidationBreakout
from .structure import SwingBreakout, BOSStrategy
from .regime_switch import RegimeSwitchStrategy
from .ensemble import EnsembleVoting
from .base import BaseStrategy, StrategyResult

ALL_STRATEGIES = {
    "EMA_TrendPullback": EMATrendPullback,
    "Donchian_Breakout": DonchianBreakout,
    "EMA_Crossover": EMACrossover,
    "Triple_EMA_Trend": TripleEMATrend,
    "RSI_Reversion": RSIReversion,
    "Bollinger_Reversion": BollingerReversion,
    "EMA_Deviation_Reversion": EMADeviationReversion,
    "Squeeze_Breakout": SqueezeBreakout,
    "Consolidation_Breakout": ConsolidationBreakout,
    "Swing_Breakout": SwingBreakout,
    "BOS_Structure": BOSStrategy,
    "Regime_Switch": RegimeSwitchStrategy,
    "Ensemble_Voting": EnsembleVoting,
}

STRATEGY_FAMILIES = {
    "trend_following": ["EMA_TrendPullback", "Donchian_Breakout", "EMA_Crossover", "Triple_EMA_Trend"],
    "mean_reversion": ["RSI_Reversion", "Bollinger_Reversion", "EMA_Deviation_Reversion"],
    "breakout": ["Squeeze_Breakout", "Consolidation_Breakout"],
    "structure": ["Swing_Breakout", "BOS_Structure"],
    "regime_switch": ["Regime_Switch"],
    "ensemble": ["Ensemble_Voting"],
}