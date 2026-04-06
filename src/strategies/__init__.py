"""
Strategy registry.

All strategy classes are registered here.
The research engine imports STRATEGY_REGISTRY to discover available strategies.
"""

from src.strategies.trend import (
    EMACrossStrategy,
    DonchianBreakoutStrategy,
    PullbackTrendStrategy,
)
from src.strategies.mean_reversion import (
    RSIReversionStrategy,
    BollingerReversionStrategy,
    EMADeviationStrategy,
)
from src.strategies.breakout import (
    SqueezeBreakoutStrategy,
    ConsolidationBreakoutStrategy,
    ATRExpansionBreakoutStrategy,
)
from src.strategies.structure import (
    SwingBreakoutStrategy,
    LiquiditySweepStrategy,
    BOSStrategy,
)
from src.strategies.regime import RegimeSwitchStrategy
from src.strategies.ensemble import MajorityVoteEnsemble, WeightedEnsemble

# Family → list of strategy classes
STRATEGY_FAMILIES = {
    "trend": [
        EMACrossStrategy,
        DonchianBreakoutStrategy,
        PullbackTrendStrategy,
    ],
    "mean_reversion": [
        RSIReversionStrategy,
        BollingerReversionStrategy,
        EMADeviationStrategy,
    ],
    "breakout": [
        SqueezeBreakoutStrategy,
        ConsolidationBreakoutStrategy,
        ATRExpansionBreakoutStrategy,
    ],
    "structure": [
        SwingBreakoutStrategy,
        LiquiditySweepStrategy,
        BOSStrategy,
    ],
    "regime": [
        RegimeSwitchStrategy,
    ],
    "ensemble": [
        MajorityVoteEnsemble,
        WeightedEnsemble,
    ],
}

# Flat registry: name → class
STRATEGY_REGISTRY = {
    cls().name: cls
    for family in STRATEGY_FAMILIES.values()
    for cls in family
}
