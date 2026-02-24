"""Price-based setup detection — directional pattern recognition."""

from market_analyzer.opportunity.setups.breakout import assess_breakout
from market_analyzer.opportunity.setups.momentum import assess_momentum
from market_analyzer.opportunity.setups.mean_reversion import assess_mean_reversion

__all__ = [
    "assess_breakout",
    "assess_momentum",
    "assess_mean_reversion",
]
