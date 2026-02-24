"""Options opportunity assessment — per-horizon go/no-go + strategy recommendation.

Subpackages:
    setups/       — Price-based directional pattern detection (breakout, momentum, mean_reversion)
    option_plays/ — Option structure recommendations by horizon (zero_dte, leap, earnings)
"""

# Re-export everything for backward compatibility
from market_analyzer.opportunity.setups.breakout import assess_breakout
from market_analyzer.opportunity.setups.momentum import assess_momentum
from market_analyzer.opportunity.setups.mean_reversion import assess_mean_reversion
from market_analyzer.opportunity.option_plays.zero_dte import assess_zero_dte
from market_analyzer.opportunity.option_plays.leap import assess_leap
from market_analyzer.opportunity.option_plays.earnings import assess_earnings_play

__all__ = [
    # Setups (price-based)
    "assess_breakout",
    "assess_momentum",
    "assess_mean_reversion",
    # Option plays (structure-specific)
    "assess_zero_dte",
    "assess_leap",
    "assess_earnings_play",
]
