"""Options opportunity assessment — per-horizon go/no-go + strategy recommendation.

Subpackages:
    setups/       — Price-based directional pattern detection (breakout, momentum, mean_reversion)
    option_plays/ — Option structure recommendations by horizon (zero_dte, leap, earnings,
                    calendar, diagonal, iron_butterfly, ratio_spread)
"""

# Re-export everything for backward compatibility
from market_analyzer.opportunity.setups.breakout import assess_breakout
from market_analyzer.opportunity.setups.momentum import assess_momentum
from market_analyzer.opportunity.setups.mean_reversion import assess_mean_reversion
from market_analyzer.opportunity.setups.orb import assess_orb
from market_analyzer.opportunity.option_plays.zero_dte import assess_zero_dte
from market_analyzer.opportunity.option_plays.leap import assess_leap
from market_analyzer.opportunity.option_plays.earnings import assess_earnings_play, EarningsOpportunity
from market_analyzer.opportunity.setups.mean_reversion import MeanReversionOpportunity
from market_analyzer.opportunity.option_plays.calendar import assess_calendar
from market_analyzer.opportunity.option_plays.diagonal import assess_diagonal
from market_analyzer.opportunity.option_plays.iron_condor import assess_iron_condor
from market_analyzer.opportunity.option_plays.iron_butterfly import assess_iron_butterfly
from market_analyzer.opportunity.option_plays.ratio_spread import assess_ratio_spread

__all__ = [
    # Setups (price-based)
    "assess_breakout",
    "assess_momentum",
    "assess_mean_reversion",
    "assess_orb",
    # Option plays (structure-specific)
    "assess_zero_dte",
    "assess_leap",
    "assess_earnings_play",
    "EarningsOpportunity",
    "MeanReversionOpportunity",
    "assess_calendar",
    "assess_diagonal",
    "assess_iron_condor",
    "assess_iron_butterfly",
    "assess_ratio_spread",
]
