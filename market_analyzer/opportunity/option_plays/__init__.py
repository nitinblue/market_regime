"""Option strategy recommendations — horizon-specific structure selection."""

from market_analyzer.opportunity.option_plays.zero_dte import assess_zero_dte
from market_analyzer.opportunity.option_plays.leap import assess_leap
from market_analyzer.opportunity.option_plays.earnings import assess_earnings_play

__all__ = [
    "assess_zero_dte",
    "assess_leap",
    "assess_earnings_play",
]
