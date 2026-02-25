"""Opening Range Breakout (ORB) — re-exports from features.patterns.orb.

Kept for backward compatibility. New code should import from
market_analyzer.features.patterns.orb or market_analyzer.features.patterns.
"""

from market_analyzer.features.patterns.orb import compute_orb  # noqa: F401

__all__ = ["compute_orb"]
