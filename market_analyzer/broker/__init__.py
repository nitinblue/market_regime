"""Broker integration — pluggable ABCs + implementations.

Provides abstract interfaces that any broker can implement.
TastyTrade implementation is included as an optional sub-package.
"""

from market_analyzer.broker.base import (
    BrokerSession,
    MarketDataProvider,
    MarketMetricsProvider,
)

__all__ = [
    "BrokerSession",
    "MarketDataProvider",
    "MarketMetricsProvider",
]
