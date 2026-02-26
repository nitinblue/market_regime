"""Abstract broker interfaces — implement for each broker.

Three ABCs because not all brokers provide all capabilities:
- BrokerSession: authentication and connection lifecycle
- MarketDataProvider: option chains, quotes, Greeks
- MarketMetricsProvider: IV rank, beta, liquidity (TastyTrade-specific today)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from market_analyzer.models.opportunity import LegSpec
    from market_analyzer.models.quotes import MarketMetrics, OptionQuote


class BrokerSession(ABC):
    """Abstract broker connection. Implement for each broker."""

    @abstractmethod
    def connect(self) -> bool:
        """Authenticate and establish session. Returns True on success."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Clean up session resources."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @property
    @abstractmethod
    def broker_name(self) -> str:
        """Human-readable name: 'tastytrade', 'schwab', 'ibkr', etc."""
        ...


class MarketDataProvider(ABC):
    """Abstract market data provider for real-time quotes and Greeks."""

    @abstractmethod
    def get_option_chain(
        self, ticker: str, expiration: date | None = None,
    ) -> list[OptionQuote]:
        """Fetch full option chain with bid/ask/IV for a ticker.

        If expiration is None, returns all available expirations.
        """
        ...

    @abstractmethod
    def get_quotes(self, legs: list[LegSpec]) -> list[OptionQuote]:
        """Fetch quotes for specific option legs (strike + exp + type).

        Used by adjustment service for precise leg pricing.
        """
        ...

    @abstractmethod
    def get_greeks(self, legs: list[LegSpec]) -> dict[str, dict]:
        """Fetch Greeks for specific option legs.

        Returns ``{leg_key: {delta, gamma, theta, vega}}``.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """'tastytrade', 'schwab', 'ibkr', etc."""
        ...


class MarketMetricsProvider(ABC):
    """Abstract provider for market-level metrics (IV rank, etc.)."""

    @abstractmethod
    def get_metrics(self, tickers: list[str]) -> dict[str, MarketMetrics]:
        """Fetch IV rank, IV percentile, beta, etc. for tickers."""
        ...
