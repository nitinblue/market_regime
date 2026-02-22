"""FundamentalService: stock fundamentals."""

from __future__ import annotations

from market_analyzer.models.fundamentals import FundamentalsSnapshot


class FundamentalService:
    """Fetch stock fundamentals for instruments."""

    def get(self, ticker: str, ttl_minutes: int | None = None) -> FundamentalsSnapshot:
        """Fetch stock fundamentals for a single instrument.

        Uses yfinance with in-memory TTL cache.
        """
        from market_analyzer.fundamentals.fetch import fetch_fundamentals

        return fetch_fundamentals(ticker, ttl_minutes=ttl_minutes)
