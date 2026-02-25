"""VolSurfaceService: volatility surface analysis via DataService."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from market_analyzer.features.vol_surface import compute_vol_surface
from market_analyzer.models.vol_surface import (
    SkewSlice,
    TermStructurePoint,
    VolatilitySurface,
)

if TYPE_CHECKING:
    from market_analyzer.data.service import DataService


class VolSurfaceService:
    """Compute volatility surfaces from options chain data."""

    def __init__(self, data_service: DataService | None = None) -> None:
        self.data_service = data_service

    def surface(self, ticker: str, as_of: date | None = None) -> VolatilitySurface:
        """Fetch options chain + OHLCV, compute vol surface."""
        if self.data_service is None:
            raise ValueError(
                "VolSurfaceService requires a DataService to fetch options chain data"
            )

        chain_df = self.data_service.get_options_chain(ticker)
        ohlcv = self.data_service.get_ohlcv(ticker)
        underlying_price = float(ohlcv["Close"].iloc[-1])

        return compute_vol_surface(chain_df, underlying_price, ticker, as_of=as_of)

    def term_structure(self, ticker: str) -> list[TermStructurePoint]:
        """Convenience: just the term structure."""
        surf = self.surface(ticker)
        return surf.term_structure

    def skew(self, ticker: str, expiration: date | None = None) -> SkewSlice | None:
        """Convenience: skew for nearest (or specified) expiration."""
        surf = self.surface(ticker)
        if not surf.skew_by_expiry:
            return None
        if expiration is None:
            return surf.skew_by_expiry[0]
        for s in surf.skew_by_expiry:
            if s.expiration == expiration:
                return s
        return None

    def calendar_edge(self, ticker: str) -> float:
        """Convenience: calendar edge score (0-1)."""
        surf = self.surface(ticker)
        return surf.calendar_edge_score
