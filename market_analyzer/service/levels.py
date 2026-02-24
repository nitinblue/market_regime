"""LevelsService: unified price levels with confluence, stop loss, and R:R."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from market_analyzer.features.levels import compute_levels
from market_analyzer.models.levels import LevelsAnalysis

if TYPE_CHECKING:
    from market_analyzer.data.service import DataService
    from market_analyzer.service.regime_service import RegimeService
    from market_analyzer.service.technical import TechnicalService


class LevelsService:
    """Synthesize all price levels into ranked support/resistance with R:R."""

    def __init__(
        self,
        technical_service: TechnicalService,
        regime_service: RegimeService | None = None,
        data_service: DataService | None = None,
    ) -> None:
        self.technical_service = technical_service
        self.regime_service = regime_service
        self.data_service = data_service

    def analyze(
        self,
        ticker: str,
        ohlcv: pd.DataFrame | None = None,
        direction: str | None = None,
        entry_price: float | None = None,
        include_orb: bool = False,
        intraday: pd.DataFrame | None = None,
    ) -> LevelsAnalysis:
        """Compute unified price levels with confluence, stop, targets, R:R.

        Args:
            ticker: Instrument ticker.
            ohlcv: Optional OHLCV DataFrame. Auto-fetched if None.
            direction: Override trade direction ("long" or "short").
            entry_price: Override entry price. Defaults to current price.
            include_orb: Include ORB levels (requires intraday data).
            intraday: Intraday DataFrame for ORB. Fetched if None and include_orb=True.

        Returns:
            LevelsAnalysis with ranked levels, stop, targets, and R:R.
        """
        technicals = self.technical_service.snapshot(ticker, ohlcv)

        regime = None
        if self.regime_service is not None:
            try:
                regime = self.regime_service.detect(ticker, ohlcv)
            except Exception:
                pass  # Regime is optional for levels

        orb = None
        if include_orb:
            try:
                orb = self.technical_service.orb(ticker, intraday, technicals.atr)
            except Exception:
                pass  # ORB is optional

        return compute_levels(technicals, regime, orb, direction, entry_price)
