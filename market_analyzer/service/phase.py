"""PhaseService: Wyckoff phase detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from market_analyzer.models.phase import PhaseResult
from market_analyzer.phases.detector import PhaseDetector

if TYPE_CHECKING:
    from market_analyzer.data.service import DataService
    from market_analyzer.service.regime import RegimeService


class PhaseService:
    """Detect Wyckoff phases (P1-P4) for instruments."""

    def __init__(
        self,
        regime_service: RegimeService | None = None,
        data_service: DataService | None = None,
    ) -> None:
        self.regime_service = regime_service
        self.data_service = data_service

    def _get_ohlcv(self, ticker: str, ohlcv: pd.DataFrame | None) -> pd.DataFrame:
        if ohlcv is not None:
            return ohlcv
        if self.data_service is None:
            raise ValueError(
                "Either provide ohlcv DataFrame or initialize PhaseService with a DataService"
            )
        return self.data_service.get_ohlcv(ticker)

    def detect(
        self, ticker: str, ohlcv: pd.DataFrame | None = None
    ) -> PhaseResult:
        """Detect Wyckoff phase for a single instrument.

        Requires regime history (auto-fits if regime_service provided).
        """
        df = self._get_ohlcv(ticker, ohlcv)
        if self.regime_service is None:
            raise ValueError(
                "PhaseService requires a RegimeService for regime history"
            )
        regime_series = self.regime_service.get_regime_history(ticker, df)
        detector = PhaseDetector()
        return detector.detect(ticker, df, regime_series)
