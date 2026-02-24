"""Tests for ScreeningService and screening functions."""

from datetime import date
from unittest.mock import MagicMock

import pytest

from market_analyzer.features.screening import (
    screen_breakout,
    screen_income,
    screen_mean_reversion,
    screen_momentum,
)
from market_analyzer.models.regime import RegimeID, RegimeResult, TrendDirection
from market_analyzer.service.screening import AVAILABLE_SCREENS, ScreeningResult, ScreeningService


def _make_regime(regime_id: RegimeID = RegimeID.R1_LOW_VOL_MR) -> RegimeResult:
    return RegimeResult(
        ticker="TEST",
        regime=regime_id,
        confidence=0.85,
        regime_probabilities={1: 0.85, 2: 0.05, 3: 0.05, 4: 0.05},
        trend_direction=TrendDirection.BULLISH,
        as_of_date=date(2026, 2, 23),
        model_version="test",
    )


class TestScreeningFunctions:
    def test_screen_breakout_passes_r3(self, sample_ohlcv_trending):
        from market_analyzer.features.technicals import compute_technicals

        regime = _make_regime(RegimeID.R3_LOW_VOL_TREND)
        technicals = compute_technicals(sample_ohlcv_trending, "TEST")
        passes, score, reason = screen_breakout(regime, technicals)
        assert isinstance(passes, bool)
        assert 0.0 <= score <= 1.0
        assert isinstance(reason, str)

    def test_screen_momentum_basic(self, sample_ohlcv_trending):
        from market_analyzer.features.technicals import compute_technicals

        regime = _make_regime(RegimeID.R3_LOW_VOL_TREND)
        technicals = compute_technicals(sample_ohlcv_trending, "TEST")
        passes, score, reason = screen_momentum(regime, technicals)
        assert isinstance(passes, bool)
        assert 0.0 <= score <= 1.0

    def test_screen_mean_reversion_basic(self, sample_ohlcv_choppy):
        from market_analyzer.features.technicals import compute_technicals

        regime = _make_regime(RegimeID.R2_HIGH_VOL_MR)
        technicals = compute_technicals(sample_ohlcv_choppy, "TEST")
        passes, score, reason = screen_mean_reversion(regime, technicals)
        assert isinstance(passes, bool)
        assert 0.0 <= score <= 1.0

    def test_screen_income_r1(self, sample_ohlcv_choppy):
        from market_analyzer.features.technicals import compute_technicals

        regime = _make_regime(RegimeID.R1_LOW_VOL_MR)
        technicals = compute_technicals(sample_ohlcv_choppy, "TEST")
        passes, score, reason = screen_income(regime, technicals)
        assert isinstance(passes, bool)
        assert 0.0 <= score <= 1.0
        # R1 should boost income score
        assert "R1" in reason or score > 0


class TestScreeningService:
    def test_requires_services(self):
        svc = ScreeningService()
        with pytest.raises(ValueError, match="requires regime and technical"):
            svc.scan(["TEST"])

    def test_scan_with_mock(self, sample_ohlcv_mixed, tmp_path):
        from market_analyzer.service.regime import RegimeService
        from market_analyzer.service.technical import TechnicalService

        regime_svc = RegimeService(model_dir=tmp_path / "models")
        tech_svc = TechnicalService()

        # Mock DataService to return our sample data
        data_svc = MagicMock()
        data_svc.get_ohlcv.return_value = sample_ohlcv_mixed

        svc = ScreeningService(
            regime_service=regime_svc,
            technical_service=tech_svc,
            data_service=data_svc,
        )
        result = svc.scan(["TEST"])
        assert isinstance(result, ScreeningResult)
        assert result.tickers_scanned == 1
        assert isinstance(result.candidates, list)

    def test_available_screens(self):
        assert "breakout" in AVAILABLE_SCREENS
        assert "momentum" in AVAILABLE_SCREENS
        assert "mean_reversion" in AVAILABLE_SCREENS
        assert "income" in AVAILABLE_SCREENS
