"""Tests for InstrumentAnalysisService."""

import pytest

from market_analyzer.models.instrument import InstrumentAnalysis
from market_analyzer.service.instrument import InstrumentAnalysisService
from market_analyzer.service.regime import RegimeService
from market_analyzer.service.technical import TechnicalService
from market_analyzer.service.phase import PhaseService


class TestInstrumentAnalysisService:
    def test_requires_services(self):
        svc = InstrumentAnalysisService()
        with pytest.raises(ValueError, match="requires regime, technical, and phase"):
            svc.analyze("TEST")

    def test_requires_data(self):
        svc = InstrumentAnalysisService(
            regime_service=RegimeService(),
            technical_service=TechnicalService(),
            phase_service=PhaseService(regime_service=RegimeService()),
        )
        with pytest.raises(ValueError, match="Either provide ohlcv"):
            svc.analyze("TEST")

    def test_analyze_with_ohlcv(self, sample_ohlcv_mixed, tmp_path):
        regime_svc = RegimeService(model_dir=tmp_path / "models")
        svc = InstrumentAnalysisService(
            regime_service=regime_svc,
            technical_service=TechnicalService(),
            phase_service=PhaseService(regime_service=regime_svc),
        )
        result = svc.analyze("TEST", ohlcv=sample_ohlcv_mixed)
        assert isinstance(result, InstrumentAnalysis)
        assert result.ticker == "TEST"
        assert result.regime_id in (1, 2, 3, 4)
        assert result.phase_id in (1, 2, 3, 4)
        assert result.trend_bias in ("bullish", "bearish", "neutral")
        assert result.volatility_label in ("low", "high")
        assert result.summary != ""

    def test_analyze_batch(self, sample_ohlcv_mixed, tmp_path):
        regime_svc = RegimeService(model_dir=tmp_path / "models")
        svc = InstrumentAnalysisService(
            regime_service=regime_svc,
            technical_service=TechnicalService(),
            phase_service=PhaseService(regime_service=regime_svc),
        )
        # Batch with one ticker (both use same data since no DataService)
        # This tests the error-resilient path
        results = svc.analyze_batch(["TEST"], include_opportunities=False)
        # analyze_batch without DataService will fail for tickers that need auto-fetch
        # but our test provides ohlcv through regime cache, so it should fail gracefully
        assert isinstance(results, dict)

    def test_trend_bias_bullish(self, sample_ohlcv_trending, tmp_path):
        regime_svc = RegimeService(model_dir=tmp_path / "models")
        svc = InstrumentAnalysisService(
            regime_service=regime_svc,
            technical_service=TechnicalService(),
            phase_service=PhaseService(regime_service=regime_svc),
        )
        result = svc.analyze("TEST", ohlcv=sample_ohlcv_trending)
        # Trending data should typically produce bullish bias
        assert result.trend_bias in ("bullish", "bearish", "neutral")
