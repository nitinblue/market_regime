"""Tests for mean reversion opportunity assessment."""

from datetime import date

import pytest

from market_analyzer.models.opportunity import Verdict
from market_analyzer.models.regime import RegimeID, RegimeResult, TrendDirection
from market_analyzer.opportunity.mean_reversion import (
    MeanReversionOpportunity,
    assess_mean_reversion,
)


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


class TestMeanReversionAssessment:
    def test_basic_assessment(self, sample_ohlcv_choppy):
        from market_analyzer.features.technicals import compute_technicals

        regime = _make_regime(RegimeID.R1_LOW_VOL_MR)
        technicals = compute_technicals(sample_ohlcv_choppy, "TEST")

        result = assess_mean_reversion(
            "TEST", regime=regime, technicals=technicals,
        )

        assert isinstance(result, MeanReversionOpportunity)
        assert result.ticker == "TEST"
        assert result.verdict in (Verdict.GO, Verdict.CAUTION, Verdict.NO_GO)
        assert 0.0 <= result.confidence <= 1.0

    def test_r4_hard_stop(self, sample_ohlcv_choppy):
        from market_analyzer.features.technicals import compute_technicals

        regime = _make_regime(RegimeID.R4_HIGH_VOL_TREND)
        technicals = compute_technicals(sample_ohlcv_choppy, "TEST")

        result = assess_mean_reversion(
            "TEST", regime=regime, technicals=technicals,
        )

        assert result.verdict == Verdict.NO_GO
        assert len(result.hard_stops) >= 1

    def test_mr_regime_boosts_score(self, sample_ohlcv_choppy):
        from market_analyzer.features.technicals import compute_technicals

        technicals = compute_technicals(sample_ohlcv_choppy, "TEST")

        # Test with MR regime
        regime_mr = _make_regime(RegimeID.R2_HIGH_VOL_MR)
        result_mr = assess_mean_reversion("TEST", regime=regime_mr, technicals=technicals)

        # Test with trending regime (not hard stopped)
        regime_trend = RegimeResult(
            ticker="TEST",
            regime=RegimeID.R3_LOW_VOL_TREND,
            confidence=0.5,
            regime_probabilities={1: 0.1, 2: 0.1, 3: 0.5, 4: 0.3},
            trend_direction=TrendDirection.BULLISH,
            as_of_date=date(2026, 2, 23),
            model_version="test",
        )
        result_trend = assess_mean_reversion("TEST", regime=regime_trend, technicals=technicals)

        # MR regime should generally score better (or equal)
        assert result_mr.confidence >= result_trend.confidence or result_trend.verdict == Verdict.NO_GO

    def test_has_summary(self, sample_ohlcv_choppy):
        from market_analyzer.features.technicals import compute_technicals

        regime = _make_regime(RegimeID.R1_LOW_VOL_MR)
        technicals = compute_technicals(sample_ohlcv_choppy, "TEST")

        result = assess_mean_reversion("TEST", regime=regime, technicals=technicals)
        assert result.summary != ""
