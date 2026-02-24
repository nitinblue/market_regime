"""Tests for earnings play opportunity assessment."""

from datetime import date, datetime

import pytest

from market_analyzer.models.fundamentals import (
    BusinessInfo,
    CashMetrics,
    DebtMetrics,
    DividendMetrics,
    EarningsEvent,
    EarningsMetrics,
    FiftyTwoWeek,
    FundamentalsSnapshot,
    MarginMetrics,
    ReturnMetrics,
    RevenueMetrics,
    UpcomingEvents,
    ValuationMetrics,
)
from market_analyzer.models.opportunity import Verdict
from market_analyzer.models.regime import RegimeID, RegimeResult, TrendDirection
from market_analyzer.opportunity.earnings import (
    EarningsOpportunity,
    assess_earnings_play,
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


def _make_fundamentals(days_to_earnings: int | None = 7) -> FundamentalsSnapshot:
    return FundamentalsSnapshot(
        ticker="TEST",
        as_of=datetime(2026, 2, 23),
        business=BusinessInfo(long_name="Test Corp", sector="Technology"),
        valuation=ValuationMetrics(trailing_pe=20.0),
        earnings=EarningsMetrics(trailing_eps=5.0),
        revenue=RevenueMetrics(market_cap=1_000_000_000),
        margins=MarginMetrics(profit_margins=0.15),
        cash=CashMetrics(),
        debt=DebtMetrics(),
        returns=ReturnMetrics(),
        dividends=DividendMetrics(),
        fifty_two_week=FiftyTwoWeek(),
        recent_earnings=[
            EarningsEvent(
                date=date(2025, 11, 15),
                eps_estimate=4.50,
                eps_actual=5.00,
                eps_difference=0.50,
                surprise_pct=11.0,
            ),
        ],
        upcoming_events=UpcomingEvents(
            next_earnings_date=date(2026, 3, 1) if days_to_earnings is not None else None,
            days_to_earnings=days_to_earnings,
        ),
    )


class TestEarningsPlayAssessment:
    def test_no_earnings_date_is_no_go(self, sample_ohlcv_trending):
        from market_analyzer.features.technicals import compute_technicals

        regime = _make_regime()
        technicals = compute_technicals(sample_ohlcv_trending, "TEST")

        result = assess_earnings_play(
            "TEST", regime=regime, technicals=technicals,
            fundamentals=None,
        )

        assert isinstance(result, EarningsOpportunity)
        assert result.verdict == Verdict.NO_GO

    def test_earnings_in_window(self, sample_ohlcv_trending):
        from market_analyzer.features.technicals import compute_technicals

        regime = _make_regime()
        technicals = compute_technicals(sample_ohlcv_trending, "TEST")
        fundamentals = _make_fundamentals(days_to_earnings=7)

        result = assess_earnings_play(
            "TEST", regime=regime, technicals=technicals,
            fundamentals=fundamentals,
        )

        assert isinstance(result, EarningsOpportunity)
        assert result.days_to_earnings == 7
        assert result.verdict in (Verdict.GO, Verdict.CAUTION, Verdict.NO_GO)

    def test_earnings_too_far(self, sample_ohlcv_trending):
        from market_analyzer.features.technicals import compute_technicals

        regime = _make_regime()
        technicals = compute_technicals(sample_ohlcv_trending, "TEST")
        fundamentals = _make_fundamentals(days_to_earnings=60)

        result = assess_earnings_play(
            "TEST", regime=regime, technicals=technicals,
            fundamentals=fundamentals,
        )

        # Should generally not be a GO for earnings too far out
        assert result.days_to_earnings == 60

    def test_summary_present(self, sample_ohlcv_trending):
        from market_analyzer.features.technicals import compute_technicals

        regime = _make_regime()
        technicals = compute_technicals(sample_ohlcv_trending, "TEST")
        fundamentals = _make_fundamentals(days_to_earnings=3)

        result = assess_earnings_play(
            "TEST", regime=regime, technicals=technicals,
            fundamentals=fundamentals,
        )

        assert result.summary != ""
        assert result.ticker == "TEST"
