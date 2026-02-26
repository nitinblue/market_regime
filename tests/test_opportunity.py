"""Tests for options opportunity assessment (0DTE + LEAP)."""

from datetime import date, datetime

import pytest

from market_analyzer.models.fundamentals import (
    BusinessInfo,
    CashMetrics,
    DebtMetrics,
    DividendMetrics,
    EarningsMetrics,
    FiftyTwoWeek,
    FundamentalsSnapshot,
    MarginMetrics,
    ReturnMetrics,
    RevenueMetrics,
    UpcomingEvents,
    ValuationMetrics,
)
from market_analyzer.models.macro import (
    MacroCalendar,
    MacroEvent,
    MacroEventImpact,
    MacroEventType,
)
from market_analyzer.models.opportunity import (
    FundamentalScore,
    LEAPOpportunity,
    LEAPStrategy,
    Verdict,
    ZeroDTEOpportunity,
    ZeroDTEStrategy,
)
from market_analyzer.models.phase import (
    PhaseEvidence,
    PhaseID,
    PhaseResult,
    PriceStructure,
)
from market_analyzer.models.regime import RegimeID, RegimeResult, TrendDirection
from market_analyzer.models.technicals import (
    BollingerBands,
    MACDData,
    MovingAverages,
    ORBData,
    ORBStatus,
    RSIData,
    StochasticData,
    SupportResistance,
    TechnicalSnapshot,
    MarketPhase,
    PhaseIndicator,
)
from market_analyzer.opportunity.option_plays.zero_dte import assess_zero_dte
from market_analyzer.opportunity.option_plays.leap import assess_leap


# --- Test helpers ---


def _make_regime(
    regime_id: int = 1,
    confidence: float = 0.75,
    trend: str | None = None,
) -> RegimeResult:
    trend_dir = TrendDirection(trend) if trend else None
    return RegimeResult(
        ticker="TEST",
        regime=RegimeID(regime_id),
        confidence=confidence,
        regime_probabilities={1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1},
        as_of_date=date(2026, 2, 22),
        model_version="test",
        trend_direction=trend_dir,
    )


def _make_technicals(
    atr_pct: float = 1.0,
    rsi: float = 50.0,
    bb_pct_b: float = 0.5,
    support: float | None = 490.0,
    resistance: float | None = 510.0,
    price: float = 500.0,
) -> TechnicalSnapshot:
    return TechnicalSnapshot(
        ticker="TEST",
        as_of_date=date(2026, 2, 22),
        current_price=price,
        atr=price * atr_pct / 100,
        atr_pct=atr_pct,
        vwma_20=price,
        moving_averages=MovingAverages(
            sma_20=price, sma_50=price * 0.98, sma_200=price * 0.95,
            ema_9=price, ema_21=price,
            price_vs_sma_20_pct=0.0, price_vs_sma_50_pct=2.0, price_vs_sma_200_pct=5.0,
        ),
        rsi=RSIData(value=rsi, is_overbought=rsi > 70, is_oversold=rsi < 30),
        bollinger=BollingerBands(
            upper=price + 10, middle=price, lower=price - 10,
            bandwidth=0.04, percent_b=bb_pct_b,
        ),
        macd=MACDData(
            macd_line=0.5, signal_line=0.3, histogram=0.2,
            is_bullish_crossover=False, is_bearish_crossover=False,
        ),
        stochastic=StochasticData(k=50.0, d=50.0, is_overbought=False, is_oversold=False),
        support_resistance=SupportResistance(
            support=support, resistance=resistance,
            price_vs_support_pct=2.0 if support else None,
            price_vs_resistance_pct=-2.0 if resistance else None,
        ),
        phase=PhaseIndicator(
            phase=MarketPhase.ACCUMULATION, confidence=0.5,
            description="Test", higher_highs=False, higher_lows=True,
            lower_highs=False, lower_lows=False, range_compression=0.3,
            volume_trend="declining", price_vs_sma_50_pct=2.0,
        ),
        signals=[],
    )


def _make_macro(
    high_event_today: bool = False,
    high_event_tomorrow: bool = False,
    events_30d: int = 0,
    today: date = date(2026, 2, 22),
) -> MacroCalendar:
    from datetime import timedelta

    events = []
    events_7d = []
    events_30d_list = []

    if high_event_today:
        ev = MacroEvent(
            event_type=MacroEventType.FOMC,
            date=today,
            name="FOMC Decision",
            impact=MacroEventImpact.HIGH,
            description="Fed rate decision",
            options_impact="IV spike",
        )
        events.append(ev)
        events_7d.append(ev)
        events_30d_list.append(ev)

    if high_event_tomorrow:
        ev = MacroEvent(
            event_type=MacroEventType.CPI,
            date=today + timedelta(days=1),
            name="CPI Release",
            impact=MacroEventImpact.HIGH,
            description="CPI data",
            options_impact="IV spike",
        )
        events.append(ev)
        events_7d.append(ev)
        events_30d_list.append(ev)

    # Pad events_30d_list to match count
    for i in range(events_30d - len(events_30d_list)):
        events_30d_list.append(MacroEvent(
            event_type=MacroEventType.GDP,
            date=today + timedelta(days=10 + i),
            name=f"GDP Report {i}",
            impact=MacroEventImpact.LOW,
            description="GDP",
            options_impact="Minimal",
        ))

    return MacroCalendar(
        events=events,
        next_event=events[0] if events else None,
        days_to_next=0 if events else None,
        next_fomc=None,
        days_to_next_fomc=None,
        events_next_7_days=events_7d,
        events_next_30_days=events_30d_list,
    )


def _make_fundamentals(
    days_to_earnings: int | None = 30,
    forward_pe: float | None = 20.0,
    earnings_growth: float | None = 0.10,
    revenue_growth: float | None = 0.10,
    profit_margins: float | None = 0.15,
    debt_to_equity: float | None = 80.0,
    pct_from_high: float | None = -10.0,
    pct_from_low: float | None = 30.0,
    high_52wk: float | None = 550.0,
    low_52wk: float | None = 400.0,
) -> FundamentalsSnapshot:
    return FundamentalsSnapshot(
        ticker="TEST",
        as_of=datetime(2026, 2, 22),
        business=BusinessInfo(long_name="Test Corp", sector="Tech", industry="Software", beta=1.1),
        valuation=ValuationMetrics(forward_pe=forward_pe),
        earnings=EarningsMetrics(earnings_growth=earnings_growth),
        revenue=RevenueMetrics(revenue_growth=revenue_growth),
        margins=MarginMetrics(profit_margins=profit_margins),
        cash=CashMetrics(),
        debt=DebtMetrics(debt_to_equity=debt_to_equity),
        returns=ReturnMetrics(),
        dividends=DividendMetrics(),
        fifty_two_week=FiftyTwoWeek(
            high=high_52wk, low=low_52wk,
            pct_from_high=pct_from_high, pct_from_low=pct_from_low,
        ),
        recent_earnings=[],
        upcoming_events=UpcomingEvents(days_to_earnings=days_to_earnings),
    )


def _make_orb(status: str = "within") -> ORBData:
    return ORBData(
        ticker="TEST",
        date=date(2026, 2, 22),
        opening_minutes=30,
        range_high=502.0,
        range_low=498.0,
        range_size=4.0,
        range_pct=0.80,
        current_price=501.0,
        status=ORBStatus(status),
        levels=[],
        session_high=505.0,
        session_low=496.0,
        session_vwap=500.5,
        opening_volume_ratio=1.2,
        range_vs_daily_atr_pct=None,
        breakout_bar_index=None,
        retest_count=0,
        signals=[],
        description="Test ORB",
    )


def _make_phase(
    phase_id: int = 1,
    confidence: float = 0.65,
    age_days: int = 15,
) -> PhaseResult:
    names = {1: "Accumulation", 2: "Markup", 3: "Distribution", 4: "Markdown"}
    return PhaseResult(
        ticker="TEST",
        phase=PhaseID(phase_id),
        phase_name=names[phase_id],
        confidence=confidence,
        phase_age_days=age_days,
        prior_phase=None,
        cycle_completion=0.3,
        price_structure=PriceStructure(
            swing_highs=[], swing_lows=[],
            higher_highs=False, higher_lows=True,
            lower_highs=False, lower_lows=False,
            range_compression=0.3, price_vs_sma=2.0,
            volume_trend="declining",
            support_level=490.0, resistance_level=510.0,
        ),
        evidence=PhaseEvidence(
            regime_signal="R1", price_signal="Higher lows",
            volume_signal="Declining", supporting=[], contradictions=[],
        ),
        transitions=[],
        strategy_comment="Test strategy",
        as_of_date=date(2026, 2, 22),
    )


# =============================================================================
# 0DTE TESTS
# =============================================================================


class TestZeroDTEHardStops:
    def test_earnings_today_is_no_go(self):
        result = assess_zero_dte(
            "TEST", _make_regime(), _make_technicals(), _make_macro(),
            fundamentals=_make_fundamentals(days_to_earnings=0),
        )
        assert result.verdict == Verdict.NO_GO
        assert any(s.name == "earnings_blackout" for s in result.hard_stops)

    def test_earnings_tomorrow_is_no_go(self):
        result = assess_zero_dte(
            "TEST", _make_regime(), _make_technicals(), _make_macro(),
            fundamentals=_make_fundamentals(days_to_earnings=1),
        )
        assert result.verdict == Verdict.NO_GO
        assert any(s.name == "earnings_blackout" for s in result.hard_stops)

    def test_macro_event_today_is_no_go(self):
        result = assess_zero_dte(
            "TEST", _make_regime(), _make_technicals(),
            _make_macro(high_event_today=True),
            as_of=date(2026, 2, 22),
        )
        assert result.verdict == Verdict.NO_GO
        assert any(s.name == "macro_event_today" for s in result.hard_stops)

    def test_atr_too_low_is_no_go(self):
        result = assess_zero_dte(
            "TEST", _make_regime(), _make_technicals(atr_pct=0.1),
            _make_macro(),
        )
        assert result.verdict == Verdict.NO_GO
        assert any(s.name == "atr_too_low" for s in result.hard_stops)

    def test_atr_too_high_is_no_go(self):
        result = assess_zero_dte(
            "TEST", _make_regime(), _make_technicals(atr_pct=5.0),
            _make_macro(),
        )
        assert result.verdict == Verdict.NO_GO
        assert any(s.name == "atr_too_high" for s in result.hard_stops)

    def test_r4_high_confidence_is_no_go(self):
        result = assess_zero_dte(
            "TEST", _make_regime(regime_id=4, confidence=0.85),
            _make_technicals(), _make_macro(),
        )
        assert result.verdict == Verdict.NO_GO
        assert any(s.name == "r4_high_confidence" for s in result.hard_stops)

    def test_no_hard_stops_can_be_go(self):
        result = assess_zero_dte(
            "TEST", _make_regime(regime_id=1, confidence=0.80),
            _make_technicals(atr_pct=1.0, rsi=50.0),
            _make_macro(),
            fundamentals=_make_fundamentals(days_to_earnings=30),
        )
        assert result.hard_stops == []
        assert result.verdict in (Verdict.GO, Verdict.CAUTION)

    def test_multiple_hard_stops_all_listed(self):
        result = assess_zero_dte(
            "TEST", _make_regime(regime_id=4, confidence=0.85),
            _make_technicals(atr_pct=5.0),
            _make_macro(high_event_today=True),
            fundamentals=_make_fundamentals(days_to_earnings=0),
            as_of=date(2026, 2, 22),
        )
        assert result.verdict == Verdict.NO_GO
        assert len(result.hard_stops) >= 3


class TestZeroDTEStrategy:
    def test_r1_within_is_iron_condor(self):
        result = assess_zero_dte(
            "TEST", _make_regime(regime_id=1), _make_technicals(),
            _make_macro(), orb=_make_orb("within"),
        )
        assert result.zero_dte_strategy == ZeroDTEStrategy.IRON_CONDOR

    def test_r1_breakout_long_is_credit_spread(self):
        result = assess_zero_dte(
            "TEST", _make_regime(regime_id=1), _make_technicals(),
            _make_macro(), orb=_make_orb("breakout_long"),
        )
        assert result.zero_dte_strategy == ZeroDTEStrategy.CREDIT_SPREAD
        assert result.strategy.direction == "bullish"

    def test_r1_breakout_short_is_credit_spread(self):
        result = assess_zero_dte(
            "TEST", _make_regime(regime_id=1), _make_technicals(),
            _make_macro(), orb=_make_orb("breakout_short"),
        )
        assert result.zero_dte_strategy == ZeroDTEStrategy.CREDIT_SPREAD
        assert result.strategy.direction == "bearish"

    def test_r2_within_is_strangle(self):
        result = assess_zero_dte(
            "TEST", _make_regime(regime_id=2), _make_technicals(),
            _make_macro(), orb=_make_orb("within"),
        )
        assert result.zero_dte_strategy == ZeroDTEStrategy.STRADDLE_STRANGLE

    def test_r3_breakout_is_directional(self):
        result = assess_zero_dte(
            "TEST", _make_regime(regime_id=3, trend="bullish"),
            _make_technicals(), _make_macro(),
            orb=_make_orb("breakout_long"),
        )
        assert result.zero_dte_strategy == ZeroDTEStrategy.DIRECTIONAL_SPREAD

    def test_r4_always_no_trade(self):
        result = assess_zero_dte(
            "TEST", _make_regime(regime_id=4, confidence=0.85),
            _make_technicals(), _make_macro(),
        )
        assert result.zero_dte_strategy == ZeroDTEStrategy.NO_TRADE

    def test_no_orb_falls_back_to_default(self):
        result = assess_zero_dte(
            "TEST", _make_regime(regime_id=1), _make_technicals(),
            _make_macro(), orb=None,
        )
        # R1 + no ORB = iron condor (default for R1 WITHIN)
        assert result.zero_dte_strategy == ZeroDTEStrategy.IRON_CONDOR


class TestZeroDTEVerdict:
    def test_high_confidence_r1_is_go(self):
        result = assess_zero_dte(
            "TEST", _make_regime(regime_id=1, confidence=0.90),
            _make_technicals(atr_pct=1.0, rsi=50.0),
            _make_macro(),
            fundamentals=_make_fundamentals(days_to_earnings=30),
        )
        assert result.verdict == Verdict.GO

    def test_hard_stop_overrides_high_confidence(self):
        result = assess_zero_dte(
            "TEST", _make_regime(regime_id=1, confidence=0.90),
            _make_technicals(atr_pct=1.0, rsi=50.0),
            _make_macro(high_event_today=True),
            as_of=date(2026, 2, 22),
        )
        assert result.verdict == Verdict.NO_GO


class TestZeroDTEConfidence:
    def test_r1_scores_higher_than_r3(self):
        r1 = assess_zero_dte(
            "TEST", _make_regime(regime_id=1), _make_technicals(),
            _make_macro(),
        )
        r3 = assess_zero_dte(
            "TEST", _make_regime(regime_id=3), _make_technicals(),
            _make_macro(),
        )
        assert r1.confidence > r3.confidence

    def test_summary_populated(self):
        result = assess_zero_dte(
            "TEST", _make_regime(), _make_technicals(), _make_macro(),
        )
        assert len(result.summary) > 0
        assert "TEST" in result.summary


# =============================================================================
# LEAP TESTS
# =============================================================================


class TestLEAPHardStops:
    def test_r4_high_confidence_is_no_go(self):
        result = assess_leap(
            "TEST", _make_regime(regime_id=4, confidence=0.85),
            _make_technicals(), _make_phase(), _make_macro(),
        )
        assert result.verdict == Verdict.NO_GO
        assert any(s.name == "iv_expensive" for s in result.hard_stops)

    def test_distribution_high_confidence_is_no_go(self):
        result = assess_leap(
            "TEST", _make_regime(), _make_technicals(),
            _make_phase(phase_id=3, confidence=0.75), _make_macro(),
        )
        assert result.verdict == Verdict.NO_GO
        assert any(s.name == "distribution_top" for s in result.hard_stops)

    def test_markdown_is_no_go(self):
        result = assess_leap(
            "TEST", _make_regime(), _make_technicals(),
            _make_phase(phase_id=4, confidence=0.75), _make_macro(),
        )
        assert result.verdict == Verdict.NO_GO
        assert any(s.name == "markdown_phase" for s in result.hard_stops)

    def test_earnings_within_5_days_is_no_go(self):
        result = assess_leap(
            "TEST", _make_regime(), _make_technicals(),
            _make_phase(), _make_macro(),
            fundamentals=_make_fundamentals(days_to_earnings=3),
        )
        assert result.verdict == Verdict.NO_GO
        assert any(s.name == "earnings_imminent" for s in result.hard_stops)

    def test_weak_fundamentals_is_no_go(self):
        result = assess_leap(
            "TEST", _make_regime(), _make_technicals(),
            _make_phase(), _make_macro(),
            fundamentals=_make_fundamentals(
                earnings_growth=-0.2, revenue_growth=-0.1,
                profit_margins=-0.05, debt_to_equity=300.0,
                forward_pe=60.0,
            ),
        )
        assert result.verdict == Verdict.NO_GO
        assert any(s.name == "weak_fundamentals" for s in result.hard_stops)

    def test_accumulation_r1_is_not_hard_stop(self):
        result = assess_leap(
            "TEST", _make_regime(regime_id=1), _make_technicals(),
            _make_phase(phase_id=1), _make_macro(),
            fundamentals=_make_fundamentals(),
        )
        assert result.hard_stops == []


class TestLEAPStrategy:
    def test_p1_r1_is_bull_call_leap(self):
        result = assess_leap(
            "TEST", _make_regime(regime_id=1), _make_technicals(),
            _make_phase(phase_id=1), _make_macro(),
            fundamentals=_make_fundamentals(),
        )
        assert result.leap_strategy == LEAPStrategy.BULL_CALL_LEAP

    def test_p1_r2_is_bull_call_spread(self):
        result = assess_leap(
            "TEST", _make_regime(regime_id=2), _make_technicals(),
            _make_phase(phase_id=1), _make_macro(),
            fundamentals=_make_fundamentals(),
        )
        assert result.leap_strategy == LEAPStrategy.BULL_CALL_SPREAD

    def test_p2_r1_is_pmcc(self):
        result = assess_leap(
            "TEST", _make_regime(regime_id=1), _make_technicals(),
            _make_phase(phase_id=2), _make_macro(),
            fundamentals=_make_fundamentals(),
        )
        assert result.leap_strategy == LEAPStrategy.PMCC

    def test_p3_r1_is_protective_put(self):
        # P3 with low confidence (below hard stop threshold) to avoid NO_GO
        result = assess_leap(
            "TEST", _make_regime(regime_id=1), _make_technicals(),
            _make_phase(phase_id=3, confidence=0.50), _make_macro(),
            fundamentals=_make_fundamentals(),
        )
        assert result.leap_strategy == LEAPStrategy.PROTECTIVE_PUT

    def test_p4_r2_is_bear_put_leap(self):
        # P4 with low confidence to avoid hard stop
        result = assess_leap(
            "TEST", _make_regime(regime_id=2), _make_technicals(),
            _make_phase(phase_id=4, confidence=0.50), _make_macro(),
            fundamentals=_make_fundamentals(),
        )
        assert result.leap_strategy == LEAPStrategy.BEAR_PUT_LEAP

    def test_p1_r4_is_no_trade(self):
        # R4 low confidence to avoid hard stop, but strategy should still be no_trade
        result = assess_leap(
            "TEST", _make_regime(regime_id=4, confidence=0.50),
            _make_technicals(),
            _make_phase(phase_id=1), _make_macro(),
            fundamentals=_make_fundamentals(),
        )
        assert result.leap_strategy == LEAPStrategy.NO_TRADE


class TestLEAPIVEnvironment:
    def test_r1_is_cheap(self):
        result = assess_leap(
            "TEST", _make_regime(regime_id=1), _make_technicals(),
            _make_phase(), _make_macro(),
        )
        assert result.iv_environment == "cheap"

    def test_r2_is_moderate(self):
        result = assess_leap(
            "TEST", _make_regime(regime_id=2), _make_technicals(),
            _make_phase(), _make_macro(),
        )
        assert result.iv_environment == "moderate"

    def test_r3_is_moderate(self):
        result = assess_leap(
            "TEST", _make_regime(regime_id=3), _make_technicals(),
            _make_phase(), _make_macro(),
        )
        assert result.iv_environment == "moderate"

    def test_r4_is_expensive(self):
        result = assess_leap(
            "TEST", _make_regime(regime_id=4, confidence=0.50),
            _make_technicals(),
            _make_phase(), _make_macro(),
        )
        assert result.iv_environment == "expensive"


class TestFundamentalScore:
    def test_strong_fundamentals_high_score(self):
        result = assess_leap(
            "TEST", _make_regime(), _make_technicals(),
            _make_phase(), _make_macro(),
            fundamentals=_make_fundamentals(
                earnings_growth=0.25, revenue_growth=0.20,
                profit_margins=0.30, debt_to_equity=30.0,
                forward_pe=12.0,
            ),
        )
        assert result.fundamental_score.score >= 0.8
        assert result.fundamental_score.earnings_growth_signal == "strong"

    def test_weak_fundamentals_low_score(self):
        result = assess_leap(
            "TEST", _make_regime(), _make_technicals(),
            _make_phase(), _make_macro(),
            fundamentals=_make_fundamentals(
                earnings_growth=-0.1, revenue_growth=-0.05,
                profit_margins=0.02, debt_to_equity=250.0,
                forward_pe=50.0,
            ),
        )
        assert result.fundamental_score.score <= 0.35
        assert result.fundamental_score.earnings_growth_signal == "negative"

    def test_missing_data_defaults_to_neutral(self):
        result = assess_leap(
            "TEST", _make_regime(), _make_technicals(),
            _make_phase(), _make_macro(),
            fundamentals=None,
        )
        assert result.fundamental_score.score == 0.5
        assert result.fundamental_score.earnings_growth_signal == "unknown"

    def test_etf_no_earnings_handled(self):
        """ETFs have None for most fundamental fields."""
        result = assess_leap(
            "TEST", _make_regime(), _make_technicals(),
            _make_phase(), _make_macro(),
            fundamentals=_make_fundamentals(
                earnings_growth=None, revenue_growth=None,
                profit_margins=None, debt_to_equity=None,
                forward_pe=None,
            ),
        )
        assert result.fundamental_score.score == 0.5  # All unknowns = neutral
        assert result.fundamental_score.description == "No fundamental data available."


class TestLEAPVerdict:
    def test_accumulation_cheap_iv_good_fundamentals_is_go(self):
        result = assess_leap(
            "TEST", _make_regime(regime_id=1, confidence=0.80),
            _make_technicals(rsi=45.0),
            _make_phase(phase_id=1, confidence=0.70),
            _make_macro(),
            fundamentals=_make_fundamentals(
                days_to_earnings=60,
                earnings_growth=0.20, revenue_growth=0.15,
                profit_margins=0.25, debt_to_equity=40.0,
                forward_pe=18.0,
            ),
        )
        assert result.verdict == Verdict.GO

    def test_distribution_expensive_iv_is_no_go(self):
        result = assess_leap(
            "TEST", _make_regime(regime_id=4, confidence=0.80),
            _make_technicals(),
            _make_phase(phase_id=3, confidence=0.70),
            _make_macro(),
        )
        assert result.verdict == Verdict.NO_GO


class TestLEAPSummary:
    def test_summary_populated(self):
        result = assess_leap(
            "TEST", _make_regime(), _make_technicals(),
            _make_phase(), _make_macro(),
        )
        assert len(result.summary) > 0
        assert "TEST" in result.summary
        assert "LEAP" in result.summary


class TestSerialization:
    def test_zero_dte_to_dict(self):
        result = assess_zero_dte(
            "TEST", _make_regime(), _make_technicals(), _make_macro(),
        )
        d = result.model_dump()
        assert d["ticker"] == "TEST"
        assert d["verdict"] in ("go", "caution", "no_go")

    def test_leap_to_dict(self):
        result = assess_leap(
            "TEST", _make_regime(), _make_technicals(),
            _make_phase(), _make_macro(),
        )
        d = result.model_dump()
        assert d["ticker"] == "TEST"
        assert "fundamental_score" in d


# =============================================================================
# IRON MAN + ORB INTEGRATION TESTS
# =============================================================================


def _make_narrow_orb() -> ORBData:
    """Narrow ORB range (<0.5%) — triggers Iron Man selection."""
    from market_analyzer.models.technicals import ORBLevel
    return ORBData(
        ticker="TEST",
        date=date(2026, 2, 22),
        opening_minutes=30,
        range_high=501.0,
        range_low=499.0,
        range_size=2.0,
        range_pct=0.40,  # <0.5% → narrow
        current_price=500.0,
        status=ORBStatus("within"),
        levels=[
            ORBLevel(label="Midpoint", price=500.0, distance_pct=0.0),
            ORBLevel(label="T1 Long (1.0x)", price=503.0, distance_pct=-0.6),
            ORBLevel(label="T1 Short (1.0x)", price=497.0, distance_pct=0.6),
            ORBLevel(label="T2 Long (1.5x)", price=504.0, distance_pct=-0.8),
            ORBLevel(label="T2 Short (1.5x)", price=496.0, distance_pct=0.8),
        ],
        session_high=501.5,
        session_low=498.5,
        session_vwap=500.2,
        opening_volume_ratio=1.4,
        range_vs_daily_atr_pct=25.0,
        breakout_bar_index=None,
        retest_count=0,
        signals=[],
        description="Narrow ORB",
    )


def _make_orb_with_levels(status: str = "within", range_pct: float = 0.80) -> ORBData:
    """ORB with extension levels for testing ORB integration."""
    from market_analyzer.models.technicals import ORBLevel
    return ORBData(
        ticker="TEST",
        date=date(2026, 2, 22),
        opening_minutes=30,
        range_high=502.0,
        range_low=498.0,
        range_size=4.0,
        range_pct=range_pct,
        current_price=501.0,
        status=ORBStatus(status),
        levels=[
            ORBLevel(label="Midpoint", price=500.0, distance_pct=0.2),
            ORBLevel(label="T1 Long (1.0x)", price=506.0, distance_pct=-1.0),
            ORBLevel(label="T1 Short (1.0x)", price=494.0, distance_pct=1.4),
            ORBLevel(label="T2 Long (1.5x)", price=508.0, distance_pct=-1.4),
            ORBLevel(label="T2 Short (1.5x)", price=492.0, distance_pct=1.8),
        ],
        session_high=505.0,
        session_low=496.0,
        session_vwap=500.5,
        opening_volume_ratio=1.3,
        range_vs_daily_atr_pct=50.0,
        breakout_bar_index=None,
        retest_count=0,
        signals=[],
        description="ORB with levels",
    )


class TestIronManStrategy:
    """Tests for Iron Man (inverse iron condor) strategy selection."""

    def test_r1_narrow_orb_selects_iron_man(self):
        result = assess_zero_dte(
            "TEST", _make_regime(1), _make_technicals(), _make_macro(),
            orb=_make_narrow_orb(),
        )
        assert result.zero_dte_strategy == ZeroDTEStrategy.IRON_MAN
        assert "Iron Man" in result.strategy.name

    def test_r2_narrow_orb_selects_iron_man(self):
        result = assess_zero_dte(
            "TEST", _make_regime(2), _make_technicals(), _make_macro(),
            orb=_make_narrow_orb(),
        )
        assert result.zero_dte_strategy == ZeroDTEStrategy.IRON_MAN

    def test_r3_narrow_orb_selects_iron_man(self):
        result = assess_zero_dte(
            "TEST", _make_regime(3, trend="bullish"), _make_technicals(), _make_macro(),
            orb=_make_narrow_orb(),
        )
        assert result.zero_dte_strategy == ZeroDTEStrategy.IRON_MAN

    def test_r1_normal_orb_selects_iron_condor(self):
        """Normal-width ORB in R1 → standard iron condor (not iron man)."""
        result = assess_zero_dte(
            "TEST", _make_regime(1), _make_technicals(), _make_macro(),
            orb=_make_orb_with_levels("within", range_pct=0.80),
        )
        assert result.zero_dte_strategy == ZeroDTEStrategy.IRON_CONDOR

    def test_iron_man_strategy_rec_has_orb_range(self):
        result = assess_zero_dte(
            "TEST", _make_regime(1), _make_technicals(), _make_macro(),
            orb=_make_narrow_orb(),
        )
        assert "499.00" in result.strategy.structure or "501.00" in result.strategy.structure
        assert result.strategy.direction == "neutral"

    def test_iron_man_risk_notes(self):
        result = assess_zero_dte(
            "TEST", _make_regime(1), _make_technicals(), _make_macro(),
            orb=_make_narrow_orb(),
        )
        assert any("debit" in note.lower() for note in result.strategy.risk_notes)


class TestORBDecision:
    """Tests for ORB decision context on ZeroDTEOpportunity."""

    def test_orb_decision_populated_when_orb_available(self):
        result = assess_zero_dte(
            "TEST", _make_regime(1), _make_technicals(), _make_macro(),
            orb=_make_orb_with_levels("within"),
        )
        assert result.orb_decision is not None
        assert result.orb_decision.status == "within"
        assert result.orb_decision.range_high == 502.0
        assert result.orb_decision.range_low == 498.0

    def test_orb_decision_none_without_orb(self):
        result = assess_zero_dte(
            "TEST", _make_regime(1), _make_technicals(), _make_macro(),
        )
        assert result.orb_decision is None

    def test_orb_decision_direction_bullish_on_breakout_long(self):
        result = assess_zero_dte(
            "TEST", _make_regime(1), _make_technicals(), _make_macro(),
            orb=_make_orb_with_levels("breakout_long"),
        )
        assert result.orb_decision is not None
        assert result.orb_decision.direction == "bullish"

    def test_orb_decision_direction_bearish_on_breakout_short(self):
        result = assess_zero_dte(
            "TEST", _make_regime(1), _make_technicals(), _make_macro(),
            orb=_make_orb_with_levels("breakout_short"),
        )
        assert result.orb_decision is not None
        assert result.orb_decision.direction == "bearish"

    def test_orb_decision_direction_bearish_on_failed_long(self):
        result = assess_zero_dte(
            "TEST", _make_regime(1), _make_technicals(), _make_macro(),
            orb=_make_orb_with_levels("failed_long"),
        )
        assert result.orb_decision is not None
        assert result.orb_decision.direction == "bearish"

    def test_orb_decision_key_levels_has_vwap(self):
        result = assess_zero_dte(
            "TEST", _make_regime(1), _make_technicals(), _make_macro(),
            orb=_make_orb_with_levels("within"),
        )
        assert "vwap" in result.orb_decision.key_levels
        assert result.orb_decision.key_levels["vwap"] == 500.5

    def test_orb_decision_key_levels_has_extensions(self):
        result = assess_zero_dte(
            "TEST", _make_regime(1), _make_technicals(), _make_macro(),
            orb=_make_orb_with_levels("within"),
        )
        levels = result.orb_decision.key_levels
        # Should have T1/T2 targets
        has_t1 = any("t1" in k for k in levels)
        assert has_t1

    def test_orb_decision_narrow_range_suggests_iron_man(self):
        result = assess_zero_dte(
            "TEST", _make_regime(1), _make_technicals(), _make_macro(),
            orb=_make_narrow_orb(),
        )
        assert "iron man" in result.orb_decision.decision.lower() or "inverse" in result.orb_decision.decision.lower()


class TestORBIntegrationAllStrategies:
    """Test ORB integration across all 0DTE strategies (user request: integrate ORB for any other ODTE)."""

    def test_iron_condor_uses_orb_range_for_strikes(self):
        """IC in R1 with ORB data → short strikes at ORB range edges."""
        result = assess_zero_dte(
            "TEST", _make_regime(1), _make_technicals(), _make_macro(),
            orb=_make_orb_with_levels("within"),
        )
        assert result.zero_dte_strategy == ZeroDTEStrategy.IRON_CONDOR
        assert result.strategy.structure is not None
        # Strategy description should reference ORB levels
        assert "ORB" in result.strategy.structure or "502" in result.strategy.structure or "498" in result.strategy.structure

    def test_credit_spread_references_orb(self):
        """Credit spread with ORB breakout → references ORB level in structure."""
        result = assess_zero_dte(
            "TEST", _make_regime(1), _make_technicals(), _make_macro(),
            orb=_make_orb_with_levels("breakout_long"),
        )
        assert result.zero_dte_strategy == ZeroDTEStrategy.CREDIT_SPREAD
        # Should reference ORB low as support
        assert "ORB" in result.strategy.structure or "498" in result.strategy.structure

    def test_directional_spread_references_orb_targets(self):
        """Directional spread in R3 breakout → mentions ORB extension targets."""
        result = assess_zero_dte(
            "TEST", _make_regime(3, trend="bullish"), _make_technicals(), _make_macro(),
            orb=_make_orb_with_levels("breakout_long"),
        )
        assert result.zero_dte_strategy == ZeroDTEStrategy.DIRECTIONAL_SPREAD
        # Should mention T1 target in structure
        assert "T1" in result.strategy.structure or "506" in result.strategy.structure

    def test_summary_includes_orb_info(self):
        """Summary should include ORB status and range when available."""
        result = assess_zero_dte(
            "TEST", _make_regime(1), _make_technicals(), _make_macro(),
            orb=_make_orb_with_levels("within"),
        )
        if result.verdict != Verdict.NO_GO:
            assert "ORB" in result.summary


class TestConfig:
    def test_opportunity_settings_load(self):
        from market_analyzer.config import get_settings, reset_settings

        reset_settings()
        settings = get_settings()
        assert hasattr(settings, "opportunity")
        assert settings.opportunity.zero_dte.earnings_blackout_days == 1
        assert settings.opportunity.leap.earnings_blackout_days == 5

    def test_zero_dte_thresholds_configurable(self):
        from market_analyzer.config import get_settings

        cfg = get_settings().opportunity.zero_dte
        assert cfg.min_atr_pct == 0.3
        assert cfg.max_atr_pct == 3.0
        assert cfg.go_threshold == 0.55

    def test_leap_thresholds_configurable(self):
        from market_analyzer.config import get_settings

        cfg = get_settings().opportunity.leap
        assert cfg.go_threshold == 0.50
        assert cfg.min_fundamental_score == 0.2
        assert cfg.pe_cheap == 15.0
