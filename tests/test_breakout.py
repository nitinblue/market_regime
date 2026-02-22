"""Tests for breakout opportunity assessment."""

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
    BreakoutOpportunity,
    BreakoutStrategy,
    BreakoutType,
    Verdict,
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
    MarketPhase,
    MovingAverages,
    OrderBlock,
    OrderBlockType,
    PhaseIndicator,
    RSIData,
    SmartMoneyData,
    StochasticData,
    SupportResistance,
    TechnicalSnapshot,
    VCPData,
    VCPStage,
)
from market_analyzer.opportunity.breakout import assess_breakout


# --- Test helpers ---


def _make_regime(
    regime_id: int = 1,
    confidence: float = 0.60,
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
    price: float = 500.0,
    atr_pct: float = 0.4,
    rsi: float = 50.0,
    bb_bandwidth: float = 0.03,
    bb_pct_b: float = 0.5,
    support: float | None = 490.0,
    resistance: float | None = 510.0,
    res_pct: float | None = -2.0,
    vcp: VCPData | None = None,
    smart_money: SmartMoneyData | None = None,
    phase_rc: float = 0.4,
    volume_trend: str = "declining",
    higher_highs: bool = False,
    higher_lows: bool = True,
    lower_highs: bool = False,
    lower_lows: bool = False,
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
            bandwidth=bb_bandwidth, percent_b=bb_pct_b,
        ),
        macd=MACDData(
            macd_line=0.5, signal_line=0.3, histogram=0.2,
            is_bullish_crossover=False, is_bearish_crossover=False,
        ),
        stochastic=StochasticData(k=50.0, d=50.0, is_overbought=False, is_oversold=False),
        support_resistance=SupportResistance(
            support=support, resistance=resistance,
            price_vs_support_pct=2.0 if support else None,
            price_vs_resistance_pct=res_pct if resistance else None,
        ),
        phase=PhaseIndicator(
            phase=MarketPhase.ACCUMULATION, confidence=0.5,
            description="Test", higher_highs=higher_highs, higher_lows=higher_lows,
            lower_highs=lower_highs, lower_lows=lower_lows,
            range_compression=phase_rc, volume_trend=volume_trend,
            price_vs_sma_50_pct=2.0,
        ),
        vcp=vcp,
        smart_money=smart_money,
        signals=[],
    )


def _make_vcp(
    stage: str = "ready",
    score: float = 0.85,
    days_in_base: int = 30,
    pivot_price: float | None = 515.0,
    pivot_distance_pct: float | None = 2.0,
) -> VCPData:
    return VCPData(
        stage=VCPStage(stage),
        contraction_count=3,
        contraction_pcts=[10.0, 6.0, 3.0],
        current_range_pct=3.0,
        range_compression=0.7,
        volume_trend="declining",
        pivot_price=pivot_price,
        pivot_distance_pct=pivot_distance_pct,
        days_in_base=days_in_base,
        above_sma_50=True,
        above_sma_200=True,
        score=score,
        description="Test VCP",
    )


def _make_smart_money(
    bullish_ob_dist: float = 2.0,
    tested: bool = False,
    broken: bool = False,
) -> SmartMoneyData:
    ob = OrderBlock(
        type=OrderBlockType.BULLISH,
        date=date(2026, 2, 20),
        high=495.0,
        low=490.0,
        volume=1000000.0,
        impulse_strength=2.5,
        is_tested=tested,
        is_broken=broken,
        distance_pct=bullish_ob_dist,
    )
    return SmartMoneyData(
        order_blocks=[ob],
        fair_value_gaps=[],
        nearest_bullish_ob=ob,
        nearest_bearish_ob=None,
        nearest_bullish_fvg=None,
        nearest_bearish_fvg=None,
        unfilled_fvg_count=0,
        active_ob_count=1,
        score=0.6,
        description="Test SM",
    )


def _make_macro() -> MacroCalendar:
    return MacroCalendar(
        events=[],
        next_event=None,
        days_to_next=None,
        next_fomc=None,
        days_to_next_fomc=None,
        events_next_7_days=[],
        events_next_30_days=[],
    )


def _make_fundamentals(
    days_to_earnings: int | None = 30,
) -> FundamentalsSnapshot:
    return FundamentalsSnapshot(
        ticker="TEST",
        as_of=datetime(2026, 2, 22),
        business=BusinessInfo(long_name="Test Corp", sector="Tech", industry="Software", beta=1.1),
        valuation=ValuationMetrics(forward_pe=20.0),
        earnings=EarningsMetrics(earnings_growth=0.10),
        revenue=RevenueMetrics(revenue_growth=0.10),
        margins=MarginMetrics(profit_margins=0.15),
        cash=CashMetrics(),
        debt=DebtMetrics(debt_to_equity=80.0),
        returns=ReturnMetrics(),
        dividends=DividendMetrics(),
        fifty_two_week=FiftyTwoWeek(
            high=550.0, low=400.0, pct_from_high=-10.0, pct_from_low=30.0,
        ),
        recent_earnings=[],
        upcoming_events=UpcomingEvents(days_to_earnings=days_to_earnings),
    )


def _make_phase(
    phase_id: int = 1,
    confidence: float = 0.65,
) -> PhaseResult:
    names = {1: "Accumulation", 2: "Markup", 3: "Distribution", 4: "Markdown"}
    return PhaseResult(
        ticker="TEST",
        phase=PhaseID(phase_id),
        phase_name=names[phase_id],
        confidence=confidence,
        phase_age_days=15,
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
# HARD STOPS
# =============================================================================


class TestBreakoutHardStops:
    def test_r4_high_confidence_is_no_go(self):
        result = assess_breakout(
            "TEST", _make_regime(regime_id=4, confidence=0.85),
            _make_technicals(), _make_phase(), _make_macro(),
        )
        assert result.verdict == Verdict.NO_GO
        assert any(s.name == "r4_high_confidence" for s in result.hard_stops)

    def test_earnings_imminent_is_no_go(self):
        result = assess_breakout(
            "TEST", _make_regime(), _make_technicals(),
            _make_phase(), _make_macro(),
            fundamentals=_make_fundamentals(days_to_earnings=1),
        )
        assert result.verdict == Verdict.NO_GO
        assert any(s.name == "earnings_imminent" for s in result.hard_stops)

    def test_no_base_established_is_no_go(self):
        # VCP none, low range compression, wide bandwidth
        tech = _make_technicals(
            bb_bandwidth=0.12,  # > 2× squeeze threshold (0.04)
            phase_rc=0.1,  # < 0.3
            vcp=_make_vcp(stage="none", score=0.0, days_in_base=3),
        )
        result = assess_breakout(
            "TEST", _make_regime(), tech, _make_phase(), _make_macro(),
        )
        assert result.verdict == Verdict.NO_GO
        assert any(s.name == "no_base_established" for s in result.hard_stops)

    def test_already_extended_is_no_go(self):
        vcp = _make_vcp(stage="breakout", score=0.7, pivot_distance_pct=7.0)
        tech = _make_technicals(bb_pct_b=1.3, vcp=vcp)
        result = assess_breakout(
            "TEST", _make_regime(), tech, _make_phase(), _make_macro(),
        )
        assert result.verdict == Verdict.NO_GO
        assert any(s.name == "already_extended" for s in result.hard_stops)

    def test_r2_very_high_confidence_is_no_go(self):
        result = assess_breakout(
            "TEST", _make_regime(regime_id=2, confidence=0.90),
            _make_technicals(), _make_phase(), _make_macro(),
        )
        assert result.verdict == Verdict.NO_GO
        assert any(s.name == "r2_very_high_confidence" for s in result.hard_stops)

    def test_no_hard_stops_with_good_setup(self):
        vcp = _make_vcp(stage="ready", score=0.85, days_in_base=30)
        tech = _make_technicals(vcp=vcp, bb_bandwidth=0.03, phase_rc=0.5)
        result = assess_breakout(
            "TEST", _make_regime(regime_id=3, confidence=0.60),
            tech, _make_phase(), _make_macro(),
            fundamentals=_make_fundamentals(days_to_earnings=30),
        )
        assert result.hard_stops == []


# =============================================================================
# STRATEGY
# =============================================================================


class TestBreakoutStrategy:
    def test_vcp_ready_high_score_is_pivot_breakout(self):
        vcp = _make_vcp(stage="ready", score=0.85)
        tech = _make_technicals(vcp=vcp, bb_bandwidth=0.03, phase_rc=0.5)
        result = assess_breakout(
            "TEST", _make_regime(regime_id=3, confidence=0.60),
            tech, _make_phase(), _make_macro(),
        )
        assert result.breakout_strategy == BreakoutStrategy.PIVOT_BREAKOUT

    def test_bb_squeeze_vcp_forming_is_squeeze_play(self):
        vcp = _make_vcp(stage="forming", score=0.4)
        tech = _make_technicals(vcp=vcp, bb_bandwidth=0.02, phase_rc=0.2)
        result = assess_breakout(
            "TEST", _make_regime(regime_id=1, confidence=0.60),
            tech, _make_phase(), _make_macro(),
        )
        assert result.breakout_strategy == BreakoutStrategy.SQUEEZE_PLAY

    def test_r3_p2_range_compression_is_bull_flag(self):
        vcp = _make_vcp(stage="none", score=0.1)
        tech = _make_technicals(vcp=vcp, bb_bandwidth=0.06, phase_rc=0.5)
        result = assess_breakout(
            "TEST", _make_regime(regime_id=3, confidence=0.60),
            tech, _make_phase(phase_id=2), _make_macro(),
        )
        assert result.breakout_strategy == BreakoutStrategy.BULL_FLAG

    def test_vcp_breakout_near_resistance_is_pullback(self):
        vcp = _make_vcp(stage="breakout", score=0.7, pivot_distance_pct=2.0)
        tech = _make_technicals(vcp=vcp, bb_bandwidth=0.06, bb_pct_b=0.8, res_pct=-1.5)
        result = assess_breakout(
            "TEST", _make_regime(regime_id=3, confidence=0.60),
            tech, _make_phase(), _make_macro(),
        )
        assert result.breakout_strategy == BreakoutStrategy.PULLBACK_TO_BREAKOUT

    def test_no_go_is_no_trade(self):
        result = assess_breakout(
            "TEST", _make_regime(regime_id=4, confidence=0.85),
            _make_technicals(), _make_phase(), _make_macro(),
        )
        assert result.breakout_strategy == BreakoutStrategy.NO_TRADE

    def test_default_r1_fallback_is_squeeze(self):
        # R1, no VCP, no squeeze — default for R1
        vcp = _make_vcp(stage="none", score=0.0, days_in_base=15)
        tech = _make_technicals(vcp=vcp, bb_bandwidth=0.06, phase_rc=0.5)
        result = assess_breakout(
            "TEST", _make_regime(regime_id=1, confidence=0.60),
            tech, _make_phase(), _make_macro(),
        )
        assert result.breakout_strategy == BreakoutStrategy.SQUEEZE_PLAY


# =============================================================================
# VERDICT
# =============================================================================


class TestBreakoutVerdict:
    def test_strong_setup_is_go(self):
        vcp = _make_vcp(stage="ready", score=0.85)
        sm = _make_smart_money(bullish_ob_dist=2.0)
        tech = _make_technicals(
            vcp=vcp, smart_money=sm,
            bb_bandwidth=0.03, phase_rc=0.5,
            atr_pct=0.3, res_pct=-1.5,
        )
        result = assess_breakout(
            "TEST", _make_regime(regime_id=3, confidence=0.60),
            tech, _make_phase(phase_id=1), _make_macro(),
        )
        assert result.verdict == Verdict.GO

    def test_hard_stop_overrides_to_no_go(self):
        vcp = _make_vcp(stage="ready", score=0.85)
        tech = _make_technicals(vcp=vcp, bb_bandwidth=0.03)
        result = assess_breakout(
            "TEST", _make_regime(regime_id=4, confidence=0.85),
            tech, _make_phase(), _make_macro(),
        )
        assert result.verdict == Verdict.NO_GO

    def test_marginal_setup_is_caution(self):
        # Some favorable signals but not enough for GO
        vcp = _make_vcp(stage="forming", score=0.3)
        tech = _make_technicals(
            vcp=vcp, bb_bandwidth=0.06, phase_rc=0.2,
            atr_pct=1.0, res_pct=-5.0,
        )
        result = assess_breakout(
            "TEST", _make_regime(regime_id=1, confidence=0.60),
            tech, _make_phase(phase_id=1), _make_macro(),
        )
        assert result.verdict in (Verdict.CAUTION, Verdict.NO_GO)


# =============================================================================
# SETUP
# =============================================================================


class TestBreakoutSetup:
    def test_setup_fields_populated(self):
        vcp = _make_vcp(stage="ready", score=0.85)
        tech = _make_technicals(vcp=vcp, bb_bandwidth=0.03)
        result = assess_breakout(
            "TEST", _make_regime(), tech, _make_phase(), _make_macro(),
        )
        assert result.setup.vcp_stage == "ready"
        assert result.setup.vcp_score == 0.85
        assert result.setup.bollinger_squeeze is True
        assert result.setup.bollinger_bandwidth == 0.03

    def test_direction_from_vcp_ready(self):
        vcp = _make_vcp(stage="ready", score=0.85)
        tech = _make_technicals(vcp=vcp)
        result = assess_breakout(
            "TEST", _make_regime(), tech, _make_phase(), _make_macro(),
        )
        assert result.breakout_type == BreakoutType.BULLISH


# =============================================================================
# SUMMARY
# =============================================================================


class TestBreakoutSummary:
    def test_summary_contains_ticker_and_verdict(self):
        result = assess_breakout(
            "TEST", _make_regime(), _make_technicals(),
            _make_phase(), _make_macro(),
        )
        assert "TEST" in result.summary
        assert "Breakout" in result.summary


# =============================================================================
# SERIALIZATION
# =============================================================================


class TestBreakoutSerialization:
    def test_model_dump_works(self):
        result = assess_breakout(
            "TEST", _make_regime(), _make_technicals(),
            _make_phase(), _make_macro(),
        )
        d = result.model_dump()
        assert d["ticker"] == "TEST"
        assert d["verdict"] in ("go", "caution", "no_go")
        assert "setup" in d
        assert "breakout_strategy" in d
