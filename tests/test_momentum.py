"""Tests for momentum opportunity assessment."""

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
    MomentumDirection,
    MomentumOpportunity,
    MomentumStrategy,
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
    PhaseIndicator,
    RSIData,
    StochasticData,
    SupportResistance,
    TechnicalSnapshot,
)
from market_analyzer.opportunity.setups.momentum import assess_momentum


# --- Test helpers ---


def _make_regime(
    regime_id: int = 3,
    confidence: float = 0.60,
    trend: str | None = "bullish",
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
    atr_pct: float = 1.0,
    rsi: float = 60.0,
    macd_hist: float = 1.5,
    macd_bullish_cross: bool = False,
    macd_bearish_cross: bool = False,
    sma_20: float = 498.0,
    sma_50: float = 490.0,
    sma_200: float = 470.0,
    stoch_k: float = 65.0,
    stoch_d: float = 55.0,
    higher_highs: bool = True,
    higher_lows: bool = True,
    lower_highs: bool = False,
    lower_lows: bool = False,
    volume_trend: str = "rising",
    phase_rc: float = 0.2,
) -> TechnicalSnapshot:
    return TechnicalSnapshot(
        ticker="TEST",
        as_of_date=date(2026, 2, 22),
        current_price=price,
        atr=price * atr_pct / 100,
        atr_pct=atr_pct,
        vwma_20=price,
        moving_averages=MovingAverages(
            sma_20=sma_20, sma_50=sma_50, sma_200=sma_200,
            ema_9=price, ema_21=price,
            price_vs_sma_20_pct=(price - sma_20) / sma_20 * 100,
            price_vs_sma_50_pct=(price - sma_50) / sma_50 * 100,
            price_vs_sma_200_pct=(price - sma_200) / sma_200 * 100,
        ),
        rsi=RSIData(value=rsi, is_overbought=rsi > 70, is_oversold=rsi < 30),
        bollinger=BollingerBands(
            upper=price + 10, middle=price, lower=price - 10,
            bandwidth=0.04, percent_b=0.5,
        ),
        macd=MACDData(
            macd_line=2.0, signal_line=1.0, histogram=macd_hist,
            is_bullish_crossover=macd_bullish_cross,
            is_bearish_crossover=macd_bearish_cross,
        ),
        stochastic=StochasticData(
            k=stoch_k, d=stoch_d,
            is_overbought=stoch_k > 80, is_oversold=stoch_k < 20,
        ),
        support_resistance=SupportResistance(
            support=490.0, resistance=510.0,
            price_vs_support_pct=2.0, price_vs_resistance_pct=-2.0,
        ),
        phase=PhaseIndicator(
            phase=MarketPhase.MARKUP, confidence=0.6,
            description="Test", higher_highs=higher_highs, higher_lows=higher_lows,
            lower_highs=lower_highs, lower_lows=lower_lows,
            range_compression=phase_rc, volume_trend=volume_trend,
            price_vs_sma_50_pct=2.0,
        ),
        signals=[],
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
    phase_id: int = 2,
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
            higher_highs=True, higher_lows=True,
            lower_highs=False, lower_lows=False,
            range_compression=0.2, price_vs_sma=2.0,
            volume_trend="rising",
            support_level=490.0, resistance_level=510.0,
        ),
        evidence=PhaseEvidence(
            regime_signal="R3", price_signal="Higher highs",
            volume_signal="Rising", supporting=[], contradictions=[],
        ),
        transitions=[],
        strategy_comment="Test strategy",
        as_of_date=date(2026, 2, 22),
    )


# =============================================================================
# HARD STOPS
# =============================================================================


class TestMomentumHardStops:
    def test_r1_high_confidence_is_no_go(self):
        result = assess_momentum(
            "TEST", _make_regime(regime_id=1, confidence=0.80),
            _make_technicals(), _make_phase(), _make_macro(),
        )
        assert result.verdict == Verdict.NO_GO
        assert any(s.name == "r1_high_confidence" for s in result.hard_stops)

    def test_earnings_imminent_is_no_go(self):
        result = assess_momentum(
            "TEST", _make_regime(), _make_technicals(),
            _make_phase(), _make_macro(),
            fundamentals=_make_fundamentals(days_to_earnings=2),
        )
        assert result.verdict == Verdict.NO_GO
        assert any(s.name == "earnings_imminent" for s in result.hard_stops)

    def test_rsi_extreme_overbought_is_no_go(self):
        result = assess_momentum(
            "TEST", _make_regime(), _make_technicals(rsi=90.0),
            _make_phase(), _make_macro(),
        )
        assert result.verdict == Verdict.NO_GO
        assert any(s.name == "rsi_extreme" for s in result.hard_stops)

    def test_macd_crossover_against_bullish_trend_is_no_go(self):
        result = assess_momentum(
            "TEST", _make_regime(regime_id=3, trend="bullish"),
            _make_technicals(macd_bearish_cross=True),
            _make_phase(), _make_macro(),
        )
        assert result.verdict == Verdict.NO_GO
        assert any(s.name == "macd_crossover_against_trend" for s in result.hard_stops)

    def test_volume_divergence_on_new_highs_is_no_go(self):
        # Higher highs + declining volume = distribution divergence
        result = assess_momentum(
            "TEST", _make_regime(regime_id=3, trend="bullish"),
            _make_technicals(
                higher_highs=True, higher_lows=True,
                volume_trend="declining",
            ),
            _make_phase(), _make_macro(),
        )
        assert result.verdict == Verdict.NO_GO
        assert any(s.name == "volume_divergence_on_new_highs" for s in result.hard_stops)

    def test_no_hard_stops_with_good_setup(self):
        result = assess_momentum(
            "TEST", _make_regime(regime_id=3, confidence=0.60, trend="bullish"),
            _make_technicals(rsi=60.0, volume_trend="rising"),
            _make_phase(phase_id=2), _make_macro(),
            fundamentals=_make_fundamentals(days_to_earnings=30),
        )
        assert result.hard_stops == []


# =============================================================================
# STRATEGY
# =============================================================================


class TestMomentumStrategy:
    def test_pullback_to_sma20_is_pullback_entry(self):
        # P2 + price near SMA20 + healthy RSI
        tech = _make_technicals(
            price=499.0, sma_20=498.0,  # ~0.2% from SMA20
            rsi=60.0, volume_trend="rising",
        )
        result = assess_momentum(
            "TEST", _make_regime(regime_id=3, trend="bullish"),
            tech, _make_phase(phase_id=2), _make_macro(),
        )
        assert result.momentum_strategy == MomentumStrategy.PULLBACK_ENTRY

    def test_r3_consolidation_is_acceleration(self):
        # R3 + range compression + MACD expanding
        tech = _make_technicals(
            macd_hist=2.0, phase_rc=0.5, volume_trend="rising",
            # Move price further from SMA20 to avoid pullback entry
            price=510.0, sma_20=498.0,
        )
        result = assess_momentum(
            "TEST", _make_regime(regime_id=3, trend="bullish"),
            tech, _make_phase(phase_id=2), _make_macro(),
        )
        assert result.momentum_strategy == MomentumStrategy.MOMENTUM_ACCELERATION

    def test_rsi_near_extreme_declining_volume_is_fade(self):
        tech = _make_technicals(
            rsi=78.0, volume_trend="declining",
            higher_highs=False,  # avoid volume_divergence hard stop
        )
        result = assess_momentum(
            "TEST", _make_regime(regime_id=3, trend="bullish"),
            tech, _make_phase(), _make_macro(),
        )
        assert result.momentum_strategy == MomentumStrategy.MOMENTUM_FADE

    def test_default_momentum_is_trend_continuation(self):
        # Price far from SMA20, no special conditions
        tech = _make_technicals(
            price=515.0, sma_20=498.0,
            rsi=60.0, volume_trend="rising",
            phase_rc=0.1,
        )
        result = assess_momentum(
            "TEST", _make_regime(regime_id=3, trend="bullish"),
            tech, _make_phase(phase_id=1), _make_macro(),
        )
        assert result.momentum_strategy == MomentumStrategy.TREND_CONTINUATION

    def test_no_go_is_no_trade(self):
        result = assess_momentum(
            "TEST", _make_regime(regime_id=1, confidence=0.80),
            _make_technicals(), _make_phase(), _make_macro(),
        )
        assert result.momentum_strategy == MomentumStrategy.NO_TRADE


# =============================================================================
# VERDICT
# =============================================================================


class TestMomentumVerdict:
    def test_r3_p2_strong_macd_is_go(self):
        tech = _make_technicals(
            rsi=60.0, macd_hist=2.0, volume_trend="rising",
            stoch_k=65.0, stoch_d=55.0,
            # Price far from SMA20 to avoid pullback, low RC to avoid acceleration
            price=515.0, sma_20=498.0, phase_rc=0.1,
        )
        result = assess_momentum(
            "TEST", _make_regime(regime_id=3, confidence=0.60, trend="bullish"),
            tech, _make_phase(phase_id=2), _make_macro(),
        )
        assert result.verdict == Verdict.GO

    def test_r1_is_no_go(self):
        result = assess_momentum(
            "TEST", _make_regime(regime_id=1, confidence=0.80),
            _make_technicals(), _make_phase(), _make_macro(),
        )
        assert result.verdict == Verdict.NO_GO

    def test_mixed_signals_is_caution_or_no_go(self):
        # R2 (moderate multiplier), some signals favorable
        tech = _make_technicals(
            rsi=55.0, macd_hist=0.5, volume_trend="stable",
            stoch_k=52.0, stoch_d=50.0,
            higher_highs=False, higher_lows=False,
        )
        result = assess_momentum(
            "TEST", _make_regime(regime_id=2, confidence=0.60, trend="bullish"),
            tech, _make_phase(phase_id=1), _make_macro(),
        )
        assert result.verdict in (Verdict.CAUTION, Verdict.NO_GO)


# =============================================================================
# SCORE
# =============================================================================


class TestMomentumScore:
    def test_score_fields_populated(self):
        result = assess_momentum(
            "TEST", _make_regime(), _make_technicals(),
            _make_phase(), _make_macro(),
        )
        assert result.score.macd_histogram_trend in ("expanding", "flat", "contracting")
        assert result.score.rsi_zone in ("oversold", "healthy_bull", "neutral", "overbought", "healthy_bear")
        assert result.score.structural_pattern in ("HH_HL", "LH_LL", "mixed")

    def test_direction_determined(self):
        result = assess_momentum(
            "TEST", _make_regime(regime_id=3, trend="bullish"),
            _make_technicals(), _make_phase(), _make_macro(),
        )
        assert result.momentum_direction == MomentumDirection.BULLISH


# =============================================================================
# SUMMARY
# =============================================================================


class TestMomentumSummary:
    def test_summary_populated(self):
        result = assess_momentum(
            "TEST", _make_regime(), _make_technicals(),
            _make_phase(), _make_macro(),
        )
        assert "TEST" in result.summary
        assert "Momentum" in result.summary


# =============================================================================
# SERIALIZATION
# =============================================================================


class TestMomentumSerialization:
    def test_model_dump_works(self):
        result = assess_momentum(
            "TEST", _make_regime(), _make_technicals(),
            _make_phase(), _make_macro(),
        )
        d = result.model_dump()
        assert d["ticker"] == "TEST"
        assert d["verdict"] in ("go", "caution", "no_go")
        assert "score" in d
        assert "momentum_strategy" in d
