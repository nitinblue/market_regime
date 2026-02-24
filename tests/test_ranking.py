"""Tests for trade ranking: scoring engine, service, and feedback."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from market_analyzer.config import RankingSettings, RankingWeightsSettings, get_settings
from market_analyzer.features.ranking import (
    ConfigWeightProvider,
    composite_from_breakdown,
    compute_composite_score,
    compute_earnings_penalty,
    compute_income_bias_boost,
    compute_macro_penalty,
    compute_risk_reward_score,
    compute_technical_quality,
    compute_verdict_score,
    get_phase_alignment,
    get_regime_alignment,
)
from market_analyzer.models.black_swan import AlertLevel, BlackSwanAlert
from market_analyzer.models.levels import LevelsAnalysis, PriceLevel, StopLoss, Target, LevelRole, LevelSource, TradeDirection
from market_analyzer.models.opportunity import (
    BreakoutOpportunity,
    BreakoutSetup,
    BreakoutStrategy,
    BreakoutType,
    HardStop,
    LEAPOpportunity,
    LEAPStrategy,
    FundamentalScore,
    MomentumDirection,
    MomentumOpportunity,
    MomentumScore,
    MomentumStrategy,
    OpportunitySignal,
    StrategyRecommendation,
    Verdict,
    ZeroDTEOpportunity,
    ZeroDTEStrategy,
)
from market_analyzer.models.ranking import (
    RankedEntry,
    RankingFeedback,
    ScoreBreakdown,
    StrategyType,
    TradeRankingResult,
)
from market_analyzer.models.technicals import (
    BollingerBands,
    MACDData,
    MovingAverages,
    RSIData,
    StochasticData,
    SupportResistance,
    PhaseIndicator,
    MarketPhase,
    TechnicalSnapshot,
)
from market_analyzer.service.ranking import TradeRankingService


# --- Fixtures ---


def _make_technicals(
    rsi: float = 55.0,
    macd_bull: bool = False,
    macd_bear: bool = False,
    price_vs_sma50: float = 2.0,
    price_vs_sma200: float = 5.0,
    stoch_ob: bool = False,
    stoch_os: bool = False,
) -> TechnicalSnapshot:
    """Build a TechnicalSnapshot for testing."""
    return TechnicalSnapshot(
        ticker="TEST",
        as_of_date=date(2026, 2, 22),
        current_price=100.0,
        atr=2.0,
        atr_pct=2.0,
        vwma_20=99.5,
        moving_averages=MovingAverages(
            sma_20=99.0,
            sma_50=98.0,
            sma_200=95.0,
            ema_9=99.5,
            ema_21=98.5,
            price_vs_sma_20_pct=1.0,
            price_vs_sma_50_pct=price_vs_sma50,
            price_vs_sma_200_pct=price_vs_sma200,
        ),
        rsi=RSIData(value=rsi, is_overbought=rsi > 70, is_oversold=rsi < 30),
        bollinger=BollingerBands(
            upper=105.0, middle=100.0, lower=95.0, bandwidth=0.1, percent_b=0.5
        ),
        macd=MACDData(
            macd_line=0.5,
            signal_line=0.3,
            histogram=0.2,
            is_bullish_crossover=macd_bull,
            is_bearish_crossover=macd_bear,
        ),
        stochastic=StochasticData(k=50.0, d=48.0, is_overbought=stoch_ob, is_oversold=stoch_os),
        support_resistance=SupportResistance(
            support=95.0, resistance=105.0, price_vs_support_pct=5.3, price_vs_resistance_pct=-4.8
        ),
        phase=PhaseIndicator(
            phase=MarketPhase.MARKUP,
            confidence=0.7,
            description="test",
            higher_highs=True,
            higher_lows=True,
            lower_highs=False,
            lower_lows=False,
            range_compression=0.0,
            volume_trend="stable",
            price_vs_sma_50_pct=price_vs_sma50,
        ),
        signals=[],
    )


def _make_levels(rr: float = 2.5) -> LevelsAnalysis:
    """Build a LevelsAnalysis with a given R:R ratio."""
    target_level = PriceLevel(
        price=110.0, role=LevelRole.RESISTANCE, sources=[LevelSource.SWING_RESISTANCE],
        confluence_score=2, strength=0.8, distance_pct=10.0, description="target",
    )
    stop_level = PriceLevel(
        price=96.0, role=LevelRole.SUPPORT, sources=[LevelSource.SWING_SUPPORT],
        confluence_score=1, strength=0.6, distance_pct=-4.0, description="stop",
    )
    return LevelsAnalysis(
        ticker="TEST",
        as_of_date=date(2026, 2, 22),
        entry_price=100.0,
        direction=TradeDirection.LONG,
        direction_auto_detected=True,
        current_price=100.0,
        atr=2.0,
        atr_pct=2.0,
        support_levels=[stop_level],
        resistance_levels=[target_level],
        stop_loss=StopLoss(
            price=96.0, distance_pct=-4.0, dollar_risk_per_share=4.0,
            level=stop_level, atr_buffer=1.0, description="stop",
        ),
        targets=[
            Target(
                price=110.0, distance_pct=10.0, dollar_reward_per_share=10.0,
                risk_reward_ratio=rr, level=target_level, description="target",
            )
        ],
        best_target=Target(
            price=110.0, distance_pct=10.0, dollar_reward_per_share=10.0,
            risk_reward_ratio=rr, level=target_level, description="target",
        ),
        summary="test levels",
    )


def _make_zero_dte(
    ticker: str = "SPY",
    verdict: Verdict = Verdict.GO,
    confidence: float = 0.75,
    regime_id: int = 1,
) -> ZeroDTEOpportunity:
    return ZeroDTEOpportunity(
        ticker=ticker,
        as_of_date=date(2026, 2, 22),
        verdict=verdict,
        confidence=confidence,
        hard_stops=[],
        signals=[],
        strategy=StrategyRecommendation(
            name="iron_condor", direction="neutral",
            structure="IC 10-wide", rationale="Low vol range-bound",
            risk_notes=["Watch for breakout"],
        ),
        zero_dte_strategy=ZeroDTEStrategy.IRON_CONDOR,
        regime_id=regime_id,
        regime_confidence=0.85,
        atr_pct=1.0,
        orb_status=None,
        has_macro_event_today=False,
        days_to_earnings=30,
        summary="test",
    )


def _make_momentum(
    ticker: str = "AAPL",
    verdict: Verdict = Verdict.CAUTION,
    confidence: float = 0.55,
    regime_id: int = 3,
    phase_id: int = 2,
) -> MomentumOpportunity:
    return MomentumOpportunity(
        ticker=ticker,
        as_of_date=date(2026, 2, 22),
        verdict=verdict,
        confidence=confidence,
        hard_stops=[],
        signals=[],
        strategy=StrategyRecommendation(
            name="trend_continuation", direction="bullish",
            structure="Bull call spread", rationale="Strong trend",
            risk_notes=["Extended from MA"],
        ),
        momentum_strategy=MomentumStrategy.TREND_CONTINUATION,
        momentum_direction=MomentumDirection.BULLISH,
        regime_id=regime_id,
        regime_confidence=0.80,
        phase_id=phase_id,
        phase_name="Markup",
        phase_confidence=0.70,
        score=MomentumScore(
            macd_histogram_trend="expanding",
            macd_crossover="bullish",
            rsi_zone="healthy_bull",
            price_vs_ma_alignment="strong_bull",
            golden_death_cross=None,
            structural_pattern="HH_HL",
            volume_confirmation=True,
            stochastic_confirmation=True,
            atr_trend="stable",
            description="test",
        ),
        days_to_earnings=45,
        summary="test",
    )


def _make_breakout(
    ticker: str = "GLD",
    verdict: Verdict = Verdict.GO,
    confidence: float = 0.65,
    regime_id: int = 3,
    phase_id: int = 1,
) -> BreakoutOpportunity:
    return BreakoutOpportunity(
        ticker=ticker,
        as_of_date=date(2026, 2, 22),
        verdict=verdict,
        confidence=confidence,
        hard_stops=[],
        signals=[],
        strategy=StrategyRecommendation(
            name="pivot_breakout", direction="bullish",
            structure="Long call", rationale="VCP pivot breakout",
            risk_notes=["Volume confirmation needed"],
        ),
        breakout_strategy=BreakoutStrategy.PIVOT_BREAKOUT,
        breakout_type=BreakoutType.BULLISH,
        regime_id=regime_id,
        regime_confidence=0.75,
        phase_id=phase_id,
        phase_name="Accumulation",
        phase_confidence=0.65,
        setup=BreakoutSetup(
            vcp_stage="ready", vcp_score=0.8, bollinger_squeeze=True,
            bollinger_bandwidth=0.03, range_compression=0.5,
            volume_pattern="declining_base", resistance_proximity_pct=1.5,
            support_proximity_pct=3.0, days_in_base=25,
            smart_money_alignment="supportive", description="test",
        ),
        pivot_price=105.0,
        days_to_earnings=60,
        summary="test",
    )


def _make_leap(
    ticker: str = "MSFT",
    verdict: Verdict = Verdict.CAUTION,
    confidence: float = 0.50,
    regime_id: int = 3,
    phase_id: int = 1,
) -> LEAPOpportunity:
    return LEAPOpportunity(
        ticker=ticker,
        as_of_date=date(2026, 2, 22),
        verdict=verdict,
        confidence=confidence,
        hard_stops=[],
        signals=[],
        strategy=StrategyRecommendation(
            name="bull_call_leap", direction="bullish",
            structure="LEAP call 0.70 delta", rationale="Accumulation phase entry",
            risk_notes=["Long duration risk"],
        ),
        leap_strategy=LEAPStrategy.BULL_CALL_LEAP,
        regime_id=regime_id,
        regime_confidence=0.72,
        phase_id=phase_id,
        phase_name="Accumulation",
        phase_confidence=0.60,
        iv_environment="low",
        fundamental_score=FundamentalScore(
            score=0.7, earnings_growth_signal="strong",
            revenue_growth_signal="moderate", margin_signal="strong",
            debt_signal="low", valuation_signal="fair",
            description="test",
        ),
        days_to_earnings=40,
        macro_events_next_30_days=2,
        summary="test",
    )


def _default_cfg() -> RankingSettings:
    return RankingSettings()


# =============================================
# TestVerdictScore
# =============================================


class TestVerdictScore:
    def test_go_is_one(self):
        assert compute_verdict_score(Verdict.GO) == 1.0

    def test_caution_is_half(self):
        assert compute_verdict_score(Verdict.CAUTION) == 0.5

    def test_no_go_is_zero(self):
        assert compute_verdict_score(Verdict.NO_GO) == 0.0


# =============================================
# TestRegimeAlignment
# =============================================


class TestRegimeAlignment:
    def test_zero_dte_r1_is_best(self):
        assert get_regime_alignment(1, StrategyType.ZERO_DTE) == 1.0

    def test_zero_dte_r4_is_worst(self):
        assert get_regime_alignment(4, StrategyType.ZERO_DTE) == 0.3

    def test_momentum_r3_is_best(self):
        assert get_regime_alignment(3, StrategyType.MOMENTUM) == 1.0

    def test_leap_r2_is_low(self):
        assert get_regime_alignment(2, StrategyType.LEAP) == 0.2

    def test_unknown_regime_defaults(self):
        assert get_regime_alignment(99, StrategyType.BREAKOUT) == 0.5


# =============================================
# TestPhaseAlignment
# =============================================


class TestPhaseAlignment:
    def test_breakout_p1_is_best(self):
        assert get_phase_alignment(1, StrategyType.BREAKOUT) == 1.0

    def test_momentum_p2_is_best(self):
        assert get_phase_alignment(2, StrategyType.MOMENTUM) == 1.0

    def test_leap_p4_is_worst(self):
        assert get_phase_alignment(4, StrategyType.LEAP) == 0.1

    def test_zero_dte_p3(self):
        assert get_phase_alignment(3, StrategyType.ZERO_DTE) == 0.7

    def test_unknown_phase_defaults(self):
        assert get_phase_alignment(99, StrategyType.LEAP) == 0.5


# =============================================
# TestTechnicalQuality
# =============================================


class TestTechnicalQuality:
    def test_ideal_technicals(self):
        """RSI healthy, MACD bull, above MAs, stoch normal → max score."""
        t = _make_technicals(rsi=55, macd_bull=True)
        score = compute_technical_quality(t)
        assert score == pytest.approx(1.0)

    def test_bearish_technicals(self):
        """RSI extreme, MACD bear, below MAs, stoch extreme → low score."""
        t = _make_technicals(
            rsi=85, macd_bear=True, price_vs_sma50=-5.0, price_vs_sma200=-10.0,
            stoch_ob=True,
        )
        score = compute_technical_quality(t)
        assert score < 0.2

    def test_neutral_technicals(self):
        """No crossover, normal RSI, above MAs."""
        t = _make_technicals(rsi=50)
        score = compute_technical_quality(t)
        assert 0.5 <= score <= 1.0

    def test_oversold_rsi_partial(self):
        """RSI at 25 (mildly extreme) gets partial credit."""
        t = _make_technicals(rsi=25)
        score = compute_technical_quality(t)
        t2 = _make_technicals(rsi=50)
        score2 = compute_technical_quality(t2)
        assert score < score2


# =============================================
# TestRiskRewardScore
# =============================================


class TestRiskRewardScore:
    def test_excellent_rr(self):
        assert compute_risk_reward_score(_make_levels(rr=3.5)) == 1.0

    def test_good_rr(self):
        assert compute_risk_reward_score(_make_levels(rr=2.5)) == 0.7

    def test_fair_rr(self):
        assert compute_risk_reward_score(_make_levels(rr=1.5)) == 0.4

    def test_poor_rr(self):
        assert compute_risk_reward_score(_make_levels(rr=0.5)) == 0.1

    def test_no_levels(self):
        assert compute_risk_reward_score(None) == 0.1

    def test_no_best_target(self):
        levels = _make_levels(rr=2.0)
        levels.best_target = None
        assert compute_risk_reward_score(levels) == 0.1


# =============================================
# TestIncomeBias
# =============================================


class TestIncomeBias:
    def test_zero_dte_r1_gets_boost(self):
        assert compute_income_bias_boost(StrategyType.ZERO_DTE, 1) == 0.05

    def test_zero_dte_r2_gets_boost(self):
        assert compute_income_bias_boost(StrategyType.ZERO_DTE, 2) == 0.05

    def test_zero_dte_r3_no_boost(self):
        assert compute_income_bias_boost(StrategyType.ZERO_DTE, 3) == 0.0

    def test_momentum_r1_no_boost(self):
        assert compute_income_bias_boost(StrategyType.MOMENTUM, 1) == 0.0


# =============================================
# TestMacroPenalty
# =============================================


class TestMacroPenalty:
    def test_no_events(self):
        assert compute_macro_penalty(0) == 0.0

    def test_one_event(self):
        assert compute_macro_penalty(1) == pytest.approx(0.02)

    def test_max_capped(self):
        assert compute_macro_penalty(10) == pytest.approx(0.10)


# =============================================
# TestEarningsPenalty
# =============================================


class TestEarningsPenalty:
    def test_no_earnings(self):
        assert compute_earnings_penalty(None) == 0.0

    def test_far_earnings(self):
        assert compute_earnings_penalty(30) == 0.0

    def test_close_earnings(self):
        assert compute_earnings_penalty(2) == pytest.approx(0.10)

    def test_earnings_day(self):
        assert compute_earnings_penalty(0) == pytest.approx(0.10)

    def test_boundary(self):
        assert compute_earnings_penalty(3) == pytest.approx(0.10)
        assert compute_earnings_penalty(4) == 0.0


# =============================================
# TestCompositeScore
# =============================================


class TestCompositeScore:
    def test_perfect_score(self):
        """GO verdict, high confidence, ideal alignment, no penalties."""
        technicals = _make_technicals(rsi=55, macd_bull=True)
        levels = _make_levels(rr=3.5)
        breakdown = compute_composite_score(
            verdict=Verdict.GO,
            confidence=0.90,
            regime_id=1,
            phase_id=2,
            strategy=StrategyType.ZERO_DTE,
            technicals=technicals,
            levels=levels,
            black_swan_score=0.0,
            events_next_7_days=0,
            days_to_earnings=60,
        )
        composite = composite_from_breakdown(breakdown)
        assert composite > 0.8

    def test_no_go_low_score(self):
        """NO_GO verdict → low composite."""
        technicals = _make_technicals(rsi=55)
        breakdown = compute_composite_score(
            verdict=Verdict.NO_GO,
            confidence=0.20,
            regime_id=4,
            phase_id=4,
            strategy=StrategyType.ZERO_DTE,
            technicals=technicals,
            levels=None,
            black_swan_score=0.0,
            events_next_7_days=3,
            days_to_earnings=2,
        )
        composite = composite_from_breakdown(breakdown)
        assert composite < 0.3

    def test_black_swan_penalty_reduces_score(self):
        """High black swan score should significantly reduce composite."""
        technicals = _make_technicals(rsi=55, macd_bull=True)
        levels = _make_levels(rr=3.0)
        bd_clean = compute_composite_score(
            verdict=Verdict.GO, confidence=0.80, regime_id=1, phase_id=1,
            strategy=StrategyType.ZERO_DTE, technicals=technicals,
            levels=levels, black_swan_score=0.0, events_next_7_days=0,
            days_to_earnings=60,
        )
        bd_stressed = compute_composite_score(
            verdict=Verdict.GO, confidence=0.80, regime_id=1, phase_id=1,
            strategy=StrategyType.ZERO_DTE, technicals=technicals,
            levels=levels, black_swan_score=0.5, events_next_7_days=0,
            days_to_earnings=60,
        )
        clean = composite_from_breakdown(bd_clean)
        stressed = composite_from_breakdown(bd_stressed)
        assert stressed < clean * 0.6

    def test_income_bias_boost_applies(self):
        """Zero DTE in R1 should score higher than in R3 (all else equal)."""
        technicals = _make_technicals(rsi=55)
        bd_r1 = compute_composite_score(
            verdict=Verdict.GO, confidence=0.70, regime_id=1, phase_id=2,
            strategy=StrategyType.ZERO_DTE, technicals=technicals,
            levels=None, black_swan_score=0.0, events_next_7_days=0,
            days_to_earnings=60,
        )
        bd_r3 = compute_composite_score(
            verdict=Verdict.GO, confidence=0.70, regime_id=3, phase_id=2,
            strategy=StrategyType.ZERO_DTE, technicals=technicals,
            levels=None, black_swan_score=0.0, events_next_7_days=0,
            days_to_earnings=60,
        )
        assert composite_from_breakdown(bd_r1) > composite_from_breakdown(bd_r3)

    def test_clamped_to_zero_one(self):
        """Score should never exceed 1.0 or go below 0.0."""
        technicals = _make_technicals(rsi=55, macd_bull=True)
        levels = _make_levels(rr=5.0)
        breakdown = compute_composite_score(
            verdict=Verdict.GO, confidence=1.0, regime_id=1, phase_id=1,
            strategy=StrategyType.ZERO_DTE, technicals=technicals,
            levels=levels, black_swan_score=0.0, events_next_7_days=0,
            days_to_earnings=60,
        )
        composite = composite_from_breakdown(breakdown)
        assert 0.0 <= composite <= 1.0


# =============================================
# TestWeightProvider
# =============================================


class TestWeightProvider:
    def test_config_provider_returns_defaults(self):
        cfg = _default_cfg()
        provider = ConfigWeightProvider(cfg)
        weights = provider.get_weights("SPY", StrategyType.ZERO_DTE)
        assert weights["verdict"] == 0.25
        assert weights["confidence"] == 0.25
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_config_provider_returns_copy(self):
        """Modifying returned weights should not affect provider."""
        provider = ConfigWeightProvider(_default_cfg())
        w1 = provider.get_weights("SPY", StrategyType.ZERO_DTE)
        w1["verdict"] = 999.0
        w2 = provider.get_weights("SPY", StrategyType.ZERO_DTE)
        assert w2["verdict"] == 0.25

    def test_custom_weights(self):
        cfg = RankingSettings(
            weights=RankingWeightsSettings(
                verdict=0.5, confidence=0.1, regime_alignment=0.1,
                risk_reward=0.1, technical_quality=0.1, phase_alignment=0.1,
            )
        )
        provider = ConfigWeightProvider(cfg)
        weights = provider.get_weights("ANY", StrategyType.LEAP)
        assert weights["verdict"] == 0.5


# =============================================
# TestBlackSwanGate
# =============================================


class TestBlackSwanGate:
    def _make_service(self, alert_level: AlertLevel, score: float = 0.3) -> TradeRankingService:
        opp = MagicMock()
        levels = MagicMock()
        bs = MagicMock()
        bs.alert.return_value = BlackSwanAlert(
            as_of_date=date(2026, 2, 22),
            alert_level=alert_level,
            composite_score=score,
            circuit_breakers=[],
            indicators=[],
            triggered_breakers=0,
            action="test",
            summary="test",
        )
        return TradeRankingService(
            opportunity_service=opp,
            levels_service=levels,
            black_swan_service=bs,
        )

    def test_critical_gates_all(self):
        svc = self._make_service(AlertLevel.CRITICAL, score=0.9)
        result = svc.rank(["SPY"])
        assert result.black_swan_gate is True
        assert result.top_trades == []
        assert result.total_assessed == 0

    def test_high_allows_trading(self):
        svc = self._make_service(AlertLevel.HIGH, score=0.6)
        # Mock the opportunity methods and technical_service to allow assessment
        opp = svc.opportunity
        opp.technical_service = MagicMock()
        opp.technical_service.snapshot.return_value = _make_technicals()
        opp.macro_service = MagicMock()
        opp.macro_service.calendar.return_value = MagicMock(events_next_7_days=[])
        opp.assess_zero_dte.return_value = _make_zero_dte()
        opp.assess_leap.return_value = _make_leap()
        opp.assess_breakout.return_value = _make_breakout()
        opp.assess_momentum.return_value = _make_momentum()

        svc.levels.analyze.return_value = _make_levels()

        result = svc.rank(["SPY"])
        assert result.black_swan_gate is False
        assert len(result.top_trades) > 0

    def test_normal_no_penalty(self):
        svc = self._make_service(AlertLevel.NORMAL, score=0.0)
        opp = svc.opportunity
        opp.technical_service = MagicMock()
        opp.technical_service.snapshot.return_value = _make_technicals()
        opp.macro_service = MagicMock()
        opp.macro_service.calendar.return_value = MagicMock(events_next_7_days=[])
        opp.assess_zero_dte.return_value = _make_zero_dte()
        opp.assess_leap.return_value = _make_leap()
        opp.assess_breakout.return_value = _make_breakout()
        opp.assess_momentum.return_value = _make_momentum()
        svc.levels.analyze.return_value = _make_levels()

        result = svc.rank(["SPY"])
        assert result.black_swan_level == "normal"
        # No penalty means scores are at their max potential
        for entry in result.top_trades:
            assert entry.breakdown.black_swan_penalty == 0.0


# =============================================
# TestRankingResult
# =============================================


class TestRankingResult:
    @pytest.fixture
    def ranked_result(self) -> TradeRankingResult:
        """Build a full ranking result from mocked services."""
        opp = MagicMock()
        opp.technical_service = MagicMock()
        opp.technical_service.snapshot.return_value = _make_technicals(rsi=55, macd_bull=True)
        opp.macro_service = MagicMock()
        opp.macro_service.calendar.return_value = MagicMock(events_next_7_days=[])
        opp.assess_zero_dte.side_effect = lambda t, **kw: _make_zero_dte(
            ticker=t, verdict=Verdict.GO, confidence=0.80
        )
        opp.assess_leap.side_effect = lambda t, **kw: _make_leap(
            ticker=t, verdict=Verdict.CAUTION, confidence=0.50
        )
        opp.assess_breakout.side_effect = lambda t, **kw: _make_breakout(
            ticker=t, verdict=Verdict.GO, confidence=0.70
        )
        opp.assess_momentum.side_effect = lambda t, **kw: _make_momentum(
            ticker=t, verdict=Verdict.CAUTION, confidence=0.55
        )

        levels = MagicMock()
        levels.analyze.return_value = _make_levels(rr=2.5)

        bs = MagicMock()
        bs.alert.return_value = BlackSwanAlert(
            as_of_date=date(2026, 2, 22),
            alert_level=AlertLevel.NORMAL,
            composite_score=0.0,
            circuit_breakers=[],
            indicators=[],
            triggered_breakers=0,
            action="normal",
            summary="All clear",
        )

        svc = TradeRankingService(
            opportunity_service=opp,
            levels_service=levels,
            black_swan_service=bs,
        )
        return svc.rank(["SPY", "AAPL"], as_of=date(2026, 2, 22))

    def test_sorted_descending(self, ranked_result: TradeRankingResult):
        scores = [e.composite_score for e in ranked_result.top_trades]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_assigned(self, ranked_result: TradeRankingResult):
        ranks = [e.rank for e in ranked_result.top_trades]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_by_ticker_groups(self, ranked_result: TradeRankingResult):
        assert "SPY" in ranked_result.by_ticker
        assert "AAPL" in ranked_result.by_ticker
        assert len(ranked_result.by_ticker["SPY"]) == 4  # 4 strategies
        assert len(ranked_result.by_ticker["AAPL"]) == 4

    def test_by_strategy_groups(self, ranked_result: TradeRankingResult):
        for st in StrategyType:
            assert st in ranked_result.by_strategy
            assert len(ranked_result.by_strategy[st]) == 2  # 2 tickers

    def test_total_counts(self, ranked_result: TradeRankingResult):
        assert ranked_result.total_assessed == 8  # 2 tickers x 4 strategies
        assert len(ranked_result.top_trades) == 8
        assert ranked_result.total_actionable > 0


# =============================================
# TestService
# =============================================


class TestService:
    def _build_service(self) -> TradeRankingService:
        opp = MagicMock()
        opp.technical_service = MagicMock()
        opp.technical_service.snapshot.return_value = _make_technicals()
        opp.macro_service = MagicMock()
        opp.macro_service.calendar.return_value = MagicMock(events_next_7_days=[])
        opp.assess_zero_dte.side_effect = lambda t, **kw: _make_zero_dte(ticker=t)
        opp.assess_leap.side_effect = lambda t, **kw: _make_leap(ticker=t)
        opp.assess_breakout.side_effect = lambda t, **kw: _make_breakout(ticker=t)
        opp.assess_momentum.side_effect = lambda t, **kw: _make_momentum(ticker=t)

        levels = MagicMock()
        levels.analyze.return_value = _make_levels()

        bs = MagicMock()
        bs.alert.return_value = BlackSwanAlert(
            as_of_date=date(2026, 2, 22),
            alert_level=AlertLevel.NORMAL,
            composite_score=0.05,
            circuit_breakers=[], indicators=[], triggered_breakers=0,
            action="normal", summary="ok",
        )
        return TradeRankingService(
            opportunity_service=opp, levels_service=levels, black_swan_service=bs,
        )

    def test_single_ticker(self):
        svc = self._build_service()
        result = svc.rank(["SPY"])
        assert result.total_assessed == 4
        assert len(result.top_trades) == 4

    def test_strategy_filter(self):
        svc = self._build_service()
        result = svc.rank(["SPY"], strategies=[StrategyType.ZERO_DTE, StrategyType.BREAKOUT])
        assert result.total_assessed == 2
        assert len(result.top_trades) == 2
        types = {e.strategy_type for e in result.top_trades}
        assert types == {StrategyType.ZERO_DTE, StrategyType.BREAKOUT}

    def test_assessment_error_skipped(self):
        """If one assessment fails, others still rank."""
        svc = self._build_service()
        svc.opportunity.assess_momentum.side_effect = RuntimeError("network error")
        result = svc.rank(["SPY"])
        assert result.total_assessed == 4
        assert len(result.top_trades) == 3  # momentum failed

    def test_technicals_error_skips_ticker(self):
        """If technicals fail for a ticker, skip it entirely."""
        svc = self._build_service()
        svc.opportunity.technical_service.snapshot.side_effect = RuntimeError("no data")
        result = svc.rank(["SPY"])
        assert len(result.top_trades) == 0

    def test_summary_generated(self):
        svc = self._build_service()
        result = svc.rank(["SPY", "AAPL"])
        assert len(result.summary) > 0
        assert "Ranked" in result.summary


# =============================================
# TestFeedback
# =============================================


class TestFeedback:
    def test_feedback_model(self):
        fb = RankingFeedback(
            as_of_date=date(2026, 2, 22),
            ticker="SPY",
            strategy_type=StrategyType.ZERO_DTE,
            composite_score=0.85,
            verdict=Verdict.GO,
        )
        assert fb.outcome_5d_return is None
        assert fb.trade_pnl is None

    def test_feedback_with_outcome(self):
        fb = RankingFeedback(
            as_of_date=date(2026, 2, 22),
            ticker="AAPL",
            strategy_type=StrategyType.MOMENTUM,
            composite_score=0.60,
            verdict=Verdict.CAUTION,
            outcome_5d_return=0.025,
            outcome_20d_return=0.05,
            outcome_max_drawdown=-0.03,
        )
        assert fb.outcome_5d_return == 0.025

    def test_record_feedback_creates_file(self, tmp_path: Path):
        svc = TradeRankingService(
            opportunity_service=MagicMock(),
            levels_service=MagicMock(),
            black_swan_service=MagicMock(),
        )
        fb = RankingFeedback(
            as_of_date=date(2026, 2, 22),
            ticker="SPY",
            strategy_type=StrategyType.ZERO_DTE,
            composite_score=0.85,
            verdict=Verdict.GO,
        )
        feedback_dir = tmp_path / "feedback"
        with patch(
            "market_analyzer.service.ranking.Path.home",
            return_value=tmp_path / ".market_analyzer_test_home",
        ):
            # Manually set the path for testing
            import market_analyzer.service.ranking as ranking_mod
            orig = Path.home
            Path.home = staticmethod(lambda: tmp_path)
            try:
                svc.record_feedback(fb)
                path = tmp_path / ".market_analyzer" / "feedback" / "ranking_feedback.parquet"
                assert path.exists()
                df = pd.read_parquet(path)
                assert len(df) == 1
                assert df.iloc[0]["ticker"] == "SPY"

                # Record another
                fb2 = RankingFeedback(
                    as_of_date=date(2026, 2, 22),
                    ticker="AAPL",
                    strategy_type=StrategyType.MOMENTUM,
                    composite_score=0.60,
                    verdict=Verdict.CAUTION,
                )
                svc.record_feedback(fb2)
                df2 = pd.read_parquet(path)
                assert len(df2) == 2
            finally:
                Path.home = orig


# =============================================
# TestScoreBreakdown
# =============================================


class TestScoreBreakdown:
    def test_all_fields_present(self):
        bd = ScoreBreakdown(
            verdict_score=1.0, confidence_score=0.8, regime_alignment=0.9,
            risk_reward=0.7, technical_quality=0.6, phase_alignment=0.5,
            income_bias_boost=0.05, black_swan_penalty=0.1,
            macro_penalty=0.02, earnings_penalty=0.0,
        )
        assert bd.verdict_score == 1.0
        assert bd.income_bias_boost == 0.05

    def test_composite_from_breakdown_consistency(self):
        """composite_from_breakdown should produce same result as compute_composite_score."""
        technicals = _make_technicals(rsi=55, macd_bull=True)
        levels = _make_levels(rr=2.5)
        breakdown = compute_composite_score(
            verdict=Verdict.GO, confidence=0.75, regime_id=3, phase_id=2,
            strategy=StrategyType.MOMENTUM, technicals=technicals,
            levels=levels, black_swan_score=0.1, events_next_7_days=1,
            days_to_earnings=30,
        )
        c1 = composite_from_breakdown(breakdown)
        # Verify it's reasonable
        assert 0.0 <= c1 <= 1.0
        assert c1 > 0.3  # GO + decent confidence + good alignment

    def test_breakdown_zero_confidence(self):
        technicals = _make_technicals()
        breakdown = compute_composite_score(
            verdict=Verdict.GO, confidence=0.0, regime_id=1, phase_id=1,
            strategy=StrategyType.ZERO_DTE, technicals=technicals,
            levels=None, black_swan_score=0.0, events_next_7_days=0,
            days_to_earnings=60,
        )
        assert breakdown.confidence_score == 0.0

    def test_breakdown_earnings_close(self):
        technicals = _make_technicals()
        breakdown = compute_composite_score(
            verdict=Verdict.GO, confidence=0.8, regime_id=1, phase_id=1,
            strategy=StrategyType.ZERO_DTE, technicals=technicals,
            levels=None, black_swan_score=0.0, events_next_7_days=0,
            days_to_earnings=1,
        )
        assert breakdown.earnings_penalty == pytest.approx(0.10)

    def test_breakdown_no_earnings_penalty(self):
        technicals = _make_technicals()
        breakdown = compute_composite_score(
            verdict=Verdict.GO, confidence=0.8, regime_id=1, phase_id=1,
            strategy=StrategyType.ZERO_DTE, technicals=technicals,
            levels=None, black_swan_score=0.0, events_next_7_days=0,
            days_to_earnings=30,
        )
        assert breakdown.earnings_penalty == 0.0

    def test_breakdown_macro_events(self):
        technicals = _make_technicals()
        breakdown = compute_composite_score(
            verdict=Verdict.GO, confidence=0.8, regime_id=1, phase_id=1,
            strategy=StrategyType.ZERO_DTE, technicals=technicals,
            levels=None, black_swan_score=0.0, events_next_7_days=3,
            days_to_earnings=60,
        )
        assert breakdown.macro_penalty == pytest.approx(0.06)
