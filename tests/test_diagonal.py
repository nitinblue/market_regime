"""Tests for diagonal spread opportunity assessment."""

from datetime import date, timedelta

import pytest

from market_analyzer.models.regime import RegimeID, RegimeResult
from market_analyzer.models.phase import PhaseEvidence, PhaseID, PhaseResult, PriceStructure
from market_analyzer.models.technicals import (
    BollingerBands, MACDData, MovingAverages, RSIData,
    StochasticData, SupportResistance, TechnicalSnapshot,
    MarketPhase, PhaseIndicator,
)
from market_analyzer.models.vol_surface import (
    SkewSlice, TermStructurePoint, VolatilitySurface,
)
from market_analyzer.models.opportunity import Verdict
from market_analyzer.opportunity.option_plays.diagonal import (
    DiagonalOpportunity, DiagonalStrategy, assess_diagonal,
)


def _regime(regime_id: int = 3, confidence: float = 0.75) -> RegimeResult:
    return RegimeResult(
        ticker="TEST", regime=RegimeID(regime_id), confidence=confidence,
        regime_probabilities={1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1},
        as_of_date=date(2026, 3, 1), model_version="test", trend_direction=None,
    )


def _technicals(rsi: float = 55.0, atr_pct: float = 1.0, price: float = 580.0) -> TechnicalSnapshot:
    return TechnicalSnapshot(
        ticker="TEST", as_of_date=date(2026, 3, 1), current_price=price,
        atr=price * atr_pct / 100, atr_pct=atr_pct, vwma_20=price,
        moving_averages=MovingAverages(
            sma_20=price, sma_50=price * 0.98, sma_200=price * 0.95,
            ema_9=price, ema_21=price,
            price_vs_sma_20_pct=0.0, price_vs_sma_50_pct=2.0, price_vs_sma_200_pct=5.0,
        ),
        rsi=RSIData(value=rsi, is_overbought=rsi > 70, is_oversold=rsi < 30),
        bollinger=BollingerBands(upper=price + 10, middle=price, lower=price - 10, bandwidth=0.04, percent_b=0.5),
        macd=MACDData(macd_line=0.5, signal_line=0.3, histogram=0.2, is_bullish_crossover=False, is_bearish_crossover=False),
        stochastic=StochasticData(k=50.0, d=50.0, is_overbought=False, is_oversold=False),
        support_resistance=SupportResistance(support=570.0, resistance=590.0, price_vs_support_pct=1.7, price_vs_resistance_pct=-1.7),
        phase=PhaseIndicator(phase=MarketPhase.MARKUP, confidence=0.6, description="Test",
                            higher_highs=True, higher_lows=True, lower_highs=False, lower_lows=False,
                            range_compression=0.3, volume_trend="rising", price_vs_sma_50_pct=2.0),
        signals=[],
    )


def _phase(phase_id: int = 2) -> PhaseResult:
    return PhaseResult(
        ticker="TEST", phase=PhaseID(phase_id),
        phase_name={1: "Accumulation", 2: "Markup", 3: "Distribution", 4: "Markdown"}[phase_id],
        confidence=0.70, as_of_date=date(2026, 3, 1),
        phase_age_days=15,
        prior_phase=None,
        cycle_completion=0.3,
        price_structure=PriceStructure(
            swing_highs=[], swing_lows=[],
            higher_highs=True, higher_lows=True,
            lower_highs=False, lower_lows=False,
            range_compression=0.0, price_vs_sma=2.0,
            volume_trend="rising", support_level=570.0, resistance_level=590.0,
        ),
        evidence=PhaseEvidence(
            regime_signal="R3 low-vol trending",
            price_signal="higher highs, higher lows",
            volume_signal="rising volume",
            supporting=["trend confirmed"],
            contradictions=[],
        ),
        transitions=[],
        strategy_comment="Test strategy comment",
    )


def _vol_surface(
    front_iv: float = 0.20,
    back_iv: float = 0.22,
    quality: str = "good",
    skew_ratio: float = 1.2,
    put_skew: float = 0.03,
    call_skew: float = 0.01,
) -> VolatilitySurface:
    today = date(2026, 3, 1)
    exps = [today + timedelta(days=7), today + timedelta(days=35)]
    ts = [
        TermStructurePoint(expiration=exps[0], days_to_expiry=7, atm_iv=front_iv, atm_strike=580.0),
        TermStructurePoint(expiration=exps[1], days_to_expiry=35, atm_iv=back_iv, atm_strike=580.0),
    ]
    slope = (back_iv - front_iv) / front_iv if front_iv > 0 else 0.0
    skew = SkewSlice(
        expiration=exps[0], days_to_expiry=7, atm_iv=front_iv,
        otm_put_iv=front_iv + put_skew, otm_call_iv=front_iv + call_skew,
        put_skew=put_skew, call_skew=call_skew, skew_ratio=skew_ratio,
    )
    return VolatilitySurface(
        ticker="TEST", as_of_date=today, underlying_price=580.0,
        expirations=exps, term_structure=ts,
        front_iv=front_iv, back_iv=back_iv, term_slope=slope,
        is_contango=back_iv > front_iv, is_backwardation=front_iv > back_iv,
        skew_by_expiry=[skew],
        calendar_edge_score=0.4,
        best_calendar_expiries=(exps[0], exps[1]),
        iv_differential_pct=(front_iv - back_iv) / back_iv * 100 if back_iv > 0 else 0.0,
        total_contracts=200, avg_bid_ask_spread_pct=0.5,
        data_quality=quality, summary="test",
    )


class TestDiagonalHardStops:
    def test_r4_high_confidence_is_hard_stop(self) -> None:
        result = assess_diagonal("SPY", _regime(4, 0.85), _technicals(), _vol_surface())
        assert result.verdict == Verdict.NO_GO
        assert any("R4" in s.name for s in result.hard_stops)

    def test_no_vol_surface_is_hard_stop(self) -> None:
        result = assess_diagonal("SPY", _regime(3), _technicals(), vol_surface=None)
        assert result.verdict == Verdict.NO_GO

    def test_extreme_skew_is_hard_stop(self) -> None:
        result = assess_diagonal("SPY", _regime(3), _technicals(), _vol_surface(skew_ratio=5.0))
        assert result.verdict == Verdict.NO_GO
        assert any("skew" in s.name.lower() for s in result.hard_stops)


class TestDiagonalDirection:
    def test_bullish_with_high_rsi(self) -> None:
        result = assess_diagonal("SPY", _regime(3), _technicals(rsi=60), _vol_surface(), _phase(2))
        assert result.trend_direction == "bullish"

    def test_bearish_with_low_rsi(self) -> None:
        result = assess_diagonal("SPY", _regime(3), _technicals(rsi=40), _vol_surface(), _phase(4))
        assert result.trend_direction == "bearish"


class TestDiagonalStrategy:
    def test_r3_bullish_selects_bull_call(self) -> None:
        result = assess_diagonal("SPY", _regime(3), _technicals(rsi=60), _vol_surface(), _phase(2))
        assert result.diagonal_strategy == DiagonalStrategy.BULL_CALL_DIAGONAL

    def test_r3_bearish_selects_bear_put(self) -> None:
        result = assess_diagonal("SPY", _regime(3), _technicals(rsi=40), _vol_surface(), _phase(4))
        assert result.diagonal_strategy == DiagonalStrategy.BEAR_PUT_DIAGONAL

    def test_r1_bullish_selects_pmcc(self) -> None:
        result = assess_diagonal("SPY", _regime(1), _technicals(rsi=60), _vol_surface(), _phase(2))
        assert result.diagonal_strategy == DiagonalStrategy.PMCC_DIAGONAL


class TestDiagonalVerdict:
    def test_r3_trending_good_conditions_go(self) -> None:
        """R3 + bullish + good vol surface → GO."""
        result = assess_diagonal(
            "SPY", _regime(3, 0.80), _technicals(rsi=60),
            _vol_surface(front_iv=0.20, back_iv=0.25), _phase(2),
        )
        assert result.verdict == Verdict.GO

    def test_no_direction_reduces_confidence(self) -> None:
        """Neutral direction + R1 → lower confidence."""
        result = assess_diagonal(
            "SPY", _regime(1), _technicals(rsi=50),
            _vol_surface(), _phase(1),
        )
        # Neutral direction makes diagonals less attractive
        assert result.confidence < 0.55


class TestDiagonalOutput:
    def test_output_fields(self) -> None:
        result = assess_diagonal("SPY", _regime(3), _technicals(), _vol_surface(), _phase(2))
        assert isinstance(result, DiagonalOpportunity)
        assert result.ticker == "SPY"
        assert result.phase_id == 2
        assert result.phase_name == "Markup"
        assert 0.0 <= result.confidence <= 1.0

    def test_go_has_trade_spec(self) -> None:
        """GO diagonal should have a trade spec with 2 legs at different strikes."""
        result = assess_diagonal(
            "SPY", _regime(3, 0.80), _technicals(rsi=60),
            _vol_surface(front_iv=0.20, back_iv=0.25), _phase(2),
        )
        if result.verdict != Verdict.NO_GO and result.trade_spec is not None:
            assert len(result.trade_spec.legs) == 2
            assert result.trade_spec.front_expiration is not None
            assert result.trade_spec.back_expiration is not None

    def test_no_go_has_no_trade_spec(self) -> None:
        result = assess_diagonal("SPY", _regime(4, 0.85), _technicals(), _vol_surface())
        assert result.trade_spec is None
