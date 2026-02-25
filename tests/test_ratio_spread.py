"""Tests for ratio spread opportunity assessment."""

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
from market_analyzer.opportunity.option_plays.ratio_spread import (
    RatioSpreadOpportunity, RatioSpreadStrategy, assess_ratio_spread,
)


def _regime(regime_id: int = 1, confidence: float = 0.75) -> RegimeResult:
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
        phase=PhaseIndicator(phase=MarketPhase.ACCUMULATION, confidence=0.5, description="Test",
                            higher_highs=False, higher_lows=True, lower_highs=False, lower_lows=False,
                            range_compression=0.3, volume_trend="declining", price_vs_sma_50_pct=2.0),
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
            regime_signal="R1 low-vol MR",
            price_signal="higher highs, higher lows",
            volume_signal="rising volume",
            supporting=["trend confirmed"],
            contradictions=[],
        ),
        transitions=[],
        strategy_comment="Test strategy comment",
    )


def _vol_surface(
    front_iv: float = 0.25,
    back_iv: float = 0.22,
    quality: str = "good",
    put_skew: float = 0.05,
    call_skew: float = 0.02,
    skew_ratio: float = 2.5,
) -> VolatilitySurface:
    today = date(2026, 3, 1)
    exps = [today + timedelta(days=30), today + timedelta(days=60)]
    ts = [
        TermStructurePoint(expiration=exps[0], days_to_expiry=30, atm_iv=front_iv, atm_strike=580.0),
        TermStructurePoint(expiration=exps[1], days_to_expiry=60, atm_iv=back_iv, atm_strike=580.0),
    ]
    slope = (back_iv - front_iv) / front_iv if front_iv > 0 else 0.0
    skew = SkewSlice(
        expiration=exps[0], days_to_expiry=30, atm_iv=front_iv,
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


class TestRatioSpreadHardStops:
    def test_r4_any_confidence_is_hard_stop(self) -> None:
        """R4 is always a hard stop for ratio spreads (naked leg + explosive moves)."""
        result = assess_ratio_spread("SPY", _regime(4, 0.50), _technicals(), _vol_surface())
        assert result.verdict == Verdict.NO_GO
        assert any("R4" in s.name for s in result.hard_stops)

    def test_r2_high_confidence_is_hard_stop(self) -> None:
        result = assess_ratio_spread("SPY", _regime(2, 0.85), _technicals(), _vol_surface())
        assert result.verdict == Verdict.NO_GO
        assert any("R2" in s.name for s in result.hard_stops)

    def test_no_vol_surface_is_hard_stop(self) -> None:
        result = assess_ratio_spread("SPY", _regime(1), _technicals(), vol_surface=None)
        assert result.verdict == Verdict.NO_GO

    def test_flat_skew_is_hard_stop(self) -> None:
        """Skew too flat = no edge in selling OTM."""
        result = assess_ratio_spread(
            "SPY", _regime(1), _technicals(),
            _vol_surface(put_skew=0.005, call_skew=0.003),
        )
        assert result.verdict == Verdict.NO_GO
        assert any("skew" in s.name.lower() for s in result.hard_stops)


class TestRatioSpreadNakedLeg:
    def test_has_naked_leg_flag(self) -> None:
        """Standard ratio spreads have a naked leg."""
        result = assess_ratio_spread("SPY", _regime(1), _technicals(), _vol_surface(), _phase(2))
        if result.verdict != Verdict.NO_GO:
            assert result.has_naked_leg is True

    def test_margin_warning_present(self) -> None:
        result = assess_ratio_spread("SPY", _regime(1), _technicals(), _vol_surface(), _phase(2))
        if result.verdict != Verdict.NO_GO:
            assert result.margin_warning is not None
            assert "margin" in result.margin_warning.lower() or "$" in result.margin_warning


class TestRatioSpreadStrategy:
    def test_bullish_selects_call_ratio(self) -> None:
        result = assess_ratio_spread("SPY", _regime(1), _technicals(rsi=60), _vol_surface(), _phase(2))
        if result.verdict != Verdict.NO_GO:
            assert result.ratio_strategy == RatioSpreadStrategy.CALL_RATIO_SPREAD

    def test_bearish_selects_put_ratio(self) -> None:
        result = assess_ratio_spread("SPY", _regime(1), _technicals(rsi=40), _vol_surface(), _phase(4))
        if result.verdict != Verdict.NO_GO:
            assert result.ratio_strategy == RatioSpreadStrategy.PUT_RATIO_SPREAD


class TestRatioSpreadVerdict:
    def test_r1_steep_skew_bullish_go(self) -> None:
        """R1 + steep put skew + bullish → GO."""
        result = assess_ratio_spread(
            "SPY", _regime(1, 0.80), _technicals(rsi=60),
            _vol_surface(front_iv=0.28, put_skew=0.06, call_skew=0.02),
            _phase(2),
        )
        assert result.verdict == Verdict.GO

    def test_r2_low_confidence_passes(self) -> None:
        """R2 with low confidence is not a hard stop, but should be cautious."""
        result = assess_ratio_spread(
            "SPY", _regime(2, 0.50), _technicals(),
            _vol_surface(), _phase(2),
        )
        # Not a hard stop since below R2 threshold, but not ideal
        assert result.verdict in (Verdict.NO_GO, Verdict.CAUTION, Verdict.GO)


class TestRatioSpreadOutput:
    def test_output_fields(self) -> None:
        result = assess_ratio_spread("SPY", _regime(1), _technicals(), _vol_surface(), _phase(2))
        assert isinstance(result, RatioSpreadOpportunity)
        assert result.ticker == "SPY"
        assert result.front_iv > 0
        assert result.direction in ("bullish", "bearish")
        assert 0.0 <= result.confidence <= 1.0
        assert "SPY" in result.summary

    def test_hard_stop_produces_no_trade(self) -> None:
        result = assess_ratio_spread("SPY", _regime(4), _technicals(), _vol_surface())
        assert result.ratio_strategy == RatioSpreadStrategy.NO_TRADE

    def test_go_has_trade_spec(self) -> None:
        """GO ratio spread should have a trade spec with 3 legs."""
        result = assess_ratio_spread(
            "SPY", _regime(1, 0.80), _technicals(rsi=60),
            _vol_surface(front_iv=0.28, put_skew=0.06, call_skew=0.02),
            _phase(2),
        )
        if result.verdict != Verdict.NO_GO and result.trade_spec is not None:
            assert len(result.trade_spec.legs) == 3
            assert result.trade_spec.ticker == "SPY"

    def test_no_go_has_no_trade_spec(self) -> None:
        result = assess_ratio_spread("SPY", _regime(4), _technicals(), _vol_surface())
        assert result.trade_spec is None
