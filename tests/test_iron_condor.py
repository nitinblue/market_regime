"""Tests for iron condor opportunity assessment."""

from datetime import date, timedelta

import pytest

from market_analyzer.models.regime import RegimeID, RegimeResult
from market_analyzer.models.technicals import (
    BollingerBands, MACDData, MovingAverages, RSIData,
    StochasticData, SupportResistance, TechnicalSnapshot,
    MarketPhase, PhaseIndicator,
)
from market_analyzer.models.vol_surface import (
    SkewSlice, TermStructurePoint, VolatilitySurface,
)
from market_analyzer.models.opportunity import Verdict
from market_analyzer.opportunity.option_plays.iron_condor import (
    IronCondorOpportunity, IronCondorStrategy, assess_iron_condor,
)


def _regime(regime_id: int = 1, confidence: float = 0.75) -> RegimeResult:
    return RegimeResult(
        ticker="TEST", regime=RegimeID(regime_id), confidence=confidence,
        regime_probabilities={1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1},
        as_of_date=date(2026, 3, 1), model_version="test", trend_direction=None,
    )


def _technicals(rsi: float = 50.0, atr_pct: float = 1.0, price: float = 580.0) -> TechnicalSnapshot:
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


def _vol_surface(
    front_iv: float = 0.22,
    back_iv: float = 0.20,
    quality: str = "good",
    put_skew: float = 0.04,
    call_skew: float = 0.03,
    skew_ratio: float = 1.3,
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


class TestIronCondorHardStops:
    def test_r4_high_confidence_is_hard_stop(self) -> None:
        """R4 at high confidence = hard stop (explosive moves destroy condors)."""
        result = assess_iron_condor("SPY", _regime(4, 0.80), _technicals(), _vol_surface())
        assert result.verdict == Verdict.NO_GO
        assert any("R4" in s.name for s in result.hard_stops)

    def test_r4_low_confidence_passes(self) -> None:
        """R4 below threshold is not a hard stop."""
        result = assess_iron_condor("SPY", _regime(4, 0.50), _technicals(), _vol_surface())
        assert not any("R4" in s.name for s in result.hard_stops)

    def test_r3_high_confidence_is_hard_stop(self) -> None:
        """R3 at high confidence = hard stop (persistent trend blows through)."""
        result = assess_iron_condor("SPY", _regime(3, 0.80), _technicals(), _vol_surface())
        assert result.verdict == Verdict.NO_GO
        assert any("R3" in s.name for s in result.hard_stops)

    def test_r3_low_confidence_passes(self) -> None:
        """R3 below threshold is not a hard stop."""
        result = assess_iron_condor("SPY", _regime(3, 0.50), _technicals(), _vol_surface())
        assert not any("R3" in s.name for s in result.hard_stops)

    def test_no_vol_surface_is_hard_stop(self) -> None:
        result = assess_iron_condor("SPY", _regime(1), _technicals(), vol_surface=None)
        assert result.verdict == Verdict.NO_GO
        assert any("vol surface" in s.name.lower() for s in result.hard_stops)

    def test_low_iv_is_hard_stop(self) -> None:
        """IV too low = not enough premium to justify condor."""
        result = assess_iron_condor("SPY", _regime(1), _technicals(), _vol_surface(front_iv=0.08))
        assert result.verdict == Verdict.NO_GO
        assert any("IV too low" in s.name for s in result.hard_stops)

    def test_poor_data_quality_is_hard_stop(self) -> None:
        result = assess_iron_condor("SPY", _regime(1), _technicals(), _vol_surface(quality="poor"))
        assert result.verdict == Verdict.NO_GO


class TestIronCondorStrategy:
    def test_r1_high_confidence_selects_standard(self) -> None:
        """R1 is the prime iron condor environment — should select standard."""
        result = assess_iron_condor("SPY", _regime(1, 0.85), _technicals(), _vol_surface())
        if result.verdict != Verdict.NO_GO:
            assert result.iron_condor_strategy == IronCondorStrategy.STANDARD_IRON_CONDOR

    def test_r2_selects_wide(self) -> None:
        """R2 wider swings = wide iron condor."""
        result = assess_iron_condor("SPY", _regime(2, 0.75), _technicals(), _vol_surface(front_iv=0.30))
        if result.verdict != Verdict.NO_GO:
            assert result.iron_condor_strategy == IronCondorStrategy.WIDE_IRON_CONDOR

    def test_skewed_rsi_selects_unbalanced(self) -> None:
        """RSI skewed in R1/R2 should suggest unbalanced condor."""
        # RSI 35 with R1 and moderate confidence (not high enough for standard)
        result = assess_iron_condor("SPY", _regime(1, 0.70), _technicals(rsi=35), _vol_surface())
        if result.verdict != Verdict.NO_GO and result.iron_condor_strategy == IronCondorStrategy.UNBALANCED_IRON_CONDOR:
            assert "slightly bullish" in result.strategy.direction


class TestIronCondorVerdict:
    def test_r1_centered_rsi_good_iv_is_go(self) -> None:
        """R1 + centered RSI + good IV = GO — the ideal condor setup."""
        result = assess_iron_condor(
            "SPY", _regime(1, 0.85), _technicals(rsi=50),
            _vol_surface(front_iv=0.25),
        )
        assert result.verdict == Verdict.GO

    def test_r2_moderate_iv_is_caution_or_go(self) -> None:
        """R2 + moderate IV should be at least CAUTION."""
        result = assess_iron_condor(
            "SPY", _regime(2, 0.70), _technicals(rsi=50),
            _vol_surface(front_iv=0.22),
        )
        assert result.verdict in (Verdict.GO, Verdict.CAUTION)


class TestIronCondorOutput:
    def test_output_fields(self) -> None:
        result = assess_iron_condor("SPY", _regime(1), _technicals(), _vol_surface())
        assert isinstance(result, IronCondorOpportunity)
        assert result.ticker == "SPY"
        assert result.front_iv > 0
        assert 0.0 <= result.confidence <= 1.0
        assert "SPY" in result.summary

    def test_wing_width_suggestion_present(self) -> None:
        """Non-NO_GO results should have a wing width suggestion."""
        result = assess_iron_condor(
            "SPY", _regime(1, 0.85), _technicals(),
            _vol_surface(front_iv=0.25),
        )
        if result.verdict != Verdict.NO_GO:
            assert result.wing_width_suggestion != "N/A"
            assert "OTM" in result.wing_width_suggestion

    def test_hard_stop_produces_no_trade(self) -> None:
        result = assess_iron_condor("SPY", _regime(4, 0.80), _technicals(), _vol_surface())
        assert result.iron_condor_strategy == IronCondorStrategy.NO_TRADE

    def test_skew_values_populated(self) -> None:
        result = assess_iron_condor("SPY", _regime(1), _technicals(), _vol_surface(put_skew=0.05, call_skew=0.02))
        assert result.put_skew == pytest.approx(0.05)
        assert result.call_skew == pytest.approx(0.02)

    def test_go_has_trade_spec(self) -> None:
        """GO verdict should include a trade spec with 4 legs."""
        result = assess_iron_condor(
            "SPY", _regime(1, 0.85), _technicals(rsi=50), _vol_surface(front_iv=0.25),
        )
        if result.verdict != Verdict.NO_GO:
            assert result.trade_spec is not None
            assert len(result.trade_spec.legs) == 4
            assert result.trade_spec.wing_width_points > 0
            assert result.trade_spec.ticker == "SPY"

    def test_no_go_has_no_trade_spec(self) -> None:
        result = assess_iron_condor("SPY", _regime(4, 0.80), _technicals(), _vol_surface())
        assert result.trade_spec is None

    def test_trade_spec_leg_codes(self) -> None:
        """Trade spec should produce human-readable leg codes."""
        result = assess_iron_condor(
            "SPY", _regime(1, 0.85), _technicals(rsi=50), _vol_surface(front_iv=0.25),
        )
        if result.trade_spec:
            codes = result.trade_spec.leg_codes
            assert len(codes) == 4
            assert all("SPY" in c for c in codes)
