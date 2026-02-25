"""Tests for iron butterfly opportunity assessment."""

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
from market_analyzer.opportunity.option_plays.iron_butterfly import (
    IronButterflyOpportunity, IronButterflyStrategy, assess_iron_butterfly,
)


def _regime(regime_id: int = 2, confidence: float = 0.80) -> RegimeResult:
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
        bollinger=BollingerBands(upper=price + 10, middle=price, lower=price - 10, bandwidth=0.06, percent_b=0.5),
        macd=MACDData(macd_line=0.5, signal_line=0.3, histogram=0.2, is_bullish_crossover=False, is_bearish_crossover=False),
        stochastic=StochasticData(k=50.0, d=50.0, is_overbought=False, is_oversold=False),
        support_resistance=SupportResistance(support=570.0, resistance=590.0, price_vs_support_pct=1.7, price_vs_resistance_pct=-1.7),
        phase=PhaseIndicator(phase=MarketPhase.ACCUMULATION, confidence=0.5, description="Test",
                            higher_highs=False, higher_lows=True, lower_highs=False, lower_lows=False,
                            range_compression=0.3, volume_trend="declining", price_vs_sma_50_pct=2.0),
        signals=[],
    )


def _vol_surface(
    front_iv: float = 0.28,
    back_iv: float = 0.25,
    quality: str = "good",
    skew_ratio: float = 1.2,
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
        otm_put_iv=front_iv + 0.03, otm_call_iv=front_iv + 0.01,
        put_skew=0.03, call_skew=0.01, skew_ratio=skew_ratio,
    )
    return VolatilitySurface(
        ticker="TEST", as_of_date=today, underlying_price=580.0,
        expirations=exps, term_structure=ts,
        front_iv=front_iv, back_iv=back_iv, term_slope=slope,
        is_contango=back_iv > front_iv, is_backwardation=front_iv > back_iv,
        skew_by_expiry=[skew],
        calendar_edge_score=0.5,
        best_calendar_expiries=(exps[0], exps[1]),
        iv_differential_pct=(front_iv - back_iv) / back_iv * 100 if back_iv > 0 else 0.0,
        total_contracts=200, avg_bid_ask_spread_pct=0.5,
        data_quality=quality, summary="test",
    )


class TestIronButterflyHardStops:
    def test_r3_high_confidence_is_hard_stop(self) -> None:
        result = assess_iron_butterfly("SPY", _regime(3, 0.85), _technicals(), _vol_surface())
        assert result.verdict == Verdict.NO_GO
        assert any("trending" in s.name.lower() for s in result.hard_stops)

    def test_r4_high_confidence_is_hard_stop(self) -> None:
        result = assess_iron_butterfly("SPY", _regime(4, 0.85), _technicals(), _vol_surface())
        assert result.verdict == Verdict.NO_GO

    def test_low_iv_is_hard_stop(self) -> None:
        result = assess_iron_butterfly("SPY", _regime(2), _technicals(), _vol_surface(front_iv=0.10))
        assert result.verdict == Verdict.NO_GO
        assert any("iv too low" in s.name.lower() for s in result.hard_stops)

    def test_no_vol_surface_is_hard_stop(self) -> None:
        result = assess_iron_butterfly("SPY", _regime(2), _technicals(), vol_surface=None)
        assert result.verdict == Verdict.NO_GO

    def test_poor_data_quality_is_hard_stop(self) -> None:
        result = assess_iron_butterfly("SPY", _regime(2), _technicals(), _vol_surface(quality="poor"))
        assert result.verdict == Verdict.NO_GO


class TestIronButterflySignals:
    def test_high_iv_favorable(self) -> None:
        result = assess_iron_butterfly("SPY", _regime(2), _technicals(), _vol_surface(front_iv=0.35))
        iv_signal = [s for s in result.signals if "excellent" in s.name.lower() or "atm iv" in s.name.lower()]
        assert any(s.favorable for s in iv_signal)

    def test_r2_mean_reverting_favorable(self) -> None:
        result = assess_iron_butterfly("SPY", _regime(2), _technicals(), _vol_surface())
        regime_signal = [s for s in result.signals if "mean-reverting" in s.name.lower()]
        assert len(regime_signal) == 1
        assert regime_signal[0].favorable is True

    def test_centered_rsi_favorable(self) -> None:
        result = assess_iron_butterfly("SPY", _regime(2), _technicals(rsi=50), _vol_surface())
        rsi_signal = [s for s in result.signals if "rsi" in s.name.lower()]
        assert any(s.favorable for s in rsi_signal)


class TestIronButterflyStrategy:
    def test_r2_high_confidence_standard(self) -> None:
        result = assess_iron_butterfly("SPY", _regime(2, 0.85), _technicals(rsi=50), _vol_surface(front_iv=0.30))
        assert result.iron_butterfly_strategy == IronButterflyStrategy.STANDARD_IRON_BUTTERFLY

    def test_r1_directional_broken_wing(self) -> None:
        result = assess_iron_butterfly("SPY", _regime(1, 0.80), _technicals(rsi=60), _vol_surface())
        assert result.iron_butterfly_strategy == IronButterflyStrategy.BROKEN_WING_BUTTERFLY

    def test_r2_lower_confidence_wide(self) -> None:
        """R2 with moderate confidence → wide butterfly."""
        result = assess_iron_butterfly("SPY", _regime(2, 0.65), _technicals(rsi=50), _vol_surface(front_iv=0.22))
        assert result.iron_butterfly_strategy in (
            IronButterflyStrategy.WIDE_IRON_BUTTERFLY,
            IronButterflyStrategy.STANDARD_IRON_BUTTERFLY,
        )


class TestIronButterflyVerdict:
    def test_r2_high_iv_centered_is_go(self) -> None:
        """R2 + high IV + centered RSI → GO."""
        result = assess_iron_butterfly(
            "SPY", _regime(2, 0.85), _technicals(rsi=50),
            _vol_surface(front_iv=0.32),
        )
        assert result.verdict == Verdict.GO

    def test_r1_moderate_iv_is_caution_or_go(self) -> None:
        result = assess_iron_butterfly(
            "SPY", _regime(1, 0.75), _technicals(rsi=50),
            _vol_surface(front_iv=0.22),
        )
        assert result.verdict in (Verdict.GO, Verdict.CAUTION)


class TestIronButterflyOutput:
    def test_output_fields(self) -> None:
        result = assess_iron_butterfly("SPY", _regime(2), _technicals(), _vol_surface())
        assert isinstance(result, IronButterflyOpportunity)
        assert result.ticker == "SPY"
        assert result.atm_iv > 0
        assert 0.0 <= result.confidence <= 1.0
        assert "SPY" in result.summary

    def test_go_has_trade_spec(self) -> None:
        """GO verdict should include a trade spec with 4 legs."""
        result = assess_iron_butterfly(
            "SPY", _regime(2, 0.85), _technicals(rsi=50), _vol_surface(front_iv=0.32),
        )
        if result.verdict != Verdict.NO_GO:
            assert result.trade_spec is not None
            assert len(result.trade_spec.legs) == 4
            assert result.trade_spec.wing_width_points > 0

    def test_no_go_has_no_trade_spec(self) -> None:
        result = assess_iron_butterfly("SPY", _regime(4, 0.85), _technicals(), _vol_surface())
        assert result.trade_spec is None
