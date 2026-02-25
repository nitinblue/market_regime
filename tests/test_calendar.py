"""Tests for calendar spread opportunity assessment."""

from datetime import date, datetime, timedelta

import pytest

from market_analyzer.models.regime import RegimeID, RegimeResult, TrendDirection
from market_analyzer.models.technicals import (
    BollingerBands, MACDData, MovingAverages, RSIData,
    StochasticData, SupportResistance, TechnicalSnapshot,
    MarketPhase, PhaseIndicator,
)
from market_analyzer.models.vol_surface import (
    SkewSlice, TermStructurePoint, VolatilitySurface,
)
from market_analyzer.models.opportunity import Verdict
from market_analyzer.opportunity.option_plays.calendar import (
    CalendarOpportunity, CalendarStrategy, assess_calendar,
)


# --- Test helpers ---

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
    back_iv: float = 0.18,
    quality: str = "good",
    spread_pct: float = 0.5,
    skew_ratio: float = 1.2,
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
        otm_put_iv=front_iv + 0.03, otm_call_iv=front_iv + 0.01,
        put_skew=0.03, call_skew=0.01, skew_ratio=skew_ratio,
    )
    return VolatilitySurface(
        ticker="TEST", as_of_date=today, underlying_price=580.0,
        expirations=exps, term_structure=ts,
        front_iv=front_iv, back_iv=back_iv, term_slope=slope,
        is_contango=back_iv > front_iv, is_backwardation=front_iv > back_iv,
        skew_by_expiry=[skew],
        calendar_edge_score=0.6 if front_iv > back_iv else 0.3,
        best_calendar_expiries=(exps[0], exps[1]),
        iv_differential_pct=(front_iv - back_iv) / back_iv * 100 if back_iv > 0 else 0.0,
        total_contracts=200, avg_bid_ask_spread_pct=spread_pct,
        data_quality=quality, summary="test",
    )


class TestCalendarHardStops:
    def test_r4_high_confidence_is_hard_stop(self) -> None:
        result = assess_calendar("SPY", _regime(4, 0.85), _technicals(), _vol_surface())
        assert result.verdict == Verdict.NO_GO
        assert any("R4" in s.name for s in result.hard_stops)

    def test_r4_low_confidence_passes(self) -> None:
        result = assess_calendar("SPY", _regime(4, 0.50), _technicals(), _vol_surface())
        assert not any("R4" in s.name for s in result.hard_stops)

    def test_no_vol_surface_is_hard_stop(self) -> None:
        result = assess_calendar("SPY", _regime(), _technicals(), vol_surface=None)
        assert result.verdict == Verdict.NO_GO
        assert any("vol surface" in s.name.lower() for s in result.hard_stops)

    def test_poor_data_quality_is_hard_stop(self) -> None:
        result = assess_calendar("SPY", _regime(), _technicals(), _vol_surface(quality="poor"))
        assert result.verdict == Verdict.NO_GO

    def test_wide_bid_ask_is_hard_stop(self) -> None:
        result = assess_calendar("SPY", _regime(), _technicals(), _vol_surface(spread_pct=5.0))
        assert result.verdict == Verdict.NO_GO
        assert any("bid-ask" in s.name.lower() for s in result.hard_stops)


class TestCalendarSignals:
    def test_backwardation_favorable(self) -> None:
        """Backwardation (front > back) is good for calendars."""
        result = assess_calendar("SPY", _regime(1), _technicals(), _vol_surface(front_iv=0.25, back_iv=0.18))
        term_signal = [s for s in result.signals if "backwardation" in s.name.lower()]
        assert len(term_signal) == 1
        assert term_signal[0].favorable is True

    def test_r1_mean_reverting_favorable(self) -> None:
        result = assess_calendar("SPY", _regime(1), _technicals(), _vol_surface())
        regime_signal = [s for s in result.signals if "mean-reverting" in s.name.lower()]
        assert len(regime_signal) == 1
        assert regime_signal[0].favorable is True

    def test_neutral_rsi_favorable(self) -> None:
        result = assess_calendar("SPY", _regime(1), _technicals(rsi=50), _vol_surface())
        rsi_signal = [s for s in result.signals if "rsi" in s.name.lower()]
        assert len(rsi_signal) == 1
        assert rsi_signal[0].favorable is True


class TestCalendarStrategy:
    def test_r1_selects_atm_calendar(self) -> None:
        result = assess_calendar("SPY", _regime(1), _technicals(), _vol_surface())
        assert result.calendar_strategy == CalendarStrategy.ATM_CALENDAR

    def test_r2_selects_double_calendar(self) -> None:
        result = assess_calendar("SPY", _regime(2, 0.80), _technicals(), _vol_surface())
        assert result.calendar_strategy == CalendarStrategy.DOUBLE_CALENDAR

    def test_r3_bullish_selects_otm_call(self) -> None:
        result = assess_calendar("SPY", _regime(3, 0.60), _technicals(rsi=60), _vol_surface())
        assert result.calendar_strategy == CalendarStrategy.OTM_CALL_CALENDAR

    def test_r3_bearish_selects_otm_put(self) -> None:
        result = assess_calendar("SPY", _regime(3, 0.60), _technicals(rsi=40), _vol_surface())
        assert result.calendar_strategy == CalendarStrategy.OTM_PUT_CALENDAR


class TestCalendarVerdict:
    def test_ideal_conditions_is_go(self) -> None:
        """R1 + backwardation + neutral RSI + good data → GO."""
        result = assess_calendar(
            "SPY", _regime(1, 0.85), _technicals(rsi=50),
            _vol_surface(front_iv=0.28, back_iv=0.18),
        )
        assert result.verdict == Verdict.GO

    def test_poor_conditions_is_no_go(self) -> None:
        """R4 low confidence + contango + extreme RSI → NO_GO or CAUTION."""
        result = assess_calendar(
            "SPY", _regime(4, 0.50), _technicals(rsi=80),
            _vol_surface(front_iv=0.10, back_iv=0.15),
        )
        assert result.verdict in (Verdict.NO_GO, Verdict.CAUTION)


class TestCalendarOutput:
    def test_output_fields_populated(self) -> None:
        result = assess_calendar("SPY", _regime(), _technicals(), _vol_surface())
        assert isinstance(result, CalendarOpportunity)
        assert result.ticker == "SPY"
        assert result.front_iv > 0
        assert result.back_iv > 0
        assert 0.0 <= result.confidence <= 1.0
        assert "SPY" in result.summary

    def test_no_trade_on_hard_stop(self) -> None:
        result = assess_calendar("SPY", _regime(), _technicals(), vol_surface=None)
        assert result.calendar_strategy == CalendarStrategy.NO_TRADE

    def test_go_has_trade_spec_with_two_expirations(self) -> None:
        """GO calendar should have front and back expiration dates."""
        result = assess_calendar(
            "SPY", _regime(1, 0.85), _technicals(rsi=50),
            _vol_surface(front_iv=0.28, back_iv=0.18),
        )
        if result.verdict != Verdict.NO_GO and result.trade_spec is not None:
            assert result.trade_spec.front_expiration is not None
            assert result.trade_spec.back_expiration is not None
            assert result.trade_spec.front_dte < result.trade_spec.back_dte
            assert result.trade_spec.iv_differential_pct is not None
            assert len(result.trade_spec.legs) == 2

    def test_no_go_has_no_trade_spec(self) -> None:
        result = assess_calendar("SPY", _regime(), _technicals(), vol_surface=None)
        assert result.trade_spec is None
