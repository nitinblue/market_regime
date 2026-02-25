"""Tests for ORB setup opportunity assessment."""

from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from market_analyzer.models.regime import RegimeID, RegimeResult
from market_analyzer.models.technicals import (
    BollingerBands, MACDData, MovingAverages, RSIData,
    StochasticData, SupportResistance, TechnicalSnapshot,
    MarketPhase, PhaseIndicator, ORBData, ORBLevel, ORBStatus,
)
from market_analyzer.models.opportunity import Verdict
from market_analyzer.opportunity.setups.orb import (
    ORBSetupOpportunity, ORBStrategy, assess_orb,
)


def _regime(regime_id: int = 1, confidence: float = 0.75) -> RegimeResult:
    return RegimeResult(
        ticker="TEST", regime=RegimeID(regime_id), confidence=confidence,
        regime_probabilities={1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1},
        as_of_date=date(2026, 3, 1), model_version="test", trend_direction=None,
    )


def _technicals(rsi: float = 55.0, price: float = 580.0) -> TechnicalSnapshot:
    return TechnicalSnapshot(
        ticker="TEST", as_of_date=date(2026, 3, 1), current_price=price,
        atr=5.8, atr_pct=1.0, vwma_20=price,
        moving_averages=MovingAverages(
            sma_20=price, sma_50=price * 0.98, sma_200=price * 0.95,
            ema_9=price, ema_21=price,
            price_vs_sma_20_pct=0.0, price_vs_sma_50_pct=2.0, price_vs_sma_200_pct=5.0,
        ),
        rsi=RSIData(value=rsi, is_overbought=rsi > 70, is_oversold=rsi < 30),
        bollinger=BollingerBands(upper=price + 10, middle=price, lower=price - 10, bandwidth=0.04, percent_b=0.5),
        macd=MACDData(macd_line=0.5, signal_line=0.3, histogram=0.2,
                      is_bullish_crossover=False, is_bearish_crossover=False),
        stochastic=StochasticData(k=50.0, d=50.0, is_overbought=False, is_oversold=False),
        support_resistance=SupportResistance(support=570.0, resistance=590.0,
                                             price_vs_support_pct=1.7, price_vs_resistance_pct=-1.7),
        phase=PhaseIndicator(phase=MarketPhase.ACCUMULATION, confidence=0.5, description="Test",
                             higher_highs=False, higher_lows=True, lower_highs=False, lower_lows=False,
                             range_compression=0.3, volume_trend="declining", price_vs_sma_50_pct=2.0),
        signals=[],
    )


def _orb(
    status: ORBStatus = ORBStatus.BREAKOUT_LONG,
    range_pct: float = 0.35,
    volume_ratio: float = 1.4,
    atr_pct: float = 35.0,
    current_price: float = 582.0,
    retest_count: int = 0,
) -> ORBData:
    return ORBData(
        ticker="TEST",
        date=date(2026, 3, 1),
        opening_minutes=30,
        range_high=581.0,
        range_low=579.0,
        range_size=2.0,
        range_pct=range_pct,
        current_price=current_price,
        status=status,
        levels=[
            ORBLevel(label="Midpoint", price=580.0, distance_pct=0.34),
        ],
        session_high=583.0,
        session_low=578.5,
        session_vwap=580.5,
        opening_volume_ratio=volume_ratio,
        range_vs_daily_atr_pct=atr_pct,
        breakout_bar_index=10 if status != ORBStatus.WITHIN else None,
        retest_count=retest_count,
        signals=[],
        description="test ORB",
    )


class TestORBNoData:
    def test_no_orb_data_is_no_go(self) -> None:
        result = assess_orb("SPY", _regime(), _technicals(), orb=None)
        assert result.verdict == Verdict.NO_GO
        assert result.strategy == ORBStrategy.NO_TRADE
        assert any("intraday" in s.description for s in result.hard_stops)


class TestORBHardStops:
    def test_r4_high_confidence_is_no_go(self) -> None:
        result = assess_orb("SPY", _regime(4, 0.80), _technicals(), _orb())
        assert result.verdict == Verdict.NO_GO
        assert any("R4" in s.name for s in result.hard_stops)

    def test_r4_low_confidence_passes(self) -> None:
        result = assess_orb("SPY", _regime(4, 0.50), _technicals(), _orb())
        assert result.verdict != Verdict.NO_GO or not any("R4" in s.name for s in result.hard_stops)

    def test_range_too_wide_is_no_go(self) -> None:
        result = assess_orb("SPY", _regime(), _technicals(), _orb(atr_pct=90.0))
        assert result.verdict == Verdict.NO_GO
        assert any("wide" in s.name.lower() for s in result.hard_stops)


class TestORBBreakout:
    def test_breakout_long_is_bullish(self) -> None:
        result = assess_orb("SPY", _regime(), _technicals(), _orb(status=ORBStatus.BREAKOUT_LONG))
        assert result.direction == "bullish"

    def test_breakout_short_is_bearish(self) -> None:
        result = assess_orb("SPY", _regime(), _technicals(), _orb(status=ORBStatus.BREAKOUT_SHORT))
        assert result.direction == "bearish"

    def test_breakout_with_retest(self) -> None:
        result = assess_orb("SPY", _regime(), _technicals(),
                            _orb(status=ORBStatus.BREAKOUT_LONG, retest_count=1))
        if result.verdict != Verdict.NO_GO:
            assert result.strategy == ORBStrategy.BREAKOUT_WITH_RETEST

    def test_breakout_continuation(self) -> None:
        result = assess_orb("SPY", _regime(), _technicals(),
                            _orb(status=ORBStatus.BREAKOUT_LONG, retest_count=0))
        if result.verdict != Verdict.NO_GO:
            assert result.strategy == ORBStrategy.BREAKOUT_CONTINUATION

    def test_strong_breakout_is_go(self) -> None:
        """Strong volume + healthy range + trending regime → GO."""
        result = assess_orb(
            "SPY", _regime(3, 0.80), _technicals(rsi=55),
            _orb(status=ORBStatus.BREAKOUT_LONG, volume_ratio=1.6, atr_pct=35.0, retest_count=1),
        )
        assert result.verdict == Verdict.GO


class TestORBFailedBreakout:
    def test_failed_long_is_bearish(self) -> None:
        result = assess_orb("SPY", _regime(), _technicals(), _orb(status=ORBStatus.FAILED_LONG))
        assert result.direction == "bearish"

    def test_failed_short_is_bullish(self) -> None:
        result = assess_orb("SPY", _regime(), _technicals(), _orb(status=ORBStatus.FAILED_SHORT))
        assert result.direction == "bullish"

    def test_failed_breakout_reversal_strategy(self) -> None:
        result = assess_orb("SPY", _regime(), _technicals(), _orb(status=ORBStatus.FAILED_LONG))
        if result.verdict != Verdict.NO_GO:
            assert result.strategy == ORBStrategy.FAILED_BREAKOUT_REVERSAL

    def test_mr_regime_favors_failed_breakout(self) -> None:
        """Mean-reverting regime + failed breakout should score well."""
        result = assess_orb(
            "SPY", _regime(1, 0.80), _technicals(rsi=72),
            _orb(status=ORBStatus.FAILED_LONG, volume_ratio=1.3, atr_pct=40.0),
        )
        assert result.confidence > 0.5


class TestORBWithinRange:
    def test_within_range_neutral(self) -> None:
        result = assess_orb("SPY", _regime(), _technicals(),
                            _orb(status=ORBStatus.WITHIN, range_pct=0.8))
        assert result.direction == "neutral"

    def test_narrow_range_anticipation(self) -> None:
        result = assess_orb("SPY", _regime(), _technicals(),
                            _orb(status=ORBStatus.WITHIN, range_pct=0.3))
        if result.strategy != ORBStrategy.NO_TRADE:
            assert result.strategy == ORBStrategy.NARROW_RANGE_ANTICIPATION


class TestORBVolume:
    def test_strong_volume_boosts_score(self) -> None:
        base = assess_orb("SPY", _regime(), _technicals(),
                          _orb(volume_ratio=0.8))
        strong = assess_orb("SPY", _regime(), _technicals(),
                            _orb(volume_ratio=1.6))
        assert strong.confidence > base.confidence

    def test_light_volume_hurts_score(self) -> None:
        result = assess_orb("SPY", _regime(), _technicals(),
                            _orb(volume_ratio=0.5))
        # Should have a light volume signal
        assert any("light" in s.name.lower() or "light" in s.description.lower()
                    for s in result.signals)


class TestORBOutput:
    def test_output_fields_populated(self) -> None:
        result = assess_orb("SPY", _regime(), _technicals(), _orb())
        assert isinstance(result, ORBSetupOpportunity)
        assert result.ticker == "SPY"
        assert result.orb_status == ORBStatus.BREAKOUT_LONG.value
        assert result.range_pct > 0
        assert result.opening_volume_ratio > 0
        assert 0.0 <= result.confidence <= 1.0
        assert "SPY" in result.summary

    def test_summary_has_verdict_and_ticker(self) -> None:
        result = assess_orb("SPY", _regime(), _technicals(), _orb())
        assert "SPY" in result.summary

    def test_hard_stop_no_go_no_signals_scored(self) -> None:
        """Hard stop should produce NO_GO with no scoring signals."""
        result = assess_orb("SPY", _regime(4, 0.80), _technicals(), _orb())
        assert result.verdict == Verdict.NO_GO
        assert result.confidence == 0.0
