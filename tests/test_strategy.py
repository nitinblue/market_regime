"""Tests for StrategyService."""

from datetime import date

import pytest

from market_analyzer.models.regime import RegimeID, RegimeResult, TrendDirection
from market_analyzer.models.strategy import (
    OptionStructureType,
    PositionSize,
    StrategyParameters,
)
from market_analyzer.service.strategy import StrategyService


def _make_regime(regime_id: RegimeID) -> RegimeResult:
    return RegimeResult(
        ticker="TEST",
        regime=regime_id,
        confidence=0.85,
        regime_probabilities={1: 0.85, 2: 0.05, 3: 0.05, 4: 0.05},
        trend_direction=TrendDirection.BULLISH,
        as_of_date=date(2026, 2, 23),
        model_version="test",
    )


class TestStrategySelection:
    def test_r1_selects_iron_condor(self, sample_ohlcv_trending):
        from market_analyzer.features.technicals import compute_technicals

        svc = StrategyService()
        regime = _make_regime(RegimeID.R1_LOW_VOL_MR)
        technicals = compute_technicals(sample_ohlcv_trending, "TEST")

        result = svc.select("TEST", regime=regime, technicals=technicals)
        assert isinstance(result, StrategyParameters)
        assert result.primary_structure.structure_type == OptionStructureType.IRON_CONDOR
        assert result.primary_structure.direction == "neutral"
        assert result.primary_structure.theta_exposure == "positive"

    def test_r2_selects_income_with_wider_wings(self, sample_ohlcv_choppy):
        from market_analyzer.features.technicals import compute_technicals

        svc = StrategyService()
        regime = _make_regime(RegimeID.R2_HIGH_VOL_MR)
        technicals = compute_technicals(sample_ohlcv_choppy, "TEST")

        result = svc.select("TEST", regime=regime, technicals=technicals)
        assert result.primary_structure.structure_type == OptionStructureType.IRON_CONDOR

    def test_r3_selects_directional(self, sample_ohlcv_trending):
        from market_analyzer.features.technicals import compute_technicals

        svc = StrategyService()
        regime = _make_regime(RegimeID.R3_LOW_VOL_TREND)
        technicals = compute_technicals(sample_ohlcv_trending, "TEST")

        result = svc.select("TEST", regime=regime, technicals=technicals)
        assert result.primary_structure.structure_type == OptionStructureType.DEBIT_SPREAD

    def test_r4_selects_risk_defined(self, sample_ohlcv_choppy):
        from market_analyzer.features.technicals import compute_technicals

        svc = StrategyService()
        regime = _make_regime(RegimeID.R4_HIGH_VOL_TREND)
        technicals = compute_technicals(sample_ohlcv_choppy, "TEST")

        result = svc.select("TEST", regime=regime, technicals=technicals)
        assert result.primary_structure.max_loss == "defined"
        assert result.primary_structure.vega_exposure == "long"

    def test_alternatives_provided(self, sample_ohlcv_trending):
        from market_analyzer.features.technicals import compute_technicals

        svc = StrategyService()
        regime = _make_regime(RegimeID.R1_LOW_VOL_MR)
        technicals = compute_technicals(sample_ohlcv_trending, "TEST")

        result = svc.select("TEST", regime=regime, technicals=technicals)
        assert len(result.alternative_structures) >= 1


class TestPositionSizing:
    def test_basic_sizing(self, sample_ohlcv_trending):
        from market_analyzer.features.technicals import compute_technicals

        svc = StrategyService(account_size=50_000, max_position_pct=0.05)
        regime = _make_regime(RegimeID.R1_LOW_VOL_MR)
        technicals = compute_technicals(sample_ohlcv_trending, "TEST")

        params = svc.select("TEST", regime=regime, technicals=technicals)
        size = svc.size(params, current_price=100.0)

        assert isinstance(size, PositionSize)
        assert size.account_size == 50_000
        assert size.max_risk_dollars == 2_500  # 5% of 50k
        assert size.suggested_contracts >= 1
        assert size.max_contracts >= size.suggested_contracts

    def test_custom_account_size(self, sample_ohlcv_trending):
        from market_analyzer.features.technicals import compute_technicals

        svc = StrategyService(account_size=50_000)
        regime = _make_regime(RegimeID.R1_LOW_VOL_MR)
        technicals = compute_technicals(sample_ohlcv_trending, "TEST")

        params = svc.select("TEST", regime=regime, technicals=technicals)
        size = svc.size(params, current_price=100.0, account_size=200_000)

        assert size.account_size == 200_000

    def test_wing_width_suggestion(self, sample_ohlcv_trending):
        from market_analyzer.features.technicals import compute_technicals

        svc = StrategyService()
        regime = _make_regime(RegimeID.R1_LOW_VOL_MR)
        technicals = compute_technicals(sample_ohlcv_trending, "TEST")

        result = svc.select("TEST", regime=regime, technicals=technicals)
        assert result.wing_width_suggestion in ("5-wide", "10-wide", "15-wide", "20-wide")
