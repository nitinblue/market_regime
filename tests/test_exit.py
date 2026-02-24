"""Tests for ExitService."""

from datetime import date

import pytest

from market_analyzer.models.exit_plan import ExitPlan, ExitReason
from market_analyzer.models.regime import RegimeID, RegimeResult, TrendDirection
from market_analyzer.models.strategy import (
    OptionStructure,
    OptionStructureType,
    StrategyParameters,
)
from market_analyzer.service.exit import ExitService


def _make_regime(regime_id: RegimeID = RegimeID.R1_LOW_VOL_MR) -> RegimeResult:
    return RegimeResult(
        ticker="TEST",
        regime=regime_id,
        confidence=0.85,
        regime_probabilities={1: 0.85, 2: 0.05, 3: 0.05, 4: 0.05},
        trend_direction=TrendDirection.BULLISH,
        as_of_date=date(2026, 2, 23),
        model_version="test",
    )


def _make_strategy(
    structure: OptionStructureType = OptionStructureType.IRON_CONDOR,
) -> StrategyParameters:
    return StrategyParameters(
        ticker="TEST",
        as_of_date=date(2026, 2, 23),
        primary_structure=OptionStructure(
            structure_type=structure,
            direction="neutral",
            max_loss="defined",
            theta_exposure="positive",
            vega_exposure="short",
            rationale="Test strategy",
        ),
        regime_rationale="Test rationale",
        wing_width_suggestion="5-wide",
    )


class TestExitService:
    def test_plan_basic(self, sample_ohlcv_trending):
        from market_analyzer.features.technicals import compute_technicals

        svc = ExitService()
        regime = _make_regime()
        technicals = compute_technicals(sample_ohlcv_trending, "TEST")
        strategy = _make_strategy()

        plan = svc.plan(
            "TEST",
            strategy=strategy,
            entry_price=100.0,
            regime=regime,
            technicals=technicals,
        )

        assert isinstance(plan, ExitPlan)
        assert plan.ticker == "TEST"
        assert plan.entry_price == 100.0
        assert plan.strategy_type == "iron_condor"
        assert len(plan.profit_targets) >= 1
        assert plan.stop_loss is not None
        assert plan.trailing_stop is not None

    def test_plan_has_profit_targets(self, sample_ohlcv_trending):
        from market_analyzer.features.technicals import compute_technicals

        svc = ExitService()
        regime = _make_regime()
        technicals = compute_technicals(sample_ohlcv_trending, "TEST")
        strategy = _make_strategy()

        plan = svc.plan(
            "TEST", strategy=strategy, entry_price=100.0,
            regime=regime, technicals=technicals,
        )

        for target in plan.profit_targets:
            assert target.reason == ExitReason.PROFIT_TARGET

    def test_plan_has_adjustments(self, sample_ohlcv_trending):
        from market_analyzer.features.technicals import compute_technicals

        svc = ExitService()
        regime = _make_regime()
        technicals = compute_technicals(sample_ohlcv_trending, "TEST")
        strategy = _make_strategy(OptionStructureType.IRON_CONDOR)

        plan = svc.plan(
            "TEST", strategy=strategy, entry_price=100.0,
            regime=regime, technicals=technicals,
        )

        # Iron condor should have roll and widen adjustments
        assert len(plan.adjustments) >= 2

    def test_plan_with_debit_spread(self, sample_ohlcv_trending):
        from market_analyzer.features.technicals import compute_technicals

        svc = ExitService()
        regime = _make_regime(RegimeID.R3_LOW_VOL_TREND)
        technicals = compute_technicals(sample_ohlcv_trending, "TEST")
        strategy = _make_strategy(OptionStructureType.DEBIT_SPREAD)

        plan = svc.plan(
            "TEST", strategy=strategy, entry_price=100.0,
            regime=regime, technicals=technicals,
        )

        assert plan.strategy_type == "debit_spread"
        # Should have partial profit adjustment
        partial = [a for a in plan.adjustments if "partial" in a.action.lower() or "50%" in a.condition]
        assert len(partial) >= 1

    def test_regime_change_action_income(self, sample_ohlcv_trending):
        from market_analyzer.features.technicals import compute_technicals

        svc = ExitService()
        regime = _make_regime(RegimeID.R1_LOW_VOL_MR)
        technicals = compute_technicals(sample_ohlcv_trending, "TEST")
        strategy = _make_strategy()

        plan = svc.plan(
            "TEST", strategy=strategy, entry_price=100.0,
            regime=regime, technicals=technicals,
        )

        assert "R3/R4" in plan.regime_change_action or "directional" in plan.regime_change_action.lower()

    def test_income_strategy_has_theta_exit(self, sample_ohlcv_trending):
        from market_analyzer.features.technicals import compute_technicals

        svc = ExitService()
        regime = _make_regime()
        technicals = compute_technicals(sample_ohlcv_trending, "TEST")
        strategy = _make_strategy(OptionStructureType.IRON_CONDOR)

        plan = svc.plan(
            "TEST", strategy=strategy, entry_price=100.0,
            regime=regime, technicals=technicals,
        )

        assert plan.theta_decay_exit_pct is not None
        assert plan.dte_exit_threshold is not None
