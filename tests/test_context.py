"""Tests for MarketContextService."""

from datetime import date
from unittest.mock import MagicMock

import pytest

from market_analyzer.models.black_swan import AlertLevel, BlackSwanAlert
from market_analyzer.models.context import IntermarketDashboard, IntermarketEntry, MarketContext
from market_analyzer.models.macro import MacroCalendar
from market_analyzer.models.regime import RegimeID, RegimeResult, TrendDirection
from market_analyzer.service.context import MarketContextService


def _make_alert(level: AlertLevel = AlertLevel.NORMAL, score: float = 0.1) -> BlackSwanAlert:
    return BlackSwanAlert(
        as_of_date=date(2026, 2, 23),
        alert_level=level,
        composite_score=score,
        circuit_breakers=[],
        indicators=[],
        triggered_breakers=0,
        action="Monitor" if level == AlertLevel.NORMAL else "Review",
        summary=f"Alert: {level}",
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


def _make_regime(regime_id: RegimeID = RegimeID.R1_LOW_VOL_MR) -> RegimeResult:
    return RegimeResult(
        ticker="SPY",
        regime=regime_id,
        confidence=0.85,
        regime_probabilities={1: 0.85, 2: 0.05, 3: 0.05, 4: 0.05},
        trend_direction=TrendDirection.BULLISH,
        as_of_date=date(2026, 2, 23),
        model_version="test",
    )


class TestMarketContextService:
    def test_requires_macro_service(self):
        svc = MarketContextService(black_swan_service=MagicMock())
        with pytest.raises(ValueError, match="requires a MacroService"):
            svc.assess()

    def test_requires_black_swan_service(self):
        svc = MarketContextService(macro_service=MagicMock())
        with pytest.raises(ValueError, match="requires a BlackSwanService"):
            svc.assess()

    def test_assess_normal_environment(self):
        macro_svc = MagicMock()
        macro_svc.calendar.return_value = _make_macro()
        bs_svc = MagicMock()
        bs_svc.alert.return_value = _make_alert(AlertLevel.NORMAL)
        regime_svc = MagicMock()
        regime_svc.detect.return_value = _make_regime(RegimeID.R1_LOW_VOL_MR)

        svc = MarketContextService(
            regime_service=regime_svc,
            macro_service=macro_svc,
            black_swan_service=bs_svc,
        )
        ctx = svc.assess(as_of=date(2026, 2, 23))

        assert isinstance(ctx, MarketContext)
        assert ctx.environment_label == "risk-on"
        assert ctx.trading_allowed is True
        assert ctx.position_size_factor == 1.0

    def test_assess_critical_halts_trading(self):
        macro_svc = MagicMock()
        macro_svc.calendar.return_value = _make_macro()
        bs_svc = MagicMock()
        bs_svc.alert.return_value = _make_alert(AlertLevel.CRITICAL, 0.9)

        svc = MarketContextService(macro_service=macro_svc, black_swan_service=bs_svc)
        ctx = svc.assess(as_of=date(2026, 2, 23))

        assert ctx.environment_label == "crisis"
        assert ctx.trading_allowed is False
        assert ctx.position_size_factor == 0.0

    def test_assess_elevated_reduces_size(self):
        macro_svc = MagicMock()
        macro_svc.calendar.return_value = _make_macro()
        bs_svc = MagicMock()
        bs_svc.alert.return_value = _make_alert(AlertLevel.ELEVATED, 0.3)

        svc = MarketContextService(macro_service=macro_svc, black_swan_service=bs_svc)
        ctx = svc.assess(as_of=date(2026, 2, 23))

        assert ctx.position_size_factor == 0.75

    def test_intermarket_without_regime_service(self):
        svc = MarketContextService()
        dashboard = svc.intermarket()
        assert isinstance(dashboard, IntermarketDashboard)
        assert dashboard.entries == []


class TestEnvironmentClassification:
    def test_risk_on(self):
        intermarket = IntermarketDashboard(entries=[], risk_on_count=3, risk_off_count=1)
        label = MarketContextService._classify_environment(AlertLevel.NORMAL, intermarket)
        assert label == "risk-on"

    def test_cautious_from_intermarket(self):
        intermarket = IntermarketDashboard(entries=[], risk_on_count=1, risk_off_count=3)
        label = MarketContextService._classify_environment(AlertLevel.NORMAL, intermarket)
        assert label == "cautious"

    def test_defensive_from_high_alert(self):
        intermarket = IntermarketDashboard(entries=[])
        label = MarketContextService._classify_environment(AlertLevel.HIGH, intermarket)
        assert label == "defensive"

    def test_crisis_from_critical(self):
        intermarket = IntermarketDashboard(entries=[])
        label = MarketContextService._classify_environment(AlertLevel.CRITICAL, intermarket)
        assert label == "crisis"
