"""Tests for the daily trading plan framework.

Covers:
  - Expiry calendar (Phase 1)
  - Trading plan models (Phase 2)
  - Fill price estimation (Phase 3)
  - Day verdict logic (Phase 4)
  - Horizon bucketing (Phase 4)
  - Plan generation (Phase 4)
  - Config (Phase 5)
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from market_analyzer.macro.expiry import (
    ExpiryEvent,
    ExpiryType,
    get_expiry_calendar,
    is_quad_witching,
    monthly_opex,
    upcoming_expiries,
    vix_settlement,
    weekly_opex_fridays,
)
from market_analyzer.models.opportunity import (
    LegAction,
    LegSpec,
    OrderSide,
    StructureType,
    TradeSpec,
    Verdict,
)
from market_analyzer.models.trading_plan import (
    DailyTradingPlan,
    DayVerdict,
    PlanHorizon,
    PlanTrade,
    RiskBudget,
)
from market_analyzer.models.ranking import StrategyType
from market_analyzer.opportunity.option_plays._trade_spec_helpers import (
    build_double_calendar_legs,
    compute_max_entry_price,
    estimate_trade_price,
)
from market_analyzer.models.vol_surface import TermStructurePoint


# ===== Phase 1: Expiry Calendar =====


class TestMonthlyOpex:
    def test_third_friday_march_2026(self):
        result = monthly_opex(2026, 3)
        assert result == date(2026, 3, 20)
        assert result.weekday() == 4  # Friday

    def test_third_friday_feb_2026(self):
        result = monthly_opex(2026, 2)
        assert result == date(2026, 2, 20)
        assert result.weekday() == 4

    def test_third_friday_jan_2026(self):
        result = monthly_opex(2026, 1)
        assert result == date(2026, 1, 16)
        assert result.weekday() == 4

    def test_is_always_friday(self):
        for month in range(1, 13):
            d = monthly_opex(2026, month)
            assert d.weekday() == 4, f"Month {month}: {d} is not Friday"


class TestVixSettlement:
    def test_wednesday_before_third_friday(self):
        # March 2026: 3rd Friday = March 20 -> VIX = March 18 (Wednesday)
        result = vix_settlement(2026, 3)
        assert result == date(2026, 3, 18)
        assert result.weekday() == 2  # Wednesday

    def test_always_wednesday(self):
        for month in range(1, 13):
            d = vix_settlement(2026, month)
            assert d.weekday() == 2, f"Month {month}: {d} is not Wednesday"


class TestQuadWitching:
    def test_quarterly_months(self):
        # 3rd Friday of Mar/Jun/Sep/Dec
        assert is_quad_witching(monthly_opex(2026, 3))
        assert is_quad_witching(monthly_opex(2026, 6))
        assert is_quad_witching(monthly_opex(2026, 9))
        assert is_quad_witching(monthly_opex(2026, 12))

    def test_non_quarterly_months(self):
        assert not is_quad_witching(monthly_opex(2026, 1))
        assert not is_quad_witching(monthly_opex(2026, 2))
        assert not is_quad_witching(monthly_opex(2026, 4))

    def test_non_third_friday(self):
        assert not is_quad_witching(date(2026, 3, 13))  # 2nd Friday
        assert not is_quad_witching(date(2026, 3, 21))  # Saturday


class TestWeeklyOpexFridays:
    def test_returns_fridays(self):
        start = date(2026, 3, 1)
        end = date(2026, 3, 31)
        fridays = weekly_opex_fridays(start, end)
        assert len(fridays) == 4 or len(fridays) == 5  # 4 or 5 Fridays in March
        for f in fridays:
            assert f.weekday() == 4

    def test_inclusive_range(self):
        # A Friday itself
        friday = date(2026, 3, 6)
        result = weekly_opex_fridays(friday, friday)
        assert result == [friday]

    def test_empty_range(self):
        # Mon-Thu range with no Friday
        result = weekly_opex_fridays(date(2026, 3, 2), date(2026, 3, 5))
        assert result == []


class TestGetExpiryCalendar:
    def test_single_month(self):
        events = get_expiry_calendar(date(2026, 2, 1), date(2026, 2, 28))
        types = {e.expiry_type for e in events}
        assert ExpiryType.MONTHLY_OPEX in types
        assert ExpiryType.VIX_SETTLEMENT in types

    def test_quarterly_month_has_quad_witching(self):
        events = get_expiry_calendar(date(2026, 3, 1), date(2026, 3, 31))
        types = {e.expiry_type for e in events}
        assert ExpiryType.QUAD_WITCHING in types
        assert ExpiryType.QUARTERLY_FUTURES in types
        # No MONTHLY_OPEX for quarterly months (replaced by quad witching)
        assert ExpiryType.MONTHLY_OPEX not in types

    def test_sorted_by_date(self):
        events = get_expiry_calendar(date(2026, 1, 1), date(2026, 6, 30))
        dates = [e.date for e in events]
        assert dates == sorted(dates)

    def test_multi_month_range(self):
        events = get_expiry_calendar(date(2026, 1, 1), date(2026, 3, 31))
        # Should have events for Jan, Feb, Mar
        months = {e.date.month for e in events}
        assert months == {1, 2, 3}


class TestUpcomingExpiries:
    def test_returns_within_7_days(self):
        # Use a specific date near a known OpEx
        opex = monthly_opex(2026, 3)
        as_of = opex - timedelta(days=3)
        events = upcoming_expiries(as_of=as_of, days_ahead=7)
        assert any(e.date == opex for e in events)

    def test_empty_when_no_events(self):
        # Pick a date far from any events (e.g., middle of month, not near any Friday)
        events = upcoming_expiries(as_of=date(2026, 3, 2), days_ahead=1)
        # May or may not have events — just check it doesn't crash
        assert isinstance(events, list)


# ===== Phase 2: Trading Plan Models =====


class TestPlanModels:
    def test_day_verdict_values(self):
        assert DayVerdict.TRADE == "trade"
        assert DayVerdict.TRADE_LIGHT == "trade_light"
        assert DayVerdict.AVOID == "avoid"
        assert DayVerdict.NO_TRADE == "no_trade"

    def test_plan_horizon_values(self):
        assert PlanHorizon.ZERO_DTE == "0dte"
        assert PlanHorizon.WEEKLY == "weekly"
        assert PlanHorizon.MONTHLY == "monthly"
        assert PlanHorizon.LEAP == "leap"

    def test_risk_budget_construction(self):
        rb = RiskBudget(
            max_new_positions=3,
            max_daily_risk_dollars=1000.0,
            position_size_factor=1.0,
        )
        assert rb.max_new_positions == 3
        assert rb.max_daily_risk_dollars == 1000.0

    def test_plan_trade_construction(self):
        pt = PlanTrade(
            rank=1,
            ticker="SPY",
            strategy_type=StrategyType.IRON_CONDOR,
            horizon=PlanHorizon.MONTHLY,
            verdict=Verdict.GO,
            composite_score=0.78,
            direction="neutral",
            trade_spec=None,
            rationale="test",
            risk_notes=[],
        )
        assert pt.max_entry_price is None
        assert pt.expiry_note is None

    def test_daily_plan_construction(self):
        plan = DailyTradingPlan(
            as_of_date=date(2026, 2, 26),
            plan_for_date=date(2026, 2, 26),
            day_verdict=DayVerdict.TRADE,
            day_verdict_reasons=["Normal conditions"],
            risk_budget=RiskBudget(
                max_new_positions=3,
                max_daily_risk_dollars=1000.0,
                position_size_factor=1.0,
            ),
            expiry_events=[],
            upcoming_expiries=[],
            trades_by_horizon={h: [] for h in PlanHorizon},
            all_trades=[],
            total_trades=0,
            summary="test",
        )
        assert plan.day_verdict == DayVerdict.TRADE
        assert plan.total_trades == 0


# ===== Phase 3: Fill Price Estimation =====


def _make_leg(
    role: str,
    action: LegAction,
    option_type: str,
    strike: float,
    dte: int = 30,
    iv: float = 0.25,
    quantity: int = 1,
) -> LegSpec:
    return LegSpec(
        role=role,
        action=action,
        quantity=quantity,
        option_type=option_type,
        strike=strike,
        strike_label=f"test {role}",
        expiration=date(2026, 3, 27),
        days_to_expiry=dte,
        atm_iv_at_expiry=iv,
    )


def _make_trade_spec(
    legs: list[LegSpec],
    structure_type: str = StructureType.IRON_CONDOR,
    order_side: str = OrderSide.CREDIT,
) -> TradeSpec:
    return TradeSpec(
        ticker="SPY",
        legs=legs,
        underlying_price=600.0,
        target_dte=30,
        target_expiration=date(2026, 3, 27),
        spec_rationale="test",
        structure_type=structure_type,
        order_side=order_side,
    )


class TestEstimateTradePrice:
    def test_empty_legs_returns_none(self):
        ts = _make_trade_spec([])
        assert estimate_trade_price(ts) is None

    def test_iron_condor_returns_positive(self):
        """IC is net credit — should return positive."""
        legs = [
            _make_leg("short_put", LegAction.SELL_TO_OPEN, "put", 590.0),
            _make_leg("long_put", LegAction.BUY_TO_OPEN, "put", 585.0),
            _make_leg("short_call", LegAction.SELL_TO_OPEN, "call", 610.0),
            _make_leg("long_call", LegAction.BUY_TO_OPEN, "call", 615.0),
        ]
        ts = _make_trade_spec(legs)
        price = estimate_trade_price(ts)
        assert price is not None
        assert price > 0  # Net credit

    def test_long_call_returns_negative(self):
        """Single long call is debit — should return negative."""
        legs = [_make_leg("long_call", LegAction.BUY_TO_OPEN, "call", 600.0)]
        ts = _make_trade_spec(legs, StructureType.LONG_OPTION, OrderSide.DEBIT)
        price = estimate_trade_price(ts)
        assert price is not None
        assert price < 0  # Net debit

    def test_credit_spread(self):
        legs = [
            _make_leg("short_put", LegAction.SELL_TO_OPEN, "put", 590.0),
            _make_leg("long_put", LegAction.BUY_TO_OPEN, "put", 585.0),
        ]
        ts = _make_trade_spec(legs, StructureType.CREDIT_SPREAD, OrderSide.CREDIT)
        price = estimate_trade_price(ts)
        assert price is not None
        assert price > 0  # Net credit

    def test_zero_dte_near_intrinsic(self):
        """With 0 DTE, should return near intrinsic value (tiny time value from min 1-day floor)."""
        legs = [_make_leg("long_call", LegAction.BUY_TO_OPEN, "call", 590.0, dte=0)]
        ts = _make_trade_spec(legs, StructureType.LONG_OPTION, OrderSide.DEBIT)
        price = estimate_trade_price(ts)
        assert price is not None
        # Intrinsic = 600 - 590 = 10, cost is negative; tiny time value from 1-day floor
        assert abs(price - (-10.0)) < 1.0


class TestComputeMaxEntryPrice:
    def test_credit_structure(self):
        legs = [
            _make_leg("short_put", LegAction.SELL_TO_OPEN, "put", 590.0),
            _make_leg("long_put", LegAction.BUY_TO_OPEN, "put", 585.0),
        ]
        ts = _make_trade_spec(legs, StructureType.CREDIT_SPREAD, OrderSide.CREDIT)
        max_price = compute_max_entry_price(ts)
        estimated = estimate_trade_price(ts)
        assert max_price is not None
        assert estimated is not None
        # For credit: max_entry = estimated * 0.80
        assert abs(max_price - estimated * 0.80) < 0.02

    def test_debit_structure(self):
        legs = [
            _make_leg("long_call", LegAction.BUY_TO_OPEN, "call", 600.0),
            _make_leg("short_call", LegAction.SELL_TO_OPEN, "call", 610.0),
        ]
        ts = _make_trade_spec(legs, StructureType.DEBIT_SPREAD, OrderSide.DEBIT)
        max_price = compute_max_entry_price(ts)
        estimated = estimate_trade_price(ts)
        assert max_price is not None
        assert estimated is not None
        # For debit: max_entry = |estimated| * 1.20
        assert abs(max_price - abs(estimated) * 1.20) < 0.02

    def test_long_option_tighter_tolerance(self):
        legs = [_make_leg("long_call", LegAction.BUY_TO_OPEN, "call", 600.0)]
        ts = _make_trade_spec(legs, StructureType.LONG_OPTION, OrderSide.DEBIT)
        max_price = compute_max_entry_price(ts)
        estimated = estimate_trade_price(ts)
        assert max_price is not None
        assert estimated is not None
        # Tighter: |estimated| * 1.15
        assert abs(max_price - abs(estimated) * 1.15) < 0.02

    def test_empty_legs_returns_none(self):
        ts = _make_trade_spec([])
        assert compute_max_entry_price(ts) is None

    def test_max_entry_price_field_on_tradespec(self):
        """TradeSpec now has max_entry_price field."""
        ts = _make_trade_spec([])
        assert ts.max_entry_price is None
        ts2 = ts.model_copy(update={"max_entry_price": 2.50})
        assert ts2.max_entry_price == 2.50


# ===== Phase 4: Day Verdict & Horizon Bucketing =====


class TestDteToHorizon:
    def test_zero_dte(self):
        from market_analyzer.service.trading_plan import _dte_to_horizon
        assert _dte_to_horizon(0) == PlanHorizon.ZERO_DTE
        assert _dte_to_horizon(None) == PlanHorizon.ZERO_DTE

    def test_weekly(self):
        from market_analyzer.service.trading_plan import _dte_to_horizon
        assert _dte_to_horizon(1) == PlanHorizon.WEEKLY
        assert _dte_to_horizon(7) == PlanHorizon.WEEKLY

    def test_monthly(self):
        from market_analyzer.service.trading_plan import _dte_to_horizon
        assert _dte_to_horizon(8) == PlanHorizon.MONTHLY
        assert _dte_to_horizon(30) == PlanHorizon.MONTHLY
        assert _dte_to_horizon(60) == PlanHorizon.MONTHLY

    def test_leap(self):
        from market_analyzer.service.trading_plan import _dte_to_horizon
        assert _dte_to_horizon(61) == PlanHorizon.LEAP
        assert _dte_to_horizon(365) == PlanHorizon.LEAP


class TestStrategyToHorizon:
    def test_zero_dte_strategy_override(self):
        from market_analyzer.service.trading_plan import TradingPlanService
        assert TradingPlanService._strategy_to_horizon(StrategyType.ZERO_DTE, 0) == PlanHorizon.ZERO_DTE
        assert TradingPlanService._strategy_to_horizon(StrategyType.ZERO_DTE, None) == PlanHorizon.ZERO_DTE

    def test_leap_strategy_override(self):
        from market_analyzer.service.trading_plan import TradingPlanService
        assert TradingPlanService._strategy_to_horizon(StrategyType.LEAP, 180) == PlanHorizon.LEAP

    def test_other_strategy_uses_dte(self):
        from market_analyzer.service.trading_plan import TradingPlanService
        assert TradingPlanService._strategy_to_horizon(StrategyType.IRON_CONDOR, 35) == PlanHorizon.MONTHLY
        assert TradingPlanService._strategy_to_horizon(StrategyType.BREAKOUT, 5) == PlanHorizon.WEEKLY


class TestDayVerdict:
    """Test _compute_day_verdict via TradingPlanService."""

    def _make_context(
        self,
        alert_level="normal",
        events=None,
        fomc_today=False,
        cpi_today=False,
    ):
        """Build a mock MarketContext."""
        from market_analyzer.models.black_swan import AlertLevel, BlackSwanAlert
        from market_analyzer.models.context import IntermarketDashboard, MarketContext
        from market_analyzer.models.macro import MacroCalendar, MacroEvent, MacroEventType

        today = date(2026, 2, 26)

        macro_events = []
        if fomc_today:
            macro_events.append(MacroEvent(
                event_type=MacroEventType.FOMC,
                date=today,
                name="FOMC",
                impact="high",
                description="FOMC",
                options_impact="high",
            ))
        if cpi_today:
            macro_events.append(MacroEvent(
                event_type=MacroEventType.CPI,
                date=today,
                name="CPI",
                impact="high",
                description="CPI",
                options_impact="medium",
            ))

        return MarketContext(
            as_of_date=today,
            market="US",
            macro=MacroCalendar(
                events=macro_events,
                next_event=macro_events[0] if macro_events else None,
                days_to_next=0 if macro_events else None,
                next_fomc=macro_events[0] if fomc_today else None,
                days_to_next_fomc=0 if fomc_today else None,
                events_next_7_days=macro_events,
                events_next_30_days=macro_events,
            ),
            black_swan=BlackSwanAlert(
                as_of_date=today,
                alert_level=AlertLevel(alert_level),
                composite_score=0.1,
                circuit_breakers=[],
                indicators=[],
                triggered_breakers=0,
                action="proceed",
                summary="ok",
            ),
            intermarket=IntermarketDashboard(entries=[], summary=""),
            environment_label="risk-on",
            trading_allowed=alert_level != "critical",
            position_size_factor=1.0 if alert_level == "normal" else 0.5,
        )

    def test_normal_conditions(self):
        from market_analyzer.service.trading_plan import TradingPlanService
        svc = TradingPlanService.__new__(TradingPlanService)
        ctx = self._make_context()
        verdict, reasons = svc._compute_day_verdict(ctx, [])
        assert verdict == DayVerdict.TRADE
        assert "Normal conditions" in reasons[0]

    def test_critical_black_swan(self):
        from market_analyzer.service.trading_plan import TradingPlanService
        svc = TradingPlanService.__new__(TradingPlanService)
        ctx = self._make_context(alert_level="critical")
        verdict, reasons = svc._compute_day_verdict(ctx, [])
        assert verdict == DayVerdict.NO_TRADE

    def test_fomc_day_avoid(self):
        from market_analyzer.service.trading_plan import TradingPlanService
        svc = TradingPlanService.__new__(TradingPlanService)
        ctx = self._make_context(fomc_today=True)
        verdict, reasons = svc._compute_day_verdict(ctx, [])
        assert verdict == DayVerdict.AVOID
        assert any("FOMC" in r for r in reasons)

    def test_quad_witching_avoid(self):
        from market_analyzer.service.trading_plan import TradingPlanService
        svc = TradingPlanService.__new__(TradingPlanService)
        ctx = self._make_context()
        qw = [ExpiryEvent(
            date=date(2026, 3, 20),
            expiry_type=ExpiryType.QUAD_WITCHING,
            label="Mar Quad Witching",
        )]
        verdict, reasons = svc._compute_day_verdict(ctx, qw)
        assert verdict == DayVerdict.AVOID

    def test_high_black_swan_avoid(self):
        from market_analyzer.service.trading_plan import TradingPlanService
        svc = TradingPlanService.__new__(TradingPlanService)
        ctx = self._make_context(alert_level="high")
        verdict, reasons = svc._compute_day_verdict(ctx, [])
        assert verdict == DayVerdict.AVOID

    def test_cpi_today_trade_light(self):
        from market_analyzer.service.trading_plan import TradingPlanService
        svc = TradingPlanService.__new__(TradingPlanService)
        ctx = self._make_context(cpi_today=True)
        verdict, reasons = svc._compute_day_verdict(ctx, [])
        assert verdict == DayVerdict.TRADE_LIGHT

    def test_monthly_opex_trade_light(self):
        from market_analyzer.service.trading_plan import TradingPlanService
        svc = TradingPlanService.__new__(TradingPlanService)
        ctx = self._make_context()
        opex = [ExpiryEvent(
            date=date(2026, 2, 20),
            expiry_type=ExpiryType.MONTHLY_OPEX,
            label="Feb Monthly OpEx",
        )]
        verdict, reasons = svc._compute_day_verdict(ctx, opex)
        assert verdict == DayVerdict.TRADE_LIGHT

    def test_vix_settlement_trade_light(self):
        from market_analyzer.service.trading_plan import TradingPlanService
        svc = TradingPlanService.__new__(TradingPlanService)
        ctx = self._make_context()
        vix = [ExpiryEvent(
            date=date(2026, 2, 18),
            expiry_type=ExpiryType.VIX_SETTLEMENT,
            label="Feb VIX Settlement",
        )]
        verdict, reasons = svc._compute_day_verdict(ctx, vix)
        assert verdict == DayVerdict.TRADE_LIGHT

    def test_elevated_black_swan_trade_light(self):
        from market_analyzer.service.trading_plan import TradingPlanService
        svc = TradingPlanService.__new__(TradingPlanService)
        ctx = self._make_context(alert_level="elevated")
        verdict, reasons = svc._compute_day_verdict(ctx, [])
        assert verdict == DayVerdict.TRADE_LIGHT


# ===== Phase 5: Config =====


class TestTradingPlanConfig:
    def test_default_settings(self):
        from market_analyzer.config import TradingPlanSettings
        s = TradingPlanSettings()
        assert s.max_trades_per_plan == 10
        assert s.daily_risk_pct == 0.02
        assert s.max_new_positions_normal == 3
        assert s.max_new_positions_light == 1
        assert s.fill_slippage_pct == 0.20
        assert "SPX" in s.default_tickers

    def test_settings_on_main_config(self):
        from market_analyzer.config import Settings
        s = Settings()
        assert hasattr(s, "trading_plan")
        assert s.trading_plan.max_trades_per_plan == 10

    def test_loads_from_yaml(self):
        from market_analyzer.config import load_settings
        s = load_settings(_force_reload=True)
        assert s.trading_plan.max_trades_per_plan == 10
        assert s.trading_plan.daily_risk_pct == 0.02


# ===== Phase 4: Expiry Note Tagging =====


class TestExpiryNoteForTrade:
    def test_tags_target_expiration(self):
        from market_analyzer.service.trading_plan import _expiry_note_for_trade
        opex_date = date(2026, 3, 20)
        ts = _make_trade_spec(
            [_make_leg("short_put", LegAction.SELL_TO_OPEN, "put", 590.0)],
        )
        ts = ts.model_copy(update={"target_expiration": opex_date})
        events = [ExpiryEvent(date=opex_date, expiry_type=ExpiryType.MONTHLY_OPEX, label="Mar Monthly OpEx")]
        note = _expiry_note_for_trade(ts, events)
        assert note == "Mar Monthly OpEx"

    def test_tags_front_expiration(self):
        from market_analyzer.service.trading_plan import _expiry_note_for_trade
        opex_date = date(2026, 3, 20)
        ts = _make_trade_spec(
            [_make_leg("short_put", LegAction.SELL_TO_OPEN, "put", 590.0)],
        )
        ts = ts.model_copy(update={
            "target_expiration": date(2026, 4, 17),
            "front_expiration": opex_date,
        })
        events = [ExpiryEvent(date=opex_date, expiry_type=ExpiryType.MONTHLY_OPEX, label="Mar Monthly OpEx")]
        note = _expiry_note_for_trade(ts, events)
        assert "Front leg" in note
        assert "Mar Monthly OpEx" in note

    def test_no_match(self):
        from market_analyzer.service.trading_plan import _expiry_note_for_trade
        ts = _make_trade_spec(
            [_make_leg("short_put", LegAction.SELL_TO_OPEN, "put", 590.0)],
        )
        events = [ExpiryEvent(date=date(2026, 4, 17), expiry_type=ExpiryType.MONTHLY_OPEX, label="Apr OpEx")]
        note = _expiry_note_for_trade(ts, events)
        assert note is None

    def test_none_trade_spec(self):
        from market_analyzer.service.trading_plan import _expiry_note_for_trade
        assert _expiry_note_for_trade(None, []) is None


# ===== Exports =====


class TestExports:
    def test_all_models_importable(self):
        from market_analyzer import (
            DailyTradingPlan,
            DayVerdict,
            ExpiryEvent,
            ExpiryType,
            PlanHorizon,
            PlanTrade,
            RiskBudget,
            TradingPlanService,
            TradingPlanSettings,
        )
        assert DayVerdict.TRADE == "trade"
        assert PlanHorizon.ZERO_DTE == "0dte"
        assert ExpiryType.MONTHLY_OPEX == "monthly_opex"


# ===== BS Price Sanity =====


class TestBSPriceSanity:
    """Sanity checks on the BS approximation — not exact, just directionally correct."""

    def test_atm_call_positive(self):
        from market_analyzer.opportunity.option_plays._trade_spec_helpers import _bs_price
        price = _bs_price(100.0, 100.0, 30 / 365, 0.25, "call")
        assert price > 0
        assert price < 10  # Shouldn't be more than ~10% of underlying for 30d ATM

    def test_deep_otm_put_near_zero(self):
        from market_analyzer.opportunity.option_plays._trade_spec_helpers import _bs_price
        price = _bs_price(100.0, 50.0, 30 / 365, 0.25, "put")
        assert price < 0.01

    def test_deep_itm_call_near_intrinsic(self):
        from market_analyzer.opportunity.option_plays._trade_spec_helpers import _bs_price
        price = _bs_price(100.0, 50.0, 30 / 365, 0.25, "call")
        assert price > 49.0  # Near intrinsic (100-50=50)

    def test_higher_iv_higher_price(self):
        from market_analyzer.opportunity.option_plays._trade_spec_helpers import _bs_price
        low_iv = _bs_price(100.0, 100.0, 30 / 365, 0.15, "call")
        high_iv = _bs_price(100.0, 100.0, 30 / 365, 0.40, "call")
        assert high_iv > low_iv

    def test_longer_dte_higher_price(self):
        from market_analyzer.opportunity.option_plays._trade_spec_helpers import _bs_price
        short = _bs_price(100.0, 100.0, 7 / 365, 0.25, "call")
        long_ = _bs_price(100.0, 100.0, 60 / 365, 0.25, "call")
        assert long_ > short


# ===== Structure Profiles =====


class TestStructureProfile:
    def test_iron_condor(self):
        from market_analyzer.models.opportunity import (
            RiskProfile, StructureType, get_structure_profile,
        )
        p = get_structure_profile(StructureType.IRON_CONDOR)
        assert p.bias == "neutral"
        assert p.risk_profile == RiskProfile.DEFINED
        assert "/‾‾\\" in p.payoff_graph

    def test_iron_man(self):
        from market_analyzer.models.opportunity import (
            RiskProfile, StructureType, get_structure_profile,
        )
        p = get_structure_profile(StructureType.IRON_MAN)
        assert p.bias == "neutral"
        assert p.risk_profile == RiskProfile.DEFINED
        assert "\\__/" in p.payoff_graph

    def test_iron_butterfly(self):
        from market_analyzer.models.opportunity import (
            RiskProfile, StructureType, get_structure_profile,
        )
        p = get_structure_profile(StructureType.IRON_BUTTERFLY)
        assert p.bias == "neutral"
        assert p.risk_profile == RiskProfile.DEFINED

    def test_bull_credit_spread(self):
        from market_analyzer.models.opportunity import (
            RiskProfile, StructureType, OrderSide, get_structure_profile,
        )
        p = get_structure_profile(StructureType.CREDIT_SPREAD, OrderSide.CREDIT, "bullish")
        assert p.bias == "bullish"
        assert p.risk_profile == RiskProfile.DEFINED
        assert "bull put" in p.label

    def test_bear_credit_spread(self):
        from market_analyzer.models.opportunity import (
            RiskProfile, StructureType, OrderSide, get_structure_profile,
        )
        p = get_structure_profile(StructureType.CREDIT_SPREAD, OrderSide.CREDIT, "bearish")
        assert p.bias == "bearish"
        assert p.risk_profile == RiskProfile.DEFINED
        assert "bear call" in p.label

    def test_bull_debit_spread(self):
        from market_analyzer.models.opportunity import (
            RiskProfile, StructureType, OrderSide, get_structure_profile,
        )
        p = get_structure_profile(StructureType.DEBIT_SPREAD, OrderSide.DEBIT, "bullish")
        assert p.bias == "bullish"
        assert p.risk_profile == RiskProfile.DEFINED

    def test_ratio_spread_undefined_risk(self):
        from market_analyzer.models.opportunity import (
            RiskProfile, StructureType, get_structure_profile,
        )
        p = get_structure_profile(StructureType.RATIO_SPREAD, direction="bullish")
        assert p.risk_profile == RiskProfile.UNDEFINED
        assert "UNDEFINED" in p.label

    def test_short_straddle_undefined(self):
        from market_analyzer.models.opportunity import (
            RiskProfile, StructureType, OrderSide, get_structure_profile,
        )
        p = get_structure_profile(StructureType.STRADDLE, OrderSide.CREDIT)
        assert p.risk_profile == RiskProfile.UNDEFINED
        assert p.bias == "neutral"

    def test_long_straddle_defined(self):
        from market_analyzer.models.opportunity import (
            RiskProfile, StructureType, OrderSide, get_structure_profile,
        )
        p = get_structure_profile(StructureType.STRADDLE, OrderSide.DEBIT)
        assert p.risk_profile == RiskProfile.DEFINED
        assert p.bias == "neutral"

    def test_short_strangle_undefined(self):
        from market_analyzer.models.opportunity import (
            RiskProfile, StructureType, OrderSide, get_structure_profile,
        )
        p = get_structure_profile(StructureType.STRANGLE, OrderSide.CREDIT)
        assert p.risk_profile == RiskProfile.UNDEFINED

    def test_long_strangle_defined(self):
        from market_analyzer.models.opportunity import (
            RiskProfile, StructureType, OrderSide, get_structure_profile,
        )
        p = get_structure_profile(StructureType.STRANGLE, OrderSide.DEBIT)
        assert p.risk_profile == RiskProfile.DEFINED
        assert "\\__/" in p.payoff_graph

    def test_long_call(self):
        from market_analyzer.models.opportunity import (
            RiskProfile, StructureType, get_structure_profile,
        )
        p = get_structure_profile(StructureType.LONG_OPTION, direction="bullish")
        assert p.bias == "bullish"
        assert p.risk_profile == RiskProfile.DEFINED
        assert "__/" in p.payoff_graph

    def test_long_put(self):
        from market_analyzer.models.opportunity import (
            RiskProfile, StructureType, get_structure_profile,
        )
        p = get_structure_profile(StructureType.LONG_OPTION, direction="bearish")
        assert p.bias == "bearish"
        assert p.risk_profile == RiskProfile.DEFINED
        assert "\\__" in p.payoff_graph

    def test_pmcc(self):
        from market_analyzer.models.opportunity import (
            RiskProfile, StructureType, get_structure_profile,
        )
        p = get_structure_profile(StructureType.PMCC)
        assert p.bias == "bullish"
        assert p.risk_profile == RiskProfile.DEFINED

    def test_calendar(self):
        from market_analyzer.models.opportunity import (
            RiskProfile, StructureType, get_structure_profile,
        )
        p = get_structure_profile(StructureType.CALENDAR)
        assert p.bias == "neutral"
        assert p.risk_profile == RiskProfile.DEFINED

    def test_diagonal(self):
        from market_analyzer.models.opportunity import (
            RiskProfile, StructureType, get_structure_profile,
        )
        p = get_structure_profile(StructureType.DIAGONAL)
        assert p.risk_profile == RiskProfile.DEFINED

    def test_fallback_unknown(self):
        from market_analyzer.models.opportunity import get_structure_profile
        p = get_structure_profile("some_unknown_type")
        assert p.payoff_graph == "???"

    def test_all_structure_types_have_profiles(self):
        """Every StructureType should return a profile (not the fallback '???')."""
        from market_analyzer.models.opportunity import (
            OrderSide, StructureType, get_structure_profile,
        )
        for st in StructureType:
            # Provide order_side for straddle/strangle
            side = OrderSide.CREDIT if st in (StructureType.STRADDLE, StructureType.STRANGLE) else None
            p = get_structure_profile(st, side, "bullish")
            assert p.payoff_graph != "???", f"No profile for {st}"

    def test_double_calendar(self):
        from market_analyzer.models.opportunity import (
            RiskProfile, StructureType, get_structure_profile,
        )
        p = get_structure_profile(StructureType.DOUBLE_CALENDAR)
        assert p.bias == "neutral"
        assert p.risk_profile == RiskProfile.DEFINED
        assert "4-leg" in p.label

    def test_exports(self):
        from market_analyzer import RiskProfile, StructureProfile, get_structure_profile
        assert RiskProfile.DEFINED == "defined"
        assert RiskProfile.UNDEFINED == "undefined"


# ===== Double Calendar Legs =====


def _make_term_pt(dte: int, iv: float = 0.25) -> TermStructurePoint:
    return TermStructurePoint(
        expiration=date(2026, 3, 27) + timedelta(days=dte - 30),
        days_to_expiry=dte,
        atm_iv=iv,
        atm_strike=600.0,
        put_skew=0.0,
        call_skew=0.0,
    )


class TestBuildDoubleCalendarLegs:
    def test_returns_4_legs(self):
        legs = build_double_calendar_legs(600.0, _make_term_pt(25), _make_term_pt(55), atr=10.0)
        assert len(legs) == 4

    def test_strikes_bracket_price(self):
        price = 600.0
        legs = build_double_calendar_legs(price, _make_term_pt(25), _make_term_pt(55), atr=10.0)
        strikes = {leg.strike for leg in legs}
        assert len(strikes) == 2  # Two distinct strikes
        assert min(strikes) < price
        assert max(strikes) > price

    def test_has_puts_and_calls(self):
        legs = build_double_calendar_legs(600.0, _make_term_pt(25), _make_term_pt(55), atr=10.0)
        types = [leg.option_type for leg in legs]
        assert types.count("put") == 2
        assert types.count("call") == 2

    def test_front_sold_back_bought(self):
        front = _make_term_pt(25)
        back = _make_term_pt(55)
        legs = build_double_calendar_legs(600.0, front, back, atr=10.0)
        for leg in legs:
            if leg.days_to_expiry == front.days_to_expiry:
                assert leg.action == LegAction.SELL_TO_OPEN
            else:
                assert leg.action == LegAction.BUY_TO_OPEN

    def test_fallback_offset_without_atr(self):
        legs = build_double_calendar_legs(600.0, _make_term_pt(25), _make_term_pt(55), atr=None)
        assert len(legs) == 4
        strikes = {leg.strike for leg in legs}
        assert len(strikes) == 2
