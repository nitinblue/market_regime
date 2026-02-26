"""Pydantic models for the daily trading plan framework."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel

from market_analyzer.macro.expiry import ExpiryEvent
from market_analyzer.models.opportunity import TradeSpec, Verdict
from market_analyzer.models.ranking import StrategyType


class DayVerdict(StrEnum):
    TRADE = "trade"
    TRADE_LIGHT = "trade_light"
    AVOID = "avoid"
    NO_TRADE = "no_trade"


class PlanHorizon(StrEnum):
    ZERO_DTE = "0dte"
    WEEKLY = "weekly"       # 1-7 DTE
    MONTHLY = "monthly"     # 8-60 DTE
    LEAP = "leap"           # 60+ DTE


class PlanTrade(BaseModel):
    """A single trade in the daily plan — wraps RankedEntry with plan context."""

    rank: int
    ticker: str
    strategy_type: StrategyType
    horizon: PlanHorizon
    verdict: Verdict
    composite_score: float
    direction: str
    trade_spec: TradeSpec | None
    max_entry_price: float | None = None
    rationale: str
    risk_notes: list[str]
    expiry_note: str | None = None


class RiskBudget(BaseModel):
    max_new_positions: int
    max_daily_risk_dollars: float
    position_size_factor: float


class DailyTradingPlan(BaseModel):
    as_of_date: date
    plan_for_date: date
    day_verdict: DayVerdict
    day_verdict_reasons: list[str]
    risk_budget: RiskBudget
    expiry_events: list[ExpiryEvent]
    upcoming_expiries: list[ExpiryEvent]
    trades_by_horizon: dict[PlanHorizon, list[PlanTrade]]
    all_trades: list[PlanTrade]
    total_trades: int
    summary: str
