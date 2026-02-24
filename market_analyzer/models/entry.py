"""Pydantic models for entry confirmation."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel

from market_analyzer.models.technicals import TechnicalSignal


class EntryTriggerType(StrEnum):
    """Types of entry triggers to confirm."""

    BREAKOUT_CONFIRMED = "breakout_confirmed"
    PULLBACK_TO_SUPPORT = "pullback_to_support"
    MOMENTUM_CONTINUATION = "momentum_continuation"
    MEAN_REVERSION_EXTREME = "mean_reversion_extreme"
    ORB_BREAKOUT = "orb_breakout"


class EntryCondition(BaseModel):
    """A single pass/fail condition for entry confirmation."""

    name: str
    met: bool
    weight: float
    description: str


class EntryConfirmation(BaseModel):
    """Result of entry signal confirmation."""

    ticker: str
    as_of_date: date
    trigger_type: EntryTriggerType
    confirmed: bool
    confidence: float               # 0.0–1.0
    conditions: list[EntryCondition]
    conditions_met: int
    conditions_total: int
    signals: list[TechnicalSignal]
    suggested_entry_price: float | None = None
    suggested_stop_price: float | None = None
    risk_per_share: float | None = None
    summary: str = ""
