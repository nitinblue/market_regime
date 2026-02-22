"""Pydantic models for macro economic calendar."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel


class MacroEventType(StrEnum):
    FOMC = "fomc"
    CPI = "cpi"
    NFP = "nfp"
    PCE = "pce"
    GDP = "gdp"


class MacroEventImpact(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MacroEvent(BaseModel):
    """A single macro economic event."""

    event_type: MacroEventType
    date: date
    name: str
    impact: MacroEventImpact
    description: str
    options_impact: str


class MacroCalendar(BaseModel):
    """Macro calendar with convenience accessors."""

    events: list[MacroEvent]
    next_event: MacroEvent | None
    days_to_next: int | None
    next_fomc: MacroEvent | None
    days_to_next_fomc: int | None
    events_next_7_days: list[MacroEvent]
    events_next_30_days: list[MacroEvent]
