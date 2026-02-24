"""Pydantic models for strategy selection and position sizing."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel


class OptionStructureType(StrEnum):
    """Option structure types for strategy selection."""

    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    CREDIT_SPREAD = "credit_spread"
    DEBIT_SPREAD = "debit_spread"
    STRANGLE = "strangle"
    STRADDLE = "straddle"
    CALENDAR_SPREAD = "calendar_spread"
    DIAGONAL_SPREAD = "diagonal_spread"
    BUTTERFLY = "butterfly"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    BULL_PUT_SPREAD = "bull_put_spread"
    BEAR_CALL_SPREAD = "bear_call_spread"
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    PMCC = "pmcc"
    PROTECTIVE_PUT = "protective_put"


class OptionStructure(BaseModel):
    """A recommended option structure with rationale."""

    structure_type: OptionStructureType
    direction: str                  # "neutral", "bullish", "bearish"
    max_loss: str                   # "defined" | "undefined"
    theta_exposure: str             # "positive" | "negative" | "neutral"
    vega_exposure: str              # "short" | "long" | "neutral"
    rationale: str
    risk_notes: list[str] = []


class StrategyParameters(BaseModel):
    """Complete strategy recommendation for a ticker."""

    ticker: str
    as_of_date: date
    primary_structure: OptionStructure
    alternative_structures: list[OptionStructure] = []
    regime_rationale: str           # Why this structure fits the regime
    setup_type: str = ""            # "breakout", "income", "momentum", etc.
    suggested_dte_range: tuple[int, int] = (30, 45)
    suggested_delta_range: tuple[float, float] = (0.20, 0.35)
    wing_width_suggestion: str = "" # e.g. "5-wide", "10-wide"
    summary: str = ""


class PositionSize(BaseModel):
    """Position sizing recommendation."""

    ticker: str
    strategy: OptionStructureType
    account_size: float
    max_risk_dollars: float         # Max loss for this position
    max_risk_pct: float             # As % of account
    suggested_contracts: int
    max_contracts: int
    margin_estimate: float | None = None
    buying_power_usage_pct: float | None = None
    rationale: str = ""
