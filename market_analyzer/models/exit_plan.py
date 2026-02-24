"""Pydantic models for exit planning."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel


class ExitReason(StrEnum):
    """Reason categories for exits."""

    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    TIME_DECAY = "time_decay"
    REGIME_CHANGE = "regime_change"
    TECHNICAL_BREAKDOWN = "technical_breakdown"
    RISK_MANAGEMENT = "risk_management"


class AdjustmentTriggerType(StrEnum):
    """Types of adjustments that can be triggered."""

    ROLL_OUT = "roll_out"               # Roll to later expiration
    ROLL_UP = "roll_up"                 # Roll strikes up
    ROLL_DOWN = "roll_down"             # Roll strikes down
    WIDEN_WINGS = "widen_wings"         # Widen spread
    ADD_HEDGE = "add_hedge"             # Add protective leg
    CLOSE_PARTIAL = "close_partial"     # Take partial profits
    CLOSE_FULL = "close_full"           # Exit entirely


class ExitTarget(BaseModel):
    """A profit-taking or stop-loss level."""

    price: float
    pct_from_entry: float
    reason: ExitReason
    action: str                     # "close 50%", "close all", "trail stop"
    description: str


class AdjustmentTrigger(BaseModel):
    """A condition that triggers a position adjustment."""

    trigger_type: AdjustmentTriggerType
    condition: str                  # Human-readable condition
    action: str                     # What to do
    urgency: str                    # "immediate" | "next_session" | "monitor"
    description: str


class ExitPlan(BaseModel):
    """Complete exit plan for a position."""

    ticker: str
    as_of_date: date
    entry_price: float
    strategy_type: str              # From OptionStructureType

    # Profit targets (ordered: nearest first)
    profit_targets: list[ExitTarget]

    # Stop losses
    stop_loss: ExitTarget | None = None
    trailing_stop: ExitTarget | None = None

    # Time-based exits
    dte_exit_threshold: int | None = None       # Close when DTE drops below this
    theta_decay_exit_pct: float | None = None   # Close at X% of max profit

    # Adjustment triggers
    adjustments: list[AdjustmentTrigger] = []

    # Regime-change actions
    regime_change_action: str = ""  # What to do if regime changes

    max_loss_dollars: float | None = None
    max_profit_dollars: float | None = None
    risk_reward_ratio: float | None = None
    summary: str = ""
