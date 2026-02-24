"""Pydantic models for tail-risk / black swan detection."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel


class AlertLevel(StrEnum):
    """Tail-risk alert level."""

    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class IndicatorStatus(StrEnum):
    """Status of an individual stress indicator."""

    NORMAL = "normal"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"
    UNAVAILABLE = "unavailable"


class StressIndicator(BaseModel):
    """A single stress indicator reading."""

    name: str
    value: float | None
    score: float  # 0.0–1.0 normalized stress
    status: IndicatorStatus
    weight: float
    description: str


class CircuitBreaker(BaseModel):
    """A circuit breaker condition. Any triggered → CRITICAL override."""

    name: str
    triggered: bool
    value: float | None
    threshold: float
    description: str


class BlackSwanAlert(BaseModel):
    """Complete tail-risk assessment. Pre-trade gate for the portfolio."""

    as_of_date: date
    alert_level: AlertLevel
    composite_score: float  # 0.0–1.0 weighted stress
    circuit_breakers: list[CircuitBreaker]
    indicators: list[StressIndicator]
    triggered_breakers: int
    action: str
    summary: str
