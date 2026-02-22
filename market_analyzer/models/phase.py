"""Wyckoff phase detection data models."""

from __future__ import annotations

from datetime import date
from enum import IntEnum

from pydantic import BaseModel


class PhaseID(IntEnum):
    ACCUMULATION = 1
    MARKUP = 2
    DISTRIBUTION = 3
    MARKDOWN = 4


class SwingPoint(BaseModel):
    """A detected swing high or low."""

    date: date
    price: float
    type: str  # "high" or "low"


class PriceStructure(BaseModel):
    """Wyckoff-relevant price structure features from OHLCV."""

    swing_highs: list[SwingPoint]
    swing_lows: list[SwingPoint]
    higher_highs: bool
    higher_lows: bool
    lower_highs: bool
    lower_lows: bool
    range_compression: float  # -1 (expanding) to +1 (compressing)
    price_vs_sma: float  # % above/below SMA
    volume_trend: str  # "declining" | "stable" | "rising"
    support_level: float | None
    resistance_level: float | None


class PhaseEvidence(BaseModel):
    """Evidence supporting the phase classification."""

    regime_signal: str  # "R1 following bearish R3 -> accumulation"
    price_signal: str  # "higher lows, range compressing"
    volume_signal: str  # "declining volume"
    supporting: list[str]
    contradictions: list[str]


class PhaseTransition(BaseModel):
    """Estimated probability of transitioning to another phase."""

    to_phase: PhaseID
    probability: float
    triggers: list[str]


class PhaseResult(BaseModel):
    """Full Wyckoff phase detection result for a single ticker."""

    ticker: str
    phase: PhaseID
    phase_name: str
    confidence: float
    phase_age_days: int
    prior_phase: PhaseID | None
    cycle_completion: float  # 0.0-1.0 rough estimate
    price_structure: PriceStructure
    evidence: PhaseEvidence
    transitions: list[PhaseTransition]
    strategy_comment: str  # LEAP-specific
    as_of_date: date
