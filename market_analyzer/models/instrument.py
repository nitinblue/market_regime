"""Pydantic models for unified per-instrument analysis."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from market_analyzer.models.fundamentals import FundamentalsSnapshot
from market_analyzer.models.levels import LevelsAnalysis
from market_analyzer.models.opportunity import (
    BreakoutOpportunity,
    LEAPOpportunity,
    MomentumOpportunity,
    ZeroDTEOpportunity,
)
from market_analyzer.models.phase import PhaseID, PhaseResult
from market_analyzer.models.regime import RegimeID, RegimeResult
from market_analyzer.models.technicals import TechnicalSnapshot


class InstrumentAnalysis(BaseModel):
    """Unified per-ticker report combining all analysis layers."""

    ticker: str
    as_of_date: date

    # Core analysis
    regime: RegimeResult
    phase: PhaseResult
    technicals: TechnicalSnapshot
    levels: LevelsAnalysis | None = None
    fundamentals: FundamentalsSnapshot | None = None

    # Optional opportunity assessments
    breakout: BreakoutOpportunity | None = None
    momentum: MomentumOpportunity | None = None
    leap: LEAPOpportunity | None = None
    zero_dte: ZeroDTEOpportunity | None = None

    # Derived summary
    regime_id: RegimeID
    phase_id: PhaseID
    trend_bias: str = ""            # "bullish" | "bearish" | "neutral"
    volatility_label: str = ""      # "low" | "high"
    actionable_setups: list[str] = []   # e.g. ["breakout", "momentum"]
    summary: str = ""
