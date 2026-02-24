"""Pydantic models for market context / environment assessment."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from market_analyzer.models.black_swan import AlertLevel, BlackSwanAlert
from market_analyzer.models.macro import MacroCalendar
from market_analyzer.models.regime import RegimeID, TrendDirection


class IntermarketEntry(BaseModel):
    """Regime read for a single reference ticker."""

    ticker: str
    regime: RegimeID
    confidence: float
    trend_direction: TrendDirection | None = None


class IntermarketDashboard(BaseModel):
    """Snapshot of reference-ticker regimes for environment assessment."""

    entries: list[IntermarketEntry]
    dominant_regime: RegimeID | None = None
    risk_on_count: int = 0      # Tickers in R1/R3
    risk_off_count: int = 0     # Tickers in R2/R4
    divergence: bool = False    # True if reference tickers disagree significantly
    summary: str = ""


class MarketContext(BaseModel):
    """Complete market environment assessment — pre-trade gate."""

    as_of_date: date
    market: str                             # "US", "India"
    macro: MacroCalendar
    black_swan: BlackSwanAlert
    intermarket: IntermarketDashboard
    environment_label: str                  # "risk-on" | "cautious" | "defensive" | "crisis"
    trading_allowed: bool
    position_size_factor: float = 1.0       # 0.0–1.0 scale-down in stressed environments
    summary: str = ""
