"""Pydantic models for options opportunity assessment."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel


class Verdict(StrEnum):
    """Go/no-go verdict for an opportunity assessment."""

    GO = "go"
    CAUTION = "caution"
    NO_GO = "no_go"


class HardStop(BaseModel):
    """A condition that forces a NO_GO verdict."""

    name: str
    description: str


class OpportunitySignal(BaseModel):
    """A single contributing signal to the opportunity assessment."""

    name: str
    favorable: bool
    weight: float
    description: str


class StrategyRecommendation(BaseModel):
    """A specific trade structure recommendation."""

    name: str
    direction: str  # "neutral", "bullish", "bearish"
    structure: str
    rationale: str
    risk_notes: list[str]


# --- 0DTE ---


class ZeroDTEStrategy(StrEnum):
    IRON_CONDOR = "iron_condor"
    CREDIT_SPREAD = "credit_spread"
    STRADDLE_STRANGLE = "straddle_strangle"
    DIRECTIONAL_SPREAD = "directional_spread"
    NO_TRADE = "no_trade"


class ZeroDTEOpportunity(BaseModel):
    """0DTE opportunity assessment for a single ticker."""

    ticker: str
    as_of_date: date
    verdict: Verdict
    confidence: float
    hard_stops: list[HardStop]
    signals: list[OpportunitySignal]
    strategy: StrategyRecommendation
    zero_dte_strategy: ZeroDTEStrategy
    regime_id: int
    regime_confidence: float
    atr_pct: float
    orb_status: str | None
    has_macro_event_today: bool
    days_to_earnings: int | None
    summary: str


# --- LEAP ---


class LEAPStrategy(StrEnum):
    BULL_CALL_LEAP = "bull_call_leap"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_LEAP = "bear_put_leap"
    PROTECTIVE_PUT = "protective_put"
    PMCC = "pmcc"
    NO_TRADE = "no_trade"


class FundamentalScore(BaseModel):
    """Composite score from fundamentals data."""

    score: float
    earnings_growth_signal: str
    revenue_growth_signal: str
    margin_signal: str
    debt_signal: str
    valuation_signal: str
    description: str


class LEAPOpportunity(BaseModel):
    """LEAP opportunity assessment for a single ticker."""

    ticker: str
    as_of_date: date
    verdict: Verdict
    confidence: float
    hard_stops: list[HardStop]
    signals: list[OpportunitySignal]
    strategy: StrategyRecommendation
    leap_strategy: LEAPStrategy
    regime_id: int
    regime_confidence: float
    phase_id: int
    phase_name: str
    phase_confidence: float
    iv_environment: str
    fundamental_score: FundamentalScore
    days_to_earnings: int | None
    macro_events_next_30_days: int
    summary: str


# --- Breakout ---


class BreakoutType(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"


class BreakoutStrategy(StrEnum):
    PIVOT_BREAKOUT = "pivot_breakout"
    SQUEEZE_PLAY = "squeeze_play"
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    PULLBACK_TO_BREAKOUT = "pullback_to_breakout"
    NO_TRADE = "no_trade"


class BreakoutSetup(BaseModel):
    """Describes the current breakout setup context."""

    vcp_stage: str
    vcp_score: float
    bollinger_squeeze: bool
    bollinger_bandwidth: float
    range_compression: float
    volume_pattern: str  # "declining_base" | "surge" | "normal" | "distribution"
    resistance_proximity_pct: float | None
    support_proximity_pct: float | None
    days_in_base: int | None
    smart_money_alignment: str  # "supportive" | "neutral" | "conflicting"
    description: str


class BreakoutOpportunity(BaseModel):
    """Breakout opportunity assessment for a single ticker."""

    ticker: str
    as_of_date: date
    verdict: Verdict
    confidence: float
    hard_stops: list[HardStop]
    signals: list[OpportunitySignal]
    strategy: StrategyRecommendation
    breakout_strategy: BreakoutStrategy
    breakout_type: BreakoutType
    regime_id: int
    regime_confidence: float
    phase_id: int
    phase_name: str
    phase_confidence: float
    setup: BreakoutSetup
    pivot_price: float | None
    days_to_earnings: int | None
    summary: str


# --- Momentum ---


class MomentumDirection(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"


class MomentumStrategy(StrEnum):
    TREND_CONTINUATION = "trend_continuation"
    PULLBACK_ENTRY = "pullback_entry"
    MOMENTUM_ACCELERATION = "momentum_acceleration"
    MOMENTUM_FADE = "momentum_fade"
    NO_TRADE = "no_trade"


class MomentumScore(BaseModel):
    """Composite momentum score details."""

    macd_histogram_trend: str  # "expanding" | "flat" | "contracting"
    macd_crossover: str  # "bullish" | "bearish" | "none"
    rsi_zone: str  # "oversold" | "healthy_bull" | "neutral" | "overbought" | "healthy_bear"
    price_vs_ma_alignment: str  # "strong_bull" | "bull" | "neutral" | "bear" | "strong_bear"
    golden_death_cross: str | None  # "golden_cross" | "death_cross" | None
    structural_pattern: str  # "HH_HL" | "LH_LL" | "mixed"
    volume_confirmation: bool
    stochastic_confirmation: bool
    atr_trend: str  # "rising" | "stable" | "falling"
    description: str


class MomentumOpportunity(BaseModel):
    """Momentum opportunity assessment for a single ticker."""

    ticker: str
    as_of_date: date
    verdict: Verdict
    confidence: float
    hard_stops: list[HardStop]
    signals: list[OpportunitySignal]
    strategy: StrategyRecommendation
    momentum_strategy: MomentumStrategy
    momentum_direction: MomentumDirection
    regime_id: int
    regime_confidence: float
    phase_id: int
    phase_name: str
    phase_confidence: float
    score: MomentumScore
    days_to_earnings: int | None
    summary: str
