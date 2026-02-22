"""Pydantic models for technical indicators."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel


class SignalDirection(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SignalStrength(StrEnum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class TechnicalSignal(BaseModel):
    name: str
    direction: SignalDirection
    strength: SignalStrength
    description: str


class MovingAverages(BaseModel):
    sma_20: float
    sma_50: float
    sma_200: float
    ema_9: float
    ema_21: float
    price_vs_sma_20_pct: float
    price_vs_sma_50_pct: float
    price_vs_sma_200_pct: float


class RSIData(BaseModel):
    value: float
    is_overbought: bool
    is_oversold: bool


class BollingerBands(BaseModel):
    upper: float
    middle: float
    lower: float
    bandwidth: float
    percent_b: float


class MACDData(BaseModel):
    macd_line: float
    signal_line: float
    histogram: float
    is_bullish_crossover: bool
    is_bearish_crossover: bool


class StochasticData(BaseModel):
    k: float
    d: float
    is_overbought: bool
    is_oversold: bool


class SupportResistance(BaseModel):
    support: float | None
    resistance: float | None
    price_vs_support_pct: float | None
    price_vs_resistance_pct: float | None


class VCPStage(StrEnum):
    NONE = "none"
    FORMING = "forming"
    MATURING = "maturing"
    READY = "ready"
    BREAKOUT = "breakout"


class VCPData(BaseModel):
    """Volatility Contraction Pattern (Minervini VCP) detection."""

    stage: VCPStage
    contraction_count: int
    contraction_pcts: list[float]
    current_range_pct: float
    range_compression: float
    volume_trend: str
    pivot_price: float | None
    pivot_distance_pct: float | None
    days_in_base: int
    above_sma_50: bool
    above_sma_200: bool
    score: float
    description: str


class MarketPhase(StrEnum):
    """Price-structure-derived market phase (no HMM required)."""

    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"


class PhaseIndicator(BaseModel):
    """Lightweight phase indicator derived purely from price structure."""

    phase: MarketPhase
    confidence: float  # 0.0–1.0
    description: str
    higher_highs: bool
    higher_lows: bool
    lower_highs: bool
    lower_lows: bool
    range_compression: float  # -1 (expanding) to +1 (compressing)
    volume_trend: str  # "declining" | "stable" | "rising"
    price_vs_sma_50_pct: float


class TechnicalSnapshot(BaseModel):
    ticker: str
    as_of_date: date
    current_price: float
    atr: float
    atr_pct: float
    vwma_20: float
    moving_averages: MovingAverages
    rsi: RSIData
    bollinger: BollingerBands
    macd: MACDData
    stochastic: StochasticData
    support_resistance: SupportResistance
    phase: PhaseIndicator
    vcp: VCPData | None = None
    signals: list[TechnicalSignal]
