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


class OrderBlockType(StrEnum):
    BULLISH = "bullish"  # Demand zone: last bearish candle before impulse up
    BEARISH = "bearish"  # Supply zone: last bullish candle before impulse down


class OrderBlock(BaseModel):
    """A single detected order block zone."""

    type: OrderBlockType
    date: date
    high: float  # Top of the OB zone
    low: float  # Bottom of the OB zone
    volume: float
    impulse_strength: float  # Impulse move as multiple of ATR
    is_tested: bool  # Has price returned to this zone?
    is_broken: bool  # Has price broken through?
    distance_pct: float  # Current price distance from zone midpoint


class FVGType(StrEnum):
    BULLISH = "bullish"  # Gap up: candle1.high < candle3.low
    BEARISH = "bearish"  # Gap down: candle1.low > candle3.high


class FairValueGap(BaseModel):
    """A single detected fair value gap."""

    type: FVGType
    date: date
    high: float  # Top of gap
    low: float  # Bottom of gap
    gap_size_pct: float  # Gap width as % of price
    is_filled: bool  # Has price completely filled the gap?
    fill_pct: float  # How much of gap has been filled (0–100)
    distance_pct: float  # Current price distance from gap midpoint


class SmartMoneyData(BaseModel):
    """Order Block and Fair Value Gap detection (Smart Money Concepts)."""

    order_blocks: list[OrderBlock]
    fair_value_gaps: list[FairValueGap]
    nearest_bullish_ob: OrderBlock | None
    nearest_bearish_ob: OrderBlock | None
    nearest_bullish_fvg: FairValueGap | None
    nearest_bearish_fvg: FairValueGap | None
    unfilled_fvg_count: int
    active_ob_count: int  # Not broken
    score: float  # 0.0–1.0 composite confluence score
    description: str


class ORBStatus(StrEnum):
    WITHIN = "within"
    BREAKOUT_LONG = "breakout_long"
    BREAKOUT_SHORT = "breakout_short"
    FAILED_LONG = "failed_long"
    FAILED_SHORT = "failed_short"


class ORBLevel(BaseModel):
    """A single ORB extension level."""

    label: str
    price: float
    distance_pct: float


class ORBData(BaseModel):
    """Opening Range Breakout analysis from intraday data."""

    ticker: str
    date: date
    opening_minutes: int
    range_high: float
    range_low: float
    range_size: float
    range_pct: float
    current_price: float
    status: ORBStatus
    levels: list[ORBLevel]
    session_high: float
    session_low: float
    session_vwap: float | None
    opening_volume_ratio: float
    range_vs_daily_atr_pct: float | None
    breakout_bar_index: int | None
    retest_count: int
    signals: list[TechnicalSignal]
    description: str


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
    smart_money: SmartMoneyData | None = None
    signals: list[TechnicalSignal]
