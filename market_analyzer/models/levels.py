"""Pydantic models for unified price levels and risk/reward analysis."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel


class LevelSource(StrEnum):
    SWING_SUPPORT = "swing_support"
    SWING_RESISTANCE = "swing_resistance"
    SMA_20 = "sma_20"
    SMA_50 = "sma_50"
    SMA_200 = "sma_200"
    EMA_9 = "ema_9"
    EMA_21 = "ema_21"
    BOLLINGER_UPPER = "bollinger_upper"
    BOLLINGER_MIDDLE = "bollinger_middle"
    BOLLINGER_LOWER = "bollinger_lower"
    VWMA_20 = "vwma_20"
    VCP_PIVOT = "vcp_pivot"
    ORDER_BLOCK_HIGH = "order_block_high"
    ORDER_BLOCK_LOW = "order_block_low"
    FVG_HIGH = "fvg_high"
    FVG_LOW = "fvg_low"
    ORB_LEVEL = "orb_level"


class LevelRole(StrEnum):
    SUPPORT = "support"
    RESISTANCE = "resistance"


class TradeDirection(StrEnum):
    LONG = "long"
    SHORT = "short"


class PriceLevel(BaseModel):
    price: float
    role: LevelRole
    sources: list[LevelSource]
    confluence_score: int
    strength: float
    distance_pct: float
    description: str


class StopLoss(BaseModel):
    price: float
    distance_pct: float
    dollar_risk_per_share: float
    level: PriceLevel
    atr_buffer: float
    description: str


class Target(BaseModel):
    price: float
    distance_pct: float
    dollar_reward_per_share: float
    risk_reward_ratio: float
    level: PriceLevel
    description: str


class LevelsAnalysis(BaseModel):
    ticker: str
    as_of_date: date
    entry_price: float
    direction: TradeDirection
    direction_auto_detected: bool
    current_price: float
    atr: float
    atr_pct: float
    support_levels: list[PriceLevel]
    resistance_levels: list[PriceLevel]
    stop_loss: StopLoss | None
    targets: list[Target]
    best_target: Target | None
    summary: str
