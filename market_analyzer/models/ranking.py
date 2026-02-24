"""Pydantic models for multi-ticker trade idea ranking."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel

from market_analyzer.models.opportunity import Verdict


class StrategyType(StrEnum):
    """Strategy types that can be ranked across tickers."""

    ZERO_DTE = "zero_dte"
    LEAP = "leap"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"


class ScoreBreakdown(BaseModel):
    """Component scores that compose the final ranking score."""

    verdict_score: float           # 0.0-1.0 from verdict (GO=1.0, CAUTION=0.5, NO_GO=0.0)
    confidence_score: float        # 0.0-1.0 from opportunity confidence
    regime_alignment: float        # 0.0-1.0 how well regime fits strategy
    risk_reward: float             # 0.0-1.0 from levels R:R (if available)
    technical_quality: float       # 0.0-1.0 from technicals (RSI, MACD, structure)
    phase_alignment: float         # 0.0-1.0 how well phase fits strategy
    income_bias_boost: float       # 0.0-0.05 bonus for theta strategies in R1/R2
    black_swan_penalty: float      # 0.0-1.0 multiplicative penalty
    macro_penalty: float           # 0.0-0.10 additive penalty for nearby events
    earnings_penalty: float        # 0.0-0.10 penalty for earnings proximity


class RankedEntry(BaseModel):
    """A single ticker x strategy entry in the ranking."""

    rank: int
    ticker: str
    strategy_type: StrategyType
    verdict: Verdict               # from opportunity assessment
    composite_score: float         # 0.0-1.0 final weighted score
    breakdown: ScoreBreakdown
    strategy_name: str             # e.g. "iron_condor", "bull_call_leap"
    direction: str                 # "neutral", "bullish", "bearish"
    rationale: str                 # from StrategyRecommendation
    risk_notes: list[str]


class TradeRankingResult(BaseModel):
    """Complete ranking result for a set of tickers."""

    as_of_date: date
    tickers: list[str]
    top_trades: list[RankedEntry]                        # All entries, sorted by score desc
    by_ticker: dict[str, list[RankedEntry]]              # Per-ticker, sorted by score desc
    by_strategy: dict[StrategyType, list[RankedEntry]]   # Per-strategy, sorted by score desc
    black_swan_level: str                                 # AlertLevel from BlackSwanService
    black_swan_gate: bool                                 # True if CRITICAL -> halt
    total_assessed: int
    total_actionable: int                                 # verdict != NO_GO
    summary: str


class RankingFeedback(BaseModel):
    """For future RL: record what was ranked and what happened."""

    as_of_date: date
    ticker: str
    strategy_type: StrategyType
    composite_score: float
    verdict: Verdict
    # Outcome (filled later from market data)
    outcome_5d_return: float | None = None
    outcome_20d_return: float | None = None
    outcome_max_drawdown: float | None = None
    trade_pnl: float | None = None          # From real trade history (future)
