"""Mean reversion opportunity assessment — go/no-go + strategy recommendation."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel

from market_analyzer.config import get_settings
from market_analyzer.models.opportunity import (
    HardStop,
    OpportunitySignal,
    StrategyRecommendation,
    TradeSpec,
    Verdict,
)
from market_analyzer.models.regime import RegimeID

if TYPE_CHECKING:
    from market_analyzer.models.fundamentals import FundamentalsSnapshot
    from market_analyzer.models.macro import MacroCalendar
    from market_analyzer.models.phase import PhaseResult
    from market_analyzer.models.regime import RegimeResult
    from market_analyzer.models.technicals import TechnicalSnapshot
    from market_analyzer.models.vol_surface import VolatilitySurface


class MeanReversionStrategy(StrEnum):
    """Strategy types for mean reversion plays."""

    OVERSOLD_BOUNCE = "oversold_bounce"
    OVERBOUGHT_FADE = "overbought_fade"
    BOLLINGER_REVERSION = "bollinger_reversion"
    NO_TRADE = "no_trade"


class MeanReversionOpportunity(BaseModel):
    """Result container for mean reversion assessment."""

    ticker: str
    as_of_date: date
    verdict: Verdict
    confidence: float
    strategy: str
    direction: str
    signals: list[OpportunitySignal]
    hard_stops: list[HardStop]
    rsi: float
    bollinger_pct_b: float
    regime_id: int
    trade_spec: TradeSpec | None = None
    summary: str


def assess_mean_reversion(
    ticker: str,
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    phase: PhaseResult | None = None,
    macro: MacroCalendar | None = None,
    fundamentals: FundamentalsSnapshot | None = None,
    vol_surface: VolatilitySurface | None = None,
    as_of: date | None = None,
) -> MeanReversionOpportunity:
    """Assess mean reversion opportunity for a single instrument.

    Pure function — consumes pre-computed analysis, produces structured assessment.
    """
    today = as_of or date.today()
    hard_stops: list[HardStop] = []
    signals: list[OpportunitySignal] = []
    score = 0.5  # Start neutral

    rsi = technicals.rsi.value
    bb_pct_b = technicals.bollinger.percent_b

    # --- Hard stops ---
    if regime.regime == RegimeID.R4_HIGH_VOL_TREND and regime.confidence >= 0.7:
        hard_stops.append(HardStop(
            name="R4 trending",
            description="High-vol trending regime — mean reversion unreliable",
        ))

    if fundamentals and fundamentals.upcoming_events.days_to_earnings is not None:
        if fundamentals.upcoming_events.days_to_earnings <= 2:
            hard_stops.append(HardStop(
                name="Earnings imminent",
                description=f"Earnings in {fundamentals.upcoming_events.days_to_earnings} days",
            ))

    if hard_stops:
        return MeanReversionOpportunity(
            ticker=ticker,
            as_of_date=today,
            verdict=Verdict.NO_GO,
            confidence=0.0,
            strategy=MeanReversionStrategy.NO_TRADE,
            direction="neutral",
            signals=signals,
            hard_stops=hard_stops,
            rsi=rsi,
            bollinger_pct_b=bb_pct_b,
            regime_id=int(regime.regime),
            summary=f"NO_GO: {hard_stops[0].description}",
        )

    # --- Signals ---
    # RSI extreme
    if rsi <= 25:
        signals.append(OpportunitySignal(
            name="RSI oversold extreme", favorable=True, weight=0.3,
            description=f"RSI {rsi:.0f} — deeply oversold",
        ))
        score += 0.25
    elif rsi <= 35:
        signals.append(OpportunitySignal(
            name="RSI approaching oversold", favorable=True, weight=0.2,
            description=f"RSI {rsi:.0f} — approaching oversold",
        ))
        score += 0.15
    elif rsi >= 75:
        signals.append(OpportunitySignal(
            name="RSI overbought extreme", favorable=True, weight=0.3,
            description=f"RSI {rsi:.0f} — deeply overbought",
        ))
        score += 0.25
    elif rsi >= 65:
        signals.append(OpportunitySignal(
            name="RSI approaching overbought", favorable=True, weight=0.2,
            description=f"RSI {rsi:.0f} — approaching overbought",
        ))
        score += 0.15

    # Bollinger band extreme
    if bb_pct_b <= 0.0:
        signals.append(OpportunitySignal(
            name="Below lower BB", favorable=True, weight=0.25,
            description=f"Bollinger %B = {bb_pct_b:.2f} — below lower band",
        ))
        score += 0.2
    elif bb_pct_b >= 1.0:
        signals.append(OpportunitySignal(
            name="Above upper BB", favorable=True, weight=0.25,
            description=f"Bollinger %B = {bb_pct_b:.2f} — above upper band",
        ))
        score += 0.2

    # Mean-reverting regime confirmation
    if regime.regime.is_mean_reverting:
        signals.append(OpportunitySignal(
            name="Mean-reverting regime", favorable=True, weight=0.2,
            description=f"R{regime.regime} supports mean reversion",
        ))
        score += 0.15

    # Stochastic confirmation
    if technicals.stochastic.is_oversold and rsi <= 35:
        signals.append(OpportunitySignal(
            name="Stochastic oversold", favorable=True, weight=0.1,
            description="Double oversold confirmation",
        ))
        score += 0.1
    elif technicals.stochastic.is_overbought and rsi >= 65:
        signals.append(OpportunitySignal(
            name="Stochastic overbought", favorable=True, weight=0.1,
            description="Double overbought confirmation",
        ))
        score += 0.1

    # Determine direction and strategy
    if rsi <= 40 or bb_pct_b <= 0.2:
        direction = "bullish"
        strategy = MeanReversionStrategy.OVERSOLD_BOUNCE
    elif rsi >= 60 or bb_pct_b >= 0.8:
        direction = "bearish"
        strategy = MeanReversionStrategy.OVERBOUGHT_FADE
    else:
        direction = "neutral"
        strategy = MeanReversionStrategy.NO_TRADE

    # Clamp score
    confidence = max(0.0, min(1.0, score))

    if confidence >= 0.55:
        verdict = Verdict.GO
    elif confidence >= 0.35:
        verdict = Verdict.CAUTION
    else:
        verdict = Verdict.NO_GO

    # --- Trade spec ---
    trade_spec = None
    if verdict != Verdict.NO_GO and strategy != MeanReversionStrategy.NO_TRADE:
        from market_analyzer.opportunity.option_plays._trade_spec_helpers import build_setup_trade_spec
        trade_spec = build_setup_trade_spec(
            ticker, technicals.current_price, technicals.atr,
            direction, int(regime.regime), vol_surface,
        )

    summary_parts = [f"{verdict.upper()}: {ticker}"]
    if signals:
        summary_parts.append(signals[0].description)
    summary_parts.append(f"Score: {confidence:.0%}")

    return MeanReversionOpportunity(
        ticker=ticker,
        as_of_date=today,
        verdict=verdict,
        confidence=confidence,
        strategy=strategy,
        direction=direction,
        signals=signals,
        hard_stops=hard_stops,
        rsi=rsi,
        bollinger_pct_b=bb_pct_b,
        regime_id=int(regime.regime),
        trade_spec=trade_spec,
        summary=" | ".join(summary_parts),
    )
