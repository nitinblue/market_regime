"""Earnings play opportunity assessment — go/no-go for earnings-related trades."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from market_analyzer.models.opportunity import (
    HardStop,
    OpportunitySignal,
    Verdict,
)
from market_analyzer.models.regime import RegimeID

if TYPE_CHECKING:
    from market_analyzer.models.fundamentals import FundamentalsSnapshot
    from market_analyzer.models.regime import RegimeResult
    from market_analyzer.models.technicals import TechnicalSnapshot


class EarningsPlayStrategy(str):
    """Strategy types for earnings plays."""

    PRE_EARNINGS_STRADDLE = "pre_earnings_straddle"
    POST_EARNINGS_DRIFT = "post_earnings_drift"
    IV_CRUSH_SELL = "iv_crush_sell"
    NO_TRADE = "no_trade"


class EarningsOpportunity:
    """Result container for earnings play assessment."""

    def __init__(
        self,
        ticker: str,
        as_of_date: date,
        verdict: Verdict,
        confidence: float,
        strategy: str,
        direction: str,
        signals: list[OpportunitySignal],
        hard_stops: list[HardStop],
        days_to_earnings: int | None,
        regime_id: int,
        summary: str,
    ):
        self.ticker = ticker
        self.as_of_date = as_of_date
        self.verdict = verdict
        self.confidence = confidence
        self.strategy = strategy
        self.direction = direction
        self.signals = signals
        self.hard_stops = hard_stops
        self.days_to_earnings = days_to_earnings
        self.regime_id = regime_id
        self.summary = summary


def assess_earnings_play(
    ticker: str,
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    fundamentals: FundamentalsSnapshot | None = None,
    as_of: date | None = None,
) -> EarningsOpportunity:
    """Assess earnings play opportunity for a single instrument.

    Pure function — consumes pre-computed analysis, produces structured assessment.
    """
    today = as_of or date.today()
    hard_stops: list[HardStop] = []
    signals: list[OpportunitySignal] = []
    score = 0.3  # Start slightly below neutral

    days_to_earnings: int | None = None
    if fundamentals is not None:
        days_to_earnings = fundamentals.upcoming_events.days_to_earnings

    # --- Hard stops ---
    if days_to_earnings is None:
        hard_stops.append(HardStop(
            name="No earnings date",
            description="No upcoming earnings date found — cannot assess earnings play",
        ))

    if hard_stops:
        return EarningsOpportunity(
            ticker=ticker,
            as_of_date=today,
            verdict=Verdict.NO_GO,
            confidence=0.0,
            strategy=EarningsPlayStrategy.NO_TRADE,
            direction="neutral",
            signals=signals,
            hard_stops=hard_stops,
            days_to_earnings=days_to_earnings,
            regime_id=int(regime.regime),
            summary=f"NO_GO: {hard_stops[0].description}",
        )

    assert days_to_earnings is not None  # Guarded by hard stop above

    # --- Proximity signals ---
    if 5 <= days_to_earnings <= 14:
        signals.append(OpportunitySignal(
            name="Earnings window", favorable=True, weight=0.3,
            description=f"Earnings in {days_to_earnings} days — IV expansion likely",
        ))
        score += 0.2
        strategy = EarningsPlayStrategy.PRE_EARNINGS_STRADDLE
    elif 1 <= days_to_earnings <= 4:
        signals.append(OpportunitySignal(
            name="Earnings imminent", favorable=True, weight=0.25,
            description=f"Earnings in {days_to_earnings} days — peak IV",
        ))
        score += 0.25
        strategy = EarningsPlayStrategy.IV_CRUSH_SELL
    elif days_to_earnings == 0:
        signals.append(OpportunitySignal(
            name="Earnings today", favorable=True, weight=0.2,
            description="Earnings today — IV crush imminent",
        ))
        score += 0.15
        strategy = EarningsPlayStrategy.IV_CRUSH_SELL
    else:
        strategy = EarningsPlayStrategy.NO_TRADE
        signals.append(OpportunitySignal(
            name="Earnings too far", favorable=False, weight=0.2,
            description=f"Earnings in {days_to_earnings} days — too far for earnings play",
        ))

    # --- Regime context ---
    if regime.regime in (RegimeID.R1_LOW_VOL_MR, RegimeID.R2_HIGH_VOL_MR):
        signals.append(OpportunitySignal(
            name="MR regime + earnings", favorable=True, weight=0.15,
            description="Mean-reverting regime — post-earnings drift less likely",
        ))
        score += 0.1
    elif regime.regime == RegimeID.R3_LOW_VOL_TREND:
        signals.append(OpportunitySignal(
            name="Trending + earnings", favorable=True, weight=0.15,
            description="Trending regime — post-earnings drift possible",
        ))
        score += 0.15

    # --- ATR context ---
    if technicals.atr_pct >= 1.5:
        signals.append(OpportunitySignal(
            name="High ATR", favorable=True, weight=0.1,
            description=f"ATR% {technicals.atr_pct:.2f} — good premium potential",
        ))
        score += 0.1

    # --- Historical earnings surprise ---
    if fundamentals is not None and fundamentals.recent_earnings:
        last_surprise = fundamentals.recent_earnings[0]
        if last_surprise.surprise_pct is not None and abs(last_surprise.surprise_pct) >= 5:
            signals.append(OpportunitySignal(
                name="Surprise history", favorable=True, weight=0.1,
                description=f"Last surprise: {last_surprise.surprise_pct:+.1f}%",
            ))
            score += 0.1

    # Direction
    direction = "neutral"  # Most earnings plays are non-directional

    # Clamp
    confidence = max(0.0, min(1.0, score))

    if confidence >= 0.55:
        verdict = Verdict.GO
    elif confidence >= 0.35:
        verdict = Verdict.CAUTION
    else:
        verdict = Verdict.NO_GO

    summary_parts = [f"{verdict.upper()}: {ticker}"]
    if days_to_earnings is not None:
        summary_parts.append(f"Earnings in {days_to_earnings}d")
    summary_parts.append(f"Score: {confidence:.0%}")

    return EarningsOpportunity(
        ticker=ticker,
        as_of_date=today,
        verdict=verdict,
        confidence=confidence,
        strategy=strategy,
        direction=direction,
        signals=signals,
        hard_stops=hard_stops,
        days_to_earnings=days_to_earnings,
        regime_id=int(regime.regime),
        summary=" | ".join(summary_parts),
    )
