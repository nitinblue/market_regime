"""Opening Range Breakout (ORB) setup assessment — go/no-go + strategy recommendation.

Evaluates intraday ORB data against regime, technicals, and phase to determine
whether the ORB breakout (or failure) presents a tradeable setup.

Requires intraday data (ORBData from features.patterns.orb). Daily-only analysis
will always produce NO_GO since ORB is an intraday pattern.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel

from market_analyzer.config import get_settings
from market_analyzer.models.opportunity import (
    HardStop,
    OpportunitySignal,
    Verdict,
)
from market_analyzer.models.regime import RegimeID
from market_analyzer.models.technicals import ORBStatus

if TYPE_CHECKING:
    from market_analyzer.models.fundamentals import FundamentalsSnapshot
    from market_analyzer.models.macro import MacroCalendar
    from market_analyzer.models.phase import PhaseResult
    from market_analyzer.models.regime import RegimeResult
    from market_analyzer.models.technicals import ORBData, TechnicalSnapshot


class ORBStrategy(StrEnum):
    """Strategy types for ORB setups."""

    BREAKOUT_CONTINUATION = "breakout_continuation"
    BREAKOUT_WITH_RETEST = "breakout_with_retest"
    FAILED_BREAKOUT_REVERSAL = "failed_breakout_reversal"
    NARROW_RANGE_ANTICIPATION = "narrow_range_anticipation"
    NO_TRADE = "no_trade"


class ORBSetupOpportunity(BaseModel):
    """Result of ORB setup assessment."""

    ticker: str
    as_of_date: date
    verdict: Verdict
    confidence: float
    strategy: ORBStrategy
    direction: str  # "bullish", "bearish", "neutral"
    signals: list[OpportunitySignal]
    hard_stops: list[HardStop]
    # ORB-specific fields
    orb_status: str  # ORBStatus value
    range_pct: float
    opening_volume_ratio: float
    range_vs_daily_atr_pct: float | None
    regime_id: int
    summary: str


def assess_orb(
    ticker: str,
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    orb: ORBData | None = None,
    phase: PhaseResult | None = None,
    macro: MacroCalendar | None = None,
    fundamentals: FundamentalsSnapshot | None = None,
    as_of: date | None = None,
) -> ORBSetupOpportunity:
    """Assess ORB setup opportunity for a single instrument.

    Pure function — consumes pre-computed ORBData + regime + technicals.
    Without ORBData, always returns NO_GO (ORB requires intraday data).
    """
    today = as_of or date.today()
    hard_stops: list[HardStop] = []
    signals: list[OpportunitySignal] = []
    score = 0.5  # Start neutral

    # --- No ORB data → immediate NO_GO ---
    if orb is None:
        return ORBSetupOpportunity(
            ticker=ticker,
            as_of_date=today,
            verdict=Verdict.NO_GO,
            confidence=0.0,
            strategy=ORBStrategy.NO_TRADE,
            direction="neutral",
            signals=[],
            hard_stops=[HardStop(
                name="No intraday data",
                description="ORB setup requires intraday (1m/5m) data",
            )],
            orb_status="none",
            range_pct=0.0,
            opening_volume_ratio=0.0,
            range_vs_daily_atr_pct=None,
            regime_id=int(regime.regime),
            summary="NO_GO: ORB requires intraday data",
        )

    # --- Hard stops ---

    # R4 high-vol trending: ORB breakouts are unreliable in explosive moves
    if regime.regime == RegimeID.R4_HIGH_VOL_TREND and regime.confidence >= 0.70:
        hard_stops.append(HardStop(
            name="R4 high-vol trend",
            description="R4 regime — ORB breakouts unreliable in explosive moves",
        ))

    # Earnings imminent: gap risk invalidates ORB levels
    if fundamentals and fundamentals.upcoming_events.days_to_earnings is not None:
        if fundamentals.upcoming_events.days_to_earnings <= 1:
            hard_stops.append(HardStop(
                name="Earnings today/tomorrow",
                description=f"Earnings in {fundamentals.upcoming_events.days_to_earnings} day(s) — gap risk",
            ))

    # Range too wide: already consumed most of daily range
    if orb.range_vs_daily_atr_pct is not None and orb.range_vs_daily_atr_pct > 85:
        hard_stops.append(HardStop(
            name="Range too wide",
            description=f"Opening range is {orb.range_vs_daily_atr_pct:.0f}% of daily ATR — little room left",
        ))

    if hard_stops:
        return ORBSetupOpportunity(
            ticker=ticker,
            as_of_date=today,
            verdict=Verdict.NO_GO,
            confidence=0.0,
            strategy=ORBStrategy.NO_TRADE,
            direction="neutral",
            signals=signals,
            hard_stops=hard_stops,
            orb_status=orb.status.value,
            range_pct=orb.range_pct,
            opening_volume_ratio=orb.opening_volume_ratio,
            range_vs_daily_atr_pct=orb.range_vs_daily_atr_pct,
            regime_id=int(regime.regime),
            summary=f"NO_GO: {hard_stops[0].description}",
        )

    # --- Scoring signals ---

    # 1. ORB status (weight 0.25)
    if orb.status in (ORBStatus.BREAKOUT_LONG, ORBStatus.BREAKOUT_SHORT):
        signals.append(OpportunitySignal(
            name="ORB breakout confirmed",
            favorable=True,
            weight=0.25,
            description=f"Price broke {'above' if orb.status == ORBStatus.BREAKOUT_LONG else 'below'} "
                        f"opening range ({orb.range_high:.2f}–{orb.range_low:.2f})",
        ))
        score += 0.20
    elif orb.status in (ORBStatus.FAILED_LONG, ORBStatus.FAILED_SHORT):
        signals.append(OpportunitySignal(
            name="ORB failed breakout",
            favorable=True,
            weight=0.20,
            description=f"Failed {'long' if orb.status == ORBStatus.FAILED_LONG else 'short'} "
                        f"breakout — reversal signal",
        ))
        score += 0.15
    elif orb.status == ORBStatus.WITHIN:
        if orb.range_pct < 0.5:
            signals.append(OpportunitySignal(
                name="Narrow opening range",
                favorable=True,
                weight=0.15,
                description=f"Range only {orb.range_pct:.2f}% — potential for directional move",
            ))
            score += 0.10
        else:
            signals.append(OpportunitySignal(
                name="Still within range",
                favorable=False,
                weight=0.15,
                description="No breakout yet — waiting for direction",
            ))
            score -= 0.10

    # 2. Opening volume (weight 0.15)
    if orb.opening_volume_ratio > 1.5:
        signals.append(OpportunitySignal(
            name="Strong opening volume",
            favorable=True,
            weight=0.15,
            description=f"Opening volume {orb.opening_volume_ratio:.1f}x session average — high conviction",
        ))
        score += 0.15
    elif orb.opening_volume_ratio > 1.2:
        signals.append(OpportunitySignal(
            name="Above-average opening volume",
            favorable=True,
            weight=0.10,
            description=f"Opening volume {orb.opening_volume_ratio:.1f}x session average",
        ))
        score += 0.08
    elif orb.opening_volume_ratio < 0.7:
        signals.append(OpportunitySignal(
            name="Light opening volume",
            favorable=False,
            weight=0.10,
            description=f"Opening volume only {orb.opening_volume_ratio:.1f}x — low conviction",
        ))
        score -= 0.10

    # 3. Range vs ATR (weight 0.15)
    if orb.range_vs_daily_atr_pct is not None:
        atr_pct = orb.range_vs_daily_atr_pct
        if 25 <= atr_pct <= 50:
            signals.append(OpportunitySignal(
                name="Healthy range size",
                favorable=True,
                weight=0.15,
                description=f"Opening range is {atr_pct:.0f}% of daily ATR — room to extend",
            ))
            score += 0.12
        elif atr_pct < 25:
            signals.append(OpportunitySignal(
                name="Tight range",
                favorable=True,
                weight=0.10,
                description=f"Opening range is only {atr_pct:.0f}% of daily ATR — coiled for move",
            ))
            score += 0.08
        elif atr_pct > 65:
            signals.append(OpportunitySignal(
                name="Wide range",
                favorable=False,
                weight=0.10,
                description=f"Opening range consumed {atr_pct:.0f}% of daily ATR — limited upside",
            ))
            score -= 0.10

    # 4. Retest confirmation (weight 0.10)
    if orb.retest_count > 0 and orb.status in (ORBStatus.BREAKOUT_LONG, ORBStatus.BREAKOUT_SHORT):
        signals.append(OpportunitySignal(
            name="Breakout retested",
            favorable=True,
            weight=0.10,
            description=f"{orb.retest_count} retest(s) of range edge — holding breakout level",
        ))
        score += 0.10

    # 5. Regime alignment (weight 0.15)
    if regime.regime in (RegimeID.R1_LOW_VOL_MR, RegimeID.R2_HIGH_VOL_MR):
        # Mean-reverting: failed breakouts are better setups
        if orb.status in (ORBStatus.FAILED_LONG, ORBStatus.FAILED_SHORT):
            signals.append(OpportunitySignal(
                name="MR regime + failed breakout",
                favorable=True,
                weight=0.15,
                description=f"R{regime.regime} mean-reverting regime favors failed ORB reversal",
            ))
            score += 0.12
        elif orb.status in (ORBStatus.BREAKOUT_LONG, ORBStatus.BREAKOUT_SHORT):
            signals.append(OpportunitySignal(
                name="MR regime caution on breakout",
                favorable=False,
                weight=0.10,
                description=f"R{regime.regime} mean-reverting — breakout may fail",
            ))
            score -= 0.05
    elif regime.regime == RegimeID.R3_LOW_VOL_TREND:
        # Trending: breakouts are strong setups
        if orb.status in (ORBStatus.BREAKOUT_LONG, ORBStatus.BREAKOUT_SHORT):
            signals.append(OpportunitySignal(
                name="Trending regime + breakout",
                favorable=True,
                weight=0.15,
                description=f"R3 trending regime supports ORB breakout continuation",
            ))
            score += 0.12

    # 6. RSI alignment (weight 0.10)
    rsi = technicals.rsi.value
    if orb.status == ORBStatus.BREAKOUT_LONG and 40 <= rsi <= 65:
        signals.append(OpportunitySignal(
            name="RSI confirms long",
            favorable=True,
            weight=0.10,
            description=f"RSI {rsi:.0f} — room to run on long breakout",
        ))
        score += 0.08
    elif orb.status == ORBStatus.BREAKOUT_SHORT and 35 <= rsi <= 60:
        signals.append(OpportunitySignal(
            name="RSI confirms short",
            favorable=True,
            weight=0.10,
            description=f"RSI {rsi:.0f} — room to run on short breakout",
        ))
        score += 0.08
    elif orb.status == ORBStatus.FAILED_LONG and rsi > 70:
        signals.append(OpportunitySignal(
            name="RSI overbought on failed long",
            favorable=True,
            weight=0.10,
            description=f"RSI {rsi:.0f} overbought — supports reversal from failed long",
        ))
        score += 0.08
    elif orb.status == ORBStatus.FAILED_SHORT and rsi < 30:
        signals.append(OpportunitySignal(
            name="RSI oversold on failed short",
            favorable=True,
            weight=0.10,
            description=f"RSI {rsi:.0f} oversold — supports reversal from failed short",
        ))
        score += 0.08

    # 7. VWAP alignment (weight 0.10)
    if orb.session_vwap is not None:
        if orb.status == ORBStatus.BREAKOUT_LONG and orb.current_price > orb.session_vwap:
            signals.append(OpportunitySignal(
                name="Above VWAP",
                favorable=True,
                weight=0.10,
                description=f"Breakout above VWAP ({orb.session_vwap:.2f}) — institutional flow",
            ))
            score += 0.08
        elif orb.status == ORBStatus.BREAKOUT_SHORT and orb.current_price < orb.session_vwap:
            signals.append(OpportunitySignal(
                name="Below VWAP",
                favorable=True,
                weight=0.10,
                description=f"Breakout below VWAP ({orb.session_vwap:.2f}) — institutional selling",
            ))
            score += 0.08

    # --- Direction ---
    if orb.status == ORBStatus.BREAKOUT_LONG:
        direction = "bullish"
    elif orb.status == ORBStatus.BREAKOUT_SHORT:
        direction = "bearish"
    elif orb.status == ORBStatus.FAILED_LONG:
        direction = "bearish"  # Reversal from failed long
    elif orb.status == ORBStatus.FAILED_SHORT:
        direction = "bullish"  # Reversal from failed short
    else:
        direction = "neutral"

    # --- Strategy selection ---
    if orb.status in (ORBStatus.BREAKOUT_LONG, ORBStatus.BREAKOUT_SHORT):
        if orb.retest_count > 0:
            strategy = ORBStrategy.BREAKOUT_WITH_RETEST
        else:
            strategy = ORBStrategy.BREAKOUT_CONTINUATION
    elif orb.status in (ORBStatus.FAILED_LONG, ORBStatus.FAILED_SHORT):
        strategy = ORBStrategy.FAILED_BREAKOUT_REVERSAL
    elif orb.status == ORBStatus.WITHIN and orb.range_pct < 0.5:
        strategy = ORBStrategy.NARROW_RANGE_ANTICIPATION
    else:
        strategy = ORBStrategy.NO_TRADE

    # --- Verdict ---
    confidence = max(0.0, min(1.0, score))

    if strategy == ORBStrategy.NO_TRADE:
        verdict = Verdict.NO_GO
    elif confidence >= 0.60:
        verdict = Verdict.GO
    elif confidence >= 0.40:
        verdict = Verdict.CAUTION
    else:
        verdict = Verdict.NO_GO

    # --- Summary ---
    summary_parts = [f"{verdict.upper()}: {ticker} ORB"]
    summary_parts.append(f"{orb.status.value.replace('_', ' ').title()}")
    summary_parts.append(f"Range {orb.range_low:.2f}–{orb.range_high:.2f} ({orb.range_pct:.1f}%)")
    if direction != "neutral":
        summary_parts.append(direction.title())
    summary_parts.append(f"Score: {confidence:.0%}")

    return ORBSetupOpportunity(
        ticker=ticker,
        as_of_date=today,
        verdict=verdict,
        confidence=confidence,
        strategy=strategy,
        direction=direction,
        signals=signals,
        hard_stops=hard_stops,
        orb_status=orb.status.value,
        range_pct=orb.range_pct,
        opening_volume_ratio=orb.opening_volume_ratio,
        range_vs_daily_atr_pct=orb.range_vs_daily_atr_pct,
        regime_id=int(regime.regime),
        summary=" | ".join(summary_parts),
    )
