"""Earnings play opportunity assessment — go/no-go for earnings-related trades."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel

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
    from market_analyzer.models.regime import RegimeResult
    from market_analyzer.models.technicals import TechnicalSnapshot
    from market_analyzer.models.vol_surface import VolatilitySurface


class EarningsPlayStrategy(StrEnum):
    """Strategy types for earnings plays."""

    PRE_EARNINGS_STRADDLE = "pre_earnings_straddle"
    POST_EARNINGS_DRIFT = "post_earnings_drift"
    IV_CRUSH_SELL = "iv_crush_sell"
    NO_TRADE = "no_trade"


class EarningsOpportunity(BaseModel):
    """Result container for earnings play assessment."""

    ticker: str
    as_of_date: date
    verdict: Verdict
    confidence: float
    strategy: str
    direction: str
    signals: list[OpportunitySignal]
    hard_stops: list[HardStop]
    days_to_earnings: int | None
    regime_id: int
    trade_spec: TradeSpec | None = None
    summary: str


def assess_earnings_play(
    ticker: str,
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    fundamentals: FundamentalsSnapshot | None = None,
    vol_surface: VolatilitySurface | None = None,
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

    # --- Trade spec ---
    trade_spec = None
    if verdict != Verdict.NO_GO and strategy != EarningsPlayStrategy.NO_TRADE:
        trade_spec = _build_earnings_trade_spec(
            ticker, technicals, strategy, vol_surface, days_to_earnings,
        )

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
        trade_spec=trade_spec,
        summary=" | ".join(summary_parts),
    )


def _build_earnings_trade_spec(
    ticker: str,
    technicals: TechnicalSnapshot,
    strategy: EarningsPlayStrategy,
    vol_surface: VolatilitySurface | None,
    days_to_earnings: int,
) -> TradeSpec | None:
    """Build trade spec for earnings play."""
    from market_analyzer.opportunity.option_plays._trade_spec_helpers import (
        build_iron_butterfly_legs,
        build_straddle_legs,
        find_best_expiration,
    )
    from market_analyzer.models.opportunity import OrderSide, StructureType

    if vol_surface is None or not vol_surface.term_structure:
        return None

    price = technicals.current_price
    atr = technicals.atr

    # Target expiration close to earnings
    target_dte = max(days_to_earnings, 1)
    exp_pt = find_best_expiration(vol_surface.term_structure, target_dte, target_dte + 7)
    if exp_pt is None:
        exp_pt = find_best_expiration(vol_surface.term_structure, 0, 30)
    if exp_pt is None:
        return None

    try:
        if strategy == EarningsPlayStrategy.PRE_EARNINGS_STRADDLE:
            legs = build_straddle_legs(
                price, "buy", exp_pt.expiration, exp_pt.days_to_expiry, exp_pt.atm_iv,
            )
            return TradeSpec(
                ticker=ticker, legs=legs, underlying_price=price,
                target_dte=exp_pt.days_to_expiry, target_expiration=exp_pt.expiration,
                spec_rationale=f"Pre-earnings long straddle. {exp_pt.days_to_expiry} DTE, earnings in {days_to_earnings}d.",
                structure_type=StructureType.STRADDLE,
                order_side=OrderSide.DEBIT,
                profit_target_pct=0.50,
                stop_loss_pct=0.50,
                max_profit_desc="Unlimited (long straddle — profits from big move either direction)",
                max_loss_desc="Net debit paid (both premiums)",
                exit_notes=["Close morning after earnings release",
                            "IV crush will reduce value — need big move to overcome",
                            "Close at 50% loss if move doesn't materialize pre-earnings"],
            )

        elif strategy == EarningsPlayStrategy.IV_CRUSH_SELL:
            legs, wing_width = build_iron_butterfly_legs(
                price, atr, 2, exp_pt.expiration, exp_pt.days_to_expiry, exp_pt.atm_iv,
            )
            return TradeSpec(
                ticker=ticker, legs=legs, underlying_price=price,
                target_dte=exp_pt.days_to_expiry, target_expiration=exp_pt.expiration,
                wing_width_points=wing_width,
                spec_rationale=f"IV crush iron butterfly. {exp_pt.days_to_expiry} DTE, earnings in {days_to_earnings}d.",
                structure_type=StructureType.IRON_BUTTERFLY,
                order_side=OrderSide.CREDIT,
                profit_target_pct=0.50,
                stop_loss_pct=2.0,
                max_profit_desc="Credit received (amplified by IV crush post-earnings)",
                max_loss_desc=f"Wing width (${wing_width:.0f}) minus credit",
                exit_notes=["Close morning after earnings release",
                            "IV crush is your edge — close quickly to capture",
                            "Close if underlying gaps beyond wing strikes"],
            )
    except Exception:
        return None

    return None
