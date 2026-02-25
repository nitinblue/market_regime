"""Ratio spread opportunity assessment — go/no-go + strategy recommendation.

Buy 1 ATM, sell 2 OTM (or other ratios like 1:2, 2:3).
Profits from premium collection + limited directional move.
Naked leg on the short side = margin-intensive, undefined risk.
Best in R1 (low vol, range-bound) or R3 (mild trend, sell against trend end).
Steep skew makes ratios attractive (selling expensive OTM).
"""

from __future__ import annotations

from datetime import date
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
    from market_analyzer.models.phase import PhaseResult
    from market_analyzer.models.regime import RegimeResult
    from market_analyzer.models.technicals import TechnicalSnapshot
    from market_analyzer.models.vol_surface import VolatilitySurface


# --- Strategy enum ---

class RatioSpreadStrategy(str):
    CALL_RATIO_SPREAD = "call_ratio_spread"
    PUT_RATIO_SPREAD = "put_ratio_spread"
    CALL_BACK_RATIO = "call_back_ratio"
    PUT_BACK_RATIO = "put_back_ratio"
    NO_TRADE = "no_trade"


# --- Opportunity model ---

class RatioSpreadOpportunity(BaseModel):
    """Ratio spread opportunity assessment."""

    ticker: str
    as_of_date: date
    verdict: Verdict
    confidence: float
    hard_stops: list[HardStop]
    signals: list[OpportunitySignal]
    strategy: StrategyRecommendation
    ratio_strategy: str
    regime_id: int
    regime_confidence: float
    direction: str  # "bullish" | "bearish"
    has_naked_leg: bool
    margin_warning: str | None
    front_iv: float
    put_skew: float
    call_skew: float
    days_to_earnings: int | None
    trade_spec: TradeSpec | None = None
    summary: str


# --- Public API ---


def assess_ratio_spread(
    ticker: str,
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    vol_surface: VolatilitySurface | None = None,
    phase: PhaseResult | None = None,
    fundamentals: FundamentalsSnapshot | None = None,
    as_of: date | None = None,
) -> RatioSpreadOpportunity:
    """Assess ratio spread opportunity for a single instrument.

    Pure function — consumes pre-computed analysis, no data fetching.
    """
    cfg = get_settings().opportunity.ratio_spread
    today = as_of or date.today()

    days_to_earnings: int | None = None
    if fundamentals is not None:
        days_to_earnings = fundamentals.upcoming_events.days_to_earnings

    front_iv = vol_surface.front_iv if vol_surface else 0.0
    put_skew = 0.0
    call_skew = 0.0
    if vol_surface and vol_surface.skew_by_expiry:
        put_skew = vol_surface.skew_by_expiry[0].put_skew
        call_skew = vol_surface.skew_by_expiry[0].call_skew

    direction = _determine_direction(regime, technicals, phase)

    # --- Hard stops ---
    hard_stops = _check_hard_stops(regime, vol_surface, days_to_earnings, put_skew, call_skew, cfg)

    if hard_stops:
        return RatioSpreadOpportunity(
            ticker=ticker,
            as_of_date=today,
            verdict=Verdict.NO_GO,
            confidence=0.0,
            hard_stops=hard_stops,
            signals=[],
            strategy=_no_trade_rec(),
            ratio_strategy=RatioSpreadStrategy.NO_TRADE,
            regime_id=int(regime.regime),
            regime_confidence=regime.confidence,
            direction=direction,
            has_naked_leg=True,
            margin_warning="Position not recommended — hard stop triggered",
            front_iv=front_iv,
            put_skew=put_skew,
            call_skew=call_skew,
            days_to_earnings=days_to_earnings,
            summary=f"NO_GO: {hard_stops[0].description}",
        )

    assert vol_surface is not None

    # --- Signals ---
    signals = _score_signals(regime, technicals, vol_surface, phase, days_to_earnings, direction, cfg)

    # --- Confidence ---
    raw = sum(s.weight for s in signals if s.favorable)
    regime_mult = cfg.regime_multipliers.get(int(regime.regime), 0.5)
    confidence = min(1.0, raw * regime_mult)

    # --- Verdict ---
    if confidence >= cfg.go_threshold:
        verdict = Verdict.GO
    elif confidence >= cfg.caution_threshold:
        verdict = Verdict.CAUTION
    else:
        verdict = Verdict.NO_GO

    # --- Strategy ---
    ratio_strat, strat_rec, has_naked, margin_warn = _select_strategy(
        regime, technicals, direction, confidence, cfg,
    )

    # --- Trade spec ---
    trade_spec = _compute_trade_spec(ticker, technicals, regime, vol_surface, direction) if verdict != Verdict.NO_GO else None

    summary = _build_summary(ticker, verdict, confidence, ratio_strat, direction, has_naked)

    return RatioSpreadOpportunity(
        ticker=ticker,
        as_of_date=today,
        verdict=verdict,
        confidence=confidence,
        hard_stops=hard_stops,
        signals=signals,
        strategy=strat_rec,
        ratio_strategy=ratio_strat,
        regime_id=int(regime.regime),
        regime_confidence=regime.confidence,
        direction=direction,
        has_naked_leg=has_naked,
        margin_warning=margin_warn,
        front_iv=front_iv,
        put_skew=put_skew,
        call_skew=call_skew,
        days_to_earnings=days_to_earnings,
        trade_spec=trade_spec,
        summary=summary,
    )


# --- Internal helpers ---


def _determine_direction(regime, technicals, phase) -> str:
    rsi = technicals.rsi.value if technicals.rsi else 50.0
    phase_id = int(phase.phase) if phase else 0

    if phase_id == 2 or rsi >= 55:
        return "bullish"
    elif phase_id == 4 or rsi <= 45:
        return "bearish"
    return "bullish"  # Default to bullish for ratio spreads


def _check_hard_stops(regime, vol_surface, days_to_earnings, put_skew, call_skew, cfg) -> list[HardStop]:
    stops: list[HardStop] = []

    # R4 at any confidence — naked leg + explosive moves = disaster
    if regime.regime == RegimeID.R4_HIGH_VOL_TREND:
        stops.append(HardStop(
            name="R4 explosive moves",
            description="R4 (high-vol trending) — naked leg + explosive moves is catastrophic",
        ))

    # R2 high confidence — wide swings hit naked leg
    if regime.regime == RegimeID.R2_HIGH_VOL_MR and regime.confidence >= cfg.r2_confidence_threshold:
        stops.append(HardStop(
            name="R2 high vol swings",
            description=f"R2 (high-vol MR) at {regime.confidence:.0%} — wide swings threaten naked leg",
        ))

    if vol_surface is None:
        stops.append(HardStop(
            name="No vol surface",
            description="No options chain/vol surface data — cannot assess ratio spread",
        ))
        return stops

    if vol_surface.data_quality == "poor":
        stops.append(HardStop(
            name="Poor data quality",
            description="Options chain data too poor for ratio spread assessment",
        ))

    # Earnings imminent
    if days_to_earnings is not None and 0 < days_to_earnings <= cfg.earnings_blackout_days:
        stops.append(HardStop(
            name="Earnings imminent",
            description=f"Earnings in {days_to_earnings} days — gap risk on naked leg",
        ))

    # Skew too flat — no edge in selling OTM
    max_skew = max(abs(put_skew), abs(call_skew))
    if max_skew < cfg.min_skew_pct:
        stops.append(HardStop(
            name="Skew too flat",
            description=f"Skew too flat (max {max_skew:.2%}) — no edge selling OTM",
        ))

    return stops


def _score_signals(regime, technicals, vol_surface, phase, days_to_earnings, direction, cfg):
    signals: list[OpportunitySignal] = []
    regime_id = int(regime.regime)

    # 1. Skew steep enough (0.20)
    if vol_surface.skew_by_expiry:
        front_skew = vol_surface.skew_by_expiry[0]
        if direction == "bullish" and front_skew.put_skew > cfg.min_skew_pct:
            signals.append(OpportunitySignal(
                name="Put skew steep",
                favorable=True,
                weight=0.20,
                description=f"Put skew {front_skew.put_skew:.2%} — OTM puts overpriced, sell them",
            ))
        elif direction == "bearish" and front_skew.call_skew > cfg.min_skew_pct:
            signals.append(OpportunitySignal(
                name="Call skew steep",
                favorable=True,
                weight=0.20,
                description=f"Call skew {front_skew.call_skew:.2%} — OTM calls overpriced, sell them",
            ))
        else:
            max_skew = max(abs(front_skew.put_skew), abs(front_skew.call_skew))
            signals.append(OpportunitySignal(
                name="Moderate skew",
                favorable=True,
                weight=0.10,
                description=f"Skew present ({max_skew:.2%}) — some edge",
            ))

    # 2. Regime R1/R3 (0.20)
    if regime_id in (1, 3):
        signals.append(OpportunitySignal(
            name="Favorable regime",
            favorable=True,
            weight=0.20,
            description=f"R{regime_id} — {'range-bound' if regime_id == 1 else 'mild trend'}, good for ratio",
        ))
    elif regime_id == 2:
        signals.append(OpportunitySignal(
            name="R2 moderate",
            favorable=True,
            weight=0.08,
            description="R2 (high-vol MR) — use with caution, wider strikes needed",
        ))
    else:
        signals.append(OpportunitySignal(
            name="Unfavorable regime",
            favorable=False,
            weight=0.20,
            description=f"R{regime_id} — not suited for ratio spreads",
        ))

    # 3. ATM IV level (0.15)
    front_iv = vol_surface.front_iv
    if front_iv >= 0.25:
        signals.append(OpportunitySignal(
            name="Elevated IV",
            favorable=True,
            weight=0.15,
            description=f"IV {front_iv:.1%} — good premium for ratio spread",
        ))
    elif front_iv >= 0.15:
        signals.append(OpportunitySignal(
            name="Moderate IV",
            favorable=True,
            weight=0.08,
            description=f"IV {front_iv:.1%} — acceptable",
        ))
    else:
        signals.append(OpportunitySignal(
            name="Low IV",
            favorable=False,
            weight=0.15,
            description=f"IV {front_iv:.1%} — thin premium for ratio",
        ))

    # 4. Direction confidence (0.10)
    rsi = technicals.rsi.value if technicals.rsi else 50.0
    if direction == "bullish" and rsi >= 50:
        signals.append(OpportunitySignal(
            name="Bullish direction confirmed",
            favorable=True,
            weight=0.10,
            description=f"RSI {rsi:.0f} confirms bullish bias for call ratio",
        ))
    elif direction == "bearish" and rsi <= 50:
        signals.append(OpportunitySignal(
            name="Bearish direction confirmed",
            favorable=True,
            weight=0.10,
            description=f"RSI {rsi:.0f} confirms bearish bias for put ratio",
        ))
    else:
        signals.append(OpportunitySignal(
            name="Weak directional signal",
            favorable=False,
            weight=0.10,
            description=f"RSI {rsi:.0f} — weak directional conviction",
        ))

    # 5. Phase alignment (0.10)
    if phase is not None:
        p = int(phase.phase)
        if (direction == "bullish" and p == 2) or (direction == "bearish" and p == 4):
            signals.append(OpportunitySignal(
                name="Phase confirms direction",
                favorable=True,
                weight=0.10,
                description=f"Phase P{p} confirms {direction} for ratio spread",
            ))

    # 6. Support/resistance as backstop (0.10)
    # Without levels data, use ATR as proxy for backstop distance
    atr_pct = technicals.atr_pct
    if atr_pct < 1.5:
        signals.append(OpportunitySignal(
            name="Contained ATR",
            favorable=True,
            weight=0.10,
            description=f"ATR {atr_pct:.2f}% — moves contained, backstop holds",
        ))
    elif atr_pct < 2.5:
        signals.append(OpportunitySignal(
            name="Moderate ATR",
            favorable=True,
            weight=0.05,
            description=f"ATR {atr_pct:.2f}% — moderate moves",
        ))

    # 7. No earnings (0.10)
    if days_to_earnings is None or days_to_earnings > 30:
        signals.append(OpportunitySignal(
            name="Clear earnings window",
            favorable=True,
            weight=0.10,
            description="No near-term earnings — naked leg safer",
        ))

    # 8. Liquidity (0.05)
    if vol_surface.data_quality == "good":
        signals.append(OpportunitySignal(
            name="Good liquidity",
            favorable=True,
            weight=0.05,
            description="Good options chain data quality",
        ))

    return signals


def _select_strategy(regime, technicals, direction, confidence, cfg):
    if confidence < 0.30:
        return RatioSpreadStrategy.NO_TRADE, _no_trade_rec(), False, None

    regime_id = int(regime.regime)

    # Standard ratio spreads (1:2 — has naked leg)
    if direction == "bullish":
        return (
            RatioSpreadStrategy.CALL_RATIO_SPREAD,
            StrategyRecommendation(
                name="Call Ratio Spread (1:2)",
                direction="bullish",
                structure="Buy 1 ATM call, sell 2 OTM calls",
                rationale=f"R{regime_id} + bullish — profit from limited move up + skew premium",
                risk_notes=[
                    "NAKED LEG: Unlimited risk above upper breakeven",
                    "Margin-intensive position",
                    "Close if price approaches upper strike",
                ],
            ),
            True,
            f"Naked call leg — ensure margin available (estimated {cfg.margin_warning_threshold} account minimum)",
        )

    if direction == "bearish":
        return (
            RatioSpreadStrategy.PUT_RATIO_SPREAD,
            StrategyRecommendation(
                name="Put Ratio Spread (1:2)",
                direction="bearish",
                structure="Buy 1 ATM put, sell 2 OTM puts",
                rationale=f"R{regime_id} + bearish — profit from limited move down + skew premium",
                risk_notes=[
                    "NAKED LEG: Significant risk below lower breakeven",
                    "Margin-intensive position",
                    "Close if price approaches lower strike",
                ],
            ),
            True,
            f"Naked put leg — ensure margin available (estimated {cfg.margin_warning_threshold} account minimum)",
        )

    return RatioSpreadStrategy.NO_TRADE, _no_trade_rec(), False, None


def _compute_trade_spec(ticker, technicals, regime, vol_surface, direction) -> TradeSpec | None:
    """Compute actionable trade parameters for ratio spread."""
    from market_analyzer.opportunity.option_plays._trade_spec_helpers import build_single_expiry_trade_spec

    return build_single_expiry_trade_spec(
        ticker=ticker,
        price=technicals.current_price,
        atr=technicals.atr,
        regime_id=int(regime.regime),
        vol_surface=vol_surface,
        structure_type="ratio_spread",
        target_dte_min=30,
        target_dte_max=45,
        direction=direction,
    )


def _no_trade_rec() -> StrategyRecommendation:
    return StrategyRecommendation(
        name="No Trade",
        direction="neutral",
        structure="No position",
        rationale="Conditions not favorable for ratio spread",
        risk_notes=[],
    )


def _build_summary(ticker, verdict, confidence, ratio_strat, direction, has_naked) -> str:
    parts = [f"{verdict.upper()}: {ticker}"]
    parts.append(f"Ratio: {ratio_strat}")
    parts.append(f"Direction: {direction}")
    if has_naked:
        parts.append("NAKED LEG")
    parts.append(f"Score: {confidence:.0%}")
    return " | ".join(parts)
