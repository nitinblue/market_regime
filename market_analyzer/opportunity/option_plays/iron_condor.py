"""Iron condor opportunity assessment — go/no-go + strategy recommendation.

Sell OTM put + OTM call (short strangle) with long further OTM wings for protection.
Defined risk, defined max profit (net credit received).
The #1 income strategy for small accounts — explicitly designed for theta harvesting.

Best in R1 (low vol, mean-reverting = stays in range, ideal for premium selling).
Good in R2 (wider wings needed, but still mean-reverting).
Poor in R3 (directional move blows through one side).
Worst in R4 (explosive moves destroy condors).
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
    from market_analyzer.models.regime import RegimeResult
    from market_analyzer.models.technicals import TechnicalSnapshot
    from market_analyzer.models.vol_surface import VolatilitySurface


# --- Strategy enum ---

class IronCondorStrategy(str):
    STANDARD_IRON_CONDOR = "standard_iron_condor"
    WIDE_IRON_CONDOR = "wide_iron_condor"
    UNBALANCED_IRON_CONDOR = "unbalanced_iron_condor"
    NARROW_IRON_CONDOR = "narrow_iron_condor"
    NO_TRADE = "no_trade"


# --- Opportunity model ---

class IronCondorOpportunity(BaseModel):
    """Iron condor opportunity assessment."""

    ticker: str
    as_of_date: date
    verdict: Verdict
    confidence: float
    hard_stops: list[HardStop]
    signals: list[OpportunitySignal]
    strategy: StrategyRecommendation
    iron_condor_strategy: str
    regime_id: int
    regime_confidence: float
    front_iv: float
    put_skew: float
    call_skew: float
    wing_width_suggestion: str
    days_to_earnings: int | None
    trade_spec: TradeSpec | None = None
    summary: str


# --- Public API ---


def assess_iron_condor(
    ticker: str,
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    vol_surface: VolatilitySurface | None = None,
    fundamentals: FundamentalsSnapshot | None = None,
    as_of: date | None = None,
) -> IronCondorOpportunity:
    """Assess iron condor opportunity for a single instrument.

    Pure function — consumes pre-computed analysis, no data fetching.
    """
    cfg = get_settings().opportunity.iron_condor
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

    # --- Hard stops ---
    hard_stops = _check_hard_stops(regime, vol_surface, days_to_earnings, front_iv, cfg)

    if hard_stops:
        return IronCondorOpportunity(
            ticker=ticker,
            as_of_date=today,
            verdict=Verdict.NO_GO,
            confidence=0.0,
            hard_stops=hard_stops,
            signals=[],
            strategy=_no_trade_rec(),
            iron_condor_strategy=IronCondorStrategy.NO_TRADE,
            regime_id=int(regime.regime),
            regime_confidence=regime.confidence,
            front_iv=front_iv,
            put_skew=put_skew,
            call_skew=call_skew,
            wing_width_suggestion="N/A",
            days_to_earnings=days_to_earnings,
            summary=f"NO_GO: {hard_stops[0].description}",
        )

    assert vol_surface is not None

    # --- Signals ---
    signals = _score_signals(regime, technicals, vol_surface, days_to_earnings, cfg)

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

    # --- Strategy + wing width ---
    ic_strat, strat_rec = _select_strategy(regime, technicals, front_iv, confidence, cfg)
    wing_width = _suggest_wing_width(regime, technicals, front_iv, cfg)

    # --- Trade spec (actionable parameters) ---
    trade_spec = _compute_trade_spec(ticker, technicals, regime, vol_surface) if verdict != Verdict.NO_GO else None

    summary = _build_summary(ticker, verdict, confidence, ic_strat, front_iv, wing_width)

    return IronCondorOpportunity(
        ticker=ticker,
        as_of_date=today,
        verdict=verdict,
        confidence=confidence,
        hard_stops=hard_stops,
        signals=signals,
        strategy=strat_rec,
        iron_condor_strategy=ic_strat,
        regime_id=int(regime.regime),
        regime_confidence=regime.confidence,
        front_iv=front_iv,
        put_skew=put_skew,
        call_skew=call_skew,
        wing_width_suggestion=wing_width,
        days_to_earnings=days_to_earnings,
        trade_spec=trade_spec,
        summary=summary,
    )


# --- Internal helpers ---


def _check_hard_stops(regime, vol_surface, days_to_earnings, front_iv, cfg) -> list[HardStop]:
    stops: list[HardStop] = []

    # R4 at high confidence — explosive moves destroy condors
    if regime.regime == RegimeID.R4_HIGH_VOL_TREND:
        if regime.confidence >= cfg.r4_confidence_threshold:
            stops.append(HardStop(
                name="R4 trending",
                description=f"R4 at {regime.confidence:.0%} — explosive moves destroy iron condors",
            ))

    # R3 at high confidence — persistent trend blows through one side
    if regime.regime == RegimeID.R3_LOW_VOL_TREND:
        if regime.confidence >= cfg.r3_confidence_threshold:
            stops.append(HardStop(
                name="R3 trending",
                description=f"R3 trending at {regime.confidence:.0%} — directional move likely breaches short strike",
            ))

    if vol_surface is None:
        stops.append(HardStop(
            name="No vol surface",
            description="No options chain/vol surface data — cannot assess iron condor",
        ))
        return stops

    # ATM IV too low — not enough premium to justify risk
    if front_iv < cfg.min_iv:
        stops.append(HardStop(
            name="IV too low",
            description=f"ATM IV {front_iv:.1%} below minimum {cfg.min_iv:.1%} — insufficient premium for condor",
        ))

    if vol_surface.data_quality == "poor":
        stops.append(HardStop(
            name="Poor data quality",
            description="Options chain data quality too poor for iron condor assessment",
        ))

    # Earnings imminent
    if days_to_earnings is not None and 0 < days_to_earnings <= cfg.earnings_blackout_days:
        stops.append(HardStop(
            name="Earnings imminent",
            description=f"Earnings in {days_to_earnings} days — gap risk destroys condors",
        ))

    return stops


def _score_signals(regime, technicals, vol_surface, days_to_earnings, cfg):
    signals: list[OpportunitySignal] = []
    regime_id = int(regime.regime)
    rsi = technicals.rsi.value if technicals.rsi else 50.0

    # 1. ATM IV elevated (0.20) — higher IV = wider credit, better premium
    front_iv = vol_surface.front_iv
    if front_iv >= cfg.iv_excellent:
        signals.append(OpportunitySignal(
            name="Excellent IV",
            favorable=True,
            weight=0.20,
            description=f"IV {front_iv:.1%} — fat premium for iron condor",
        ))
    elif front_iv >= cfg.iv_good:
        signals.append(OpportunitySignal(
            name="Good IV",
            favorable=True,
            weight=0.14,
            description=f"IV {front_iv:.1%} — decent premium for condor",
        ))
    elif front_iv >= cfg.min_iv:
        signals.append(OpportunitySignal(
            name="Adequate IV",
            favorable=True,
            weight=0.08,
            description=f"IV {front_iv:.1%} — marginally adequate premium",
        ))
    else:
        signals.append(OpportunitySignal(
            name="Low IV",
            favorable=False,
            weight=0.20,
            description=f"IV {front_iv:.1%} — insufficient premium",
        ))

    # 2. Regime R1/R2 (0.25) — mean-reverting is the core edge
    if regime_id == 1:
        signals.append(OpportunitySignal(
            name="R1 ideal regime",
            favorable=True,
            weight=0.25,
            description="R1 low-vol MR — prime iron condor environment",
        ))
    elif regime_id == 2:
        signals.append(OpportunitySignal(
            name="R2 acceptable regime",
            favorable=True,
            weight=0.20,
            description="R2 high-vol MR — wider wings needed, but still range-bound",
        ))
    elif regime_id == 3:
        signals.append(OpportunitySignal(
            name="R3 trending regime",
            favorable=False,
            weight=0.25,
            description="R3 trending — one side likely breached",
        ))
    else:
        signals.append(OpportunitySignal(
            name="R4 hostile regime",
            favorable=False,
            weight=0.25,
            description="R4 — worst environment for iron condors",
        ))

    # 3. RSI neutral (0.15) — centered price = both short strikes safe
    if 40 <= rsi <= 60:
        signals.append(OpportunitySignal(
            name="Centered RSI",
            favorable=True,
            weight=0.15,
            description=f"RSI {rsi:.0f} — centered, both wings safe",
        ))
    elif 30 <= rsi <= 70:
        signals.append(OpportunitySignal(
            name="RSI acceptable",
            favorable=True,
            weight=0.08,
            description=f"RSI {rsi:.0f} — somewhat centered",
        ))
    else:
        signals.append(OpportunitySignal(
            name="Extreme RSI",
            favorable=False,
            weight=0.15,
            description=f"RSI {rsi:.0f} — price likely to move directionally, one side at risk",
        ))

    # 4. Bollinger bandwidth (0.10) — moderate is ideal
    if hasattr(technicals, "bollinger") and technicals.bollinger:
        bw = technicals.bollinger.bandwidth if hasattr(technicals.bollinger, "bandwidth") else 0.0
        if 0.03 < bw <= 0.08:
            signals.append(OpportunitySignal(
                name="Moderate BB width",
                favorable=True,
                weight=0.10,
                description=f"BB width {bw:.2%} — moderate vol, good for condors",
            ))
        elif bw > 0.08:
            signals.append(OpportunitySignal(
                name="Wide BB bands",
                favorable=True,
                weight=0.06,
                description=f"BB width {bw:.2%} — high vol, wider wings needed",
            ))
        else:
            signals.append(OpportunitySignal(
                name="Compressed BB",
                favorable=False,
                weight=0.10,
                description=f"BB width {bw:.2%} — too compressed, breakout risk",
            ))

    # 5. Skew balance (0.10) — balanced = both wings well-priced
    if vol_surface.skew_by_expiry:
        skew = vol_surface.skew_by_expiry[0]
        ratio = skew.skew_ratio
        if 0.5 <= ratio <= 2.0:
            signals.append(OpportunitySignal(
                name="Balanced skew",
                favorable=True,
                weight=0.10,
                description=f"Skew ratio {ratio:.1f} — both wings fairly priced",
            ))
        else:
            signals.append(OpportunitySignal(
                name="Skewed pricing",
                favorable=False,
                weight=0.10,
                description=f"Skew ratio {ratio:.1f} — consider unbalanced condor",
            ))

    # 6. No earnings (0.10)
    if days_to_earnings is None or days_to_earnings > 30:
        signals.append(OpportunitySignal(
            name="Clear earnings window",
            favorable=True,
            weight=0.10,
            description="No near-term earnings — no gap risk",
        ))
    elif days_to_earnings > cfg.earnings_blackout_days:
        signals.append(OpportunitySignal(
            name="Earnings proximity",
            favorable=False,
            weight=0.10,
            description=f"Earnings in {days_to_earnings} days — elevated gap risk",
        ))

    # 7. Term structure contango (0.05)
    if vol_surface.is_contango:
        signals.append(OpportunitySignal(
            name="Contango term structure",
            favorable=True,
            weight=0.05,
            description="Term structure in contango — time decay favorable",
        ))
    elif vol_surface.is_backwardation:
        signals.append(OpportunitySignal(
            name="Backwardation warning",
            favorable=False,
            weight=0.05,
            description="Term structure inverted — near-term event risk elevated",
        ))

    # 8. Liquidity (0.05)
    if vol_surface.data_quality == "good":
        signals.append(OpportunitySignal(
            name="Good liquidity",
            favorable=True,
            weight=0.05,
            description="Good options chain liquidity — tight fills expected",
        ))
    elif vol_surface.data_quality == "fair":
        signals.append(OpportunitySignal(
            name="Fair liquidity",
            favorable=True,
            weight=0.02,
            description="Fair liquidity — wider fills possible",
        ))

    return signals


def _select_strategy(regime, technicals, front_iv, confidence, cfg):
    if confidence < 0.30:
        return IronCondorStrategy.NO_TRADE, _no_trade_rec()

    regime_id = int(regime.regime)
    rsi = technicals.rsi.value if technicals.rsi else 50.0

    # R1 high confidence → standard iron condor (the bread and butter)
    if regime_id == 1 and confidence >= 0.55:
        return IronCondorStrategy.STANDARD_IRON_CONDOR, StrategyRecommendation(
            name="Standard Iron Condor",
            direction="neutral",
            structure="Sell OTM put + OTM call, buy further OTM put + call (equal-width wings)",
            rationale="R1 low-vol MR — prime environment for premium selling with defined risk",
            risk_notes=[
                "Max loss = wing width - credit received",
                "Profit zone = between short strikes",
                "Manage at 50% max profit or 21 DTE",
            ],
        )

    # R2 → wide iron condor (bigger swings need wider strikes)
    if regime_id == 2 and confidence >= 0.45:
        return IronCondorStrategy.WIDE_IRON_CONDOR, StrategyRecommendation(
            name="Wide Iron Condor",
            direction="neutral",
            structure="Sell wider OTM put + OTM call, buy further OTM wings",
            rationale="R2 high-vol MR — wider short strikes to accommodate bigger swings",
            risk_notes=[
                "Wider strikes = less premium but more room",
                "Consider 2x wing width vs standard",
                "More patient management — may need to hold longer",
            ],
        )

    # RSI skewed → unbalanced condor (bias toward the neutral side)
    if regime_id in (1, 2) and (rsi > 60 or rsi < 40):
        direction = "slightly bearish" if rsi > 60 else "slightly bullish"
        wide_side = "call" if rsi > 60 else "put"
        return IronCondorStrategy.UNBALANCED_IRON_CONDOR, StrategyRecommendation(
            name="Unbalanced Iron Condor",
            direction=direction,
            structure=f"Wider {wide_side} wing — directional tilt to account for RSI {rsi:.0f}",
            rationale=f"RSI {rsi:.0f} suggests mild bias — widen the vulnerable side",
            risk_notes=[
                "Asymmetric risk/reward profile",
                "More premium on the side closer to price",
                "Monitor RSI reversion for early exit",
            ],
        )

    # Low IV → narrow condor (tighter strikes for meaningful premium)
    if front_iv < cfg.iv_good:
        return IronCondorStrategy.NARROW_IRON_CONDOR, StrategyRecommendation(
            name="Narrow Iron Condor",
            direction="neutral",
            structure="Tighter short strikes for higher premium-to-risk ratio",
            rationale=f"IV at {front_iv:.1%} — narrower strikes collect more premium relative to risk",
            risk_notes=[
                "Smaller profit zone — requires tighter management",
                "Higher probability of touching a short strike",
                "Consider smaller position size to compensate",
            ],
        )

    # Fallback for R1/R2
    if regime_id in (1, 2):
        return IronCondorStrategy.STANDARD_IRON_CONDOR, StrategyRecommendation(
            name="Standard Iron Condor",
            direction="neutral",
            structure="Sell OTM put + OTM call, buy further OTM put + call",
            rationale="Mean-reverting regime — standard premium collection",
            risk_notes=["Max loss = wing width - credit", "Manage at 50% profit"],
        )

    return IronCondorStrategy.NO_TRADE, _no_trade_rec()


def _suggest_wing_width(regime, technicals, front_iv, cfg) -> str:
    """Suggest wing width based on regime and ATR."""
    atr_pct = technicals.atr_pct if hasattr(technicals, "atr_pct") else 1.0
    regime_id = int(regime.regime)

    if regime_id == 1:
        # R1: short strikes at ~1 ATR, wings at ~1.5 ATR
        return f"Short strikes ~{atr_pct:.1f}% OTM ({atr_pct * technicals.current_price / 100:.0f}pt), wings +{atr_pct * 0.5:.1f}% beyond"
    elif regime_id == 2:
        # R2: wider — short strikes at ~1.5 ATR, wings at ~2 ATR
        mult = 1.5
        return f"Short strikes ~{atr_pct * mult:.1f}% OTM ({atr_pct * mult * technicals.current_price / 100:.0f}pt), wings +{atr_pct * 0.5:.1f}% beyond"
    else:
        return f"Short strikes ~{atr_pct * 1.2:.1f}% OTM, wings +{atr_pct * 0.5:.1f}% beyond"


def _no_trade_rec() -> StrategyRecommendation:
    return StrategyRecommendation(
        name="No Trade",
        direction="neutral",
        structure="No position",
        rationale="Conditions not favorable for iron condor",
        risk_notes=[],
    )


def _compute_trade_spec(ticker, technicals, regime, vol_surface) -> TradeSpec | None:
    """Compute actionable trade parameters for iron condor."""
    from market_analyzer.opportunity.option_plays._trade_spec_helpers import build_single_expiry_trade_spec

    return build_single_expiry_trade_spec(
        ticker=ticker,
        price=technicals.current_price,
        atr=technicals.atr,
        regime_id=int(regime.regime),
        vol_surface=vol_surface,
        structure_type="iron_condor",
        target_dte_min=30,
        target_dte_max=45,
    )


def _build_summary(ticker, verdict, confidence, ic_strat, front_iv, wing_width) -> str:
    parts = [f"{verdict.upper()}: {ticker}"]
    parts.append(f"Iron Condor: {ic_strat}")
    parts.append(f"IV: {front_iv:.1%}")
    parts.append(f"Score: {confidence:.0%}")
    if verdict != Verdict.NO_GO:
        parts.append(f"Wings: {wing_width}")
    return " | ".join(parts)
