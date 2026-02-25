"""Iron butterfly opportunity assessment — go/no-go + strategy recommendation.

Short ATM straddle + long OTM wings (defined risk).
Maximum premium collection at ATM, wings cap risk.
Best in R2 (high IV = fat premium, mean-reverting = stays near center).
Dangerous in R3/R4 (directional moves blow through wings).
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

class IronButterflyStrategy(str):
    STANDARD_IRON_BUTTERFLY = "standard_iron_butterfly"
    BROKEN_WING_BUTTERFLY = "broken_wing_butterfly"
    WIDE_IRON_BUTTERFLY = "wide_iron_butterfly"
    NO_TRADE = "no_trade"


# --- Opportunity model ---

class IronButterflyOpportunity(BaseModel):
    """Iron butterfly opportunity assessment."""

    ticker: str
    as_of_date: date
    verdict: Verdict
    confidence: float
    hard_stops: list[HardStop]
    signals: list[OpportunitySignal]
    strategy: StrategyRecommendation
    iron_butterfly_strategy: str
    regime_id: int
    regime_confidence: float
    atm_iv: float
    front_iv: float
    days_to_earnings: int | None
    trade_spec: TradeSpec | None = None
    summary: str


# --- Public API ---


def assess_iron_butterfly(
    ticker: str,
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    vol_surface: VolatilitySurface | None = None,
    fundamentals: FundamentalsSnapshot | None = None,
    as_of: date | None = None,
) -> IronButterflyOpportunity:
    """Assess iron butterfly opportunity for a single instrument.

    Pure function — consumes pre-computed analysis, no data fetching.
    """
    cfg = get_settings().opportunity.iron_butterfly
    today = as_of or date.today()

    days_to_earnings: int | None = None
    if fundamentals is not None:
        days_to_earnings = fundamentals.upcoming_events.days_to_earnings

    front_iv = vol_surface.front_iv if vol_surface else 0.0
    atm_iv = front_iv  # Best approximation from vol surface

    # --- Hard stops ---
    hard_stops = _check_hard_stops(regime, vol_surface, days_to_earnings, front_iv, cfg)

    if hard_stops:
        return IronButterflyOpportunity(
            ticker=ticker,
            as_of_date=today,
            verdict=Verdict.NO_GO,
            confidence=0.0,
            hard_stops=hard_stops,
            signals=[],
            strategy=_no_trade_rec(),
            iron_butterfly_strategy=IronButterflyStrategy.NO_TRADE,
            regime_id=int(regime.regime),
            regime_confidence=regime.confidence,
            atm_iv=atm_iv,
            front_iv=front_iv,
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

    # --- Strategy ---
    ifly_strat, strat_rec = _select_strategy(regime, technicals, confidence)

    # --- Trade spec ---
    trade_spec = _compute_trade_spec(ticker, technicals, regime, vol_surface) if verdict != Verdict.NO_GO else None

    summary = _build_summary(ticker, verdict, confidence, ifly_strat, atm_iv)

    return IronButterflyOpportunity(
        ticker=ticker,
        as_of_date=today,
        verdict=verdict,
        confidence=confidence,
        hard_stops=hard_stops,
        signals=signals,
        strategy=strat_rec,
        iron_butterfly_strategy=ifly_strat,
        regime_id=int(regime.regime),
        regime_confidence=regime.confidence,
        atm_iv=atm_iv,
        front_iv=front_iv,
        days_to_earnings=days_to_earnings,
        trade_spec=trade_spec,
        summary=summary,
    )


# --- Internal helpers ---


def _check_hard_stops(regime, vol_surface, days_to_earnings, front_iv, cfg) -> list[HardStop]:
    stops: list[HardStop] = []

    # R3/R4 at high confidence — trending kills butterflies
    if regime.regime in (RegimeID.R3_LOW_VOL_TREND, RegimeID.R4_HIGH_VOL_TREND):
        if regime.confidence >= cfg.trending_confidence_threshold:
            stops.append(HardStop(
                name="Trending regime",
                description=f"R{int(regime.regime)} trending at {regime.confidence:.0%} — directional moves destroy butterflies",
            ))

    if vol_surface is None:
        stops.append(HardStop(
            name="No vol surface",
            description="No options chain/vol surface data — cannot assess iron butterfly",
        ))
        return stops

    # ATM IV too low
    if front_iv < cfg.min_atm_iv:
        stops.append(HardStop(
            name="IV too low",
            description=f"ATM IV {front_iv:.1%} below minimum {cfg.min_atm_iv:.1%} — not enough premium",
        ))

    if vol_surface.data_quality == "poor":
        stops.append(HardStop(
            name="Poor data quality",
            description="Options chain data quality too poor for iron butterfly assessment",
        ))

    # Earnings imminent
    if days_to_earnings is not None and 0 < days_to_earnings <= cfg.earnings_blackout_days:
        stops.append(HardStop(
            name="Earnings imminent",
            description=f"Earnings in {days_to_earnings} days — gap risk destroys butterflies",
        ))

    return stops


def _score_signals(regime, technicals, vol_surface, days_to_earnings, cfg):
    signals: list[OpportunitySignal] = []
    regime_id = int(regime.regime)
    rsi = technicals.rsi.value if technicals.rsi else 50.0

    # 1. ATM IV elevated (0.25) — core edge
    front_iv = vol_surface.front_iv
    if front_iv >= cfg.atm_iv_excellent:
        signals.append(OpportunitySignal(
            name="Excellent ATM IV",
            favorable=True,
            weight=0.25,
            description=f"ATM IV {front_iv:.1%} — very fat premium for butterfly",
        ))
    elif front_iv >= cfg.atm_iv_good:
        signals.append(OpportunitySignal(
            name="Good ATM IV",
            favorable=True,
            weight=0.18,
            description=f"ATM IV {front_iv:.1%} — decent premium",
        ))
    elif front_iv >= cfg.min_atm_iv:
        signals.append(OpportunitySignal(
            name="Adequate ATM IV",
            favorable=True,
            weight=0.10,
            description=f"ATM IV {front_iv:.1%} — marginally adequate",
        ))
    else:
        signals.append(OpportunitySignal(
            name="Low ATM IV",
            favorable=False,
            weight=0.25,
            description=f"ATM IV {front_iv:.1%} — insufficient premium",
        ))

    # 2. Regime R1/R2 (0.25) — mean-reverting critical
    if regime_id in (1, 2):
        signals.append(OpportunitySignal(
            name="Mean-reverting regime",
            favorable=True,
            weight=0.25,
            description=f"R{regime_id} mean-reverting — ideal for iron butterfly",
        ))
    elif regime_id == 3:
        signals.append(OpportunitySignal(
            name="Mild trend regime",
            favorable=False,
            weight=0.25,
            description="R3 trending — butterflies at risk of directional move",
        ))
    else:
        signals.append(OpportunitySignal(
            name="High-vol trend regime",
            favorable=False,
            weight=0.25,
            description="R4 — worst environment for iron butterfly",
        ))

    # 3. RSI neutral (0.15) — 40-60 range ideal
    if 40 <= rsi <= 60:
        signals.append(OpportunitySignal(
            name="Centered RSI",
            favorable=True,
            weight=0.15,
            description=f"RSI {rsi:.0f} — centered, ideal for ATM butterfly",
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
            description=f"RSI {rsi:.0f} — price likely to move directionally",
        ))

    # 4. Bollinger bandwidth elevated (0.10)
    if hasattr(technicals, "bollinger") and technicals.bollinger:
        bb = technicals.bollinger
        bw = bb.bandwidth if hasattr(bb, "bandwidth") else 0.0
        if bw > 0.06:
            signals.append(OpportunitySignal(
                name="Wide Bollinger bands",
                favorable=True,
                weight=0.10,
                description=f"BB width {bw:.2%} — high vol = good premium",
            ))
        elif bw > 0.03:
            signals.append(OpportunitySignal(
                name="Moderate BB width",
                favorable=True,
                weight=0.05,
                description=f"BB width {bw:.2%} — moderate premium",
            ))

    # 5. Skew symmetry (0.10) — balanced put/call preferred
    if vol_surface.skew_by_expiry:
        front_skew = vol_surface.skew_by_expiry[0]
        if abs(front_skew.skew_ratio) < 1.5:
            signals.append(OpportunitySignal(
                name="Balanced skew",
                favorable=True,
                weight=0.10,
                description=f"Skew ratio {front_skew.skew_ratio:.1f} — balanced wings",
            ))
        else:
            signals.append(OpportunitySignal(
                name="Asymmetric skew",
                favorable=False,
                weight=0.10,
                description=f"Skew ratio {front_skew.skew_ratio:.1f} — consider broken wing",
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

    # 7. Liquidity (0.05)
    if vol_surface.data_quality == "good":
        signals.append(OpportunitySignal(
            name="Good liquidity",
            favorable=True,
            weight=0.05,
            description="Good options chain liquidity",
        ))

    return signals


def _select_strategy(regime, technicals, confidence):
    if confidence < 0.30:
        return IronButterflyStrategy.NO_TRADE, _no_trade_rec()

    regime_id = int(regime.regime)
    rsi = technicals.rsi.value if technicals.rsi else 50.0

    # R2 high confidence → standard iron butterfly (maximum premium)
    if regime_id == 2 and confidence >= 0.55:
        return IronButterflyStrategy.STANDARD_IRON_BUTTERFLY, StrategyRecommendation(
            name="Standard Iron Butterfly",
            direction="neutral",
            structure="Short ATM call + put, long OTM call + put (equal-width wings)",
            rationale="R2 high-vol MR — maximum premium collection at ATM",
            risk_notes=["Max loss = wing width - credit received", "Profit zone narrow around ATM"],
        )

    # R1 + slight directional bias → broken wing
    if regime_id == 1 and (rsi > 55 or rsi < 45):
        direction = "bullish" if rsi > 55 else "bearish"
        return IronButterflyStrategy.BROKEN_WING_BUTTERFLY, StrategyRecommendation(
            name="Broken Wing Butterfly",
            direction=direction,
            structure=f"Short ATM straddle + wider {'put' if direction == 'bullish' else 'call'} wing",
            rationale=f"R1 MR + {direction} tilt — broken wing adds directional bias",
            risk_notes=["Asymmetric risk profile", "Unlimited risk on unprotected side"],
        )

    # R2 lower confidence → wide iron butterfly (more room)
    if regime_id in (1, 2):
        return IronButterflyStrategy.WIDE_IRON_BUTTERFLY, StrategyRecommendation(
            name="Wide Iron Butterfly",
            direction="neutral",
            structure="Short ATM straddle + wider OTM wings for extra room",
            rationale="Mean-reverting regime — wider wings for safety margin",
            risk_notes=["Less premium than standard", "More room for price movement"],
        )

    return IronButterflyStrategy.NO_TRADE, _no_trade_rec()


def _no_trade_rec() -> StrategyRecommendation:
    return StrategyRecommendation(
        name="No Trade",
        direction="neutral",
        structure="No position",
        rationale="Conditions not favorable for iron butterfly",
        risk_notes=[],
    )


def _compute_trade_spec(ticker, technicals, regime, vol_surface) -> TradeSpec | None:
    """Compute actionable trade parameters for iron butterfly."""
    from market_analyzer.opportunity.option_plays._trade_spec_helpers import build_single_expiry_trade_spec

    return build_single_expiry_trade_spec(
        ticker=ticker,
        price=technicals.current_price,
        atr=technicals.atr,
        regime_id=int(regime.regime),
        vol_surface=vol_surface,
        structure_type="iron_butterfly",
        target_dte_min=30,
        target_dte_max=45,
    )


def _build_summary(ticker, verdict, confidence, ifly_strat, atm_iv) -> str:
    parts = [f"{verdict.upper()}: {ticker}"]
    parts.append(f"Iron Butterfly: {ifly_strat}")
    parts.append(f"ATM IV: {atm_iv:.1%}")
    parts.append(f"Score: {confidence:.0%}")
    return " | ".join(parts)
