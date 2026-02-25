"""Diagonal spread opportunity assessment — go/no-go + strategy recommendation.

Diagonal = calendar with different strikes (sell OTM front, buy ATM/ITM back).
Combines theta decay (front) + directional bias (strike difference).
Best in R3 (mild trend + time decay) or P2 markup (directional conviction).
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

class DiagonalStrategy(str):
    BULL_CALL_DIAGONAL = "bull_call_diagonal"
    BEAR_PUT_DIAGONAL = "bear_put_diagonal"
    PMCC_DIAGONAL = "pmcc_diagonal"
    NO_TRADE = "no_trade"


# --- Opportunity model ---

class DiagonalOpportunity(BaseModel):
    """Diagonal spread opportunity assessment."""

    ticker: str
    as_of_date: date
    verdict: Verdict
    confidence: float
    hard_stops: list[HardStop]
    signals: list[OpportunitySignal]
    strategy: StrategyRecommendation
    diagonal_strategy: str
    regime_id: int
    regime_confidence: float
    phase_id: int
    phase_name: str
    trend_direction: str  # "bullish" | "bearish" | "neutral"
    front_iv: float
    back_iv: float
    term_slope: float
    days_to_earnings: int | None
    trade_spec: TradeSpec | None = None
    summary: str


# --- Public API ---


def assess_diagonal(
    ticker: str,
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    vol_surface: VolatilitySurface | None = None,
    phase: PhaseResult | None = None,
    fundamentals: FundamentalsSnapshot | None = None,
    as_of: date | None = None,
) -> DiagonalOpportunity:
    """Assess diagonal spread opportunity for a single instrument.

    Pure function — consumes pre-computed analysis, no data fetching.
    """
    cfg = get_settings().opportunity.diagonal
    today = as_of or date.today()

    days_to_earnings: int | None = None
    if fundamentals is not None:
        days_to_earnings = fundamentals.upcoming_events.days_to_earnings

    phase_id = int(phase.phase) if phase else 0
    phase_name = phase.phase_name if phase else "unknown"

    # --- Determine trend direction ---
    trend_dir = _determine_direction(regime, technicals, phase)

    # --- Hard stops ---
    hard_stops = _check_hard_stops(regime, vol_surface, days_to_earnings, cfg)

    front_iv = vol_surface.front_iv if vol_surface else 0.0
    back_iv = vol_surface.back_iv if vol_surface else 0.0
    term_slope = vol_surface.term_slope if vol_surface else 0.0

    if hard_stops:
        return DiagonalOpportunity(
            ticker=ticker,
            as_of_date=today,
            verdict=Verdict.NO_GO,
            confidence=0.0,
            hard_stops=hard_stops,
            signals=[],
            strategy=_no_trade_rec(),
            diagonal_strategy=DiagonalStrategy.NO_TRADE,
            regime_id=int(regime.regime),
            regime_confidence=regime.confidence,
            phase_id=phase_id,
            phase_name=phase_name,
            trend_direction=trend_dir,
            front_iv=front_iv,
            back_iv=back_iv,
            term_slope=term_slope,
            days_to_earnings=days_to_earnings,
            summary=f"NO_GO: {hard_stops[0].description}",
        )

    assert vol_surface is not None

    # --- Signals ---
    signals = _score_signals(regime, technicals, vol_surface, phase, days_to_earnings, trend_dir, cfg)

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
    diag_strat, strat_rec = _select_strategy(regime, technicals, trend_dir, confidence)

    # --- Trade spec ---
    trade_spec = _compute_trade_spec(ticker, technicals, vol_surface, diag_strat, trend_dir) if verdict != Verdict.NO_GO else None

    summary = _build_summary(ticker, verdict, confidence, diag_strat, trend_dir, front_iv, back_iv)

    return DiagonalOpportunity(
        ticker=ticker,
        as_of_date=today,
        verdict=verdict,
        confidence=confidence,
        hard_stops=hard_stops,
        signals=signals,
        strategy=strat_rec,
        diagonal_strategy=diag_strat,
        regime_id=int(regime.regime),
        regime_confidence=regime.confidence,
        phase_id=phase_id,
        phase_name=phase_name,
        trend_direction=trend_dir,
        front_iv=front_iv,
        back_iv=back_iv,
        term_slope=term_slope,
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
    return "neutral"


def _check_hard_stops(regime, vol_surface, days_to_earnings, cfg) -> list[HardStop]:
    stops: list[HardStop] = []

    if regime.regime == RegimeID.R4_HIGH_VOL_TREND and regime.confidence >= cfg.r4_confidence_threshold:
        stops.append(HardStop(
            name="R4 explosive moves",
            description="R4 (high-vol trending) — explosive moves blow through diagonal strikes",
        ))

    if vol_surface is None:
        stops.append(HardStop(
            name="No vol surface",
            description="No options chain/vol surface data — cannot assess diagonal",
        ))
        return stops

    if vol_surface.data_quality == "poor":
        stops.append(HardStop(
            name="Poor data quality",
            description="Options chain data quality too poor for diagonal assessment",
        ))

    # Extreme skew makes one leg too expensive
    if vol_surface.skew_by_expiry:
        front_skew = vol_surface.skew_by_expiry[0]
        if abs(front_skew.skew_ratio) > cfg.max_skew_ratio:
            stops.append(HardStop(
                name="Extreme skew",
                description=f"Skew ratio {front_skew.skew_ratio:.1f} — makes one leg overpriced",
            ))

    if days_to_earnings is not None and 0 < days_to_earnings < cfg.earnings_blackout_days:
        stops.append(HardStop(
            name="Earnings imminent",
            description=f"Earnings in {days_to_earnings} days — IV crush asymmetry risk",
        ))

    return stops


def _score_signals(regime, technicals, vol_surface, phase, days_to_earnings, trend_dir, cfg):
    signals: list[OpportunitySignal] = []

    # 1. Term structure favorable (0.15)
    if vol_surface.is_contango:
        signals.append(OpportunitySignal(
            name="Contango term structure",
            favorable=True,
            weight=0.15,
            description=f"Term slope {vol_surface.term_slope:.1%} — helps diagonal structure",
        ))
    elif vol_surface.is_backwardation and abs(vol_surface.term_slope) < 0.15:
        signals.append(OpportunitySignal(
            name="Mild backwardation",
            favorable=True,
            weight=0.08,
            description="Mild backwardation — acceptable for diagonals",
        ))
    else:
        signals.append(OpportunitySignal(
            name="Unfavorable term structure",
            favorable=False,
            weight=0.15,
            description="Steep backwardation — front month overpriced relative to back",
        ))

    # 2. Trend direction confirmed (0.20)
    regime_id = int(regime.regime)
    if regime_id == 3 and trend_dir != "neutral":
        signals.append(OpportunitySignal(
            name="R3 + directional bias",
            favorable=True,
            weight=0.20,
            description=f"R3 trending + {trend_dir} — ideal diagonal environment",
        ))
    elif trend_dir != "neutral":
        signals.append(OpportunitySignal(
            name="Directional bias present",
            favorable=True,
            weight=0.12,
            description=f"Trend direction: {trend_dir}",
        ))
    else:
        signals.append(OpportunitySignal(
            name="No directional bias",
            favorable=False,
            weight=0.20,
            description="Neutral direction — diagonals need directional conviction",
        ))

    # 3. Skew favorable for direction (0.15)
    if vol_surface.skew_by_expiry:
        front_skew = vol_surface.skew_by_expiry[0]
        if trend_dir == "bullish" and front_skew.put_skew > 0:
            signals.append(OpportunitySignal(
                name="Put skew helps bull diagonal",
                favorable=True,
                weight=0.15,
                description="Elevated put skew — cheap call side for bull diagonal",
            ))
        elif trend_dir == "bearish" and front_skew.call_skew > 0:
            signals.append(OpportunitySignal(
                name="Call skew helps bear diagonal",
                favorable=True,
                weight=0.15,
                description="Elevated call skew — cheap put side for bear diagonal",
            ))
        else:
            signals.append(OpportunitySignal(
                name="Skew neutral",
                favorable=True,
                weight=0.05,
                description="Skew doesn't particularly favor or hinder diagonal",
            ))

    # 4. Regime alignment (0.15)
    if regime_id == 3:
        signals.append(OpportunitySignal(
            name="Ideal regime",
            favorable=True,
            weight=0.15,
            description="R3 (low-vol trending) — ideal for diagonals",
        ))
    elif regime_id == 1:
        signals.append(OpportunitySignal(
            name="Acceptable regime",
            favorable=True,
            weight=0.08,
            description="R1 (low-vol MR) — PMCC variant possible",
        ))
    else:
        signals.append(OpportunitySignal(
            name="Suboptimal regime",
            favorable=False,
            weight=0.15,
            description=f"R{regime_id} — not ideal for diagonals",
        ))

    # 5. Phase alignment (0.10)
    if phase is not None:
        p = int(phase.phase)
        if (trend_dir == "bullish" and p == 2) or (trend_dir == "bearish" and p == 4):
            signals.append(OpportunitySignal(
                name="Phase confirms direction",
                favorable=True,
                weight=0.10,
                description=f"Phase P{p} confirms {trend_dir} bias",
            ))
        elif p in (2, 4):
            signals.append(OpportunitySignal(
                name="Phase partially aligns",
                favorable=True,
                weight=0.05,
                description=f"Phase P{p}",
            ))

    # 6. ATM IV level (0.10)
    avg_iv = (vol_surface.front_iv + vol_surface.back_iv) / 2
    if avg_iv >= 0.25:
        signals.append(OpportunitySignal(
            name="Good IV level",
            favorable=True,
            weight=0.10,
            description=f"Avg IV {avg_iv:.1%} — good premium environment",
        ))
    elif avg_iv >= 0.15:
        signals.append(OpportunitySignal(
            name="Moderate IV",
            favorable=True,
            weight=0.05,
            description=f"Avg IV {avg_iv:.1%} — acceptable",
        ))

    # 7. No earnings between legs (0.10)
    if days_to_earnings is None or days_to_earnings > 45:
        signals.append(OpportunitySignal(
            name="Clear earnings window",
            favorable=True,
            weight=0.10,
            description="No earnings in diagonal window",
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


def _select_strategy(regime, technicals, trend_dir, confidence):
    if confidence < 0.30:
        return DiagonalStrategy.NO_TRADE, _no_trade_rec()

    regime_id = int(regime.regime)

    # PMCC for R1 with mild bullish bias
    if regime_id == 1 and trend_dir == "bullish":
        return DiagonalStrategy.PMCC_DIAGONAL, StrategyRecommendation(
            name="Poor Man's Covered Call",
            direction="bullish",
            structure="Buy deep ITM back-month call, sell OTM front-month call",
            rationale="R1 + bullish bias — PMCC captures upside with limited capital",
            risk_notes=["Max loss = net debit", "Requires back-month call to retain value"],
        )

    if trend_dir == "bullish":
        return DiagonalStrategy.BULL_CALL_DIAGONAL, StrategyRecommendation(
            name="Bull Call Diagonal",
            direction="bullish",
            structure="Sell OTM front-month call, buy ATM/ITM back-month call",
            rationale=f"R{regime_id} + bullish — diagonal captures trend + theta",
            risk_notes=["Max loss = net debit", "Short call can be assigned if deep ITM"],
        )

    if trend_dir == "bearish":
        return DiagonalStrategy.BEAR_PUT_DIAGONAL, StrategyRecommendation(
            name="Bear Put Diagonal",
            direction="bearish",
            structure="Sell OTM front-month put, buy ATM/ITM back-month put",
            rationale=f"R{regime_id} + bearish — diagonal captures downtrend + theta",
            risk_notes=["Max loss = net debit", "Short put can be assigned if deep ITM"],
        )

    return DiagonalStrategy.NO_TRADE, _no_trade_rec()


def _compute_trade_spec(ticker, technicals, vol_surface, diag_strat, trend_dir) -> TradeSpec | None:
    """Compute actionable trade parameters for diagonal spread."""
    from market_analyzer.opportunity.option_plays._trade_spec_helpers import build_dual_expiry_trade_spec

    return build_dual_expiry_trade_spec(
        ticker=ticker,
        price=technicals.current_price,
        atr=technicals.atr,
        vol_surface=vol_surface,
        structure_type="diagonal",
        strategy_type=diag_strat,
        front_dte_min=20,
        front_dte_max=30,
        back_dte_min=50,
        back_dte_max=90,
        trend_direction=trend_dir,
    )


def _no_trade_rec() -> StrategyRecommendation:
    return StrategyRecommendation(
        name="No Trade",
        direction="neutral",
        structure="No position",
        rationale="Conditions not favorable for diagonal spread",
        risk_notes=[],
    )


def _build_summary(ticker, verdict, confidence, diag_strat, trend_dir, front_iv, back_iv) -> str:
    parts = [f"{verdict.upper()}: {ticker}"]
    parts.append(f"Diagonal: {diag_strat}")
    parts.append(f"Direction: {trend_dir}")
    parts.append(f"IV: {front_iv:.1%}/{back_iv:.1%}")
    parts.append(f"Score: {confidence:.0%}")
    return " | ".join(parts)
