"""Breakout opportunity assessment — go/no-go + strategy recommendation."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from market_analyzer.config import get_settings
from market_analyzer.models.opportunity import (
    BreakoutOpportunity,
    BreakoutSetup,
    BreakoutStrategy,
    BreakoutType,
    HardStop,
    OpportunitySignal,
    StrategyRecommendation,
    Verdict,
)

if TYPE_CHECKING:
    from market_analyzer.models.fundamentals import FundamentalsSnapshot
    from market_analyzer.models.macro import MacroCalendar
    from market_analyzer.models.phase import PhaseResult
    from market_analyzer.models.regime import RegimeResult
    from market_analyzer.models.technicals import TechnicalSnapshot


def assess_breakout(
    ticker: str,
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    phase: PhaseResult,
    macro: MacroCalendar,
    fundamentals: FundamentalsSnapshot | None = None,
    as_of: date | None = None,
) -> BreakoutOpportunity:
    """Assess breakout opportunity for a single instrument.

    Pure function — consumes pre-computed analysis, produces structured assessment.
    No data fetching.
    """
    cfg = get_settings().opportunity.breakout
    today = as_of or date.today()

    # --- Days to earnings ---
    days_to_earnings: int | None = None
    if fundamentals is not None:
        days_to_earnings = fundamentals.upcoming_events.days_to_earnings

    # --- Build setup context ---
    setup = _build_setup(technicals)

    # --- Hard stops ---
    hard_stops = _check_hard_stops(
        regime, technicals, days_to_earnings, cfg,
    )

    # --- Scoring signals ---
    signals = _score_signals(regime, technicals, phase, cfg)

    # --- Confidence ---
    regime_mult = cfg.regime_multipliers.get(int(regime.regime), 0.5)
    raw_confidence = sum(s.weight for s in signals if s.favorable)
    confidence = min(1.0, raw_confidence * regime_mult)

    # --- Verdict ---
    if hard_stops:
        verdict = Verdict.NO_GO
    elif confidence >= cfg.go_threshold:
        verdict = Verdict.GO
    elif confidence >= cfg.caution_threshold:
        verdict = Verdict.CAUTION
    else:
        verdict = Verdict.NO_GO

    # --- Direction ---
    breakout_type = _determine_direction(regime, technicals)

    # --- Strategy selection ---
    breakout_strategy, strategy_rec = _select_strategy(
        regime, technicals, phase, setup, verdict, breakout_type,
    )

    # --- Pivot price ---
    pivot_price = None
    if technicals.vcp is not None and technicals.vcp.pivot_price is not None:
        pivot_price = technicals.vcp.pivot_price

    # --- Summary ---
    summary = _build_summary(
        ticker, verdict, confidence, breakout_strategy, hard_stops,
        regime, phase, setup,
    )

    return BreakoutOpportunity(
        ticker=ticker,
        as_of_date=today,
        verdict=verdict,
        confidence=round(confidence, 2),
        hard_stops=hard_stops,
        signals=signals,
        strategy=strategy_rec,
        breakout_strategy=breakout_strategy,
        breakout_type=breakout_type,
        regime_id=int(regime.regime),
        regime_confidence=round(regime.confidence, 2),
        phase_id=int(phase.phase),
        phase_name=phase.phase_name,
        phase_confidence=round(phase.confidence, 2),
        setup=setup,
        pivot_price=pivot_price,
        days_to_earnings=days_to_earnings,
        summary=summary,
    )


# --- Private helpers ---


def _build_setup(technicals: TechnicalSnapshot) -> BreakoutSetup:
    """Build breakout setup context from technicals."""
    vcp = technicals.vcp
    vcp_stage = vcp.stage.value if vcp is not None else "none"
    vcp_score = vcp.score if vcp is not None else 0.0
    days_in_base = vcp.days_in_base if vcp is not None else None

    bb = technicals.bollinger
    squeeze = bb.bandwidth < 0.04

    rc = technicals.phase.range_compression

    # Volume pattern
    vol_trend = technicals.phase.volume_trend
    if vcp is not None and vcp_stage in ("forming", "maturing", "ready") and vol_trend == "declining":
        volume_pattern = "declining_base"
    elif vol_trend == "rising":
        volume_pattern = "surge"
    elif vol_trend == "declining":
        volume_pattern = "distribution"
    else:
        volume_pattern = "normal"

    # S/R proximity
    sr = technicals.support_resistance
    res_prox = sr.price_vs_resistance_pct if sr.resistance is not None else None
    sup_prox = sr.price_vs_support_pct if sr.support is not None else None

    # Smart money alignment
    sm = technicals.smart_money
    if sm is not None and sm.nearest_bullish_ob is not None:
        ob = sm.nearest_bullish_ob
        if not ob.is_tested and not ob.is_broken and abs(ob.distance_pct) <= 3.0:
            sm_align = "supportive"
        elif ob.is_broken:
            sm_align = "conflicting"
        else:
            sm_align = "neutral"
    else:
        sm_align = "neutral"

    # Description
    parts = []
    if vcp_stage != "none":
        parts.append(f"VCP {vcp_stage} (score {vcp_score:.2f})")
    if squeeze:
        parts.append("Bollinger squeeze")
    if rc > 0.3:
        parts.append(f"range compression {rc:.2f}")
    if volume_pattern == "declining_base":
        parts.append("volume declining in base")
    desc = "; ".join(parts) if parts else "No clear breakout setup."

    return BreakoutSetup(
        vcp_stage=vcp_stage,
        vcp_score=vcp_score,
        bollinger_squeeze=squeeze,
        bollinger_bandwidth=round(bb.bandwidth, 4),
        range_compression=round(rc, 2),
        volume_pattern=volume_pattern,
        resistance_proximity_pct=round(res_prox, 2) if res_prox is not None else None,
        support_proximity_pct=round(sup_prox, 2) if sup_prox is not None else None,
        days_in_base=days_in_base,
        smart_money_alignment=sm_align,
        description=desc,
    )


def _check_hard_stops(
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    days_to_earnings: int | None,
    cfg,
) -> list[HardStop]:
    stops: list[HardStop] = []
    rid = int(regime.regime)

    # R4 high confidence
    if rid == 4 and regime.confidence > cfg.r4_confidence_threshold:
        stops.append(HardStop(
            name="r4_high_confidence",
            description=(
                f"R4 (High-Vol Trending) at {regime.confidence:.0%} — "
                f"breakouts unreliable in explosive regimes."
            ),
        ))

    # R2 very high confidence
    if rid == 2 and regime.confidence > cfg.r2_confidence_threshold:
        stops.append(HardStop(
            name="r2_very_high_confidence",
            description=(
                f"R2 (High-Vol MR) at {regime.confidence:.0%} — "
                f"breakouts fail in high-vol mean-reverting regimes."
            ),
        ))

    # Earnings imminent
    if days_to_earnings is not None and days_to_earnings <= cfg.earnings_blackout_days:
        stops.append(HardStop(
            name="earnings_imminent",
            description=f"Earnings in {days_to_earnings} day(s) — breakout invalidated by gap risk.",
        ))

    # No base established
    vcp = technicals.vcp
    bb = technicals.bollinger
    phase_ind = technicals.phase
    if (
        (vcp is None or vcp.stage.value == "none")
        and (vcp is None or vcp.days_in_base < cfg.min_base_days)
        and phase_ind.range_compression < cfg.range_compression_threshold
        and bb.bandwidth > cfg.bollinger_squeeze_bandwidth * 2
    ):
        stops.append(HardStop(
            name="no_base_established",
            description="No consolidation base — VCP none, low range compression, wide Bollinger bands.",
        ))

    # Already extended
    if vcp is not None and vcp.stage.value == "breakout":
        if bb.percent_b > 1.2 and vcp.pivot_distance_pct is not None and vcp.pivot_distance_pct > 5.0:
            stops.append(HardStop(
                name="already_extended",
                description=(
                    f"Already extended past breakout — %B={bb.percent_b:.2f}, "
                    f"pivot distance {vcp.pivot_distance_pct:.1f}%."
                ),
            ))

    return stops


def _score_signals(
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    phase: PhaseResult,
    cfg,
) -> list[OpportunitySignal]:
    signals: list[OpportunitySignal] = []
    vcp = technicals.vcp

    # 1. VCP setup quality (0.20)
    vcp_fav = (
        vcp is not None
        and vcp.stage.value in ("ready", "breakout")
        and vcp.score >= cfg.vcp_ready_min_score
    )
    signals.append(OpportunitySignal(
        name="vcp_setup_quality",
        favorable=vcp_fav,
        weight=0.20,
        description=(
            f"VCP {vcp.stage.value} with score {vcp.score:.2f}"
            if vcp is not None and vcp.stage.value != "none"
            else "No VCP setup detected"
        ),
    ))

    # 2. Bollinger squeeze (0.15)
    bb = technicals.bollinger
    squeeze_fav = bb.bandwidth < cfg.bollinger_squeeze_bandwidth
    signals.append(OpportunitySignal(
        name="bollinger_squeeze",
        favorable=squeeze_fav,
        weight=0.15,
        description=(
            f"Bollinger squeeze — bandwidth {bb.bandwidth:.4f}"
            if squeeze_fav
            else f"No squeeze — bandwidth {bb.bandwidth:.4f}"
        ),
    ))

    # 3. S/R proximity (0.15)
    sr = technicals.support_resistance
    sr_fav = (
        sr.resistance is not None
        and sr.price_vs_resistance_pct is not None
        and abs(sr.price_vs_resistance_pct) <= cfg.resistance_proximity_pct
    )
    signals.append(OpportunitySignal(
        name="sr_proximity",
        favorable=sr_fav,
        weight=0.15,
        description=(
            f"Price within {abs(sr.price_vs_resistance_pct):.1f}% of resistance"
            if sr_fav
            else "Not near resistance level"
        ),
    ))

    # 4. Range compression (0.10)
    rc = technicals.phase.range_compression
    rc_fav = rc > cfg.range_compression_threshold
    signals.append(OpportunitySignal(
        name="range_compression",
        favorable=rc_fav,
        weight=0.10,
        description=(
            f"Range compressing ({rc:.2f})"
            if rc_fav
            else f"Range not compressed ({rc:.2f})"
        ),
    ))

    # 5. Volume pattern (0.10)
    vol_trend = technicals.phase.volume_trend
    vol_fav = (
        vol_trend == "declining"
        and vcp is not None
        and vcp.stage.value in ("forming", "maturing", "ready")
    )
    signals.append(OpportunitySignal(
        name="volume_pattern",
        favorable=vol_fav,
        weight=0.10,
        description=(
            "Volume declining during base formation"
            if vol_fav
            else f"Volume trend: {vol_trend}"
        ),
    ))

    # 6. Smart money alignment (0.10)
    sm = technicals.smart_money
    sm_fav = False
    if sm is not None and sm.nearest_bullish_ob is not None:
        ob = sm.nearest_bullish_ob
        sm_fav = not ob.is_tested and not ob.is_broken and abs(ob.distance_pct) <= 3.0
    signals.append(OpportunitySignal(
        name="smart_money_alignment",
        favorable=sm_fav,
        weight=0.10,
        description=(
            "Bullish order block supports breakout"
            if sm_fav
            else "No supportive order block nearby"
        ),
    ))

    # 7. Phase alignment (0.10)
    pid = int(phase.phase)
    phase_fav = pid in (1, 2)
    signals.append(OpportunitySignal(
        name="phase_alignment",
        favorable=phase_fav,
        weight=0.10,
        description=(
            f"P{pid} {phase.phase_name} — favorable for breakout"
            if phase_fav
            else f"P{pid} {phase.phase_name} — less favorable"
        ),
    ))

    # 8. ATR context (0.05)
    atr_fav = technicals.atr_pct < cfg.atr_low_baseline_pct
    signals.append(OpportunitySignal(
        name="atr_context",
        favorable=atr_fav,
        weight=0.05,
        description=(
            f"ATR {technicals.atr_pct:.2f}% — low-vol base forming"
            if atr_fav
            else f"ATR {technicals.atr_pct:.2f}% — elevated"
        ),
    ))

    # 9. Regime favorable (0.05)
    rid = int(regime.regime)
    regime_fav = rid in (1, 3)
    signals.append(OpportunitySignal(
        name="regime_favorable",
        favorable=regime_fav,
        weight=0.05,
        description=(
            f"R{rid} — favorable for breakouts"
            if regime_fav
            else f"R{rid} — less favorable for breakouts"
        ),
    ))

    return signals


def _determine_direction(
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
) -> BreakoutType:
    """Determine breakout direction."""
    vcp = technicals.vcp
    # VCP READY/BREAKOUT → BULLISH
    if vcp is not None and vcp.stage.value in ("ready", "breakout"):
        return BreakoutType.BULLISH

    # Trend direction from regime
    if regime.trend_direction is not None:
        if regime.trend_direction.value == "bearish":
            return BreakoutType.BEARISH
        return BreakoutType.BULLISH

    # MA structure fallback
    ma = technicals.moving_averages
    if ma.price_vs_sma_50_pct > 0:
        return BreakoutType.BULLISH
    return BreakoutType.BEARISH


def _select_strategy(
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    phase: PhaseResult,
    setup: BreakoutSetup,
    verdict: Verdict,
    breakout_type: BreakoutType,
) -> tuple[BreakoutStrategy, StrategyRecommendation]:
    """Select breakout strategy based on setup conditions."""
    rid = int(regime.regime)
    pid = int(phase.phase)
    cfg = get_settings().opportunity.breakout

    if verdict == Verdict.NO_GO:
        return BreakoutStrategy.NO_TRADE, StrategyRecommendation(
            name="No Trade",
            direction="neutral",
            structure="Do not enter breakout trade.",
            rationale="Conditions unfavorable for breakout.",
            risk_notes=["Wait for better setup."],
        )

    vcp = technicals.vcp
    bb = technicals.bollinger

    # VCP READY/BREAKOUT with high score → PIVOT_BREAKOUT
    if (
        vcp is not None
        and vcp.stage.value in ("ready", "breakout")
        and vcp.score >= cfg.vcp_breakout_confirmation
    ):
        direction = breakout_type.value
        return BreakoutStrategy.PIVOT_BREAKOUT, StrategyRecommendation(
            name="Pivot Breakout",
            direction=direction,
            structure=f"Buy above VCP pivot at {vcp.pivot_price:.2f}." if vcp.pivot_price else "Buy above VCP pivot.",
            rationale=f"VCP {vcp.stage.value} with score {vcp.score:.2f} — high-probability breakout.",
            risk_notes=["Stop below last contraction low.", "Volume must confirm on breakout day."],
        )

    # VCP BREAKOUT + near old resistance → PULLBACK_TO_BREAKOUT
    sr = technicals.support_resistance
    if (
        vcp is not None
        and vcp.stage.value == "breakout"
        and sr.resistance is not None
        and sr.price_vs_resistance_pct is not None
        and abs(sr.price_vs_resistance_pct) <= cfg.resistance_proximity_pct
    ):
        return BreakoutStrategy.PULLBACK_TO_BREAKOUT, StrategyRecommendation(
            name="Pullback to Breakout",
            direction=breakout_type.value,
            structure="Re-enter at S/R retest after breakout.",
            rationale="VCP breakout retesting old resistance — re-entry opportunity.",
            risk_notes=["Failed retest = exit.", "Volume should be lighter on pullback."],
        )

    # Bollinger squeeze + VCP forming/maturing → SQUEEZE_PLAY
    if (
        bb.bandwidth < cfg.bollinger_squeeze_bandwidth
        and vcp is not None
        and vcp.stage.value in ("forming", "maturing")
    ):
        return BreakoutStrategy.SQUEEZE_PLAY, StrategyRecommendation(
            name="Squeeze Play",
            direction=breakout_type.value,
            structure="Position for Bollinger squeeze breakout.",
            rationale=f"Bollinger squeeze (bandwidth {bb.bandwidth:.4f}) with VCP {vcp.stage.value}.",
            risk_notes=["Breakout direction uncertain.", "Use defined risk (spreads)."],
        )

    # P2 + R3 + range compression → BULL_FLAG / BEAR_FLAG
    if (
        pid == 2 and rid == 3
        and technicals.phase.range_compression > cfg.range_compression_threshold
    ):
        if breakout_type == BreakoutType.BEARISH:
            return BreakoutStrategy.BEAR_FLAG, StrategyRecommendation(
                name="Bear Flag",
                direction="bearish",
                structure="Put debit spread on flag breakdown.",
                rationale="P2 + R3 with range compression — bear flag pattern.",
                risk_notes=["Counter-trend pattern, keep size small."],
            )
        return BreakoutStrategy.BULL_FLAG, StrategyRecommendation(
            name="Bull Flag",
            direction="bullish",
            structure="Call debit spread on flag breakout.",
            rationale="P2 + R3 with range compression — bull flag continuation.",
            risk_notes=["Set stop below flag low.", "Target measured move."],
        )

    # Default: R1/R3 with some compression → SQUEEZE_PLAY
    if rid in (1, 3):
        return BreakoutStrategy.SQUEEZE_PLAY, StrategyRecommendation(
            name="Squeeze Play",
            direction=breakout_type.value,
            structure="Position for volatility expansion breakout.",
            rationale=f"R{rid} with some consolidation — pre-positioning for breakout.",
            risk_notes=["Breakout timing uncertain.", "Use defined risk."],
        )

    # Fallback
    return BreakoutStrategy.NO_TRADE, StrategyRecommendation(
        name="No Trade",
        direction="neutral",
        structure="No clear breakout pattern.",
        rationale="Conditions do not support a breakout trade.",
        risk_notes=["Wait for clearer setup."],
    )


def _build_summary(
    ticker: str,
    verdict: Verdict,
    confidence: float,
    strategy: BreakoutStrategy,
    hard_stops: list[HardStop],
    regime: RegimeResult,
    phase: PhaseResult,
    setup: BreakoutSetup,
) -> str:
    if verdict == Verdict.NO_GO:
        reasons = "; ".join(s.name for s in hard_stops) if hard_stops else "low confidence"
        return (
            f"{ticker} Breakout: NO-GO ({reasons}). "
            f"R{int(regime.regime)}, {phase.phase_name}."
        )
    elif verdict == Verdict.CAUTION:
        return (
            f"{ticker} Breakout: CAUTION ({confidence:.0%}). "
            f"R{int(regime.regime)} {phase.phase_name}, "
            f"VCP {setup.vcp_stage}. "
            f"Consider {strategy.value} with reduced size."
        )
    else:
        return (
            f"{ticker} Breakout: GO ({confidence:.0%}). "
            f"R{int(regime.regime)} {phase.phase_name}, "
            f"VCP {setup.vcp_stage}. "
            f"Recommended: {strategy.value}."
        )
