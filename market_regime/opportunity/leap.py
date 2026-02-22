"""LEAP opportunity assessment — go/no-go + strategy recommendation."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from market_regime.config import get_settings
from market_regime.models.opportunity import (
    FundamentalScore,
    HardStop,
    LEAPOpportunity,
    LEAPStrategy,
    OpportunitySignal,
    StrategyRecommendation,
    Verdict,
)

if TYPE_CHECKING:
    from market_regime.models.fundamentals import FundamentalsSnapshot
    from market_regime.models.macro import MacroCalendar
    from market_regime.models.phase import PhaseResult
    from market_regime.models.regime import RegimeResult
    from market_regime.models.technicals import TechnicalSnapshot


def assess_leap(
    ticker: str,
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    phase: PhaseResult,
    macro: MacroCalendar,
    fundamentals: FundamentalsSnapshot | None = None,
    as_of: date | None = None,
) -> LEAPOpportunity:
    """Assess LEAP opportunity for a single instrument.

    Pure function — consumes pre-computed analysis, produces structured assessment.
    No data fetching.
    """
    cfg = get_settings().opportunity.leap
    today = as_of or date.today()

    # --- Days to earnings ---
    days_to_earnings: int | None = None
    if fundamentals is not None:
        days_to_earnings = fundamentals.upcoming_events.days_to_earnings

    # --- IV environment ---
    iv_env = _iv_environment(int(regime.regime))

    # --- Fundamental score ---
    fund_score = _compute_fundamental_score(fundamentals, cfg)

    # --- Hard stops ---
    hard_stops = _check_hard_stops(
        regime, phase, days_to_earnings, fund_score, cfg,
    )

    # --- Scoring signals ---
    signals = _score_signals(
        regime, technicals, phase, macro, fundamentals, fund_score, cfg,
    )

    # --- Confidence ---
    raw_confidence = sum(s.weight for s in signals if s.favorable)
    confidence = min(1.0, raw_confidence)

    # --- Verdict ---
    if hard_stops:
        verdict = Verdict.NO_GO
    elif confidence >= cfg.go_threshold:
        verdict = Verdict.GO
    elif confidence >= cfg.caution_threshold:
        verdict = Verdict.CAUTION
    else:
        verdict = Verdict.NO_GO

    # --- Strategy selection ---
    leap_strategy, strategy_rec = _select_strategy(
        regime, phase, iv_env, verdict,
    )

    # --- Summary ---
    summary = _build_summary(
        ticker, verdict, confidence, leap_strategy, hard_stops,
        regime, phase, iv_env, fund_score,
    )

    return LEAPOpportunity(
        ticker=ticker,
        as_of_date=today,
        verdict=verdict,
        confidence=round(confidence, 2),
        hard_stops=hard_stops,
        signals=signals,
        strategy=strategy_rec,
        leap_strategy=leap_strategy,
        regime_id=int(regime.regime),
        regime_confidence=round(regime.confidence, 2),
        phase_id=int(phase.phase),
        phase_name=phase.phase_name,
        phase_confidence=round(phase.confidence, 2),
        iv_environment=iv_env,
        fundamental_score=fund_score,
        days_to_earnings=days_to_earnings,
        macro_events_next_30_days=len(macro.events_next_30_days),
        summary=summary,
    )


# --- Private helpers ---


def _iv_environment(regime_id: int) -> str:
    """Map regime to IV environment label."""
    if regime_id == 1:
        return "cheap"
    elif regime_id in (2, 3):
        return "moderate"
    else:
        return "expensive"


def _compute_fundamental_score(
    fundamentals: FundamentalsSnapshot | None,
    cfg,
) -> FundamentalScore:
    """Compute composite fundamental score from 5 sub-signals."""
    if fundamentals is None:
        return FundamentalScore(
            score=0.5,
            earnings_growth_signal="unknown",
            revenue_growth_signal="unknown",
            margin_signal="unknown",
            debt_signal="unknown",
            valuation_signal="unknown",
            description="Fundamentals unavailable — using neutral score.",
        )

    # 1. Earnings growth
    eg = fundamentals.earnings.earnings_growth
    if eg is not None:
        if eg > cfg.earnings_growth_strong:
            eg_score, eg_sig = 1.0, "strong"
        elif eg > cfg.earnings_growth_moderate:
            eg_score, eg_sig = 0.7, "moderate"
        elif eg > 0:
            eg_score, eg_sig = 0.4, "weak"
        else:
            eg_score, eg_sig = 0.1, "negative"
    else:
        eg_score, eg_sig = 0.5, "unknown"

    # 2. Revenue growth
    rg = fundamentals.revenue.revenue_growth
    if rg is not None:
        if rg > cfg.revenue_growth_strong:
            rg_score, rg_sig = 1.0, "strong"
        elif rg > cfg.revenue_growth_moderate:
            rg_score, rg_sig = 0.7, "moderate"
        elif rg > 0:
            rg_score, rg_sig = 0.4, "weak"
        else:
            rg_score, rg_sig = 0.1, "negative"
    else:
        rg_score, rg_sig = 0.5, "unknown"

    # 3. Margins
    mg = fundamentals.margins.profit_margins
    if mg is not None:
        if mg > cfg.margin_strong:
            mg_score, mg_sig = 1.0, "expanding"
        elif mg > cfg.margin_moderate:
            mg_score, mg_sig = 0.7, "stable"
        elif mg > 0:
            mg_score, mg_sig = 0.4, "contracting"
        else:
            mg_score, mg_sig = 0.1, "contracting"
    else:
        mg_score, mg_sig = 0.5, "unknown"

    # 4. Debt health
    d2e = fundamentals.debt.debt_to_equity
    if d2e is not None:
        if d2e < cfg.debt_low:
            dt_score, dt_sig = 1.0, "low"
        elif d2e < cfg.debt_moderate:
            dt_score, dt_sig = 0.7, "moderate"
        elif d2e < cfg.debt_high:
            dt_score, dt_sig = 0.4, "high"
        else:
            dt_score, dt_sig = 0.1, "high"
    else:
        dt_score, dt_sig = 0.5, "unknown"

    # 5. Valuation (forward PE)
    fpe = fundamentals.valuation.forward_pe
    if fpe is not None:
        if fpe < cfg.pe_cheap:
            vl_score, vl_sig = 1.0, "cheap"
        elif fpe < cfg.pe_fair:
            vl_score, vl_sig = 0.7, "fair"
        elif fpe < cfg.pe_expensive:
            vl_score, vl_sig = 0.4, "expensive"
        else:
            vl_score, vl_sig = 0.2, "expensive"
    else:
        vl_score, vl_sig = 0.5, "unknown"

    # Weighted composite
    score = (
        eg_score * 0.25
        + rg_score * 0.25
        + mg_score * 0.20
        + dt_score * 0.15
        + vl_score * 0.15
    )

    # Build description
    parts = []
    if eg_sig not in ("unknown",):
        parts.append(f"earnings {eg_sig}")
    if rg_sig not in ("unknown",):
        parts.append(f"revenue {rg_sig}")
    if mg_sig not in ("unknown",):
        parts.append(f"margins {mg_sig}")
    if dt_sig not in ("unknown",):
        parts.append(f"debt {dt_sig}")
    if vl_sig not in ("unknown",):
        parts.append(f"valuation {vl_sig}")
    desc = ", ".join(parts) if parts else "No fundamental data available."

    return FundamentalScore(
        score=round(score, 2),
        earnings_growth_signal=eg_sig,
        revenue_growth_signal=rg_sig,
        margin_signal=mg_sig,
        debt_signal=dt_sig,
        valuation_signal=vl_sig,
        description=desc.capitalize() + "." if parts else desc,
    )


def _check_hard_stops(
    regime: RegimeResult,
    phase: PhaseResult,
    days_to_earnings: int | None,
    fund_score: FundamentalScore,
    cfg,
) -> list[HardStop]:
    stops: list[HardStop] = []

    # IV too expensive (R4)
    if int(regime.regime) == 4 and regime.confidence > cfg.r4_confidence_threshold:
        stops.append(HardStop(
            name="iv_expensive",
            description=(
                f"R4 at {regime.confidence:.0%} — IV too expensive for buying LEAPs. "
                f"Wait for IV compression."
            ),
        ))

    # Distribution top (P3)
    if int(phase.phase) == 3 and phase.confidence > cfg.distribution_confidence_threshold:
        stops.append(HardStop(
            name="distribution_top",
            description=(
                f"Distribution phase at {phase.confidence:.0%} — "
                f"buying bullish LEAPs at distribution top is high risk."
            ),
        ))

    # Markdown phase (P4)
    if int(phase.phase) == 4 and phase.confidence > cfg.markdown_confidence_threshold:
        stops.append(HardStop(
            name="markdown_phase",
            description=(
                f"Markdown phase at {phase.confidence:.0%} — "
                f"bullish LEAPs face strong headwinds."
            ),
        ))

    # Earnings imminent
    if days_to_earnings is not None and days_to_earnings <= cfg.earnings_blackout_days:
        stops.append(HardStop(
            name="earnings_imminent",
            description=(
                f"Earnings in {days_to_earnings} day(s) — "
                f"wait for earnings clarity before committing to LEAP."
            ),
        ))

    # Weak fundamentals
    if fund_score.score < cfg.min_fundamental_score:
        stops.append(HardStop(
            name="weak_fundamentals",
            description=(
                f"Fundamental score {fund_score.score:.0%} < {cfg.min_fundamental_score:.0%} — "
                f"business too weak for 1-2yr commitment."
            ),
        ))

    return stops


def _score_signals(
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    phase: PhaseResult,
    macro: MacroCalendar,
    fundamentals: FundamentalsSnapshot | None,
    fund_score: FundamentalScore,
    cfg,
) -> list[OpportunitySignal]:
    signals: list[OpportunitySignal] = []

    # 1. Phase entry zone (P1 = accumulation)
    pid = int(phase.phase)
    phase_fav = pid == 1
    signals.append(OpportunitySignal(
        name="phase_entry_zone",
        favorable=phase_fav,
        weight=0.25,
        description=(
            f"P{pid} {phase.phase_name} — {'ideal LEAP entry zone' if phase_fav else 'not optimal entry'}"
        ),
    ))

    # 2. IV cheap
    rid = int(regime.regime)
    iv_fav = rid in (1, 3)
    signals.append(OpportunitySignal(
        name="iv_cheap",
        favorable=iv_fav,
        weight=0.20,
        description=(
            f"R{rid} — IV {'cheap, LEAPs affordable' if rid == 1 else 'moderate' if iv_fav else 'elevated, LEAPs expensive'}"
        ),
    ))

    # 3. Fundamental quality
    fund_fav = fund_score.score >= 0.6
    signals.append(OpportunitySignal(
        name="fundamental_quality",
        favorable=fund_fav,
        weight=0.20,
        description=(
            f"Fundamental score {fund_score.score:.0%} — "
            f"{'strong enough' if fund_fav else 'weak'} for LEAP commitment"
        ),
    ))

    # 4. Trend alignment
    trend_fav = False
    if pid in (1, 2):
        # Accumulation/Markup — bullish LEAPs aligned
        trend_fav = True
    elif pid in (3, 4) and regime.trend_direction is not None:
        # Distribution/Markdown — bearish LEAPs could be aligned
        trend_fav = regime.trend_direction.value == "bearish"
    signals.append(OpportunitySignal(
        name="trend_alignment",
        favorable=trend_fav,
        weight=0.10,
        description=(
            f"Phase {phase.phase_name} {'aligns' if trend_fav else 'conflicts'} with LEAP direction"
        ),
    ))

    # 5. 52-week position
    ftwk_fav = False
    if fundamentals is not None and fundamentals.fifty_two_week.pct_from_low is not None:
        pct_from_low = fundamentals.fifty_two_week.pct_from_low
        # pct_from_low is % above the 52-week low
        # Lower 40% of range = pct_from_low < 40% of total range
        if fundamentals.fifty_two_week.high is not None and fundamentals.fifty_two_week.low is not None:
            high = fundamentals.fifty_two_week.high
            low = fundamentals.fifty_two_week.low
            total_range = high - low
            if total_range > 0:
                position_pct = (technicals.current_price - low) / total_range * 100
                ftwk_fav = position_pct <= cfg.bull_entry_52wk_pct
                desc_52 = f"Price at {position_pct:.0f}th percentile of 52-week range"
            else:
                desc_52 = "52-week range is zero"
        else:
            desc_52 = "52-week data incomplete"
    else:
        desc_52 = "52-week data unavailable"

    signals.append(OpportunitySignal(
        name="52wk_position",
        favorable=ftwk_fav,
        weight=0.10,
        description=desc_52,
    ))

    # 6. Macro clear
    from market_regime.models.macro import MacroEventImpact

    high_events_7d = [
        e for e in macro.events_next_7_days if e.impact == MacroEventImpact.HIGH
    ]
    macro_fav = len(high_events_7d) == 0
    signals.append(OpportunitySignal(
        name="macro_clear",
        favorable=macro_fav,
        weight=0.05,
        description=(
            "No HIGH-impact macro events in next 7 days"
            if macro_fav
            else f"{len(high_events_7d)} HIGH-impact event(s) in next 7 days"
        ),
    ))

    # 7. RSI not overbought (for bull LEAPs)
    rsi = technicals.rsi.value
    rsi_fav = rsi < 65.0
    signals.append(OpportunitySignal(
        name="rsi_not_overbought",
        favorable=rsi_fav,
        weight=0.05,
        description=(
            f"RSI {rsi:.0f} — not overbought"
            if rsi_fav
            else f"RSI {rsi:.0f} — overbought, LEAP entry at premium"
        ),
    ))

    return signals


def _select_strategy(
    regime: RegimeResult,
    phase: PhaseResult,
    iv_env: str,
    verdict: Verdict,
) -> tuple[LEAPStrategy, StrategyRecommendation]:
    """Select LEAP strategy from phase × regime matrix."""
    rid = int(regime.regime)
    pid = int(phase.phase)

    if verdict == Verdict.NO_GO:
        return LEAPStrategy.NO_TRADE, StrategyRecommendation(
            name="No Trade",
            direction="neutral",
            structure="Do not enter LEAP position.",
            rationale="Conditions unfavorable for LEAP entry.",
            risk_notes=["Wait for better conditions."],
        )

    # Phase × Regime matrix
    if pid == 1:  # Accumulation
        if rid == 4:
            return _no_trade("IV too expensive in R4 for accumulation LEAP entry.")
        elif rid == 2:
            return LEAPStrategy.BULL_CALL_SPREAD, StrategyRecommendation(
                name="Bull Call Spread (LEAP)",
                direction="bullish",
                structure="Buy ATM call, sell OTM call — 12-18 months out.",
                rationale="Accumulation base + moderate IV — spread reduces vega risk.",
                risk_notes=["Spread caps upside.", "IV compression benefits the position."],
            )
        else:  # R1 or R3
            return LEAPStrategy.BULL_CALL_LEAP, StrategyRecommendation(
                name="Bull Call LEAP",
                direction="bullish",
                structure="Buy ATM or slightly ITM call — 18-24 months out.",
                rationale=(
                    f"Accumulation phase + {'cheap' if rid == 1 else 'moderate'} IV — "
                    f"ideal LEAP entry zone."
                ),
                risk_notes=[
                    "Phase could extend before breakout.",
                    "Set time-based stop if no markup within 6 months.",
                ],
            )

    elif pid == 2:  # Markup
        if rid == 4:
            return _no_trade("IV too expensive in R4 for markup LEAP entry.")
        elif rid in (1, 2):
            return LEAPStrategy.PMCC, StrategyRecommendation(
                name="Poor Man's Covered Call (PMCC)",
                direction="bullish",
                structure="Buy deep ITM LEAP call, sell near-term OTM calls monthly.",
                rationale="Markup phase — ride the trend with income from short calls.",
                risk_notes=[
                    "Short call assignment risk if ITM at expiry.",
                    "LEAP must be deep ITM (80+ delta).",
                ],
            )
        else:  # R3
            return LEAPStrategy.BULL_CALL_LEAP, StrategyRecommendation(
                name="Bull Call LEAP",
                direction="bullish",
                structure="Buy ATM call — 12-18 months out.",
                rationale="Markup + low-vol trend — LEAP benefits from delta and time.",
                risk_notes=["Late markup entry has less upside.", "Trail stops."],
            )

    elif pid == 3:  # Distribution
        if rid in (1, 2):
            return LEAPStrategy.PROTECTIVE_PUT, StrategyRecommendation(
                name="Protective Put (LEAP)",
                direction="bearish",
                structure="Buy OTM put LEAP — 12-18 months out as portfolio hedge.",
                rationale="Distribution phase — protect existing positions with long puts.",
                risk_notes=["Theta decay on long puts.", "Size appropriately as insurance."],
            )
        elif rid == 4:
            return LEAPStrategy.BEAR_PUT_LEAP, StrategyRecommendation(
                name="Bear Put LEAP",
                direction="bearish",
                structure="Buy ATM put — 18-24 months out.",
                rationale="Distribution + high vol trending — bearish LEAP for downside.",
                risk_notes=["IV is expensive — consider spread to reduce cost."],
            )
        else:
            return _no_trade("Distribution phase with moderate IV — wait for clarity.")

    else:  # pid == 4, Markdown
        if rid in (2, 4):
            return LEAPStrategy.BEAR_PUT_LEAP, StrategyRecommendation(
                name="Bear Put LEAP",
                direction="bearish",
                structure="Buy ATM put — 18-24 months out.",
                rationale="Markdown phase — bearish LEAP for continued downside.",
                risk_notes=[
                    "Watch for accumulation signs (P1 transition).",
                    "IV elevated — bear put spread may be more efficient.",
                ],
            )
        else:
            return _no_trade("Markdown phase — wait for accumulation (P1) before bullish LEAPs.")


def _no_trade(rationale: str) -> tuple[LEAPStrategy, StrategyRecommendation]:
    return LEAPStrategy.NO_TRADE, StrategyRecommendation(
        name="No Trade",
        direction="neutral",
        structure="Do not enter LEAP position.",
        rationale=rationale,
        risk_notes=["Wait for better conditions."],
    )


def _build_summary(
    ticker: str,
    verdict: Verdict,
    confidence: float,
    strategy: LEAPStrategy,
    hard_stops: list[HardStop],
    regime: RegimeResult,
    phase: PhaseResult,
    iv_env: str,
    fund_score: FundamentalScore,
) -> str:
    if verdict == Verdict.NO_GO:
        reasons = "; ".join(s.name for s in hard_stops) if hard_stops else "low confidence"
        return (
            f"{ticker} LEAP: NO-GO ({reasons}). "
            f"R{int(regime.regime)}, {phase.phase_name}, IV {iv_env}."
        )
    elif verdict == Verdict.CAUTION:
        return (
            f"{ticker} LEAP: CAUTION ({confidence:.0%}). "
            f"R{int(regime.regime)} {phase.phase_name}, IV {iv_env}, "
            f"fundamentals {fund_score.score:.0%}. "
            f"Consider {strategy.value} with reduced size."
        )
    else:
        return (
            f"{ticker} LEAP: GO ({confidence:.0%}). "
            f"R{int(regime.regime)} {phase.phase_name}, IV {iv_env}, "
            f"fundamentals {fund_score.score:.0%}. "
            f"Recommended: {strategy.value}."
        )
