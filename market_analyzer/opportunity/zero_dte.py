"""0DTE opportunity assessment — go/no-go + strategy recommendation."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from market_analyzer.config import get_settings
from market_analyzer.models.opportunity import (
    HardStop,
    OpportunitySignal,
    StrategyRecommendation,
    Verdict,
    ZeroDTEOpportunity,
    ZeroDTEStrategy,
)

if TYPE_CHECKING:
    from market_analyzer.models.fundamentals import FundamentalsSnapshot
    from market_analyzer.models.macro import MacroCalendar, MacroEventImpact
    from market_analyzer.models.regime import RegimeResult
    from market_analyzer.models.technicals import ORBData, TechnicalSnapshot


def assess_zero_dte(
    ticker: str,
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    macro: MacroCalendar,
    fundamentals: FundamentalsSnapshot | None = None,
    orb: ORBData | None = None,
    as_of: date | None = None,
) -> ZeroDTEOpportunity:
    """Assess 0DTE opportunity for a single instrument.

    Pure function — consumes pre-computed analysis, produces structured assessment.
    No data fetching.
    """
    cfg = get_settings().opportunity.zero_dte
    today = as_of or date.today()

    # --- Days to earnings ---
    days_to_earnings: int | None = None
    if fundamentals is not None:
        days_to_earnings = fundamentals.upcoming_events.days_to_earnings

    # --- Hard stops ---
    hard_stops = _check_hard_stops(
        regime, technicals, macro, days_to_earnings, today, cfg,
    )

    # --- Scoring signals ---
    signals = _score_signals(regime, technicals, macro, orb, today, cfg)

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

    # --- Macro event today ---
    has_macro_today = _has_high_impact_today(macro, today)

    # --- Strategy selection ---
    orb_status_str = orb.status.value if orb is not None else None
    zero_dte_strategy, strategy_rec = _select_strategy(
        regime, orb_status_str, technicals, verdict,
    )

    # --- Summary ---
    summary = _build_summary(
        ticker, verdict, confidence, zero_dte_strategy, hard_stops, regime,
    )

    return ZeroDTEOpportunity(
        ticker=ticker,
        as_of_date=today,
        verdict=verdict,
        confidence=round(confidence, 2),
        hard_stops=hard_stops,
        signals=signals,
        strategy=strategy_rec,
        zero_dte_strategy=zero_dte_strategy,
        regime_id=int(regime.regime),
        regime_confidence=round(regime.confidence, 2),
        atr_pct=round(technicals.atr_pct, 2),
        orb_status=orb_status_str,
        has_macro_event_today=has_macro_today,
        days_to_earnings=days_to_earnings,
        summary=summary,
    )


# --- Private helpers ---


def _check_hard_stops(
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    macro: MacroCalendar,
    days_to_earnings: int | None,
    today: date,
    cfg,
) -> list[HardStop]:
    stops: list[HardStop] = []

    # Earnings blackout
    if days_to_earnings is not None and days_to_earnings <= cfg.earnings_blackout_days:
        stops.append(HardStop(
            name="earnings_blackout",
            description=f"Earnings in {days_to_earnings} day(s) — 0DTE too risky near earnings.",
        ))

    # Macro event today
    if _has_high_impact_today(macro, today):
        events = _high_impact_events_on(macro, today)
        names = ", ".join(e.name for e in events)
        stops.append(HardStop(
            name="macro_event_today",
            description=f"HIGH impact macro event today: {names}.",
        ))

    # ATR too low
    if technicals.atr_pct < cfg.min_atr_pct:
        stops.append(HardStop(
            name="atr_too_low",
            description=f"ATR {technicals.atr_pct:.2f}% < {cfg.min_atr_pct}% — no premium worth selling.",
        ))

    # ATR too high
    if technicals.atr_pct > cfg.max_atr_pct:
        stops.append(HardStop(
            name="atr_too_high",
            description=f"ATR {technicals.atr_pct:.2f}% > {cfg.max_atr_pct}% — moves too large for 0DTE.",
        ))

    # R4 high confidence
    if int(regime.regime) == 4 and regime.confidence > cfg.r4_confidence_threshold:
        stops.append(HardStop(
            name="r4_high_confidence",
            description=(
                f"R4 (High-Vol Trending) at {regime.confidence:.0%} confidence "
                f"— explosive moves, 0DTE is too dangerous."
            ),
        ))

    return stops


def _has_high_impact_today(macro: MacroCalendar, today: date) -> bool:
    return len(_high_impact_events_on(macro, today)) > 0


def _high_impact_events_on(macro: MacroCalendar, today: date) -> list:
    from market_analyzer.models.macro import MacroEventImpact

    return [
        e for e in macro.events_next_7_days
        if e.date == today and e.impact == MacroEventImpact.HIGH
    ]


def _has_high_impact_tomorrow(macro: MacroCalendar, today: date) -> bool:
    from datetime import timedelta
    from market_analyzer.models.macro import MacroEventImpact

    tomorrow = today + timedelta(days=1)
    return any(
        e.date == tomorrow and e.impact == MacroEventImpact.HIGH
        for e in macro.events_next_7_days
    )


def _score_signals(
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    macro: MacroCalendar,
    orb: ORBData | None,
    today: date,
    cfg,
) -> list[OpportunitySignal]:
    signals: list[OpportunitySignal] = []

    # 1. Regime favorable (R1/R2)
    rid = int(regime.regime)
    regime_fav = rid in (1, 2)
    signals.append(OpportunitySignal(
        name="regime_favorable",
        favorable=regime_fav,
        weight=0.25,
        description=(
            f"R{rid} ({'ideal' if rid == 1 else 'acceptable'} for 0DTE)"
            if regime_fav
            else f"R{rid} ({'trending' if rid == 3 else 'high-vol trending'} — less ideal)"
        ),
    ))

    # 2. ATR sweet spot
    atr = technicals.atr_pct
    atr_fav = cfg.atr_sweet_low <= atr <= cfg.atr_sweet_high
    signals.append(OpportunitySignal(
        name="atr_sweet_spot",
        favorable=atr_fav,
        weight=0.15,
        description=(
            f"ATR {atr:.2f}% in sweet spot ({cfg.atr_sweet_low}-{cfg.atr_sweet_high}%)"
            if atr_fav
            else f"ATR {atr:.2f}% outside sweet spot"
        ),
    ))

    # 3. ORB alignment
    if orb is not None:
        from market_analyzer.models.technicals import ORBStatus

        orb_fav = orb.status in (
            ORBStatus.WITHIN, ORBStatus.BREAKOUT_LONG, ORBStatus.BREAKOUT_SHORT,
        )
        signals.append(OpportunitySignal(
            name="orb_alignment",
            favorable=orb_fav,
            weight=0.15,
            description=f"ORB status: {orb.status.value}",
        ))
    else:
        # No ORB data — neutral, reduce weight
        signals.append(OpportunitySignal(
            name="orb_alignment",
            favorable=True,
            weight=0.05,
            description="No intraday data — ORB not available",
        ))

    # 4. RSI not extreme
    rsi = technicals.rsi.value
    rsi_fav = 30.0 <= rsi <= 70.0
    signals.append(OpportunitySignal(
        name="rsi_not_extreme",
        favorable=rsi_fav,
        weight=0.10,
        description=(
            f"RSI {rsi:.0f} in normal range"
            if rsi_fav
            else f"RSI {rsi:.0f} — extreme, momentum could override mean reversion"
        ),
    ))

    # 5. Bollinger position
    pct_b = technicals.bollinger.percent_b
    bb_fav = 0.1 <= pct_b <= 0.9
    signals.append(OpportunitySignal(
        name="bollinger_favorable",
        favorable=bb_fav,
        weight=0.10,
        description=(
            f"Price within Bollinger Bands (%B={pct_b:.2f})"
            if bb_fav
            else f"Price at Bollinger Band extreme (%B={pct_b:.2f})"
        ),
    ))

    # 6. No macro tomorrow
    no_macro_tmrw = not _has_high_impact_tomorrow(macro, today)
    signals.append(OpportunitySignal(
        name="no_macro_tomorrow",
        favorable=no_macro_tmrw,
        weight=0.10,
        description=(
            "No HIGH-impact macro events tomorrow"
            if no_macro_tmrw
            else "HIGH-impact macro event tomorrow — end-of-day positioning risk"
        ),
    ))

    # 7. Support/resistance defined
    sr = technicals.support_resistance
    sr_fav = sr.support is not None and sr.resistance is not None
    signals.append(OpportunitySignal(
        name="sr_levels_defined",
        favorable=sr_fav,
        weight=0.10,
        description=(
            f"Support {sr.support:.2f} / Resistance {sr.resistance:.2f} — clear strike levels"
            if sr_fav
            else "Support or resistance undefined — harder to place strikes"
        ),
    ))

    # 8. Volume normal
    # Check if any volume-related signal is extreme
    vol_signals = [s for s in technicals.signals if "volume" in s.name.lower()]
    vol_fav = len(vol_signals) == 0  # No volume anomaly signals = normal
    signals.append(OpportunitySignal(
        name="volume_normal",
        favorable=vol_fav,
        weight=0.05,
        description=(
            "Volume is normal"
            if vol_fav
            else "Unusual volume detected"
        ),
    ))

    return signals


def _select_strategy(
    regime: RegimeResult,
    orb_status: str | None,
    technicals: TechnicalSnapshot,
    verdict: Verdict,
) -> tuple[ZeroDTEStrategy, StrategyRecommendation]:
    """Select 0DTE strategy from regime × ORB matrix."""
    rid = int(regime.regime)

    # R4 or NO_GO → no trade
    if rid == 4 or verdict == Verdict.NO_GO:
        return ZeroDTEStrategy.NO_TRADE, StrategyRecommendation(
            name="No Trade",
            direction="neutral",
            structure="Do not trade 0DTE today.",
            rationale="Conditions unfavorable for 0DTE.",
            risk_notes=["Wait for better conditions."],
        )

    # Default ORB status for matrix lookup
    orb = orb_status or "within"

    # Strategy matrix
    if rid == 1:
        return _r1_strategy(orb, technicals)
    elif rid == 2:
        return _r2_strategy(orb, technicals)
    else:  # rid == 3
        return _r3_strategy(orb, regime, technicals)


def _r1_strategy(
    orb: str, technicals: TechnicalSnapshot,
) -> tuple[ZeroDTEStrategy, StrategyRecommendation]:
    """R1 (Low-Vol MR): theta is primary."""
    sr = technicals.support_resistance

    if orb in ("within", "failed_long", "failed_short"):
        strike_note = ""
        if sr.support is not None and sr.resistance is not None:
            strike_note = f" Sell put at {sr.support:.2f}, sell call at {sr.resistance:.2f}."
        return ZeroDTEStrategy.IRON_CONDOR, StrategyRecommendation(
            name="Iron Condor",
            direction="neutral",
            structure=f"Sell OTM put and call spreads around current price.{strike_note}",
            rationale="R1 mean-reverting, low vol — ideal for premium selling.",
            risk_notes=["Max loss = spread width minus premium.", "Close if range breaks."],
        )
    elif orb == "breakout_long":
        return ZeroDTEStrategy.CREDIT_SPREAD, StrategyRecommendation(
            name="Credit Put Spread",
            direction="bullish",
            structure="Sell put spread below breakout level.",
            rationale="R1 with upside ORB breakout — sell put spread below range.",
            risk_notes=["Failed breakout reverses quickly in R1."],
        )
    else:  # breakout_short
        return ZeroDTEStrategy.CREDIT_SPREAD, StrategyRecommendation(
            name="Credit Call Spread",
            direction="bearish",
            structure="Sell call spread above breakout level.",
            rationale="R1 with downside ORB breakout — sell call spread above range.",
            risk_notes=["Failed breakout reverses quickly in R1."],
        )


def _r2_strategy(
    orb: str, technicals: TechnicalSnapshot,
) -> tuple[ZeroDTEStrategy, StrategyRecommendation]:
    """R2 (High-Vol MR): wider wings, defined risk."""
    if orb in ("within", "failed_long", "failed_short"):
        return ZeroDTEStrategy.STRADDLE_STRANGLE, StrategyRecommendation(
            name="Short Strangle (Wide Wings)",
            direction="neutral",
            structure="Sell wide OTM strangle with defined-risk wings.",
            rationale="R2 high-vol MR — wider premium but mean-reverting.",
            risk_notes=["Use wider strikes than R1.", "Define max loss with wings."],
        )
    else:
        direction = "bullish" if orb == "breakout_long" else "bearish"
        return ZeroDTEStrategy.DIRECTIONAL_SPREAD, StrategyRecommendation(
            name="Directional Spread (Defined Risk)",
            direction=direction,
            structure=f"{'Debit call' if direction == 'bullish' else 'Debit put'} spread in breakout direction.",
            rationale=f"R2 with ORB {orb} — defined-risk directional play.",
            risk_notes=["R2 is mean-reverting; breakout may fade.", "Keep size small."],
        )


def _r3_strategy(
    orb: str, regime: RegimeResult, technicals: TechnicalSnapshot,
) -> tuple[ZeroDTEStrategy, StrategyRecommendation]:
    """R3 (Low-Vol Trending): directional with trend."""
    trend = regime.trend_direction

    if orb in ("within",):
        # Trade with the trend direction
        if trend is not None:
            direction = "bullish" if trend.value == "bullish" else "bearish"
        else:
            direction = "bullish"  # default
        return ZeroDTEStrategy.DIRECTIONAL_SPREAD, StrategyRecommendation(
            name=f"Directional {'Call' if direction == 'bullish' else 'Put'} Spread",
            direction=direction,
            structure=f"{'Debit call' if direction == 'bullish' else 'Debit put'} spread with trend.",
            rationale=f"R3 low-vol trending ({direction}) — ride the trend.",
            risk_notes=["Trend reversal on 0DTE is rare but devastating."],
        )
    elif orb in ("breakout_long", "breakout_short"):
        direction = "bullish" if orb == "breakout_long" else "bearish"
        return ZeroDTEStrategy.DIRECTIONAL_SPREAD, StrategyRecommendation(
            name=f"Directional {'Call' if direction == 'bullish' else 'Put'} Spread",
            direction=direction,
            structure=f"{'Debit call' if direction == 'bullish' else 'Debit put'} spread in breakout direction.",
            rationale=f"R3 trending with ORB {orb} confirmation.",
            risk_notes=["Target ORB extension levels for exits."],
        )
    else:  # failed breakout
        direction = "bearish" if orb == "failed_long" else "bullish"
        return ZeroDTEStrategy.CREDIT_SPREAD, StrategyRecommendation(
            name=f"Credit {'Call' if direction == 'bearish' else 'Put'} Spread (Fade)",
            direction=direction,
            structure=f"Sell {'call' if direction == 'bearish' else 'put'} spread — fade the failed breakout.",
            rationale=f"ORB {orb} in R3 — fade the failure back to range.",
            risk_notes=["Counter-trend; keep size small."],
        )


def _build_summary(
    ticker: str,
    verdict: Verdict,
    confidence: float,
    strategy: ZeroDTEStrategy,
    hard_stops: list[HardStop],
    regime: RegimeResult,
) -> str:
    if verdict == Verdict.NO_GO:
        reasons = "; ".join(s.name for s in hard_stops) if hard_stops else "low confidence"
        return f"{ticker} 0DTE: NO-GO ({reasons}). R{int(regime.regime)}."
    elif verdict == Verdict.CAUTION:
        return (
            f"{ticker} 0DTE: CAUTION ({confidence:.0%}). "
            f"R{int(regime.regime)}. Consider {strategy.value} with reduced size."
        )
    else:
        return (
            f"{ticker} 0DTE: GO ({confidence:.0%}). "
            f"R{int(regime.regime)}. Recommended: {strategy.value}."
        )
