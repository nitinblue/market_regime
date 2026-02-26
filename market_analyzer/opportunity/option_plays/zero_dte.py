"""0DTE opportunity assessment — go/no-go + strategy recommendation.

Integrates ORB levels into every strategy for ORB-aware strike placement:
- IRON_CONDOR: short strikes at ORB range edges
- IRON_MAN (inverse IC): long strikes at ORB range edges, profits from breakout
- CREDIT_SPREAD: short strike beyond ORB level in breakout direction
- DIRECTIONAL_SPREAD: entry near ORB breakout, target extension levels
- STRADDLE_STRANGLE: OTM offset informed by ORB range width
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from market_analyzer.config import get_settings
from market_analyzer.models.opportunity import (
    HardStop,
    ORBDecision,
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
    from market_analyzer.models.vol_surface import VolatilitySurface


def assess_zero_dte(
    ticker: str,
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    macro: MacroCalendar,
    fundamentals: FundamentalsSnapshot | None = None,
    orb: ORBData | None = None,
    vol_surface: VolatilitySurface | None = None,
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

    # --- ORB decision ---
    orb_decision = _build_orb_decision(orb) if orb is not None else None

    # --- Strategy selection (ORB-aware) ---
    orb_status_str = orb.status.value if orb is not None else None
    zero_dte_strategy, strategy_rec = _select_strategy(
        regime, orb_status_str, technicals, verdict, orb,
    )

    # --- Trade spec (ORB-aware strike placement) ---
    trade_spec = None
    if verdict != Verdict.NO_GO and zero_dte_strategy != ZeroDTEStrategy.NO_TRADE:
        trade_spec = _build_trade_spec(
            ticker, technicals, zero_dte_strategy, strategy_rec,
            int(regime.regime), vol_surface, orb,
        )

    # --- Summary ---
    summary = _build_summary(
        ticker, verdict, confidence, zero_dte_strategy, hard_stops, regime, orb,
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
        orb_decision=orb_decision,
        has_macro_event_today=has_macro_today,
        days_to_earnings=days_to_earnings,
        trade_spec=trade_spec,
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


def _is_narrow_orb(orb: ORBData | None) -> bool:
    """Check if ORB range is narrow (coiled for breakout)."""
    if orb is None:
        return False
    # Narrow range: <0.5% or <30% of daily ATR
    if orb.range_pct < 0.5:
        return True
    if orb.range_vs_daily_atr_pct is not None and orb.range_vs_daily_atr_pct < 30:
        return True
    return False


def _build_orb_decision(orb: ORBData) -> ORBDecision:
    """Build ORB decision context from ORB data."""
    from market_analyzer.models.technicals import ORBStatus

    # Direction from ORB status
    if orb.status == ORBStatus.BREAKOUT_LONG:
        direction = "bullish"
    elif orb.status == ORBStatus.BREAKOUT_SHORT:
        direction = "bearish"
    elif orb.status == ORBStatus.FAILED_LONG:
        direction = "bearish"
    elif orb.status == ORBStatus.FAILED_SHORT:
        direction = "bullish"
    else:
        direction = "neutral"

    # Key levels from ORB
    key_levels: dict[str, float] = {
        "range_high": orb.range_high,
        "range_low": orb.range_low,
    }
    if orb.session_vwap is not None:
        key_levels["vwap"] = orb.session_vwap
    for level in orb.levels:
        # Convert labels like "T1 Long (1.0x)" to "T1_long"
        key = level.label.split("(")[0].strip().lower().replace(" ", "_")
        key_levels[key] = level.price

    # Decision text
    status_decisions = {
        ORBStatus.WITHIN: (
            "Within ORB range — wait for breakout or sell premium at range edges."
            if not _is_narrow_orb(orb)
            else "Narrow ORB range — coiled for breakout, consider Iron Man (inverse IC)."
        ),
        ORBStatus.BREAKOUT_LONG: (
            f"ORB breakout long above {orb.range_high:.2f}. "
            f"Trade with direction — credit put spread or debit call spread. "
            f"Target: T1 extension."
        ),
        ORBStatus.BREAKOUT_SHORT: (
            f"ORB breakout short below {orb.range_low:.2f}. "
            f"Trade with direction — credit call spread or debit put spread. "
            f"Target: T1 extension."
        ),
        ORBStatus.FAILED_LONG: (
            f"Failed ORB breakout long — bearish reversal. "
            f"Fade the failure: credit call spread above {orb.range_high:.2f}."
        ),
        ORBStatus.FAILED_SHORT: (
            f"Failed ORB breakout short — bullish reversal. "
            f"Fade the failure: credit put spread below {orb.range_low:.2f}."
        ),
    }

    return ORBDecision(
        status=orb.status.value,
        range_high=orb.range_high,
        range_low=orb.range_low,
        range_pct=orb.range_pct,
        direction=direction,
        decision=status_decisions[orb.status],
        key_levels=key_levels,
    )


def _select_strategy(
    regime: RegimeResult,
    orb_status: str | None,
    technicals: TechnicalSnapshot,
    verdict: Verdict,
    orb: ORBData | None = None,
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
    orb_st = orb_status or "within"
    narrow = _is_narrow_orb(orb)

    # Strategy matrix
    if rid == 1:
        return _r1_strategy(orb_st, technicals, narrow, orb)
    elif rid == 2:
        return _r2_strategy(orb_st, technicals, narrow, orb)
    else:  # rid == 3
        return _r3_strategy(orb_st, regime, technicals, narrow, orb)


def _r1_strategy(
    orb: str, technicals: TechnicalSnapshot,
    narrow: bool, orb_data: ORBData | None,
) -> tuple[ZeroDTEStrategy, StrategyRecommendation]:
    """R1 (Low-Vol MR): theta is primary. Iron Man on narrow ORB range."""
    sr = technicals.support_resistance

    if narrow and orb_data is not None:
        # Narrow ORB in R1 — range is compressed, expect breakout
        return ZeroDTEStrategy.IRON_MAN, StrategyRecommendation(
            name="Iron Man (Inverse Iron Condor)",
            direction="neutral",
            structure=(
                f"Buy put spread + call spread around ORB range "
                f"({orb_data.range_low:.2f}–{orb_data.range_high:.2f}). "
                f"Net debit. Profits from big move in either direction."
            ),
            rationale=(
                f"R1 with narrow ORB range ({orb_data.range_pct:.2f}%) — "
                f"coiled for breakout despite mean-reverting regime."
            ),
            risk_notes=[
                "Max loss = net debit paid.",
                "Need price to move past ORB range edges by session end.",
                "R1 mean reversion may cap the move — keep size small.",
            ],
        )

    if orb in ("within", "failed_long", "failed_short"):
        strike_note = ""
        if orb_data is not None:
            strike_note = (
                f" Short put near ORB low {orb_data.range_low:.2f}, "
                f"short call near ORB high {orb_data.range_high:.2f}."
            )
        elif sr.support is not None and sr.resistance is not None:
            strike_note = f" Sell put at {sr.support:.2f}, sell call at {sr.resistance:.2f}."
        return ZeroDTEStrategy.IRON_CONDOR, StrategyRecommendation(
            name="Iron Condor",
            direction="neutral",
            structure=f"Sell OTM put and call spreads around current price.{strike_note}",
            rationale="R1 mean-reverting, low vol — ideal for premium selling.",
            risk_notes=["Max loss = spread width minus premium.", "Close if range breaks."],
        )
    elif orb == "breakout_long":
        target = ""
        if orb_data is not None:
            target = f" Put spread below ORB low {orb_data.range_low:.2f}."
        return ZeroDTEStrategy.CREDIT_SPREAD, StrategyRecommendation(
            name="Credit Put Spread",
            direction="bullish",
            structure=f"Sell put spread below breakout level.{target}",
            rationale="R1 with upside ORB breakout — sell put spread below range.",
            risk_notes=["Failed breakout reverses quickly in R1."],
        )
    else:  # breakout_short
        target = ""
        if orb_data is not None:
            target = f" Call spread above ORB high {orb_data.range_high:.2f}."
        return ZeroDTEStrategy.CREDIT_SPREAD, StrategyRecommendation(
            name="Credit Call Spread",
            direction="bearish",
            structure=f"Sell call spread above breakout level.{target}",
            rationale="R1 with downside ORB breakout — sell call spread above range.",
            risk_notes=["Failed breakout reverses quickly in R1."],
        )


def _r2_strategy(
    orb: str, technicals: TechnicalSnapshot,
    narrow: bool, orb_data: ORBData | None,
) -> tuple[ZeroDTEStrategy, StrategyRecommendation]:
    """R2 (High-Vol MR): wider wings. Iron Man on narrow range (big move brewing)."""

    if narrow and orb_data is not None:
        # Narrow ORB in R2 (high vol but compressed range) — explosive breakout likely
        return ZeroDTEStrategy.IRON_MAN, StrategyRecommendation(
            name="Iron Man (Inverse Iron Condor)",
            direction="neutral",
            structure=(
                f"Buy put spread + call spread around ORB range "
                f"({orb_data.range_low:.2f}–{orb_data.range_high:.2f}). "
                f"Net debit. R2 high vol amplifies breakout potential."
            ),
            rationale=(
                f"R2 high-vol with narrow ORB range ({orb_data.range_pct:.2f}%) — "
                f"compressed range in high-vol regime = explosive breakout setup."
            ),
            risk_notes=[
                "Max loss = net debit paid.",
                "R2 high vol means bigger moves — better R:R for Iron Man.",
                "Close by 3PM ET if no breakout materializes.",
            ],
        )

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
        target = ""
        if orb_data is not None:
            # Find T1 extension level
            for level in orb_data.levels:
                if "T1" in level.label and {"bullish": "long", "bearish": "short"}.get(direction, "") in level.label.lower():
                    target = f" Target T1: {level.price:.2f}."
                    break
        return ZeroDTEStrategy.DIRECTIONAL_SPREAD, StrategyRecommendation(
            name="Directional Spread (Defined Risk)",
            direction=direction,
            structure=(
                f"{'Debit call' if direction == 'bullish' else 'Debit put'} "
                f"spread in breakout direction.{target}"
            ),
            rationale=f"R2 with ORB {orb} — defined-risk directional play.",
            risk_notes=["R2 is mean-reverting; breakout may fade.", "Keep size small."],
        )


def _r3_strategy(
    orb: str, regime: RegimeResult, technicals: TechnicalSnapshot,
    narrow: bool, orb_data: ORBData | None,
) -> tuple[ZeroDTEStrategy, StrategyRecommendation]:
    """R3 (Low-Vol Trending): directional with trend. Iron Man on narrow range."""
    trend = regime.trend_direction

    if narrow and orb_data is not None:
        # Narrow ORB in R3 (trending) — breakout likely to follow trend
        return ZeroDTEStrategy.IRON_MAN, StrategyRecommendation(
            name="Iron Man (Inverse Iron Condor)",
            direction="neutral",
            structure=(
                f"Buy put spread + call spread around ORB range "
                f"({orb_data.range_low:.2f}–{orb_data.range_high:.2f}). "
                f"Trending regime should produce directional breakout."
            ),
            rationale=(
                f"R3 trending with narrow ORB range ({orb_data.range_pct:.2f}%) — "
                f"trend energy compressed, breakout imminent."
            ),
            risk_notes=[
                "Max loss = net debit paid.",
                "R3 trend likely picks a direction — may convert to directional after breakout.",
            ],
        )

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
        target = ""
        if orb_data is not None:
            for level in orb_data.levels:
                if "T1" in level.label and {"bullish": "long", "bearish": "short"}.get(direction, "") in level.label.lower():
                    target = f" Target T1: {level.price:.2f}."
                    break
        return ZeroDTEStrategy.DIRECTIONAL_SPREAD, StrategyRecommendation(
            name=f"Directional {'Call' if direction == 'bullish' else 'Put'} Spread",
            direction=direction,
            structure=(
                f"{'Debit call' if direction == 'bullish' else 'Debit put'} "
                f"spread in breakout direction.{target}"
            ),
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


def _build_trade_spec(
    ticker: str,
    technicals: TechnicalSnapshot,
    zero_dte_strategy: ZeroDTEStrategy,
    strategy_rec: StrategyRecommendation,
    regime_id: int,
    vol_surface: VolatilitySurface | None,
    orb: ORBData | None = None,
):
    """Build 0DTE trade spec with ORB-aware strike placement."""
    from market_analyzer.models.opportunity import (
        LegAction, LegSpec, OrderSide, StructureType, TradeSpec,
    )
    from market_analyzer.opportunity.option_plays._trade_spec_helpers import (
        build_credit_spread_legs,
        build_debit_spread_legs,
        build_inverse_iron_condor_legs,
        build_iron_condor_legs,
        build_straddle_legs,
        compute_atm_strike,
        find_best_expiration,
        snap_strike,
    )

    price = technicals.current_price
    atr = technicals.atr
    today = date.today()

    # Find 0DTE expiration from vol_surface
    if vol_surface is not None and vol_surface.term_structure:
        exp_pt = find_best_expiration(vol_surface.term_structure, 0, 1)
        if exp_pt is not None:
            exp_date = exp_pt.expiration
            dte = exp_pt.days_to_expiry
            atm_iv = exp_pt.atm_iv
        else:
            exp_date = today
            dte = 0
            atm_iv = 0.20
    else:
        exp_date = today
        dte = 0
        atm_iv = 0.20

    direction = strategy_rec.direction

    # ORB range for strike placement
    orb_high = orb.range_high if orb is not None else None
    orb_low = orb.range_low if orb is not None else None

    try:
        if zero_dte_strategy == ZeroDTEStrategy.IRON_MAN:
            # Inverse iron condor — long strikes at ORB range edges
            legs, wing_width = build_inverse_iron_condor_legs(
                price, atr, regime_id, exp_date, dte, atm_iv,
                orb_range_high=orb_high, orb_range_low=orb_low,
            )
            rationale = f"0DTE Iron Man (inverse IC). R{regime_id}."
            if orb is not None:
                rationale += (
                    f" Long strikes at ORB range "
                    f"({orb.range_low:.2f}–{orb.range_high:.2f})."
                )
            return TradeSpec(
                ticker=ticker, legs=legs, underlying_price=price,
                target_dte=dte, target_expiration=exp_date,
                wing_width_points=wing_width,
                max_risk_per_spread=f"Net debit (wing width ${wing_width * 100:.0f} minus debit)",
                spec_rationale=rationale,
                structure_type=StructureType.IRON_MAN,
                order_side=OrderSide.DEBIT,
                profit_target_pct=1.0,
                stop_loss_pct=1.0,
                exit_dte=0,
                max_profit_desc=f"Wing width (${wing_width:.0f}) minus debit paid",
                max_loss_desc="Net debit paid",
                exit_notes=["0DTE: close by 3PM ET if no breakout",
                            "Profit when underlying moves past long strikes",
                            "Max loss = net debit if price stays within range"],
            )

        elif zero_dte_strategy == ZeroDTEStrategy.IRON_CONDOR:
            # ORB-aware IC: short strikes at ORB range edges if available
            if orb_high is not None and orb_low is not None:
                # Override short strikes with ORB range levels
                short_put = snap_strike(orb_low, price)
                short_call = snap_strike(orb_high, price)
                wing_mult = 0.5 if regime_id == 1 else 0.7
                wing_width = atr * wing_mult
                long_put = snap_strike(short_put - wing_width, price)
                long_call = snap_strike(short_call + wing_width, price)
                wing_width_pts = short_put - long_put
                legs = [
                    LegSpec(
                        role="short_put", action=LegAction.SELL_TO_OPEN,
                        option_type="put", strike=short_put,
                        strike_label=f"ORB low ({orb_low:.2f})",
                        expiration=exp_date, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
                    ),
                    LegSpec(
                        role="long_put", action=LegAction.BUY_TO_OPEN,
                        option_type="put", strike=long_put,
                        strike_label=f"wing {wing_mult:.1f} ATR below ORB low",
                        expiration=exp_date, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
                    ),
                    LegSpec(
                        role="short_call", action=LegAction.SELL_TO_OPEN,
                        option_type="call", strike=short_call,
                        strike_label=f"ORB high ({orb_high:.2f})",
                        expiration=exp_date, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
                    ),
                    LegSpec(
                        role="long_call", action=LegAction.BUY_TO_OPEN,
                        option_type="call", strike=long_call,
                        strike_label=f"wing {wing_mult:.1f} ATR above ORB high",
                        expiration=exp_date, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
                    ),
                ]
                return TradeSpec(
                    ticker=ticker, legs=legs, underlying_price=price,
                    target_dte=dte, target_expiration=exp_date,
                    wing_width_points=wing_width_pts,
                    spec_rationale=(
                        f"0DTE iron condor. R{regime_id}. "
                        f"Short strikes at ORB range ({orb_low:.2f}–{orb_high:.2f})."
                    ),
                    structure_type=StructureType.IRON_CONDOR,
                    order_side=OrderSide.CREDIT,
                    profit_target_pct=0.50,
                    stop_loss_pct=2.0,
                    exit_dte=0,
                    max_profit_desc="Credit received",
                    max_loss_desc=f"Wing width (${wing_width_pts:.0f}) minus credit",
                    exit_notes=["0DTE: close by 3PM ET or at 50% profit",
                                "Close if price breaks ORB range on either side",
                                "Short strikes at ORB edges — range break = stop"],
                )
            else:
                legs, wing_width = build_iron_condor_legs(
                    price, atr, regime_id, exp_date, dte, atm_iv,
                )
                return TradeSpec(
                    ticker=ticker, legs=legs, underlying_price=price,
                    target_dte=dte, target_expiration=exp_date,
                    wing_width_points=wing_width,
                    spec_rationale=f"0DTE iron condor. R{regime_id}.",
                    structure_type=StructureType.IRON_CONDOR,
                    order_side=OrderSide.CREDIT,
                    profit_target_pct=0.50,
                    stop_loss_pct=2.0,
                    exit_dte=0,
                    max_profit_desc="Credit received",
                    max_loss_desc=f"Wing width (${wing_width:.0f}) minus credit",
                    exit_notes=["0DTE: close by 3PM ET or at 50% profit",
                                "Close if short strike tested on either side"],
                )

        elif zero_dte_strategy == ZeroDTEStrategy.CREDIT_SPREAD:
            cr_dir = direction if direction in ("bullish", "bearish") else "bullish"
            # ORB-aware: place short strike at ORB level
            if orb is not None:
                if cr_dir == "bullish" and orb_low is not None:
                    # Bull put spread: short strike at/near ORB low
                    short_mult = abs(price - orb_low) / atr if atr > 0 else 1.0
                    short_mult = max(0.3, min(short_mult, 2.0))
                else:
                    short_mult = 1.0
                    if orb_high is not None and cr_dir == "bearish":
                        short_mult = abs(orb_high - price) / atr if atr > 0 else 1.0
                        short_mult = max(0.3, min(short_mult, 2.0))
            else:
                short_mult = 1.0
            legs, wing_pts = build_credit_spread_legs(
                price, atr, cr_dir, exp_date, dte, atm_iv,
                short_multiplier=short_mult,
            )
            rationale = f"0DTE {cr_dir} credit spread. R{regime_id}."
            if orb is not None:
                level = orb_low if cr_dir == "bullish" else orb_high
                if level is not None:
                    rationale += f" Short strike near ORB {'low' if cr_dir == 'bullish' else 'high'} {level:.2f}."
            return TradeSpec(
                ticker=ticker, legs=legs, underlying_price=price,
                target_dte=dte, target_expiration=exp_date,
                wing_width_points=wing_pts,
                spec_rationale=rationale,
                structure_type=StructureType.CREDIT_SPREAD,
                order_side=OrderSide.CREDIT,
                profit_target_pct=0.50,
                stop_loss_pct=2.0,
                exit_dte=0,
                max_profit_desc="Credit received",
                max_loss_desc=f"Wing width (${wing_pts:.0f}) minus credit",
                exit_notes=["0DTE: close by 3PM ET or at 50% profit",
                            "Close if short strike tested"],
            )

        elif zero_dte_strategy == ZeroDTEStrategy.STRADDLE_STRANGLE:
            # ORB-aware: use ORB range width to set OTM offset
            if orb is not None and orb.range_size > 0:
                otm_mult = orb.range_size / atr if atr > 0 else 1.0
                otm_mult = max(0.5, min(otm_mult, 2.0))
            else:
                otm_mult = 1.0
            legs = build_straddle_legs(
                price, "sell", exp_date, dte, atm_iv,
                otm_offset_multiplier=otm_mult, atr=atr,
            )
            rationale = f"0DTE short strangle. R{regime_id}."
            if orb is not None:
                rationale += f" OTM offset based on ORB range ({orb.range_pct:.1f}%)."
            return TradeSpec(
                ticker=ticker, legs=legs, underlying_price=price,
                target_dte=dte, target_expiration=exp_date,
                spec_rationale=rationale,
                structure_type=StructureType.STRANGLE,
                order_side=OrderSide.CREDIT,
                profit_target_pct=0.50,
                stop_loss_pct=1.5,
                exit_dte=0,
                max_profit_desc="Credit received",
                max_loss_desc="UNDEFINED — loss beyond short strikes is unlimited without wings",
                exit_notes=["0DTE: close by 3PM ET or at 50% profit",
                            "Close immediately if short strike tested",
                            "UNDEFINED RISK: consider adding wings for defined risk"],
            )

        elif zero_dte_strategy == ZeroDTEStrategy.DIRECTIONAL_SPREAD:
            db_dir = direction if direction in ("bullish", "bearish") else "bullish"
            legs = build_debit_spread_legs(
                price, atr, db_dir, exp_date, dte, atm_iv,
            )
            rationale = f"0DTE {db_dir} debit spread. R{regime_id}."
            if orb is not None:
                # Add ORB target levels to rationale
                for level in orb.levels:
                    if "T1" in level.label and {"bullish": "long", "bearish": "short"}.get(db_dir, "") in level.label.lower():
                        rationale += f" Target T1: {level.price:.2f}."
                        break
            return TradeSpec(
                ticker=ticker, legs=legs, underlying_price=price,
                target_dte=dte, target_expiration=exp_date,
                spec_rationale=rationale,
                structure_type=StructureType.DEBIT_SPREAD,
                order_side=OrderSide.DEBIT,
                profit_target_pct=0.50,
                stop_loss_pct=0.50,
                exit_dte=0,
                max_profit_desc="Spread width minus debit paid",
                max_loss_desc="Net debit paid",
                exit_notes=["0DTE: close by 3PM ET",
                            "Target 50% of max profit or ORB extension levels",
                            "Close at 50% loss of debit paid"],
            )
    except Exception:
        return None

    return None


def _build_summary(
    ticker: str,
    verdict: Verdict,
    confidence: float,
    strategy: ZeroDTEStrategy,
    hard_stops: list[HardStop],
    regime: RegimeResult,
    orb: ORBData | None = None,
) -> str:
    if verdict == Verdict.NO_GO:
        reasons = "; ".join(s.name for s in hard_stops) if hard_stops else "low confidence"
        return f"{ticker} 0DTE: NO-GO ({reasons}). R{int(regime.regime)}."
    elif verdict == Verdict.CAUTION:
        orb_note = ""
        if orb is not None:
            orb_note = f" ORB: {orb.status.value} ({orb.range_low:.2f}–{orb.range_high:.2f})."
        return (
            f"{ticker} 0DTE: CAUTION ({confidence:.0%}). "
            f"R{int(regime.regime)}. Consider {strategy.value} with reduced size.{orb_note}"
        )
    else:
        orb_note = ""
        if orb is not None:
            orb_note = f" ORB: {orb.status.value} ({orb.range_low:.2f}–{orb.range_high:.2f})."
        return (
            f"{ticker} 0DTE: GO ({confidence:.0%}). "
            f"R{int(regime.regime)}. Recommended: {strategy.value}.{orb_note}"
        )
