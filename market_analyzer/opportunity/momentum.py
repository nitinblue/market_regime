"""Momentum opportunity assessment — go/no-go + strategy recommendation."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from market_analyzer.config import get_settings
from market_analyzer.models.opportunity import (
    HardStop,
    MomentumDirection,
    MomentumOpportunity,
    MomentumScore,
    MomentumStrategy,
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


def assess_momentum(
    ticker: str,
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    phase: PhaseResult,
    macro: MacroCalendar,
    fundamentals: FundamentalsSnapshot | None = None,
    as_of: date | None = None,
) -> MomentumOpportunity:
    """Assess momentum opportunity for a single instrument.

    Pure function — consumes pre-computed analysis, produces structured assessment.
    No data fetching.
    """
    cfg = get_settings().opportunity.momentum
    today = as_of or date.today()

    # --- Days to earnings ---
    days_to_earnings: int | None = None
    if fundamentals is not None:
        days_to_earnings = fundamentals.upcoming_events.days_to_earnings

    # --- Direction ---
    momentum_dir = _determine_direction(regime, technicals, phase)
    is_bullish = momentum_dir == MomentumDirection.BULLISH

    # --- Score context ---
    score = _build_score(technicals, regime, is_bullish)

    # --- Hard stops ---
    hard_stops = _check_hard_stops(
        regime, technicals, days_to_earnings, is_bullish, cfg,
    )

    # --- Scoring signals ---
    signals = _score_signals(regime, technicals, phase, is_bullish, cfg)

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

    # --- Strategy selection ---
    momentum_strategy, strategy_rec = _select_strategy(
        regime, technicals, phase, verdict, momentum_dir, cfg,
    )

    # --- Summary ---
    summary = _build_summary(
        ticker, verdict, confidence, momentum_strategy, hard_stops,
        regime, phase, momentum_dir,
    )

    return MomentumOpportunity(
        ticker=ticker,
        as_of_date=today,
        verdict=verdict,
        confidence=round(confidence, 2),
        hard_stops=hard_stops,
        signals=signals,
        strategy=strategy_rec,
        momentum_strategy=momentum_strategy,
        momentum_direction=momentum_dir,
        regime_id=int(regime.regime),
        regime_confidence=round(regime.confidence, 2),
        phase_id=int(phase.phase),
        phase_name=phase.phase_name,
        phase_confidence=round(phase.confidence, 2),
        score=score,
        days_to_earnings=days_to_earnings,
        summary=summary,
    )


# --- Private helpers ---


def _determine_direction(
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    phase: PhaseResult,
) -> MomentumDirection:
    """Determine momentum direction from regime, MAs, phase, MACD."""
    # 1. Regime trend_direction (primary)
    if regime.trend_direction is not None:
        if regime.trend_direction.value == "bearish":
            return MomentumDirection.BEARISH
        return MomentumDirection.BULLISH

    # 2. MA structure fallback
    ma = technicals.moving_averages
    if ma.sma_20 > ma.sma_50 and ma.sma_50 > ma.sma_200:
        return MomentumDirection.BULLISH
    if ma.sma_20 < ma.sma_50 and ma.sma_50 < ma.sma_200:
        return MomentumDirection.BEARISH

    # 3. Phase fallback
    pid = int(phase.phase)
    if pid == 2:
        return MomentumDirection.BULLISH
    if pid == 4:
        return MomentumDirection.BEARISH

    # 4. MACD fallback
    if technicals.macd.histogram > 0:
        return MomentumDirection.BULLISH
    return MomentumDirection.BEARISH


def _build_score(
    technicals: TechnicalSnapshot,
    regime: RegimeResult,
    is_bullish: bool,
) -> MomentumScore:
    """Build momentum score context from technicals."""
    macd = technicals.macd
    rsi = technicals.rsi.value
    ma = technicals.moving_averages
    stoch = technicals.stochastic
    phase_ind = technicals.phase

    # MACD histogram trend
    hist = macd.histogram
    if is_bullish:
        hist_trend = "expanding" if hist > 0 else "contracting"
    else:
        hist_trend = "expanding" if hist < 0 else "contracting"
    if abs(hist) < 0.01:
        hist_trend = "flat"

    # MACD crossover
    if macd.is_bullish_crossover:
        macd_cross = "bullish"
    elif macd.is_bearish_crossover:
        macd_cross = "bearish"
    else:
        macd_cross = "none"

    # RSI zone
    if rsi > 85:
        rsi_zone = "overbought"
    elif rsi >= 50 and rsi <= 70:
        rsi_zone = "healthy_bull"
    elif rsi >= 30 and rsi < 50:
        rsi_zone = "healthy_bear"
    elif rsi < 15:
        rsi_zone = "oversold"
    else:
        rsi_zone = "neutral"

    # Price vs MA alignment
    if ma.price_vs_sma_20_pct > 0 and ma.price_vs_sma_50_pct > 0 and ma.price_vs_sma_200_pct > 0:
        if ma.sma_20 > ma.sma_50 > ma.sma_200:
            ma_align = "strong_bull"
        else:
            ma_align = "bull"
    elif ma.price_vs_sma_20_pct < 0 and ma.price_vs_sma_50_pct < 0 and ma.price_vs_sma_200_pct < 0:
        if ma.sma_20 < ma.sma_50 < ma.sma_200:
            ma_align = "strong_bear"
        else:
            ma_align = "bear"
    else:
        ma_align = "neutral"

    # Golden/death cross
    if ma.sma_50 > ma.sma_200:
        gd_cross = "golden_cross"
    elif ma.sma_50 < ma.sma_200:
        gd_cross = "death_cross"
    else:
        gd_cross = None

    # Structural pattern
    if phase_ind.higher_highs and phase_ind.higher_lows:
        struct = "HH_HL"
    elif phase_ind.lower_highs and phase_ind.lower_lows:
        struct = "LH_LL"
    else:
        struct = "mixed"

    # Volume confirmation
    vol_conf = (
        (is_bullish and phase_ind.volume_trend == "rising")
        or (not is_bullish and phase_ind.volume_trend == "rising")
    )

    # Stochastic confirmation
    stoch_conf = (
        (is_bullish and stoch.k > stoch.d)
        or (not is_bullish and stoch.k < stoch.d)
    )

    # ATR trend (approximate from phase)
    # Use atr_pct context — if low-vol base, ATR falling; if trending, ATR rising/stable
    atr_pct = technicals.atr_pct
    if atr_pct > 2.0:
        atr_trend = "rising"
    elif atr_pct < 0.5:
        atr_trend = "falling"
    else:
        atr_trend = "stable"

    # Description
    parts = [f"MACD {hist_trend}", f"RSI {rsi_zone}", f"MA {ma_align}"]
    if struct != "mixed":
        parts.append(f"structure {struct}")
    desc = ", ".join(parts) + "."

    return MomentumScore(
        macd_histogram_trend=hist_trend,
        macd_crossover=macd_cross,
        rsi_zone=rsi_zone,
        price_vs_ma_alignment=ma_align,
        golden_death_cross=gd_cross,
        structural_pattern=struct,
        volume_confirmation=vol_conf,
        stochastic_confirmation=stoch_conf,
        atr_trend=atr_trend,
        description=desc,
    )


def _check_hard_stops(
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    days_to_earnings: int | None,
    is_bullish: bool,
    cfg,
) -> list[HardStop]:
    stops: list[HardStop] = []
    rid = int(regime.regime)

    # R1 high confidence (mean-reverting = no momentum)
    if rid == 1 and regime.confidence > cfg.r1_confidence_threshold:
        stops.append(HardStop(
            name="r1_high_confidence",
            description=(
                f"R1 (Low-Vol MR) at {regime.confidence:.0%} — "
                f"mean-reverting regime kills momentum."
            ),
        ))

    # Earnings imminent
    if days_to_earnings is not None and days_to_earnings <= cfg.earnings_blackout_days:
        stops.append(HardStop(
            name="earnings_imminent",
            description=f"Earnings in {days_to_earnings} day(s) — momentum may reverse on earnings.",
        ))

    # RSI extreme
    rsi = technicals.rsi.value
    if rsi > cfg.rsi_extreme_overbought:
        stops.append(HardStop(
            name="rsi_extreme",
            description=f"RSI {rsi:.0f} — blow-off top territory.",
        ))
    elif rsi < cfg.rsi_extreme_oversold:
        stops.append(HardStop(
            name="rsi_extreme",
            description=f"RSI {rsi:.0f} — capitulation territory.",
        ))

    # MACD crossover against trend
    macd = technicals.macd
    if is_bullish and macd.is_bearish_crossover:
        stops.append(HardStop(
            name="macd_crossover_against_trend",
            description="MACD bearish crossover while trend is bullish — momentum stalling.",
        ))
    elif not is_bullish and macd.is_bullish_crossover:
        stops.append(HardStop(
            name="macd_crossover_against_trend",
            description="MACD bullish crossover while trend is bearish — momentum stalling.",
        ))

    # Volume divergence on new highs
    phase_ind = technicals.phase
    if (
        is_bullish
        and phase_ind.higher_highs
        and phase_ind.volume_trend == "declining"
    ):
        stops.append(HardStop(
            name="volume_divergence_on_new_highs",
            description="Higher highs with declining volume — distribution divergence.",
        ))
    elif (
        not is_bullish
        and phase_ind.lower_lows
        and phase_ind.volume_trend == "declining"
    ):
        stops.append(HardStop(
            name="volume_divergence_on_new_highs",
            description="Lower lows with declining volume — exhaustion divergence.",
        ))

    return stops


def _score_signals(
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    phase: PhaseResult,
    is_bullish: bool,
    cfg,
) -> list[OpportunitySignal]:
    signals: list[OpportunitySignal] = []
    macd = technicals.macd
    rsi = technicals.rsi.value
    ma = technicals.moving_averages
    stoch = technicals.stochastic
    phase_ind = technicals.phase
    rid = int(regime.regime)
    pid = int(phase.phase)

    # 1. MACD histogram trend (0.15)
    if is_bullish:
        macd_fav = macd.histogram > 0
    else:
        macd_fav = macd.histogram < 0
    signals.append(OpportunitySignal(
        name="macd_histogram_trend",
        favorable=macd_fav,
        weight=0.15,
        description=(
            f"MACD histogram {macd.histogram:+.2f} — {'confirming' if macd_fav else 'diverging from'} trend"
        ),
    ))

    # 2. Price vs MA alignment (0.15)
    if is_bullish:
        ma_fav = ma.price_vs_sma_20_pct > 0 and ma.sma_20 > ma.sma_50
    else:
        ma_fav = ma.price_vs_sma_20_pct < 0 and ma.sma_20 < ma.sma_50
    signals.append(OpportunitySignal(
        name="price_vs_ma_alignment",
        favorable=ma_fav,
        weight=0.15,
        description=(
            f"Price {'above rising' if is_bullish and ma_fav else 'below falling' if not is_bullish and ma_fav else 'misaligned with'} SMA20/SMA50"
        ),
    ))

    # 3. RSI zone (0.12)
    if is_bullish:
        rsi_fav = cfg.rsi_healthy_bull_low <= rsi <= cfg.rsi_healthy_bull_high
    else:
        rsi_fav = cfg.rsi_healthy_bear_low <= rsi <= cfg.rsi_healthy_bear_high
    signals.append(OpportunitySignal(
        name="rsi_zone",
        favorable=rsi_fav,
        weight=0.12,
        description=(
            f"RSI {rsi:.0f} — healthy {'bull' if is_bullish else 'bear'} zone"
            if rsi_fav
            else f"RSI {rsi:.0f} — outside ideal zone"
        ),
    ))

    # 4. Structural trend (0.12)
    if is_bullish:
        struct_fav = phase_ind.higher_highs and phase_ind.higher_lows
    else:
        struct_fav = phase_ind.lower_highs and phase_ind.lower_lows
    signals.append(OpportunitySignal(
        name="structural_trend",
        favorable=struct_fav,
        weight=0.12,
        description=(
            "HH + HL structure — bullish momentum"
            if is_bullish and struct_fav
            else "LH + LL structure — bearish momentum"
            if not is_bullish and struct_fav
            else "Mixed price structure"
        ),
    ))

    # 5. Golden/death cross (0.08)
    if is_bullish:
        gd_fav = ma.sma_50 > ma.sma_200
    else:
        gd_fav = ma.sma_50 < ma.sma_200
    signals.append(OpportunitySignal(
        name="golden_death_cross",
        favorable=gd_fav,
        weight=0.08,
        description=(
            f"SMA50 {'>' if ma.sma_50 > ma.sma_200 else '<'} SMA200 — "
            f"{'golden cross' if ma.sma_50 > ma.sma_200 else 'death cross'}"
        ),
    ))

    # 6. Volume confirmation (0.08)
    vol_fav = phase_ind.volume_trend == "rising"
    signals.append(OpportunitySignal(
        name="volume_confirmation",
        favorable=vol_fav,
        weight=0.08,
        description=(
            "Rising volume confirms momentum"
            if vol_fav
            else f"Volume {phase_ind.volume_trend} — lacks confirmation"
        ),
    ))

    # 7. Trend regime alignment (0.08)
    regime_fav = rid in (3, 4) and (
        (is_bullish and regime.trend_direction is not None and regime.trend_direction.value == "bullish")
        or (not is_bullish and regime.trend_direction is not None and regime.trend_direction.value == "bearish")
    )
    signals.append(OpportunitySignal(
        name="trend_regime_alignment",
        favorable=regime_fav,
        weight=0.08,
        description=(
            f"R{rid} with matching trend direction"
            if regime_fav
            else f"R{rid} — regime does not confirm momentum direction"
        ),
    ))

    # 8. Phase alignment (0.08)
    if is_bullish:
        phase_fav = pid == 2
    else:
        phase_fav = pid == 4
    signals.append(OpportunitySignal(
        name="phase_alignment",
        favorable=phase_fav,
        weight=0.08,
        description=(
            f"P{pid} {phase.phase_name} — aligns with momentum"
            if phase_fav
            else f"P{pid} {phase.phase_name} — not ideal for momentum"
        ),
    ))

    # 9. Stochastic confirmation (0.07)
    if is_bullish:
        stoch_fav = stoch.k > stoch.d
    else:
        stoch_fav = stoch.k < stoch.d
    signals.append(OpportunitySignal(
        name="stochastic_confirmation",
        favorable=stoch_fav,
        weight=0.07,
        description=(
            f"Stochastic K({stoch.k:.0f}) {'>' if is_bullish else '<'} D({stoch.d:.0f}) — confirms"
            if stoch_fav
            else f"Stochastic K({stoch.k:.0f}) vs D({stoch.d:.0f}) — diverges"
        ),
    ))

    # 10. ATR stability (0.07)
    atr = technicals.atr_pct
    atr_fav = atr >= 0.5  # Sustained momentum needs some volatility
    signals.append(OpportunitySignal(
        name="atr_stability",
        favorable=atr_fav,
        weight=0.07,
        description=(
            f"ATR {atr:.2f}% — sustained momentum"
            if atr_fav
            else f"ATR {atr:.2f}% — too low for momentum"
        ),
    ))

    return signals


def _select_strategy(
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    phase: PhaseResult,
    verdict: Verdict,
    momentum_dir: MomentumDirection,
    cfg,
) -> tuple[MomentumStrategy, StrategyRecommendation]:
    """Select momentum strategy based on conditions."""
    rid = int(regime.regime)
    pid = int(phase.phase)
    is_bullish = momentum_dir == MomentumDirection.BULLISH
    direction = momentum_dir.value

    if verdict == Verdict.NO_GO:
        return MomentumStrategy.NO_TRADE, StrategyRecommendation(
            name="No Trade",
            direction="neutral",
            structure="Do not enter momentum trade.",
            rationale="Conditions unfavorable for momentum.",
            risk_notes=["Wait for better conditions."],
        )

    ma = technicals.moving_averages
    rsi = technicals.rsi.value
    macd = technicals.macd

    # Pullback entry: P2/P4 + price near SMA20 + healthy RSI
    near_sma20 = abs(ma.price_vs_sma_20_pct) <= cfg.pullback_to_ma_pct
    healthy_rsi = (
        (is_bullish and cfg.rsi_healthy_bull_low <= rsi <= cfg.rsi_healthy_bull_high)
        or (not is_bullish and cfg.rsi_healthy_bear_low <= rsi <= cfg.rsi_healthy_bear_high)
    )
    if pid in (2, 4) and near_sma20 and healthy_rsi:
        return MomentumStrategy.PULLBACK_ENTRY, StrategyRecommendation(
            name="Pullback Entry",
            direction=direction,
            structure=f"{'Call' if is_bullish else 'Put'} debit spread on pullback to SMA20.",
            rationale=f"P{pid} {phase.phase_name} with price near SMA20 and healthy RSI.",
            risk_notes=["Stop below SMA50.", "Target previous swing high/low."],
        )

    # Momentum acceleration: R3 + range_compression + MACD expanding
    phase_ind = technicals.phase
    macd_expanding = (is_bullish and macd.histogram > 0) or (not is_bullish and macd.histogram < 0)
    if (
        rid == 3
        and phase_ind.range_compression > cfg.pullback_to_ma_pct / 5  # ~0.3
        and macd_expanding
    ):
        return MomentumStrategy.MOMENTUM_ACCELERATION, StrategyRecommendation(
            name="Momentum Acceleration",
            direction=direction,
            structure=f"{'Call' if is_bullish else 'Put'} debit spread on consolidation breakout.",
            rationale=f"R3 trending with range compression + expanding MACD.",
            risk_notes=["Trend continuation assumed.", "Trail stops."],
        )

    # Momentum fade: RSI divergence or near-extreme + declining volume
    rsi_near_extreme = rsi > 75 or rsi < 25
    if rsi_near_extreme and phase_ind.volume_trend == "declining":
        fade_dir = "bearish" if rsi > 75 else "bullish"
        return MomentumStrategy.MOMENTUM_FADE, StrategyRecommendation(
            name="Momentum Fade",
            direction=fade_dir,
            structure=f"{'Put' if rsi > 75 else 'Call'} debit spread — counter-trend reversal.",
            rationale=f"RSI {rsi:.0f} with declining volume — exhaustion signal.",
            risk_notes=["Counter-trend trade, strict risk management.", "Small size."],
        )

    # Default: trend continuation
    return MomentumStrategy.TREND_CONTINUATION, StrategyRecommendation(
        name="Trend Continuation",
        direction=direction,
        structure=f"{'Call' if is_bullish else 'Put'} debit spread in trend direction.",
        rationale="Confirmed momentum — ride the trend.",
        risk_notes=["Trail stops.", "Monitor for divergence signals."],
    )


def _build_summary(
    ticker: str,
    verdict: Verdict,
    confidence: float,
    strategy: MomentumStrategy,
    hard_stops: list[HardStop],
    regime: RegimeResult,
    phase: PhaseResult,
    momentum_dir: MomentumDirection,
) -> str:
    if verdict == Verdict.NO_GO:
        reasons = "; ".join(s.name for s in hard_stops) if hard_stops else "low confidence"
        return (
            f"{ticker} Momentum: NO-GO ({reasons}). "
            f"R{int(regime.regime)}, {phase.phase_name}."
        )
    elif verdict == Verdict.CAUTION:
        return (
            f"{ticker} Momentum: CAUTION ({confidence:.0%}). "
            f"R{int(regime.regime)} {phase.phase_name}, "
            f"{momentum_dir.value}. "
            f"Consider {strategy.value} with reduced size."
        )
    else:
        return (
            f"{ticker} Momentum: GO ({confidence:.0%}). "
            f"R{int(regime.regime)} {phase.phase_name}, "
            f"{momentum_dir.value}. "
            f"Recommended: {strategy.value}."
        )
