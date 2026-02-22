"""Wyckoff phase detector — state machine driven by regime sequence + price structure.

Algorithm:
1. Regime sequence analysis (current regime, age, prior regime, prior trend direction)
2. Compute price structure (swings, volume, range)
3. Primary classification (regime-driven, price-structure fallback)
4. Confidence scoring (volume/range/swing confirmation)
5. Min duration gate (prevent phase flicker)
6. Transition probability estimation
7. Strategy comment generation
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from market_analyzer.config import PhaseSettings, get_settings
from market_analyzer.models.phase import (
    PhaseEvidence,
    PhaseID,
    PhaseResult,
    PhaseTransition,
)
from market_analyzer.models.regime import RegimeID, RegimeTimeSeries, TrendDirection
from market_analyzer.phases.price_structure import compute_price_structure


def _analyze_regime_sequence(
    regime_series: RegimeTimeSeries,
    min_run_days: int,
) -> dict:
    """Walk regime series backward to extract current/prior regime info."""
    entries = regime_series.entries
    if not entries:
        return {
            "current_regime": None,
            "current_regime_age": 0,
            "prior_regime": None,
            "prior_trend_direction": None,
        }

    current_regime = entries[-1].regime
    current_trend = entries[-1].trend_direction

    # Walk backward to find how long current regime has lasted
    age = 0
    for i in range(len(entries) - 1, -1, -1):
        if entries[i].regime == current_regime:
            age += 1
        else:
            break

    # Find prior regime (the one before current run)
    prior_regime = None
    prior_trend_direction = None
    prior_idx = len(entries) - age - 1
    if prior_idx >= 0:
        prior_regime = entries[prior_idx].regime
        prior_trend_direction = entries[prior_idx].trend_direction

        # Walk backward through prior regime run to get its trend direction
        # (use the majority direction in the run)
        prior_run_start = prior_idx
        for i in range(prior_idx, -1, -1):
            if entries[i].regime == prior_regime:
                prior_run_start = i
            else:
                break

        # Use the direction from the middle of the prior run
        if prior_run_start <= prior_idx:
            mid = (prior_run_start + prior_idx) // 2
            prior_trend_direction = entries[mid].trend_direction

    return {
        "current_regime": RegimeID(current_regime),
        "current_regime_age": age,
        "prior_regime": RegimeID(prior_regime) if prior_regime is not None else None,
        "prior_trend_direction": prior_trend_direction,
    }


def _classify_phase(
    current_regime: RegimeID,
    current_trend: TrendDirection | None,
    prior_regime: RegimeID | None,
    prior_trend: TrendDirection | None,
    price_structure,
) -> tuple[PhaseID, str, str]:
    """Primary classification. Returns (phase, regime_signal, price_signal)."""
    # Trending regimes -> markup or markdown based on direction
    if current_regime.is_trending:
        if current_trend == TrendDirection.BULLISH:
            return (
                PhaseID.MARKUP,
                f"R{current_regime} bullish -> markup",
                "trending bullish",
            )
        elif current_trend == TrendDirection.BEARISH:
            return (
                PhaseID.MARKDOWN,
                f"R{current_regime} bearish -> markdown",
                "trending bearish",
            )
        # Trend regime without clear direction — use price structure
        if price_structure.higher_highs and price_structure.higher_lows:
            return (
                PhaseID.MARKUP,
                f"R{current_regime} (no clear direction) + HH/HL -> markup",
                "higher highs and higher lows",
            )
        if price_structure.lower_highs and price_structure.lower_lows:
            return (
                PhaseID.MARKDOWN,
                f"R{current_regime} (no clear direction) + LH/LL -> markdown",
                "lower highs and lower lows",
            )
        # Default: use price vs SMA
        if price_structure.price_vs_sma >= 0:
            return (
                PhaseID.MARKUP,
                f"R{current_regime} + price above SMA -> markup",
                f"price {price_structure.price_vs_sma:+.1f}% vs SMA",
            )
        return (
            PhaseID.MARKDOWN,
            f"R{current_regime} + price below SMA -> markdown",
            f"price {price_structure.price_vs_sma:+.1f}% vs SMA",
        )

    # MR regimes -> accumulation or distribution based on prior trend
    if current_regime.is_mean_reverting:
        if prior_regime is not None and prior_regime.is_trending:
            if prior_trend == TrendDirection.BEARISH:
                return (
                    PhaseID.ACCUMULATION,
                    f"R{current_regime} following bearish R{prior_regime} -> accumulation",
                    "base-building after decline",
                )
            elif prior_trend == TrendDirection.BULLISH:
                return (
                    PhaseID.DISTRIBUTION,
                    f"R{current_regime} following bullish R{prior_regime} -> distribution",
                    "topping after advance",
                )

        # MR after MR (no prior trend) — use price structure
        if price_structure.higher_lows:
            return (
                PhaseID.ACCUMULATION,
                f"R{current_regime} (no prior trend) + higher lows -> accumulation",
                "higher lows, range compressing" if price_structure.range_compression > 0 else "higher lows",
            )
        if price_structure.lower_highs:
            return (
                PhaseID.DISTRIBUTION,
                f"R{current_regime} (no prior trend) + lower highs -> distribution",
                "lower highs, range expanding" if price_structure.range_compression < 0 else "lower highs",
            )

        # Final fallback: price vs SMA
        if price_structure.price_vs_sma < 0:
            return (
                PhaseID.ACCUMULATION,
                f"R{current_regime} + price below SMA -> accumulation",
                f"price {price_structure.price_vs_sma:+.1f}% vs SMA",
            )
        return (
            PhaseID.DISTRIBUTION,
            f"R{current_regime} + price above SMA -> distribution",
            f"price {price_structure.price_vs_sma:+.1f}% vs SMA",
        )

    # Should not reach here, but safe fallback
    return (PhaseID.ACCUMULATION, "unknown regime context", "no signal")


def _score_confidence(
    phase: PhaseID,
    price_structure,
    phase_age: int,
    settings: PhaseSettings,
) -> float:
    """Score confidence starting at 0.5, adjusting based on confirmations."""
    conf = 0.50

    # Volume confirmation
    if phase == PhaseID.ACCUMULATION:
        if price_structure.volume_trend == "declining":
            conf += 0.10
        elif price_structure.volume_trend == "rising":
            conf -= 0.10
    elif phase == PhaseID.DISTRIBUTION:
        if price_structure.volume_trend == "rising":
            conf += 0.10
        elif price_structure.volume_trend == "declining":
            conf -= 0.10
    elif phase == PhaseID.MARKUP:
        if price_structure.volume_trend == "rising":
            conf += 0.05
    elif phase == PhaseID.MARKDOWN:
        if price_structure.volume_trend == "rising":
            conf += 0.05

    # Range confirmation
    if phase == PhaseID.ACCUMULATION and price_structure.range_compression > 0:
        conf += 0.10
    elif phase == PhaseID.DISTRIBUTION and price_structure.range_compression < 0:
        conf += 0.05

    # Swing confirmation
    if phase == PhaseID.MARKUP:
        if price_structure.higher_highs and price_structure.higher_lows:
            conf += 0.10
        elif price_structure.higher_highs or price_structure.higher_lows:
            conf += 0.05
    elif phase == PhaseID.MARKDOWN:
        if price_structure.lower_highs and price_structure.lower_lows:
            conf += 0.10
        elif price_structure.lower_highs or price_structure.lower_lows:
            conf += 0.05

    # Phase age: more days -> slightly more confident
    if phase_age >= settings.min_phase_days:
        conf += 0.05

    return float(max(0.10, min(0.95, conf)))


def _build_evidence(
    phase: PhaseID,
    regime_signal: str,
    price_signal: str,
    price_structure,
    phase_age: int,
    settings: PhaseSettings,
) -> PhaseEvidence:
    """Build human-readable evidence with supporting/contradicting factors."""
    vol_signal = f"{price_structure.volume_trend} volume"

    supporting: list[str] = []
    contradictions: list[str] = []

    # Volume evidence
    if phase == PhaseID.ACCUMULATION:
        if price_structure.volume_trend == "declining":
            supporting.append("Volume drying up (classic accumulation)")
        elif price_structure.volume_trend == "rising":
            contradictions.append("Rising volume unusual for accumulation")
    elif phase == PhaseID.DISTRIBUTION:
        if price_structure.volume_trend == "rising":
            supporting.append("Rising volume (distribution selling)")
        elif price_structure.volume_trend == "declining":
            contradictions.append("Declining volume unusual for distribution")
    elif phase == PhaseID.MARKUP:
        if price_structure.volume_trend == "rising":
            supporting.append("Volume confirming markup")
    elif phase == PhaseID.MARKDOWN:
        if price_structure.volume_trend == "rising":
            supporting.append("Panic volume confirming markdown")

    # Range evidence
    if phase in (PhaseID.ACCUMULATION, PhaseID.DISTRIBUTION):
        if price_structure.range_compression > 0.2:
            supporting.append("Range narrowing -> spring loading")
        elif price_structure.range_compression < -0.2:
            if phase == PhaseID.ACCUMULATION:
                contradictions.append("Range expanding (not typical accumulation)")

    # Swing evidence
    if phase == PhaseID.MARKUP:
        if price_structure.higher_highs and price_structure.higher_lows:
            supporting.append("Higher highs and higher lows confirmed")
        elif not price_structure.higher_highs:
            contradictions.append("No higher highs yet")
    elif phase == PhaseID.MARKDOWN:
        if price_structure.lower_highs and price_structure.lower_lows:
            supporting.append("Lower highs and lower lows confirmed")
        elif not price_structure.lower_lows:
            contradictions.append("No lower lows yet")

    # Support/resistance
    if phase == PhaseID.ACCUMULATION and price_structure.support_level is not None:
        supporting.append(f"Price testing support at ${price_structure.support_level:.2f}")
    if phase == PhaseID.DISTRIBUTION and price_structure.resistance_level is not None:
        supporting.append(f"Price testing resistance at ${price_structure.resistance_level:.2f}")

    # Phase age
    if phase_age < settings.min_phase_days:
        contradictions.append(
            f"Short phase duration ({phase_age} days < min {settings.min_phase_days})"
        )

    return PhaseEvidence(
        regime_signal=regime_signal,
        price_signal=price_signal,
        volume_signal=vol_signal,
        supporting=supporting,
        contradictions=contradictions,
    )


def _estimate_transitions(
    phase: PhaseID,
    phase_age: int,
    price_structure,
    settings: PhaseSettings,
) -> list[PhaseTransition]:
    """Estimate transition probabilities based on phase + age + price signals."""
    transitions: list[PhaseTransition] = []

    if phase == PhaseID.ACCUMULATION:
        # Accumulation -> Markup probability increases with age and compression
        markup_prob = 0.25
        if phase_age > 30:
            markup_prob += 0.10
        if price_structure.range_compression > 0.3:
            markup_prob += 0.10
        if price_structure.higher_lows:
            markup_prob += 0.10
        markdown_prob = 0.10
        stay_prob = 1.0 - markup_prob - markdown_prob

        transitions.append(PhaseTransition(
            to_phase=PhaseID.MARKUP,
            probability=round(markup_prob, 2),
            triggers=["breakout above resistance on volume"],
        ))
        transitions.append(PhaseTransition(
            to_phase=PhaseID.ACCUMULATION,
            probability=round(stay_prob, 2),
            triggers=["range continuation"],
        ))
        transitions.append(PhaseTransition(
            to_phase=PhaseID.MARKDOWN,
            probability=round(markdown_prob, 2),
            triggers=["break below support, failed accumulation"],
        ))

    elif phase == PhaseID.MARKUP:
        distrib_prob = 0.15
        if phase_age > 60:
            distrib_prob += 0.10
        if not price_structure.higher_highs:
            distrib_prob += 0.10
        stay_prob = 1.0 - distrib_prob - 0.05

        transitions.append(PhaseTransition(
            to_phase=PhaseID.DISTRIBUTION,
            probability=round(distrib_prob, 2),
            triggers=["failed new high, volume surge on decline"],
        ))
        transitions.append(PhaseTransition(
            to_phase=PhaseID.MARKUP,
            probability=round(stay_prob, 2),
            triggers=["trend continuation, new highs"],
        ))
        transitions.append(PhaseTransition(
            to_phase=PhaseID.MARKDOWN,
            probability=0.05,
            triggers=["sharp reversal (climactic)"],
        ))

    elif phase == PhaseID.DISTRIBUTION:
        markdown_prob = 0.25
        if phase_age > 30:
            markdown_prob += 0.10
        if price_structure.lower_highs:
            markdown_prob += 0.10
        accumulation_prob = 0.10
        stay_prob = 1.0 - markdown_prob - accumulation_prob

        transitions.append(PhaseTransition(
            to_phase=PhaseID.MARKDOWN,
            probability=round(markdown_prob, 2),
            triggers=["break below support on volume"],
        ))
        transitions.append(PhaseTransition(
            to_phase=PhaseID.DISTRIBUTION,
            probability=round(stay_prob, 2),
            triggers=["range continuation, choppy action"],
        ))
        transitions.append(PhaseTransition(
            to_phase=PhaseID.ACCUMULATION,
            probability=round(accumulation_prob, 2),
            triggers=["upside breakout (failed distribution)"],
        ))

    elif phase == PhaseID.MARKDOWN:
        accum_prob = 0.15
        if phase_age > 60:
            accum_prob += 0.10
        if not price_structure.lower_lows:
            accum_prob += 0.10
        stay_prob = 1.0 - accum_prob - 0.05

        transitions.append(PhaseTransition(
            to_phase=PhaseID.ACCUMULATION,
            probability=round(accum_prob, 2),
            triggers=["volume dries up, range compression near lows"],
        ))
        transitions.append(PhaseTransition(
            to_phase=PhaseID.MARKDOWN,
            probability=round(stay_prob, 2),
            triggers=["trend continuation, new lows"],
        ))
        transitions.append(PhaseTransition(
            to_phase=PhaseID.MARKUP,
            probability=0.05,
            triggers=["V-reversal (climactic selling exhaustion)"],
        ))

    return transitions


def _cycle_completion(phase: PhaseID, phase_age: int) -> float:
    """Rough estimate of cycle completion (0.0-1.0).

    Full Wyckoff cycle: accumulation -> markup -> distribution -> markdown.
    """
    # Map phase to base cycle position
    base = {
        PhaseID.ACCUMULATION: 0.0,
        PhaseID.MARKUP: 0.25,
        PhaseID.DISTRIBUTION: 0.50,
        PhaseID.MARKDOWN: 0.75,
    }
    # Add fraction within phase based on age (assume ~60-day avg phase)
    phase_fraction = min(phase_age / 60.0, 1.0) * 0.25
    return min(base[phase] + phase_fraction, 1.0)


class PhaseDetector:
    """Wyckoff phase detection using regime sequence + price structure."""

    def __init__(self, settings: PhaseSettings | None = None) -> None:
        self.settings = settings or get_settings().phases

    def detect(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        regime_series: RegimeTimeSeries,
    ) -> PhaseResult:
        """Detect current Wyckoff phase for a ticker.

        Args:
            ticker: Instrument ticker.
            ohlcv: OHLCV DataFrame (needs enough history for price structure).
            regime_series: Full regime time series from HMM inference.

        Returns:
            PhaseResult with classification, evidence, and transitions.
        """
        # Step 1: Regime sequence analysis
        seq = _analyze_regime_sequence(regime_series, self.settings.min_regime_run_days)

        current_regime = seq["current_regime"]
        if current_regime is None:
            # No regime data — can't detect phase
            price_structure = compute_price_structure(ohlcv, self.settings)
            return PhaseResult(
                ticker=ticker,
                phase=PhaseID.ACCUMULATION,
                phase_name=self.settings.names[1],
                confidence=0.10,
                phase_age_days=0,
                prior_phase=None,
                cycle_completion=0.0,
                price_structure=price_structure,
                evidence=PhaseEvidence(
                    regime_signal="no regime data",
                    price_signal="unknown",
                    volume_signal="unknown",
                    supporting=[],
                    contradictions=["No regime data available"],
                ),
                transitions=[],
                strategy_comment=self.settings.strategies.get(1, ""),
                as_of_date=date.today(),
            )

        # Step 2: Compute price structure
        price_structure = compute_price_structure(ohlcv, self.settings)

        # Step 3: Primary classification
        # Get current trend direction from latest entry
        current_trend = regime_series.entries[-1].trend_direction if regime_series.entries else None
        phase, regime_signal, price_signal = _classify_phase(
            current_regime,
            current_trend,
            seq["prior_regime"],
            seq["prior_trend_direction"],
            price_structure,
        )

        phase_age = seq["current_regime_age"]

        # Step 4: Confidence scoring
        confidence = _score_confidence(phase, price_structure, phase_age, self.settings)

        # Step 5: Min duration gate — not applied here since we don't track
        # prior phase state across calls. The phase_age_days field and
        # contradictions list signal short duration to callers.

        # Step 6: Transition probabilities
        transitions = _estimate_transitions(phase, phase_age, price_structure, self.settings)

        # Step 7: Evidence
        evidence = _build_evidence(
            phase, regime_signal, price_signal,
            price_structure, phase_age, self.settings,
        )

        # Strategy comment
        strategy = self.settings.strategies.get(int(phase), "")

        # Determine prior phase from prior regime context
        prior_phase: PhaseID | None = None
        if seq["prior_regime"] is not None:
            prior_trend = seq["prior_trend_direction"]
            if seq["prior_regime"].is_trending:
                if prior_trend == TrendDirection.BULLISH:
                    prior_phase = PhaseID.MARKUP
                else:
                    prior_phase = PhaseID.MARKDOWN
            else:
                # Prior was MR — could be accum or distrib; use opposite of current
                if phase in (PhaseID.MARKUP, PhaseID.ACCUMULATION):
                    prior_phase = PhaseID.MARKDOWN
                else:
                    prior_phase = PhaseID.MARKUP

        cycle = _cycle_completion(phase, phase_age)

        as_of = (
            regime_series.entries[-1].date
            if regime_series.entries
            else date.today()
        )

        return PhaseResult(
            ticker=ticker,
            phase=phase,
            phase_name=self.settings.names.get(int(phase), f"P{phase}"),
            confidence=confidence,
            phase_age_days=phase_age,
            prior_phase=prior_phase,
            cycle_completion=cycle,
            price_structure=price_structure,
            evidence=evidence,
            transitions=transitions,
            strategy_comment=strategy,
            as_of_date=as_of,
        )
