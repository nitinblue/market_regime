"""Volatility Contraction Pattern (Minervini VCP) detection.

Scans swing highs/lows for successively tightening contractions,
declining volume, and proximity to a pivot breakout level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_analyzer.config import TechnicalsSettings, get_settings
from market_analyzer.models.technicals import (
    SignalDirection,
    SignalStrength,
    TechnicalSignal,
    VCPData,
    VCPStage,
)


def compute_vcp(
    ohlcv: pd.DataFrame,
    price: float,
    sma_50_val: float,
    sma_200_val: float,
    volume: pd.Series,
    settings: TechnicalsSettings,
) -> VCPData | None:
    """Detect Volatility Contraction Pattern (Minervini VCP).

    Returns None if insufficient data for VCP analysis.
    """
    from market_analyzer.phases.price_structure import (
        compute_range_compression,
        compute_volume_trend,
        detect_swing_highs,
        detect_swing_lows,
    )

    lookback = settings.vcp_lookback_days
    if len(ohlcv) < lookback:
        return None

    window = ohlcv.iloc[-lookback:]
    phase_settings = get_settings().phases

    swing_highs = detect_swing_highs(
        window["High"], phase_settings.swing_lookback, phase_settings.swing_threshold_pct,
    )
    swing_lows = detect_swing_lows(
        window["Low"], phase_settings.swing_lookback, phase_settings.swing_threshold_pct,
    )

    # Shared computations used by both the no-pattern and full-pattern paths
    above_50 = price > sma_50_val if not np.isnan(sma_50_val) else False
    above_200 = price > sma_200_val if not np.isnan(sma_200_val) else False
    range_compression = compute_range_compression(
        window, phase_settings.range_analysis_window,
    )
    vol_trend = compute_volume_trend(
        volume.iloc[-lookback:],
        phase_settings.volume_trend_window,
        phase_settings.volume_decline_threshold,
    )

    def _no_pattern() -> VCPData:
        """Return a NONE-stage VCPData when no tightening sequence is found."""
        return VCPData(
            stage=VCPStage.NONE,
            contraction_count=0,
            contraction_pcts=[],
            current_range_pct=0.0,
            range_compression=range_compression,
            volume_trend=vol_trend,
            pivot_price=None,
            pivot_distance_pct=None,
            days_in_base=0,
            above_sma_50=above_50,
            above_sma_200=above_200,
            score=0.0,
            description="No VCP pattern detected.",
        )

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return _no_pattern()

    # Build contraction pairs: match each swing high with the nearest
    # subsequent swing low to form (high, low) contraction segments.
    contractions: list[tuple[float, float]] = []  # (high_price, low_price)
    contraction_dates: list = []  # date of swing high (start of contraction)
    low_idx = 0
    for sh in swing_highs:
        # Find the first swing low that comes after this swing high
        while low_idx < len(swing_lows) and swing_lows[low_idx].date <= sh.date:
            low_idx += 1
        if low_idx < len(swing_lows):
            sl = swing_lows[low_idx]
            contractions.append((sh.price, sl.price))
            contraction_dates.append(sh.date)

    if len(contractions) < 2:
        return _no_pattern()

    # Compute range % for each contraction
    contraction_pcts: list[float] = []
    for h, l in contractions:
        mid = (h + l) / 2
        if mid == 0:
            contraction_pcts.append(0.0)
        else:
            contraction_pcts.append((h - l) / mid * 100)

    # Find the longest tightening sequence ending at the most recent contraction.
    # Walk backward: each prior contraction must be wider.
    tightening_end = len(contraction_pcts) - 1
    tightening_start = tightening_end
    ratio = settings.vcp_tightening_ratio
    for i in range(tightening_end - 1, -1, -1):
        if contraction_pcts[i] > contraction_pcts[i + 1] * (1.0 / ratio):
            # Prior contraction is meaningfully wider — extend the sequence
            tightening_start = i
        else:
            break

    t_count = tightening_end - tightening_start + 1
    tightening_pcts = contraction_pcts[tightening_start:tightening_end + 1]
    current_range_pct = tightening_pcts[-1] if tightening_pcts else 0.0
    first_range_pct = tightening_pcts[0] if tightening_pcts else 0.0

    # Reject if first contraction is too small (no real base)
    if first_range_pct < settings.vcp_min_contraction_pct:
        t_count = 0
        tightening_pcts = []

    # Pivot = highest swing high in the tightening sequence
    if t_count >= settings.vcp_min_contractions:
        pivot_candidates = contractions[tightening_start:tightening_end + 1]
        pivot_price = max(h for h, _ in pivot_candidates)
    else:
        pivot_price = None

    # Pivot distance
    pivot_distance_pct: float | None = None
    if pivot_price is not None and pivot_price != 0:
        pivot_distance_pct = (pivot_price - price) / pivot_price * 100

    # Days in base
    if t_count >= settings.vcp_min_contractions and contraction_dates:
        first_date = contraction_dates[tightening_start]
        last_date = window.index[-1].date() if hasattr(window.index[-1], "date") else window.index[-1]
        days_in_base = (last_date - first_date).days
    else:
        days_in_base = 0

    # Volume surge check for breakout
    vol_avg_20 = float(volume.iloc[-20:].mean()) if len(volume) >= 20 else 0.0
    latest_vol = float(volume.iloc[-1]) if len(volume) > 0 else 0.0
    vol_surge = (vol_avg_20 > 0) and (latest_vol > 1.5 * vol_avg_20)

    # Classify stage
    if t_count < settings.vcp_min_contractions:
        stage = VCPStage.NONE
    elif (
        pivot_price is not None
        and price > pivot_price
        and vol_surge
    ):
        stage = VCPStage.BREAKOUT
    elif (
        current_range_pct <= settings.vcp_ready_range_pct
        and pivot_distance_pct is not None
        and pivot_distance_pct <= settings.vcp_pivot_proximity_pct
        and vol_trend == "declining"
    ):
        stage = VCPStage.READY
    elif t_count > settings.vcp_min_contractions or range_compression > 0.3:
        stage = VCPStage.MATURING
    else:
        stage = VCPStage.FORMING

    # Composite score (0–1)
    if stage == VCPStage.NONE:
        score = 0.0
    else:
        contraction_score = min(t_count / 4.0, 1.0) * 0.30
        tightness_score = (
            (1.0 - current_range_pct / first_range_pct) * 0.25
            if first_range_pct > 0
            else 0.0
        )
        tightness_score = max(tightness_score, 0.0)
        volume_score = (
            1.0 if vol_trend == "declining"
            else 0.5 if vol_trend == "stable"
            else 0.0
        ) * 0.20
        ma_score = (float(above_50) + float(above_200)) / 2.0 * 0.15
        proximity_score = (
            max(0.0, 1.0 - abs(pivot_distance_pct) / 10.0) * 0.10
            if pivot_distance_pct is not None
            else 0.0
        )
        score = contraction_score + tightness_score + volume_score + ma_score + proximity_score

    # Description
    if stage == VCPStage.NONE:
        description = "No VCP pattern detected."
    elif stage == VCPStage.BREAKOUT:
        description = (
            f"VCP breakout: price cleared pivot {pivot_price:.2f} on volume surge. "
            f"{t_count} contractions ({', '.join(f'{p:.1f}%' for p in tightening_pcts)})."
        )
    elif stage == VCPStage.READY:
        description = (
            f"VCP ready: {t_count} contractions tightening to {current_range_pct:.1f}% range, "
            f"declining volume, pivot at {pivot_price:.2f} ({abs(pivot_distance_pct or 0):.1f}% away)."
        )
    elif stage == VCPStage.MATURING:
        description = (
            f"VCP maturing: {t_count} contractions "
            f"({', '.join(f'{p:.1f}%' for p in tightening_pcts)}). "
            f"Volume {vol_trend}. Pivot at {pivot_price:.2f}."
            if pivot_price
            else f"VCP maturing: {t_count} contractions. Volume {vol_trend}."
        )
    else:
        description = (
            f"VCP forming: {t_count} contractions detected. "
            f"Range: {current_range_pct:.1f}%. Volume {vol_trend}."
        )

    return VCPData(
        stage=stage,
        contraction_count=t_count,
        contraction_pcts=tightening_pcts,
        current_range_pct=current_range_pct,
        range_compression=range_compression,
        volume_trend=vol_trend,
        pivot_price=pivot_price,
        pivot_distance_pct=pivot_distance_pct,
        days_in_base=days_in_base,
        above_sma_50=above_50,
        above_sma_200=above_200,
        score=score,
        description=description,
    )


def generate_vcp_signals(vcp: VCPData | None) -> list[TechnicalSignal]:
    """Generate TechnicalSignal entries for VCP pattern."""
    if vcp is None or vcp.stage == VCPStage.NONE:
        return []

    signals: list[TechnicalSignal] = []

    if vcp.stage == VCPStage.BREAKOUT:
        signals.append(TechnicalSignal(
            name="VCP Breakout",
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.STRONG,
            description=vcp.description,
        ))
    elif vcp.stage == VCPStage.READY:
        signals.append(TechnicalSignal(
            name="VCP Ready",
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.STRONG,
            description=vcp.description,
        ))
    elif vcp.stage == VCPStage.MATURING:
        signals.append(TechnicalSignal(
            name="VCP Maturing",
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.MODERATE,
            description=vcp.description,
        ))
    elif vcp.stage == VCPStage.FORMING:
        signals.append(TechnicalSignal(
            name="VCP Forming",
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.WEAK,
            description=vcp.description,
        ))

    return signals
