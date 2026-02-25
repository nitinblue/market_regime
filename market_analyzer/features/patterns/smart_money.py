"""Smart Money Concepts — Order Blocks and Fair Value Gaps.

Order Blocks: Last candle before a strong impulse move (demand/supply zones).
Fair Value Gaps: 3-candle imbalances where wicks don't overlap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_analyzer.config import TechnicalsSettings
from market_analyzer.models.technicals import (
    FairValueGap,
    FVGType,
    OrderBlock,
    OrderBlockType,
    SignalDirection,
    SignalStrength,
    SmartMoneyData,
    TechnicalSignal,
)


def compute_order_blocks(
    ohlcv: pd.DataFrame,
    price: float,
    atr_series: pd.Series,
    settings: TechnicalsSettings,
) -> list[OrderBlock]:
    """Detect order blocks — the last candle before a strong impulse move.

    Bullish OB: last bearish candle before an impulsive up-move (demand zone).
    Bearish OB: last bullish candle before an impulsive down-move (supply zone).

    Reuses ATR from the main technicals computation.
    """
    lookback = settings.ob_lookback_days
    if len(ohlcv) < lookback:
        return []

    window = ohlcv.iloc[-lookback:]
    atr_window = atr_series.iloc[-lookback:]

    opens = window["Open"].values
    highs = window["High"].values
    lows = window["Low"].values
    closes = window["Close"].values
    volumes = window["Volume"].values
    dates = window.index
    atrs = atr_window.values

    vol_avg = np.nanmean(volumes)
    blocks: list[OrderBlock] = []

    # Walk each bar looking for impulse moves on the NEXT bar
    for i in range(len(window) - 2):
        atr_val = atrs[i + 1]
        if np.isnan(atr_val) or atr_val == 0:
            continue

        next_move = closes[i + 1] - closes[i]
        impulse_multiple = abs(next_move) / atr_val

        if impulse_multiple < settings.ob_impulse_atr_multiple:
            continue

        # Volume confirmation on impulse bar
        if vol_avg > 0 and volumes[i + 1] < vol_avg * settings.ob_min_volume_factor:
            continue

        is_candle_bearish = closes[i] < opens[i]
        is_candle_bullish = closes[i] > opens[i]

        ob_type: OrderBlockType | None = None
        if next_move > 0 and is_candle_bearish:
            # Bullish OB: bearish candle before up impulse
            ob_type = OrderBlockType.BULLISH
        elif next_move < 0 and is_candle_bullish:
            # Bearish OB: bullish candle before down impulse
            ob_type = OrderBlockType.BEARISH

        if ob_type is None:
            continue

        ob_high = float(highs[i])
        ob_low = float(lows[i])
        ob_mid = (ob_high + ob_low) / 2

        # Check if price has tested or broken the OB zone (scan forward)
        is_tested = False
        is_broken = False
        for j in range(i + 2, len(window)):
            if ob_type == OrderBlockType.BULLISH:
                if lows[j] <= ob_high:
                    is_tested = True
                if closes[j] < ob_low:
                    is_broken = True
                    break
            else:
                if highs[j] >= ob_low:
                    is_tested = True
                if closes[j] > ob_high:
                    is_broken = True
                    break

        dist_pct = (price - ob_mid) / ob_mid * 100 if ob_mid != 0 else 0.0
        dt = dates[i]
        ob_date = dt.date() if hasattr(dt, "date") else dt

        blocks.append(OrderBlock(
            type=ob_type,
            date=ob_date,
            high=ob_high,
            low=ob_low,
            volume=float(volumes[i]),
            impulse_strength=round(impulse_multiple, 2),
            is_tested=is_tested,
            is_broken=is_broken,
            distance_pct=round(dist_pct, 2),
        ))

    # Keep only active (not broken) blocks, most recent first, limited count
    active = [b for b in blocks if not b.is_broken]
    active.sort(key=lambda b: abs(b.distance_pct))
    return active[:settings.ob_max_blocks]


def compute_fair_value_gaps(
    ohlcv: pd.DataFrame,
    price: float,
    settings: TechnicalsSettings,
) -> list[FairValueGap]:
    """Detect fair value gaps — 3-candle imbalances where wicks don't overlap.

    Bullish FVG: candle[i-2].high < candle[i].low (gap up).
    Bearish FVG: candle[i-2].low > candle[i].high (gap down).
    """
    lookback = settings.fvg_lookback_days
    if len(ohlcv) < lookback:
        return []

    window = ohlcv.iloc[-lookback:]
    highs = window["High"].values
    lows = window["Low"].values
    closes = window["Close"].values
    dates = window.index

    gaps: list[FairValueGap] = []

    for i in range(2, len(window)):
        candle1_high = highs[i - 2]
        candle1_low = lows[i - 2]
        candle3_high = highs[i]
        candle3_low = lows[i]

        fvg_type: FVGType | None = None
        gap_high: float
        gap_low: float

        if candle1_high < candle3_low:
            # Bullish FVG: gap between candle1 high and candle3 low
            fvg_type = FVGType.BULLISH
            gap_high = float(candle3_low)
            gap_low = float(candle1_high)
        elif candle1_low > candle3_high:
            # Bearish FVG: gap between candle3 high and candle1 low
            fvg_type = FVGType.BEARISH
            gap_high = float(candle1_low)
            gap_low = float(candle3_high)
        else:
            continue

        gap_mid = (gap_high + gap_low) / 2
        gap_size_pct = (gap_high - gap_low) / gap_mid * 100 if gap_mid != 0 else 0.0

        if gap_size_pct < settings.fvg_min_gap_pct:
            continue

        # Check fill status: scan bars after the gap
        is_filled = False
        max_fill = 0.0
        gap_range = gap_high - gap_low
        for j in range(i + 1, len(window)):
            if fvg_type == FVGType.BULLISH:
                # Filled when price drops into the gap
                if lows[j] <= gap_high:
                    penetration = gap_high - max(lows[j], gap_low)
                    fill = penetration / gap_range * 100 if gap_range > 0 else 100.0
                    max_fill = max(max_fill, fill)
                    if lows[j] <= gap_low:
                        is_filled = True
                        max_fill = 100.0
                        break
            else:
                # Filled when price rises into the gap
                if highs[j] >= gap_low:
                    penetration = min(highs[j], gap_high) - gap_low
                    fill = penetration / gap_range * 100 if gap_range > 0 else 100.0
                    max_fill = max(max_fill, fill)
                    if highs[j] >= gap_high:
                        is_filled = True
                        max_fill = 100.0
                        break

        dist_pct = (price - gap_mid) / gap_mid * 100 if gap_mid != 0 else 0.0

        dt = dates[i - 1]  # Middle candle date = gap date
        fvg_date = dt.date() if hasattr(dt, "date") else dt

        gaps.append(FairValueGap(
            type=fvg_type,
            date=fvg_date,
            high=round(gap_high, 2),
            low=round(gap_low, 2),
            gap_size_pct=round(gap_size_pct, 2),
            is_filled=is_filled,
            fill_pct=round(min(max_fill, 100.0), 1),
            distance_pct=round(dist_pct, 2),
        ))

    # Keep unfilled first, then partially filled, sorted by proximity
    gaps.sort(key=lambda g: (g.is_filled, abs(g.distance_pct)))
    return gaps[:settings.fvg_max_gaps]


def compute_smart_money(
    ohlcv: pd.DataFrame,
    price: float,
    atr_series: pd.Series,
    settings: TechnicalsSettings,
) -> SmartMoneyData | None:
    """Compute Order Blocks and Fair Value Gaps (Smart Money Concepts).

    Returns None if insufficient data.
    """
    min_bars = max(settings.ob_lookback_days, settings.fvg_lookback_days)
    if len(ohlcv) < min_bars:
        return None

    order_blocks = compute_order_blocks(ohlcv, price, atr_series, settings)
    fair_value_gaps = compute_fair_value_gaps(ohlcv, price, settings)

    # Find nearest zones by type
    nearest_bull_ob = next(
        (b for b in order_blocks if b.type == OrderBlockType.BULLISH), None
    )
    nearest_bear_ob = next(
        (b for b in order_blocks if b.type == OrderBlockType.BEARISH), None
    )
    nearest_bull_fvg = next(
        (g for g in fair_value_gaps if g.type == FVGType.BULLISH and not g.is_filled), None
    )
    nearest_bear_fvg = next(
        (g for g in fair_value_gaps if g.type == FVGType.BEARISH and not g.is_filled), None
    )

    unfilled_count = sum(1 for g in fair_value_gaps if not g.is_filled)
    active_ob_count = len(order_blocks)  # Already filtered to non-broken

    # Confluence score (0–1)
    score = 0.0
    desc_parts: list[str] = []

    # OB proximity score: price near an untested OB is high-value
    if nearest_bull_ob and not nearest_bull_ob.is_tested and abs(nearest_bull_ob.distance_pct) < 3.0:
        score += 0.30
        desc_parts.append(
            f"Bullish OB at ${nearest_bull_ob.low:.2f}–${nearest_bull_ob.high:.2f} "
            f"({nearest_bull_ob.distance_pct:+.1f}%, untested)"
        )
    elif nearest_bull_ob and abs(nearest_bull_ob.distance_pct) < 5.0:
        score += 0.15
        desc_parts.append(
            f"Bullish OB at ${nearest_bull_ob.low:.2f}–${nearest_bull_ob.high:.2f} "
            f"({nearest_bull_ob.distance_pct:+.1f}%)"
        )

    if nearest_bear_ob and not nearest_bear_ob.is_tested and abs(nearest_bear_ob.distance_pct) < 3.0:
        score += 0.30
        desc_parts.append(
            f"Bearish OB at ${nearest_bear_ob.low:.2f}–${nearest_bear_ob.high:.2f} "
            f"({nearest_bear_ob.distance_pct:+.1f}%, untested)"
        )
    elif nearest_bear_ob and abs(nearest_bear_ob.distance_pct) < 5.0:
        score += 0.15
        desc_parts.append(
            f"Bearish OB at ${nearest_bear_ob.low:.2f}–${nearest_bear_ob.high:.2f} "
            f"({nearest_bear_ob.distance_pct:+.1f}%)"
        )

    # FVG proximity score
    if nearest_bull_fvg and abs(nearest_bull_fvg.distance_pct) < 5.0:
        score += 0.20
        desc_parts.append(
            f"Unfilled bullish FVG at ${nearest_bull_fvg.low:.2f}–${nearest_bull_fvg.high:.2f}"
        )
    if nearest_bear_fvg and abs(nearest_bear_fvg.distance_pct) < 5.0:
        score += 0.20
        desc_parts.append(
            f"Unfilled bearish FVG at ${nearest_bear_fvg.low:.2f}–${nearest_bear_fvg.high:.2f}"
        )

    score = min(score, 1.0)

    if not desc_parts:
        description = "No significant order blocks or fair value gaps near current price."
    else:
        description = " | ".join(desc_parts)

    return SmartMoneyData(
        order_blocks=order_blocks,
        fair_value_gaps=fair_value_gaps,
        nearest_bullish_ob=nearest_bull_ob,
        nearest_bearish_ob=nearest_bear_ob,
        nearest_bullish_fvg=nearest_bull_fvg,
        nearest_bearish_fvg=nearest_bear_fvg,
        unfilled_fvg_count=unfilled_count,
        active_ob_count=active_ob_count,
        score=round(score, 2),
        description=description,
    )


def generate_smart_money_signals(smc: SmartMoneyData | None) -> list[TechnicalSignal]:
    """Generate TechnicalSignal entries for order blocks and FVGs."""
    if smc is None:
        return []

    signals: list[TechnicalSignal] = []

    # OB signals: price near an active OB
    if smc.nearest_bullish_ob and abs(smc.nearest_bullish_ob.distance_pct) < 5.0:
        ob = smc.nearest_bullish_ob
        strength = SignalStrength.STRONG if not ob.is_tested else SignalStrength.MODERATE
        signals.append(TechnicalSignal(
            name="Bullish Order Block",
            direction=SignalDirection.BULLISH,
            strength=strength,
            description=(
                f"Demand zone ${ob.low:.2f}–${ob.high:.2f} "
                f"({ob.distance_pct:+.1f}% away, "
                f"impulse {ob.impulse_strength:.1f}x ATR"
                f"{', untested' if not ob.is_tested else ', tested'})"
            ),
        ))

    if smc.nearest_bearish_ob and abs(smc.nearest_bearish_ob.distance_pct) < 5.0:
        ob = smc.nearest_bearish_ob
        strength = SignalStrength.STRONG if not ob.is_tested else SignalStrength.MODERATE
        signals.append(TechnicalSignal(
            name="Bearish Order Block",
            direction=SignalDirection.BEARISH,
            strength=strength,
            description=(
                f"Supply zone ${ob.low:.2f}–${ob.high:.2f} "
                f"({ob.distance_pct:+.1f}% away, "
                f"impulse {ob.impulse_strength:.1f}x ATR"
                f"{', untested' if not ob.is_tested else ', tested'})"
            ),
        ))

    # FVG signals: unfilled gaps near price
    if smc.nearest_bullish_fvg and abs(smc.nearest_bullish_fvg.distance_pct) < 5.0:
        fvg = smc.nearest_bullish_fvg
        signals.append(TechnicalSignal(
            name="Bullish Fair Value Gap",
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.MODERATE,
            description=(
                f"Unfilled gap ${fvg.low:.2f}–${fvg.high:.2f} "
                f"({fvg.gap_size_pct:.1f}% wide, {fvg.fill_pct:.0f}% filled)"
            ),
        ))

    if smc.nearest_bearish_fvg and abs(smc.nearest_bearish_fvg.distance_pct) < 5.0:
        fvg = smc.nearest_bearish_fvg
        signals.append(TechnicalSignal(
            name="Bearish Fair Value Gap",
            direction=SignalDirection.BEARISH,
            strength=SignalStrength.MODERATE,
            description=(
                f"Unfilled gap ${fvg.low:.2f}–${fvg.high:.2f} "
                f"({fvg.gap_size_pct:.1f}% wide, {fvg.fill_pct:.0f}% filled)"
            ),
        ))

    return signals
