"""Opening Range Breakout (ORB) analysis from intraday data."""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from market_analyzer.config import get_settings
from market_analyzer.models.technicals import (
    ORBData,
    ORBLevel,
    ORBStatus,
    SignalDirection,
    SignalStrength,
    TechnicalSignal,
)


def compute_orb(
    intraday: pd.DataFrame,
    ticker: str,
    opening_minutes: int | None = None,
    daily_atr: float | None = None,
    extensions: list[float] | None = None,
) -> ORBData:
    """Compute Opening Range Breakout levels from intraday OHLCV data.

    Args:
        intraday: Intraday OHLCV DataFrame (1m/5m bars) with DatetimeIndex.
        ticker: Instrument ticker symbol.
        opening_minutes: Minutes for opening range (default: 30 from config).
        daily_atr: Optional daily ATR for context (range_vs_daily_atr_pct).
        extensions: Extension multipliers for target levels (default: [1.0, 1.5, 2.0]).

    Returns:
        ORBData with range, status, levels, and signals.

    Raises:
        ValueError: If required columns missing, DataFrame empty, or no time component.
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(intraday.columns)
    if missing:
        raise ValueError(f"Intraday DataFrame missing columns: {missing}")
    if intraday.empty:
        raise ValueError("Intraday DataFrame is empty")
    if not isinstance(intraday.index, pd.DatetimeIndex):
        raise ValueError("Intraday DataFrame must have a DatetimeIndex")

    settings = get_settings().orb
    if opening_minutes is None:
        opening_minutes = settings.opening_minutes
    if extensions is None:
        extensions = list(settings.extensions)

    market_open = time(settings.market_open_hour, settings.market_open_minute)
    market_close = time(16, 0)

    # Normalize index to ET-naive for filtering
    idx = intraday.index
    if idx.tz is not None:
        try:
            idx = idx.tz_convert("US/Eastern").tz_localize(None)
        except Exception:
            idx = idx.tz_localize(None)

    df = intraday.copy()
    df.index = idx

    # Extract session date(s) — use latest date
    dates = df.index.normalize().unique()
    session_date = dates[-1]
    session = df[df.index.normalize() == session_date]

    # Filter to market hours
    session = session[
        (session.index.time >= market_open) & (session.index.time <= market_close)
    ]
    if session.empty:
        raise ValueError(f"No bars found within market hours for {session_date.date()}")

    # Opening range: first N minutes from market open
    opening_end = time(
        market_open.hour + (market_open.minute + opening_minutes) // 60,
        (market_open.minute + opening_minutes) % 60,
    )
    opening_bars = session[session.index.time < opening_end]
    if opening_bars.empty:
        raise ValueError(
            f"No bars found in opening {opening_minutes}-minute window"
        )

    range_high = float(opening_bars["High"].max())
    range_low = float(opening_bars["Low"].min())
    range_size = range_high - range_low
    midpoint = (range_high + range_low) / 2
    range_pct = (range_size / midpoint * 100) if midpoint != 0 else 0.0

    current_price = float(session["Close"].iloc[-1])
    session_high = float(session["High"].max())
    session_low = float(session["Low"].min())

    # VWAP
    typical_price = (session["High"] + session["Low"] + session["Close"]) / 3
    cum_tp_vol = (typical_price * session["Volume"]).cumsum()
    cum_vol = session["Volume"].cumsum()
    session_vwap: float | None = None
    if float(cum_vol.iloc[-1]) > 0:
        session_vwap = float(cum_tp_vol.iloc[-1] / cum_vol.iloc[-1])

    # Volume ratio: avg volume per bar in opening vs full session
    bars_in_opening = len(opening_bars)
    bars_in_session = len(session)
    avg_vol_opening = float(opening_bars["Volume"].mean()) if bars_in_opening > 0 else 0.0
    avg_vol_session = float(session["Volume"].mean()) if bars_in_session > 0 else 0.0
    opening_volume_ratio = (
        avg_vol_opening / avg_vol_session if avg_vol_session > 0 else 0.0
    )

    # ATR context
    range_vs_daily_atr_pct: float | None = None
    if daily_atr is not None and daily_atr > 0:
        range_vs_daily_atr_pct = range_size / daily_atr * 100

    # Status determination: walk bars after opening range
    post_opening = session[session.index.time >= opening_end]
    status = ORBStatus.WITHIN
    breakout_bar_index: int | None = None
    retest_count = 0
    broke_high = False
    broke_low = False
    returned_inside = False

    for i, (_, bar) in enumerate(post_opening.iterrows()):
        bar_high = float(bar["High"])
        bar_low = float(bar["Low"])
        bar_close = float(bar["Close"])

        if not broke_high and not broke_low:
            # Looking for initial breakout
            if bar_high > range_high:
                broke_high = True
                breakout_bar_index = len(opening_bars) + i
                status = ORBStatus.BREAKOUT_LONG
            elif bar_low < range_low:
                broke_low = True
                breakout_bar_index = len(opening_bars) + i
                status = ORBStatus.BREAKOUT_SHORT
        else:
            # After breakout: check for failure or retest
            if broke_high:
                if bar_close < range_high and bar_close >= range_low:
                    returned_inside = True
                    status = ORBStatus.FAILED_LONG
                elif bar_low <= range_high and bar_close > range_high:
                    # Retested range edge then bounced
                    retest_count += 1
            elif broke_low:
                if bar_close > range_low and bar_close <= range_high:
                    returned_inside = True
                    status = ORBStatus.FAILED_SHORT
                elif bar_high >= range_low and bar_close < range_low:
                    retest_count += 1

    # Extension levels
    levels: list[ORBLevel] = []
    # Midpoint
    mid_price = midpoint
    mid_dist = (current_price - mid_price) / mid_price * 100 if mid_price != 0 else 0.0
    levels.append(ORBLevel(label="Midpoint", price=round(mid_price, 2), distance_pct=round(mid_dist, 2)))

    for ext in extensions:
        # Long target
        long_price = range_high + ext * range_size
        long_dist = (
            (current_price - long_price) / long_price * 100
            if long_price != 0
            else 0.0
        )
        levels.append(
            ORBLevel(
                label=f"T{extensions.index(ext) + 1} Long ({ext}x)",
                price=round(long_price, 2),
                distance_pct=round(long_dist, 2),
            )
        )
        # Short target
        short_price = range_low - ext * range_size
        short_dist = (
            (current_price - short_price) / short_price * 100
            if short_price != 0
            else 0.0
        )
        levels.append(
            ORBLevel(
                label=f"T{extensions.index(ext) + 1} Short ({ext}x)",
                price=round(short_price, 2),
                distance_pct=round(short_dist, 2),
            )
        )

    # Signals
    signals = _generate_orb_signals(
        status, range_high, range_low, range_pct, current_price,
        opening_volume_ratio, range_vs_daily_atr_pct,
    )

    # Description
    description = _build_description(
        ticker, status, range_high, range_low, range_pct, current_price,
        opening_volume_ratio, range_vs_daily_atr_pct, retest_count,
    )

    return ORBData(
        ticker=ticker,
        date=session_date.date() if hasattr(session_date, "date") else session_date,
        opening_minutes=opening_minutes,
        range_high=round(range_high, 2),
        range_low=round(range_low, 2),
        range_size=round(range_size, 2),
        range_pct=round(range_pct, 2),
        current_price=round(current_price, 2),
        status=status,
        levels=levels,
        session_high=round(session_high, 2),
        session_low=round(session_low, 2),
        session_vwap=round(session_vwap, 2) if session_vwap is not None else None,
        opening_volume_ratio=round(opening_volume_ratio, 2),
        range_vs_daily_atr_pct=(
            round(range_vs_daily_atr_pct, 1) if range_vs_daily_atr_pct is not None else None
        ),
        breakout_bar_index=breakout_bar_index,
        retest_count=retest_count,
        signals=signals,
        description=description,
    )


def _generate_orb_signals(
    status: ORBStatus,
    range_high: float,
    range_low: float,
    range_pct: float,
    current_price: float,
    volume_ratio: float,
    atr_pct: float | None,
) -> list[TechnicalSignal]:
    """Generate technical signals from ORB analysis."""
    signals: list[TechnicalSignal] = []

    if status == ORBStatus.BREAKOUT_LONG:
        strength = SignalStrength.STRONG if volume_ratio > 1.3 else SignalStrength.MODERATE
        signals.append(TechnicalSignal(
            name="ORB Breakout Long",
            direction=SignalDirection.BULLISH,
            strength=strength,
            description=(
                f"Price broke above opening range high ({range_high:.2f}). "
                f"Current: {current_price:.2f}"
            ),
        ))
    elif status == ORBStatus.BREAKOUT_SHORT:
        strength = SignalStrength.STRONG if volume_ratio > 1.3 else SignalStrength.MODERATE
        signals.append(TechnicalSignal(
            name="ORB Breakout Short",
            direction=SignalDirection.BEARISH,
            strength=strength,
            description=(
                f"Price broke below opening range low ({range_low:.2f}). "
                f"Current: {current_price:.2f}"
            ),
        ))
    elif status == ORBStatus.FAILED_LONG:
        signals.append(TechnicalSignal(
            name="ORB Failed Breakout Long",
            direction=SignalDirection.BEARISH,
            strength=SignalStrength.MODERATE,
            description=(
                f"Broke above {range_high:.2f} then returned inside range. "
                f"Bearish reversal signal."
            ),
        ))
    elif status == ORBStatus.FAILED_SHORT:
        signals.append(TechnicalSignal(
            name="ORB Failed Breakout Short",
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.MODERATE,
            description=(
                f"Broke below {range_low:.2f} then returned inside range. "
                f"Bullish reversal signal."
            ),
        ))

    # Narrow range signal (potential for big move)
    if range_pct < 0.5 and status == ORBStatus.WITHIN:
        signals.append(TechnicalSignal(
            name="ORB Narrow Range",
            direction=SignalDirection.NEUTRAL,
            strength=SignalStrength.WEAK,
            description=f"Opening range is only {range_pct:.2f}% — watch for directional breakout.",
        ))

    return signals


def _build_description(
    ticker: str,
    status: ORBStatus,
    range_high: float,
    range_low: float,
    range_pct: float,
    current_price: float,
    volume_ratio: float,
    atr_pct: float | None,
    retest_count: int,
) -> str:
    """Build human-readable ORB description."""
    parts = [f"{ticker} ORB: {range_low:.2f}–{range_high:.2f} ({range_pct:.2f}% range)."]

    status_text = {
        ORBStatus.WITHIN: "Price within range.",
        ORBStatus.BREAKOUT_LONG: "Breakout above range high.",
        ORBStatus.BREAKOUT_SHORT: "Breakout below range low.",
        ORBStatus.FAILED_LONG: "Failed breakout above — returned inside.",
        ORBStatus.FAILED_SHORT: "Failed breakout below — returned inside.",
    }
    parts.append(status_text[status])

    if volume_ratio > 1.3:
        parts.append(f"Opening volume {volume_ratio:.1f}x session average (strong).")
    elif volume_ratio < 0.7:
        parts.append(f"Opening volume {volume_ratio:.1f}x session average (light).")

    if atr_pct is not None:
        if atr_pct > 80:
            parts.append(f"Range is {atr_pct:.0f}% of daily ATR (wide).")
        elif atr_pct < 30:
            parts.append(f"Range is {atr_pct:.0f}% of daily ATR (tight — breakout potential).")

    if retest_count > 0:
        parts.append(f"{retest_count} retest(s) of range edge after breakout.")

    return " ".join(parts)
