"""Wyckoff-relevant price structure analysis from raw OHLCV.

Computes swing points, HH/HL/LH/LL patterns, range compression,
volume trend, and support/resistance — all independent of the HMM
feature pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_regime.config import PhaseSettings
from market_regime.models.phase import PriceStructure, SwingPoint


def detect_swing_highs(
    high: pd.Series,
    lookback: int,
    threshold_pct: float,
) -> list[SwingPoint]:
    """Detect swing highs using N-bar pivot method with noise filter."""
    swings: list[SwingPoint] = []
    n = len(high)
    for i in range(lookback, n - lookback):
        window = high.iloc[i - lookback : i + lookback + 1]
        if high.iloc[i] == window.max():
            # Noise filter: swing must be threshold_pct above avg of surrounding bars
            surrounding = pd.concat([
                high.iloc[i - lookback : i],
                high.iloc[i + 1 : i + lookback + 1],
            ])
            if high.iloc[i] >= surrounding.mean() * (1 + threshold_pct / 100):
                swings.append(SwingPoint(
                    date=high.index[i].date() if hasattr(high.index[i], "date") else high.index[i],
                    price=float(high.iloc[i]),
                    type="high",
                ))
    return swings


def detect_swing_lows(
    low: pd.Series,
    lookback: int,
    threshold_pct: float,
) -> list[SwingPoint]:
    """Detect swing lows using N-bar pivot method with noise filter."""
    swings: list[SwingPoint] = []
    n = len(low)
    for i in range(lookback, n - lookback):
        window = low.iloc[i - lookback : i + lookback + 1]
        if low.iloc[i] == window.min():
            surrounding = pd.concat([
                low.iloc[i - lookback : i],
                low.iloc[i + 1 : i + lookback + 1],
            ])
            if low.iloc[i] <= surrounding.mean() * (1 - threshold_pct / 100):
                swings.append(SwingPoint(
                    date=low.index[i].date() if hasattr(low.index[i], "date") else low.index[i],
                    price=float(low.iloc[i]),
                    type="low",
                ))
    return swings


def _check_ascending(prices: list[float], count: int = 3) -> bool:
    """Check if last `count` values are ascending."""
    if len(prices) < count:
        return False
    recent = prices[-count:]
    return all(recent[i] < recent[i + 1] for i in range(len(recent) - 1))


def _check_descending(prices: list[float], count: int = 3) -> bool:
    """Check if last `count` values are descending."""
    if len(prices) < count:
        return False
    recent = prices[-count:]
    return all(recent[i] > recent[i + 1] for i in range(len(recent) - 1))


def compute_range_compression(ohlcv: pd.DataFrame, window: int) -> float:
    """Compute range compression: +1 = compressing, -1 = expanding.

    Compares ATR of recent half-window to ATR of earlier half-window.
    """
    if len(ohlcv) < window:
        return 0.0

    tr = pd.concat([
        ohlcv["High"] - ohlcv["Low"],
        (ohlcv["High"] - ohlcv["Close"].shift(1)).abs(),
        (ohlcv["Low"] - ohlcv["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)

    recent = tr.iloc[-window:]
    half = window // 2
    early_atr = recent.iloc[:half].mean()
    late_atr = recent.iloc[half:].mean()

    if early_atr == 0:
        return 0.0

    # Positive = compressing (late ATR < early ATR)
    ratio = (early_atr - late_atr) / early_atr
    return float(np.clip(ratio, -1.0, 1.0))


def compute_volume_trend(volume: pd.Series, window: int, decline_threshold: float) -> str:
    """Classify volume trend over window: declining / stable / rising."""
    if len(volume) < window:
        return "stable"

    recent = volume.iloc[-window:]
    half = window // 2
    early_mean = recent.iloc[:half].mean()
    late_mean = recent.iloc[half:].mean()

    if early_mean == 0:
        return "stable"

    ratio = late_mean / early_mean
    if ratio < decline_threshold:
        return "declining"
    elif ratio > 1.0 / decline_threshold:
        return "rising"
    return "stable"


def compute_price_structure(ohlcv: pd.DataFrame, settings: PhaseSettings) -> PriceStructure:
    """Compute full Wyckoff price structure from OHLCV data."""
    swing_highs = detect_swing_highs(
        ohlcv["High"], settings.swing_lookback, settings.swing_threshold_pct
    )
    swing_lows = detect_swing_lows(
        ohlcv["Low"], settings.swing_lookback, settings.swing_threshold_pct
    )

    high_prices = [s.price for s in swing_highs]
    low_prices = [s.price for s in swing_lows]

    higher_highs = _check_ascending(high_prices)
    higher_lows = _check_ascending(low_prices)
    lower_highs = _check_descending(high_prices)
    lower_lows = _check_descending(low_prices)

    range_compression = compute_range_compression(ohlcv, settings.range_analysis_window)

    # Price vs SMA
    sma = ohlcv["Close"].rolling(settings.sma_period).mean()
    current_close = ohlcv["Close"].iloc[-1]
    current_sma = sma.iloc[-1]
    if pd.isna(current_sma) or current_sma == 0:
        price_vs_sma = 0.0
    else:
        price_vs_sma = float((current_close - current_sma) / current_sma * 100)

    volume_trend = compute_volume_trend(
        ohlcv["Volume"], settings.volume_trend_window, settings.volume_decline_threshold
    )

    # Support/resistance from most recent unbroken swing points
    support_level = low_prices[-1] if low_prices else None
    resistance_level = high_prices[-1] if high_prices else None

    return PriceStructure(
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        higher_highs=higher_highs,
        higher_lows=higher_lows,
        lower_highs=lower_highs,
        lower_lows=lower_lows,
        range_compression=range_compression,
        price_vs_sma=price_vs_sma,
        volume_trend=volume_trend,
        support_level=support_level,
        resistance_level=resistance_level,
    )
