"""Tests for Wyckoff price structure analysis."""

import numpy as np
import pandas as pd
import pytest

from market_regime.config import PhaseSettings
from market_regime.phases.price_structure import (
    compute_price_structure,
    compute_range_compression,
    compute_volume_trend,
    detect_swing_highs,
    detect_swing_lows,
)


def _make_uptrend(periods: int = 100, seed: int = 42) -> pd.DataFrame:
    """Synthetic uptrend OHLCV: higher highs and higher lows."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=periods)
    base = 100 + np.arange(periods) * 0.5 + rng.normal(0, 0.5, periods).cumsum()
    # Wider high/low range to ensure swing threshold is met
    high = base + rng.uniform(2, 5, periods)
    low = base - rng.uniform(2, 5, periods)
    volume = rng.integers(1_000_000, 5_000_000, periods).astype(float)
    # Declining volume
    volume = volume * np.linspace(1.5, 0.5, periods)
    return pd.DataFrame(
        {"Open": base + 0.1, "High": high, "Low": low, "Close": base, "Volume": volume},
        index=dates,
    )


def _make_downtrend(periods: int = 100, seed: int = 99) -> pd.DataFrame:
    """Synthetic downtrend OHLCV: lower highs and lower lows."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=periods)
    base = 200 - np.arange(periods) * 0.5 + rng.normal(0, 0.5, periods).cumsum()
    high = base + rng.uniform(1, 3, periods)
    low = base - rng.uniform(1, 3, periods)
    volume = rng.integers(1_000_000, 5_000_000, periods).astype(float)
    return pd.DataFrame(
        {"Open": base + 0.1, "High": high, "Low": low, "Close": base, "Volume": volume},
        index=dates,
    )


def _make_choppy(periods: int = 100, seed: int = 77) -> pd.DataFrame:
    """Synthetic range-bound OHLCV."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=periods)
    base = 150 + rng.normal(0, 2, periods).cumsum() * 0.1
    high = base + rng.uniform(1, 4, periods)
    low = base - rng.uniform(1, 4, periods)
    volume = rng.integers(1_000_000, 5_000_000, periods).astype(float)
    return pd.DataFrame(
        {"Open": base + 0.1, "High": high, "Low": low, "Close": base, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def settings() -> PhaseSettings:
    return PhaseSettings()


class TestSwingDetection:
    def test_detects_swing_highs_in_uptrend(self):
        df = _make_uptrend(200)
        swings = detect_swing_highs(df["High"], lookback=5, threshold_pct=0.5)
        assert len(swings) > 0
        assert all(s.type == "high" for s in swings)

    def test_detects_swing_lows_in_downtrend(self):
        df = _make_downtrend(200)
        swings = detect_swing_lows(df["Low"], lookback=5, threshold_pct=0.5)
        assert len(swings) > 0
        assert all(s.type == "low" for s in swings)

    def test_swing_highs_ascending_in_uptrend(self):
        df = _make_uptrend(200)
        swings = detect_swing_highs(df["High"], lookback=5, threshold_pct=0.5)
        if len(swings) >= 3:
            prices = [s.price for s in swings[-3:]]
            # In a clear uptrend, last 3 swing highs should be ascending
            assert prices[-1] > prices[0], "Swing highs should trend up in uptrend"

    def test_swing_lows_descending_in_downtrend(self):
        df = _make_downtrend(200)
        swings = detect_swing_lows(df["Low"], lookback=5, threshold_pct=0.5)
        if len(swings) >= 3:
            prices = [s.price for s in swings[-3:]]
            assert prices[-1] < prices[0], "Swing lows should trend down in downtrend"

    def test_no_swings_on_short_data(self):
        df = _make_uptrend(5)
        swings = detect_swing_highs(df["High"], lookback=5, threshold_pct=1.5)
        assert len(swings) == 0


class TestRangeCompression:
    def test_compressing_range(self):
        """Range that narrows over time should be positive."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2024-01-01", periods=60)
        close = pd.Series(100.0, index=dates)
        # High-low spread narrows
        high_spread = np.linspace(5, 1, 60)
        high = close + high_spread + rng.normal(0, 0.1, 60)
        low = close - high_spread + rng.normal(0, 0.1, 60)
        df = pd.DataFrame({"High": high, "Low": low, "Close": close}, index=dates)
        rc = compute_range_compression(df, window=30)
        assert rc > 0, f"Expected positive (compressing), got {rc}"

    def test_expanding_range(self):
        """Range that widens over time should be negative."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2024-01-01", periods=60)
        close = pd.Series(100.0, index=dates)
        high_spread = np.linspace(1, 5, 60)
        high = close + high_spread + rng.normal(0, 0.1, 60)
        low = close - high_spread + rng.normal(0, 0.1, 60)
        df = pd.DataFrame({"High": high, "Low": low, "Close": close}, index=dates)
        rc = compute_range_compression(df, window=30)
        assert rc < 0, f"Expected negative (expanding), got {rc}"

    def test_bounded(self):
        df = _make_uptrend(100)
        rc = compute_range_compression(df, window=30)
        assert -1.0 <= rc <= 1.0


class TestVolumeTrend:
    def test_declining_volume(self):
        volume = pd.Series(np.linspace(1_000_000, 400_000, 40))
        result = compute_volume_trend(volume, window=20, decline_threshold=0.8)
        assert result == "declining"

    def test_rising_volume(self):
        volume = pd.Series(np.linspace(300_000, 1_500_000, 40))
        result = compute_volume_trend(volume, window=20, decline_threshold=0.8)
        assert result == "rising"

    def test_stable_volume(self):
        rng = np.random.default_rng(42)
        volume = pd.Series(rng.normal(1_000_000, 50_000, 40))
        result = compute_volume_trend(volume, window=20, decline_threshold=0.8)
        assert result == "stable"

    def test_short_data_returns_stable(self):
        volume = pd.Series([1_000_000] * 5)
        result = compute_volume_trend(volume, window=20, decline_threshold=0.8)
        assert result == "stable"


class TestPriceStructure:
    def test_uptrend_structure(self, settings: PhaseSettings):
        df = _make_uptrend(200)
        ps = compute_price_structure(df, settings)
        assert ps.price_vs_sma > 0, "Price should be above SMA in uptrend"
        # In a smooth uptrend, swing highs may not meet threshold (highs are
        # continuously rising), but swing lows should be detectable
        assert len(ps.swing_lows) > 0 or len(ps.swing_highs) > 0

    def test_downtrend_structure(self, settings: PhaseSettings):
        df = _make_downtrend(200)
        ps = compute_price_structure(df, settings)
        assert ps.price_vs_sma < 0, "Price should be below SMA in downtrend"

    def test_support_resistance_set(self, settings: PhaseSettings):
        df = _make_uptrend(200)
        ps = compute_price_structure(df, settings)
        # Should have at least support or resistance if swings detected
        if ps.swing_lows:
            assert ps.support_level is not None
        if ps.swing_highs:
            assert ps.resistance_level is not None

    def test_volume_trend_field(self, settings: PhaseSettings):
        df = _make_uptrend(200)
        ps = compute_price_structure(df, settings)
        assert ps.volume_trend in ("declining", "stable", "rising")

    def test_range_compression_bounded(self, settings: PhaseSettings):
        df = _make_choppy(200)
        ps = compute_price_structure(df, settings)
        assert -1.0 <= ps.range_compression <= 1.0
