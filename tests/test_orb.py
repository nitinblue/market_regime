"""Tests for Opening Range Breakout (ORB) analysis."""

import numpy as np
import pandas as pd
import pytest

from market_regime.features.orb import compute_orb
from market_regime.models.technicals import ORBData, ORBLevel, ORBStatus, SignalDirection


def _make_intraday(
    date: str = "2026-02-20",
    freq: str = "5min",
    bars: int = 78,  # 6.5 hours of 5-min bars
    base_price: float = 500.0,
    opening_high_offset: float = 2.0,
    opening_low_offset: float = -2.0,
    post_opening_drift: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic intraday OHLCV data.

    Args:
        date: Trading date.
        freq: Bar frequency.
        bars: Number of bars in session.
        base_price: Midpoint of opening range.
        opening_high_offset: Max offset above base in opening bars.
        opening_low_offset: Max offset below base in opening bars.
        post_opening_drift: Drift applied after opening period.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(f"{date} 09:30:00")
    index = pd.date_range(start=start, periods=bars, freq=freq)

    # Generate prices
    prices = np.full(bars, base_price, dtype=float)
    noise = rng.normal(0, 0.5, bars)

    # First 6 bars = opening 30 minutes (5-min bars)
    opening_bars = 6
    for i in range(opening_bars):
        prices[i] = base_price + rng.uniform(opening_low_offset, opening_high_offset)

    # Post-opening bars: apply drift
    for i in range(opening_bars, bars):
        prices[i] = prices[i - 1] + post_opening_drift + noise[i]

    spread = rng.uniform(0.3, 1.0, bars)
    high = prices + spread
    low = prices - spread
    open_prices = prices + rng.normal(0, 0.3, bars)
    volume = rng.integers(50_000, 500_000, bars).astype(float)

    # Ensure opening range has clear high/low
    high[2] = base_price + opening_high_offset + 0.5  # Opening range high bar
    low[4] = base_price + opening_low_offset - 0.5    # Opening range low bar

    return pd.DataFrame(
        {
            "Open": open_prices,
            "High": high,
            "Low": low,
            "Close": prices,
            "Volume": volume,
        },
        index=index,
    )


class TestOpeningRange:
    def test_opening_range_computed(self):
        df = _make_intraday()
        orb = compute_orb(df, "TEST")

        assert orb.range_high > orb.range_low
        assert orb.range_size > 0
        assert orb.range_pct > 0
        assert orb.opening_minutes == 30

    def test_range_high_low_from_opening_bars(self):
        """Range high/low should come from the first 30 min bars only."""
        df = _make_intraday(base_price=500.0, opening_high_offset=3.0, opening_low_offset=-3.0)
        orb = compute_orb(df, "TEST")

        # Opening bars are first 6 (5-min bars × 30 min)
        opening = df.iloc[:6]
        assert orb.range_high == round(float(opening["High"].max()), 2)
        assert orb.range_low == round(float(opening["Low"].min()), 2)

    def test_ticker_and_date_populated(self):
        df = _make_intraday(date="2026-02-20")
        orb = compute_orb(df, "SPY")
        assert orb.ticker == "SPY"
        assert str(orb.date) == "2026-02-20"


class TestBreakoutDetection:
    def test_breakout_long_detected(self):
        """Price drifting up after opening should trigger BREAKOUT_LONG."""
        df = _make_intraday(post_opening_drift=1.0, seed=10)
        orb = compute_orb(df, "TEST")
        assert orb.status == ORBStatus.BREAKOUT_LONG
        assert orb.breakout_bar_index is not None

    def test_breakout_short_detected(self):
        """Price drifting down after opening should trigger BREAKOUT_SHORT."""
        df = _make_intraday(post_opening_drift=-0.5, seed=10)
        orb = compute_orb(df, "TEST")
        assert orb.status == ORBStatus.BREAKOUT_SHORT
        assert orb.breakout_bar_index is not None

    def test_within_range(self):
        """Price staying inside opening range → WITHIN."""
        # Very wide opening range + no drift = stays within
        df = _make_intraday(
            opening_high_offset=10.0,
            opening_low_offset=-10.0,
            post_opening_drift=0.0,
            seed=42,
        )
        orb = compute_orb(df, "TEST")
        assert orb.status == ORBStatus.WITHIN
        assert orb.breakout_bar_index is None

    def test_failed_breakout_long(self):
        """Price breaks above then returns inside → FAILED_LONG."""
        df = _make_intraday(
            base_price=500.0,
            opening_high_offset=1.0,
            opening_low_offset=-1.0,
            seed=42,
        )
        # Manually create a failed breakout: spike up then return
        opening_high = float(df.iloc[:6]["High"].max())
        opening_low = float(df.iloc[:6]["Low"].min())

        # Bar 7: break above range_high
        df.iloc[7, df.columns.get_loc("High")] = opening_high + 2.0
        df.iloc[7, df.columns.get_loc("Close")] = opening_high + 1.5
        # Bar 10+: return inside range
        for i in range(10, len(df)):
            mid = (opening_high + opening_low) / 2
            df.iloc[i, df.columns.get_loc("High")] = mid + 0.3
            df.iloc[i, df.columns.get_loc("Low")] = mid - 0.3
            df.iloc[i, df.columns.get_loc("Close")] = mid

        orb = compute_orb(df, "TEST")
        assert orb.status == ORBStatus.FAILED_LONG

    def test_failed_breakout_short(self):
        """Price breaks below then returns inside → FAILED_SHORT."""
        df = _make_intraday(
            base_price=500.0,
            opening_high_offset=1.0,
            opening_low_offset=-1.0,
            seed=42,
        )
        opening_high = float(df.iloc[:6]["High"].max())
        opening_low = float(df.iloc[:6]["Low"].min())

        # Bar 7: break below range_low
        df.iloc[7, df.columns.get_loc("Low")] = opening_low - 2.0
        df.iloc[7, df.columns.get_loc("Close")] = opening_low - 1.5
        # Bar 10+: return inside range
        for i in range(10, len(df)):
            mid = (opening_high + opening_low) / 2
            df.iloc[i, df.columns.get_loc("High")] = mid + 0.3
            df.iloc[i, df.columns.get_loc("Low")] = mid - 0.3
            df.iloc[i, df.columns.get_loc("Close")] = mid

        orb = compute_orb(df, "TEST")
        assert orb.status == ORBStatus.FAILED_SHORT


class TestExtensionLevels:
    def test_extension_levels_count(self):
        """Default extensions [1.0, 1.5, 2.0] → 7 levels (midpoint + 3 long + 3 short)."""
        df = _make_intraday()
        orb = compute_orb(df, "TEST")
        assert len(orb.levels) == 7

    def test_extension_levels_correct(self):
        """Extension levels computed correctly from range."""
        df = _make_intraday()
        orb = compute_orb(df, "TEST")
        rs = orb.range_size

        # Find T1 Long
        t1_long = next(lv for lv in orb.levels if "T1 Long" in lv.label)
        assert abs(t1_long.price - (orb.range_high + 1.0 * rs)) < 0.02

        # Find T2 Short
        t2_short = next(lv for lv in orb.levels if "T2 Short" in lv.label)
        assert abs(t2_short.price - (orb.range_low - 1.5 * rs)) < 0.02

    def test_midpoint_level(self):
        """Midpoint level is average of range high and low."""
        df = _make_intraday()
        orb = compute_orb(df, "TEST")
        mid = next(lv for lv in orb.levels if lv.label == "Midpoint")
        expected = round((orb.range_high + orb.range_low) / 2, 2)
        assert mid.price == expected

    def test_custom_extensions(self):
        """Custom extension list changes level count."""
        df = _make_intraday()
        orb = compute_orb(df, "TEST", extensions=[0.5, 1.0])
        # 1 midpoint + 2 long + 2 short = 5
        assert len(orb.levels) == 5

    def test_distance_pct_populated(self):
        """All levels should have distance_pct from current price."""
        df = _make_intraday()
        orb = compute_orb(df, "TEST")
        for lv in orb.levels:
            assert isinstance(lv.distance_pct, float)


class TestVWAP:
    def test_vwap_computed(self):
        df = _make_intraday()
        orb = compute_orb(df, "TEST")
        assert orb.session_vwap is not None
        # VWAP should be in a reasonable range
        assert orb.session_low <= orb.session_vwap <= orb.session_high


class TestVolumeRatio:
    def test_volume_ratio_positive(self):
        df = _make_intraday()
        orb = compute_orb(df, "TEST")
        assert orb.opening_volume_ratio > 0

    def test_high_opening_volume(self):
        """Higher opening volume → ratio > 1."""
        df = _make_intraday()
        # Boost opening bar volumes
        for i in range(6):
            df.iloc[i, df.columns.get_loc("Volume")] *= 5
        orb = compute_orb(df, "TEST")
        assert orb.opening_volume_ratio > 1.0


class TestATRContext:
    def test_atr_context_when_provided(self):
        df = _make_intraday()
        orb = compute_orb(df, "TEST", daily_atr=5.0)
        assert orb.range_vs_daily_atr_pct is not None
        assert orb.range_vs_daily_atr_pct > 0

    def test_atr_context_none_when_not_provided(self):
        df = _make_intraday()
        orb = compute_orb(df, "TEST")
        assert orb.range_vs_daily_atr_pct is None


class TestRetestCount:
    def test_retest_count_on_breakout(self):
        """Retests after breakout should be counted."""
        df = _make_intraday(
            base_price=500.0,
            opening_high_offset=1.0,
            opening_low_offset=-1.0,
            seed=42,
        )
        opening_high = float(df.iloc[:6]["High"].max())

        # Create a clean breakout with retests:
        # Bar 7: break above
        df.iloc[7, df.columns.get_loc("High")] = opening_high + 3.0
        df.iloc[7, df.columns.get_loc("Close")] = opening_high + 2.5
        # Bar 8: retest — dips to touch range_high, closes above
        df.iloc[8, df.columns.get_loc("Low")] = opening_high - 0.1
        df.iloc[8, df.columns.get_loc("High")] = opening_high + 2.0
        df.iloc[8, df.columns.get_loc("Close")] = opening_high + 1.5
        # Bar 9: another retest
        df.iloc[9, df.columns.get_loc("Low")] = opening_high - 0.1
        df.iloc[9, df.columns.get_loc("High")] = opening_high + 2.5
        df.iloc[9, df.columns.get_loc("Close")] = opening_high + 2.0
        # Remaining bars: stay above
        for i in range(10, len(df)):
            df.iloc[i, df.columns.get_loc("High")] = opening_high + 3.0
            df.iloc[i, df.columns.get_loc("Low")] = opening_high + 1.0
            df.iloc[i, df.columns.get_loc("Close")] = opening_high + 2.0

        orb = compute_orb(df, "TEST")
        assert orb.status == ORBStatus.BREAKOUT_LONG
        assert orb.retest_count >= 2


class TestSignals:
    def test_breakout_long_signal(self):
        df = _make_intraday(post_opening_drift=0.5, seed=10)
        orb = compute_orb(df, "TEST")
        if orb.status == ORBStatus.BREAKOUT_LONG:
            signal_names = [s.name for s in orb.signals]
            assert "ORB Breakout Long" in signal_names
            sig = next(s for s in orb.signals if s.name == "ORB Breakout Long")
            assert sig.direction == SignalDirection.BULLISH

    def test_breakout_short_signal(self):
        df = _make_intraday(post_opening_drift=-0.5, seed=10)
        orb = compute_orb(df, "TEST")
        if orb.status == ORBStatus.BREAKOUT_SHORT:
            signal_names = [s.name for s in orb.signals]
            assert "ORB Breakout Short" in signal_names
            sig = next(s for s in orb.signals if s.name == "ORB Breakout Short")
            assert sig.direction == SignalDirection.BEARISH

    def test_signals_always_list(self):
        df = _make_intraday()
        orb = compute_orb(df, "TEST")
        assert isinstance(orb.signals, list)


class TestTimezoneHandling:
    def test_tz_naive_works(self):
        """Timezone-naive timestamps (assumed ET) should work."""
        df = _make_intraday()
        assert df.index.tz is None
        orb = compute_orb(df, "TEST")
        assert orb.range_high > 0

    def test_tz_aware_works(self):
        """Timezone-aware timestamps should be handled correctly."""
        df = _make_intraday()
        df.index = df.index.tz_localize("US/Eastern")
        orb = compute_orb(df, "TEST")
        assert orb.range_high > 0

    def test_utc_tz_converted(self):
        """UTC timestamps should be converted to ET."""
        df = _make_intraday()
        # Shift times to UTC equivalent (+5 hours during EST)
        df.index = df.index + pd.Timedelta(hours=5)
        df.index = df.index.tz_localize("UTC")
        orb = compute_orb(df, "TEST")
        assert orb.range_high > 0


class TestMultiDay:
    def test_multi_day_uses_latest(self):
        """With multiple days, analyze only the latest session."""
        day1 = _make_intraday(date="2026-02-19", base_price=490.0)
        day2 = _make_intraday(date="2026-02-20", base_price=510.0)
        combined = pd.concat([day1, day2])

        orb = compute_orb(combined, "TEST")
        assert str(orb.date) == "2026-02-20"
        # Range should be based on day2's prices (around 510), not day1's (around 490)
        assert orb.range_high > 500


class TestValidation:
    def test_missing_columns_raises(self):
        df = pd.DataFrame({"Close": [1, 2, 3]}, index=pd.date_range("2026-01-01", periods=3, freq="5min"))
        with pytest.raises(ValueError, match="missing columns"):
            compute_orb(df, "TEST")

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df.index = pd.DatetimeIndex([])
        with pytest.raises(ValueError, match="empty"):
            compute_orb(df, "TEST")

    def test_no_datetime_index_raises(self):
        df = pd.DataFrame({
            "Open": [1], "High": [2], "Low": [0.5], "Close": [1.5], "Volume": [100]
        })
        with pytest.raises(ValueError, match="DatetimeIndex"):
            compute_orb(df, "TEST")


class TestDescription:
    def test_description_contains_ticker(self):
        df = _make_intraday()
        orb = compute_orb(df, "SPY")
        assert "SPY" in orb.description

    def test_description_contains_range(self):
        df = _make_intraday()
        orb = compute_orb(df, "TEST")
        assert "ORB" in orb.description
        assert str(orb.range_low) in orb.description or "range" in orb.description.lower()
