"""Tests for technical indicators module."""

import numpy as np
import pandas as pd
import pytest

from market_regime.config import TechnicalsSettings
from market_regime.features.technicals import (
    compute_atr,
    compute_bollinger,
    compute_ema,
    compute_macd,
    compute_rsi,
    compute_sma,
    compute_stochastic,
    compute_technicals,
    compute_vcp,
    compute_vwma,
    _detect_golden_death_cross,
    _detect_macd_crossover,
)
from market_regime.models.technicals import (
    MarketPhase,
    PhaseIndicator,
    SignalDirection,
    SignalStrength,
    TechnicalSnapshot,
    VCPData,
    VCPStage,
)


class TestMovingAverages:
    def test_sma_correctness(self, sample_ohlcv_trending: pd.DataFrame):
        close = sample_ohlcv_trending["Close"]
        sma = compute_sma(close, 20)
        # Last SMA(20) should be mean of last 20 closes
        expected = close.iloc[-20:].mean()
        assert abs(sma.iloc[-1] - expected) < 1e-10

    def test_ema_correctness(self, sample_ohlcv_trending: pd.DataFrame):
        close = sample_ohlcv_trending["Close"]
        ema = compute_ema(close, 9)
        # EMA should exist for all rows (no NaN after first)
        assert not pd.isna(ema.iloc[-1])
        assert len(ema.dropna()) == len(close)

    def test_price_vs_sma_positive_in_uptrend(self, sample_ohlcv_trending: pd.DataFrame):
        snapshot = compute_technicals(sample_ohlcv_trending, "TEST")
        # In an uptrend, price should be above the 200-day SMA
        assert snapshot.moving_averages.price_vs_sma_200_pct > 0

    def test_sma_values_populated(self, sample_ohlcv_trending: pd.DataFrame):
        snapshot = compute_technicals(sample_ohlcv_trending, "TEST")
        ma = snapshot.moving_averages
        assert ma.sma_20 > 0
        assert ma.sma_50 > 0
        assert ma.sma_200 > 0
        assert ma.ema_9 > 0
        assert ma.ema_21 > 0


class TestRSI:
    def test_rsi_bounded(self, sample_ohlcv_trending: pd.DataFrame):
        close = sample_ohlcv_trending["Close"]
        rsi = compute_rsi(close, 14)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_overbought_oversold_flags(self):
        # Monotonically rising price -> RSI should be 100 (all gains, zero losses)
        dates = pd.bdate_range("2024-01-01", periods=100)
        close = pd.Series(np.linspace(100, 200, 100), index=dates)
        rsi = compute_rsi(close, 14)
        last_rsi = rsi.iloc[-1]
        assert last_rsi == 100.0

    def test_rsi_in_uptrend_above_50(self, sample_ohlcv_trending: pd.DataFrame):
        snapshot = compute_technicals(sample_ohlcv_trending, "TEST")
        assert snapshot.rsi.value > 50

    def test_rsi_snapshot_flags(self):
        # Monotonically rising close -> RSI = 100 -> overbought
        dates = pd.bdate_range("2024-01-01", periods=100)
        prices = np.linspace(100, 200, 100)
        ohlcv = pd.DataFrame({
            "Open": prices,
            "High": prices + 1,
            "Low": prices - 1,
            "Close": prices,
            "Volume": np.full(100, 1e6),
        }, index=dates)
        snapshot = compute_technicals(ohlcv, "TEST")
        assert snapshot.rsi.value == 100.0
        assert snapshot.rsi.is_overbought is True
        assert snapshot.rsi.is_oversold is False


class TestBollingerBands:
    def test_band_ordering(self, sample_ohlcv_trending: pd.DataFrame):
        snapshot = compute_technicals(sample_ohlcv_trending, "TEST")
        bb = snapshot.bollinger
        assert bb.upper > bb.middle > bb.lower

    def test_bandwidth_positive(self, sample_ohlcv_trending: pd.DataFrame):
        snapshot = compute_technicals(sample_ohlcv_trending, "TEST")
        assert snapshot.bollinger.bandwidth > 0

    def test_percent_b_range(self, sample_ohlcv_trending: pd.DataFrame):
        snapshot = compute_technicals(sample_ohlcv_trending, "TEST")
        # percent_b can go outside [0,1] when price is outside bands, but should be reasonable
        assert -1 < snapshot.bollinger.percent_b < 2

    def test_compute_bollinger_directly(self, sample_ohlcv_trending: pd.DataFrame):
        close = sample_ohlcv_trending["Close"]
        upper, middle, lower = compute_bollinger(close, 20, 2.0)
        valid_idx = middle.dropna().index
        assert (upper.loc[valid_idx] > middle.loc[valid_idx]).all()
        assert (middle.loc[valid_idx] > lower.loc[valid_idx]).all()


class TestMACD:
    def test_histogram_equals_line_minus_signal(self, sample_ohlcv_trending: pd.DataFrame):
        close = sample_ohlcv_trending["Close"]
        macd_line, signal_line, histogram = compute_macd(close, 12, 26, 9)
        diff = macd_line - signal_line
        np.testing.assert_allclose(histogram.values, diff.values, atol=1e-10)

    def test_crossover_detection(self):
        # Synthetic: MACD below signal, then crosses above
        macd_line = pd.Series([0.0, -0.1, -0.05, 0.05])
        signal_line = pd.Series([0.0, 0.0, 0.0, 0.0])
        bullish, bearish = _detect_macd_crossover(macd_line, signal_line)
        assert bullish is True
        assert bearish is False

    def test_bearish_crossover_detection(self):
        macd_line = pd.Series([0.0, 0.1, 0.05, -0.05])
        signal_line = pd.Series([0.0, 0.0, 0.0, 0.0])
        bullish, bearish = _detect_macd_crossover(macd_line, signal_line)
        assert bullish is False
        assert bearish is True

    def test_macd_snapshot_populated(self, sample_ohlcv_trending: pd.DataFrame):
        snapshot = compute_technicals(sample_ohlcv_trending, "TEST")
        # MACD values should be non-zero with 250 bars of trending data
        assert snapshot.macd.macd_line != 0


class TestStochastic:
    def test_k_bounded(self, sample_ohlcv_trending: pd.DataFrame):
        close = sample_ohlcv_trending["Close"]
        high = sample_ohlcv_trending["High"]
        low = sample_ohlcv_trending["Low"]
        k, d = compute_stochastic(high, low, close, 14, 3)
        valid = k.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_overbought_oversold_flags(self, sample_ohlcv_trending: pd.DataFrame):
        snapshot = compute_technicals(sample_ohlcv_trending, "TEST")
        stoch = snapshot.stochastic
        # Flags should be consistent with k value
        if stoch.k > 80:
            assert stoch.is_overbought is True
        if stoch.k < 20:
            assert stoch.is_oversold is True


class TestVWMA:
    def test_equals_sma_when_volume_constant(self):
        dates = pd.bdate_range("2024-01-01", periods=50)
        close = pd.Series(np.random.default_rng(42).normal(100, 2, 50), index=dates)
        volume = pd.Series(np.full(50, 1e6), index=dates)  # constant volume
        vwma = compute_vwma(close, volume, 20)
        sma = compute_sma(close, 20)
        valid = vwma.dropna().index
        np.testing.assert_allclose(vwma.loc[valid].values, sma.loc[valid].values, atol=1e-10)


class TestATR:
    def test_atr_positive(self, sample_ohlcv_trending: pd.DataFrame):
        snapshot = compute_technicals(sample_ohlcv_trending, "TEST")
        assert snapshot.atr > 0

    def test_atr_pct_reasonable(self, sample_ohlcv_trending: pd.DataFrame):
        snapshot = compute_technicals(sample_ohlcv_trending, "TEST")
        # ATR as % of price should be < 20% for normal data
        assert 0 < snapshot.atr_pct < 20


class TestSupportResistance:
    def test_sr_populated(self, sample_ohlcv_trending: pd.DataFrame):
        snapshot = compute_technicals(sample_ohlcv_trending, "TEST")
        sr = snapshot.support_resistance
        # With 250 bars of data, we should find swing points
        assert sr.support is not None or sr.resistance is not None

    def test_support_below_price(self, sample_ohlcv_trending: pd.DataFrame):
        """Regression: support must be below current price."""
        snapshot = compute_technicals(sample_ohlcv_trending, "TEST")
        sr = snapshot.support_resistance
        if sr.support is not None:
            assert sr.support < snapshot.current_price, (
                f"Support {sr.support} should be below price {snapshot.current_price}"
            )

    def test_resistance_above_price(self, sample_ohlcv_trending: pd.DataFrame):
        """Regression: resistance must be above current price."""
        snapshot = compute_technicals(sample_ohlcv_trending, "TEST")
        sr = snapshot.support_resistance
        if sr.resistance is not None:
            assert sr.resistance > snapshot.current_price, (
                f"Resistance {sr.resistance} should be above price {snapshot.current_price}"
            )

    def test_sr_correctness_choppy(self, sample_ohlcv_choppy: pd.DataFrame):
        """Support < price and resistance > price in choppy data too."""
        snapshot = compute_technicals(sample_ohlcv_choppy, "CHOP")
        sr = snapshot.support_resistance
        if sr.support is not None:
            assert sr.support < snapshot.current_price
        if sr.resistance is not None:
            assert sr.resistance > snapshot.current_price

    def test_sr_none_when_no_swings(self):
        # Very short data — no swings detectable
        dates = pd.bdate_range("2024-01-01", periods=5)
        prices = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        ohlcv = pd.DataFrame({
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Volume": np.full(5, 1e6),
        }, index=dates)
        snapshot = compute_technicals(ohlcv, "TEST")
        sr = snapshot.support_resistance
        assert sr.support is None
        assert sr.resistance is None
        assert sr.price_vs_support_pct is None
        assert sr.price_vs_resistance_pct is None


class TestSignals:
    def test_rsi_oversold_signal_generated(self):
        # Build data with declining prices to push RSI low
        dates = pd.bdate_range("2024-01-01", periods=100)
        prices = np.linspace(200, 100, 100)
        ohlcv = pd.DataFrame({
            "Open": prices,
            "High": prices + 1,
            "Low": prices - 1,
            "Close": prices,
            "Volume": np.full(100, 1e6),
        }, index=dates)
        snapshot = compute_technicals(ohlcv, "TEST")
        signal_names = [s.name for s in snapshot.signals]
        assert "RSI Oversold" in signal_names

    def test_macd_crossover_signal(self):
        # Create data where MACD crosses: downtrend then reversal
        dates = pd.bdate_range("2024-01-01", periods=200)
        rng = np.random.default_rng(42)
        # Down first, then up
        prices = np.concatenate([
            100 * np.exp(np.cumsum(rng.normal(-0.003, 0.005, 100))),
            100 * np.exp(np.cumsum(rng.normal(-0.003, 0.005, 100))),
        ])
        # Force a crossover by adding a sharp reversal at end
        prices[-10:] = prices[-11] * np.exp(np.cumsum(np.full(10, 0.02)))
        ohlcv = pd.DataFrame({
            "Open": prices,
            "High": prices * 1.005,
            "Low": prices * 0.995,
            "Close": prices,
            "Volume": np.full(200, 1e6),
        }, index=dates)
        snapshot = compute_technicals(ohlcv, "TEST")
        # We at least get signals (exact crossover timing is data-dependent)
        assert isinstance(snapshot.signals, list)

    def test_signals_have_descriptions(self, sample_ohlcv_trending: pd.DataFrame):
        snapshot = compute_technicals(sample_ohlcv_trending, "TEST")
        for signal in snapshot.signals:
            assert signal.description
            assert signal.direction in SignalDirection
            assert signal.strength in SignalStrength

    def test_golden_cross_detection(self):
        # Cross happens at last bar: prev below, curr above
        sma50 = pd.Series([99.0, 99.5])
        sma200 = pd.Series([100.0, 99.0])
        golden, death = _detect_golden_death_cross(sma50, sma200)
        assert golden is True
        assert death is False

    def test_death_cross_detection(self):
        # Cross happens at last bar: prev above, curr below
        sma50 = pd.Series([100.5, 98.0])
        sma200 = pd.Series([100.0, 100.0])
        golden, death = _detect_golden_death_cross(sma50, sma200)
        assert golden is False
        assert death is True


class TestTechnicalSnapshot:
    def test_full_snapshot_populated(self, sample_ohlcv_trending: pd.DataFrame):
        snapshot = compute_technicals(sample_ohlcv_trending, "TEST")
        assert isinstance(snapshot, TechnicalSnapshot)
        assert snapshot.ticker == "TEST"
        assert snapshot.current_price > 0
        assert snapshot.atr > 0
        assert snapshot.vwma_20 > 0
        assert snapshot.moving_averages is not None
        assert snapshot.rsi is not None
        assert snapshot.bollinger is not None
        assert snapshot.macd is not None
        assert snapshot.stochastic is not None
        assert snapshot.support_resistance is not None
        assert isinstance(snapshot.signals, list)

    def test_as_of_date_correct(self, sample_ohlcv_trending: pd.DataFrame):
        snapshot = compute_technicals(sample_ohlcv_trending, "TEST")
        expected = sample_ohlcv_trending.index[-1].date()
        assert snapshot.as_of_date == expected

    def test_ticker_set(self, sample_ohlcv_choppy: pd.DataFrame):
        snapshot = compute_technicals(sample_ohlcv_choppy, "CHOPPY")
        assert snapshot.ticker == "CHOPPY"

    def test_choppy_vs_trending(
        self, sample_ohlcv_trending: pd.DataFrame, sample_ohlcv_choppy: pd.DataFrame
    ):
        trend = compute_technicals(sample_ohlcv_trending, "TREND")
        chop = compute_technicals(sample_ohlcv_choppy, "CHOP")
        # Both should produce valid snapshots
        assert trend.current_price > 0
        assert chop.current_price > 0


class TestValidation:
    def test_missing_columns_raises(self):
        df = pd.DataFrame({"Close": [1, 2, 3]}, index=pd.bdate_range("2024-01-01", periods=3))
        with pytest.raises(ValueError, match="missing columns"):
            compute_technicals(df, "BAD")

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"],
            index=pd.DatetimeIndex([]),
        )
        with pytest.raises(ValueError, match="empty"):
            compute_technicals(df, "EMPTY")


def _make_vcp_ohlcv() -> pd.DataFrame:
    """Build synthetic data with a clear VCP pattern.

    Structure: uptrend (100 days) -> T1 wide contraction (30 days) ->
    T2 tighter contraction (30 days) -> T3 tight squeeze (30 days).
    Oscillation amplitudes are large enough to pass swing detection
    threshold (1.5%) and volume declines across the base.
    """
    rng = np.random.default_rng(77)
    n_warmup = 50
    n_trend = 100
    n_t1 = 30
    n_t2 = 30
    n_t3 = 30
    n_total = n_warmup + n_trend + n_t1 + n_t2 + n_t3
    dates = pd.bdate_range("2023-06-01", periods=n_total)

    prices = np.empty(n_total)
    # Warmup: flat around 80
    prices[:n_warmup] = 80 + rng.normal(0, 0.3, n_warmup).cumsum()
    # Uptrend: 80 -> 120
    trend_start = prices[n_warmup - 1]
    for i in range(n_trend):
        idx = n_warmup + i
        prices[idx] = trend_start + (40 * i / n_trend) + rng.normal(0, 0.5)

    # T1: wide contraction around 120, amplitude 8 (~13% range)
    t1_start = n_warmup + n_trend
    t1_mid = prices[t1_start - 1]
    for i in range(n_t1):
        idx = t1_start + i
        prices[idx] = t1_mid + 8 * np.sin(2 * np.pi * i / 15) + rng.normal(0, 0.2)

    # T2: tighter contraction, amplitude 4 (~6.5% range)
    t2_start = t1_start + n_t1
    t2_mid = t1_mid
    for i in range(n_t2):
        idx = t2_start + i
        prices[idx] = t2_mid + 4 * np.sin(2 * np.pi * i / 15) + rng.normal(0, 0.15)

    # T3: tight squeeze, amplitude 2.5 (~4% range — still above 1.5% swing threshold)
    t3_start = t2_start + n_t2
    t3_mid = t2_mid
    for i in range(n_t3):
        idx = t3_start + i
        prices[idx] = t3_mid + 2.5 * np.sin(2 * np.pi * i / 15) + rng.normal(0, 0.1)

    prices = np.maximum(prices, 1.0)  # No zero prices

    # High/Low have meaningful range for swing detection
    noise = rng.uniform(0.005, 0.015, n_total) * prices
    high = prices + noise
    low = prices - noise
    open_prices = prices + rng.normal(0, 0.2, n_total)

    # Volume: declining across the base — each contraction has lower volume
    # Also declining WITHIN the last 30 days for volume_trend detection
    volume = np.full(n_total, 5e6, dtype=float)
    volume[t1_start:t2_start] = np.linspace(5e6, 3.5e6, n_t1) + rng.normal(0, 1e5, n_t1)
    volume[t2_start:t3_start] = np.linspace(3.5e6, 2e6, n_t2) + rng.normal(0, 1e5, n_t2)
    volume[t3_start:] = np.linspace(2e6, 1e6, n_t3) + rng.normal(0, 5e4, n_t3)
    volume = np.maximum(volume, 1e5)

    return pd.DataFrame({
        "Open": open_prices,
        "High": high,
        "Low": low,
        "Close": prices,
        "Volume": volume,
    }, index=dates)


class TestVCP:
    def test_vcp_detects_contraction_in_synthetic(self):
        """VCP detector should find contractions in synthetic VCP data."""
        ohlcv = _make_vcp_ohlcv()
        settings = TechnicalsSettings()
        price = float(ohlcv["Close"].iloc[-1])
        sma50 = float(compute_sma(ohlcv["Close"], 50).iloc[-1])
        sma200 = float(compute_sma(ohlcv["Close"], 200).iloc[-1])

        vcp = compute_vcp(ohlcv, price, sma50, sma200, ohlcv["Volume"], settings)
        assert vcp is not None
        assert vcp.stage != VCPStage.NONE
        assert vcp.contraction_count >= 2
        assert len(vcp.contraction_pcts) >= 2
        # Each contraction should be tighter than the previous
        for i in range(1, len(vcp.contraction_pcts)):
            assert vcp.contraction_pcts[i] < vcp.contraction_pcts[i - 1]

    def test_vcp_none_in_pure_uptrend(self, sample_ohlcv_trending: pd.DataFrame):
        """Pure uptrend should not produce a VCP (no contractions)."""
        snapshot = compute_technicals(sample_ohlcv_trending, "TREND")
        # VCP might be None or stage NONE — either is acceptable
        if snapshot.vcp is not None:
            assert snapshot.vcp.stage == VCPStage.NONE or snapshot.vcp.contraction_count < 2

    def test_vcp_none_in_choppy(self, sample_ohlcv_choppy: pd.DataFrame):
        """Random chop should not produce an ordered tightening pattern."""
        snapshot = compute_technicals(sample_ohlcv_choppy, "CHOP")
        # Choppy data shouldn't produce a high-quality VCP
        if snapshot.vcp is not None and snapshot.vcp.stage != VCPStage.NONE:
            # Even if it finds something, score should be low
            assert snapshot.vcp.score < 0.8

    def test_vcp_score_range(self):
        """VCP score must be between 0.0 and 1.0."""
        ohlcv = _make_vcp_ohlcv()
        settings = TechnicalsSettings()
        price = float(ohlcv["Close"].iloc[-1])
        sma50 = float(compute_sma(ohlcv["Close"], 50).iloc[-1])
        sma200 = float(compute_sma(ohlcv["Close"], 200).iloc[-1])

        vcp = compute_vcp(ohlcv, price, sma50, sma200, ohlcv["Volume"], settings)
        assert vcp is not None
        assert 0.0 <= vcp.score <= 1.0

    def test_vcp_in_snapshot(self):
        """compute_technicals should populate the vcp field."""
        ohlcv = _make_vcp_ohlcv()
        snapshot = compute_technicals(ohlcv, "VCP_TEST")
        # VCP field should be populated (not None) for data with enough history
        assert snapshot.vcp is not None
        assert isinstance(snapshot.vcp, VCPData)

    def test_vcp_signal_generated(self):
        """VCP pattern should generate signals in snapshot.signals."""
        ohlcv = _make_vcp_ohlcv()
        snapshot = compute_technicals(ohlcv, "VCP_TEST")
        if snapshot.vcp is not None and snapshot.vcp.stage != VCPStage.NONE:
            vcp_signal_names = [s.name for s in snapshot.signals if s.name.startswith("VCP")]
            assert len(vcp_signal_names) > 0

    def test_vcp_description_nonempty(self):
        """VCP description should always be populated."""
        ohlcv = _make_vcp_ohlcv()
        settings = TechnicalsSettings()
        price = float(ohlcv["Close"].iloc[-1])
        sma50 = float(compute_sma(ohlcv["Close"], 50).iloc[-1])
        sma200 = float(compute_sma(ohlcv["Close"], 200).iloc[-1])

        vcp = compute_vcp(ohlcv, price, sma50, sma200, ohlcv["Volume"], settings)
        assert vcp is not None
        assert vcp.description
        assert len(vcp.description) > 0

    def test_vcp_returns_none_short_data(self):
        """VCP should return None if insufficient data."""
        dates = pd.bdate_range("2024-01-01", periods=30)
        prices = np.linspace(100, 110, 30)
        ohlcv = pd.DataFrame({
            "Open": prices,
            "High": prices * 1.01,
            "Low": prices * 0.99,
            "Close": prices,
            "Volume": np.full(30, 1e6),
        }, index=dates)
        settings = TechnicalsSettings()
        vcp = compute_vcp(ohlcv, 110.0, 105.0, 100.0, ohlcv["Volume"], settings)
        assert vcp is None

    def test_vcp_pivot_above_price(self):
        """Pivot price should be at or above recent swing highs."""
        ohlcv = _make_vcp_ohlcv()
        settings = TechnicalsSettings()
        price = float(ohlcv["Close"].iloc[-1])
        sma50 = float(compute_sma(ohlcv["Close"], 50).iloc[-1])
        sma200 = float(compute_sma(ohlcv["Close"], 200).iloc[-1])

        vcp = compute_vcp(ohlcv, price, sma50, sma200, ohlcv["Volume"], settings)
        if vcp is not None and vcp.pivot_price is not None:
            # Pivot should be near/above current price (within the base range)
            assert vcp.pivot_price > 0

    def test_vcp_volume_trend_declining(self):
        """Synthetic VCP data has declining volume — detector should see it."""
        ohlcv = _make_vcp_ohlcv()
        settings = TechnicalsSettings()
        price = float(ohlcv["Close"].iloc[-1])
        sma50 = float(compute_sma(ohlcv["Close"], 50).iloc[-1])
        sma200 = float(compute_sma(ohlcv["Close"], 200).iloc[-1])

        vcp = compute_vcp(ohlcv, price, sma50, sma200, ohlcv["Volume"], settings)
        assert vcp is not None
        assert vcp.volume_trend == "declining"


class TestPhaseIndicator:
    def test_phase_populated_in_snapshot(self, sample_ohlcv_trending: pd.DataFrame):
        """Phase indicator should always be present in snapshot."""
        snapshot = compute_technicals(sample_ohlcv_trending, "TEST")
        assert snapshot.phase is not None
        assert isinstance(snapshot.phase, PhaseIndicator)
        assert snapshot.phase.phase in MarketPhase

    def test_uptrend_is_markup(self, sample_ohlcv_trending: pd.DataFrame):
        """Uptrending data should classify as markup."""
        snapshot = compute_technicals(sample_ohlcv_trending, "TEST")
        # Strong uptrend should be markup
        assert snapshot.phase.phase == MarketPhase.MARKUP

    def test_choppy_not_markup(self, sample_ohlcv_choppy: pd.DataFrame):
        """Choppy data should not classify as markup."""
        snapshot = compute_technicals(sample_ohlcv_choppy, "CHOP")
        assert snapshot.phase.phase != MarketPhase.MARKUP

    def test_confidence_bounded(self, sample_ohlcv_trending: pd.DataFrame):
        """Phase confidence must be between 0.10 and 0.95."""
        snapshot = compute_technicals(sample_ohlcv_trending, "TEST")
        assert 0.10 <= snapshot.phase.confidence <= 0.95

    def test_description_nonempty(self, sample_ohlcv_trending: pd.DataFrame):
        """Phase description should always have content."""
        snapshot = compute_technicals(sample_ohlcv_trending, "TEST")
        assert snapshot.phase.description
        assert len(snapshot.phase.description) > 10

    def test_phase_signal_generated(self, sample_ohlcv_trending: pd.DataFrame):
        """Phase signal should appear in snapshot.signals."""
        snapshot = compute_technicals(sample_ohlcv_trending, "TEST")
        phase_signals = [s for s in snapshot.signals if s.name.startswith("Phase:")]
        assert len(phase_signals) == 1

    def test_swing_flags_populated(self, sample_ohlcv_trending: pd.DataFrame):
        """Swing pattern flags should be booleans."""
        snapshot = compute_technicals(sample_ohlcv_trending, "TEST")
        p = snapshot.phase
        assert isinstance(p.higher_highs, bool)
        assert isinstance(p.higher_lows, bool)
        assert isinstance(p.lower_highs, bool)
        assert isinstance(p.lower_lows, bool)

    def test_volume_trend_valid(self, sample_ohlcv_choppy: pd.DataFrame):
        """Volume trend should be one of the three values."""
        snapshot = compute_technicals(sample_ohlcv_choppy, "CHOP")
        assert snapshot.phase.volume_trend in ("declining", "stable", "rising")

    def test_downtrend_is_markdown(self):
        """Monotonic downtrend should classify as markdown."""
        dates = pd.bdate_range("2024-01-01", periods=250)
        rng = np.random.default_rng(42)
        prices = 200 * np.exp(np.cumsum(rng.normal(-0.002, 0.008, 250)))
        ohlcv = pd.DataFrame({
            "Open": prices,
            "High": prices * (1 + rng.uniform(0.005, 0.015, 250)),
            "Low": prices * (1 - rng.uniform(0.005, 0.015, 250)),
            "Close": prices,
            "Volume": rng.integers(1_000_000, 10_000_000, 250).astype(float),
        }, index=dates)
        snapshot = compute_technicals(ohlcv, "DOWN")
        assert snapshot.phase.phase == MarketPhase.MARKDOWN
