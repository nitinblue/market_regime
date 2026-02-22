"""Tests for Wyckoff phase detection."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from market_analyzer.config import PhaseSettings, get_settings, reset_settings
from market_analyzer.models.phase import PhaseID, PhaseResult
from market_analyzer.models.regime import RegimeID, RegimeTimeSeries, RegimeTimeSeriesEntry, TrendDirection
from market_analyzer.phases.detector import PhaseDetector


def _make_regime_series(
    regimes: list[tuple[int, TrendDirection | None, int]],
    start_date: str = "2024-01-01",
) -> RegimeTimeSeries:
    """Build a RegimeTimeSeries from a compact spec.

    Args:
        regimes: List of (regime_id, trend_direction, n_days)
        start_date: Starting date.
    """
    entries = []
    d = date.fromisoformat(start_date)
    for regime_id, trend, n_days in regimes:
        for _ in range(n_days):
            entries.append(RegimeTimeSeriesEntry(
                date=d,
                regime=RegimeID(regime_id),
                confidence=0.85,
                probabilities={1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05, regime_id: 0.85},
                trend_direction=trend,
            ))
            d += timedelta(days=1)
    return RegimeTimeSeries(ticker="TEST", entries=entries)


def _make_ohlcv(
    periods: int = 200,
    trend: float = 0.0,
    volatility: float = 0.01,
    base: float = 100.0,
    seed: int = 42,
    volume_trend: str = "stable",
) -> pd.DataFrame:
    """Make synthetic OHLCV with controlled properties."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=periods)
    returns = rng.normal(trend, volatility, periods)
    prices = base * np.exp(np.cumsum(returns))
    daily_range = prices * volatility * rng.uniform(0.5, 2.0, periods)
    high = prices + daily_range / 2
    low = prices - daily_range / 2
    open_prices = prices + rng.normal(0, volatility * prices * 0.3, periods)

    if volume_trend == "declining":
        volume = rng.integers(1_000_000, 5_000_000, periods).astype(float) * np.linspace(2.0, 0.3, periods)
    elif volume_trend == "rising":
        volume = rng.integers(1_000_000, 5_000_000, periods).astype(float) * np.linspace(0.3, 2.0, periods)
    else:
        volume = rng.integers(1_000_000, 5_000_000, periods).astype(float)

    return pd.DataFrame(
        {"Open": open_prices, "High": high, "Low": low, "Close": prices, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def detector() -> PhaseDetector:
    return PhaseDetector()


class TestMarkup:
    def test_bullish_r3_is_markup(self, detector: PhaseDetector):
        """Bullish R3 -> MARKUP."""
        series = _make_regime_series([(3, TrendDirection.BULLISH, 30)])
        ohlcv = _make_ohlcv(200, trend=0.002)
        result = detector.detect("TEST", ohlcv, series)
        assert result.phase == PhaseID.MARKUP

    def test_bullish_r4_is_markup(self, detector: PhaseDetector):
        """Bullish R4 -> MARKUP."""
        series = _make_regime_series([(4, TrendDirection.BULLISH, 30)])
        ohlcv = _make_ohlcv(200, trend=0.003, volatility=0.02)
        result = detector.detect("TEST", ohlcv, series)
        assert result.phase == PhaseID.MARKUP


class TestMarkdown:
    def test_bearish_r3_is_markdown(self, detector: PhaseDetector):
        """Bearish R3 -> MARKDOWN."""
        series = _make_regime_series([(3, TrendDirection.BEARISH, 30)])
        ohlcv = _make_ohlcv(200, trend=-0.002)
        result = detector.detect("TEST", ohlcv, series)
        assert result.phase == PhaseID.MARKDOWN

    def test_bearish_r4_is_markdown(self, detector: PhaseDetector):
        """Bearish R4 -> MARKDOWN."""
        series = _make_regime_series([(4, TrendDirection.BEARISH, 30)])
        ohlcv = _make_ohlcv(200, trend=-0.003, volatility=0.02)
        result = detector.detect("TEST", ohlcv, series)
        assert result.phase == PhaseID.MARKDOWN


class TestAccumulation:
    def test_r1_after_bearish_r3(self, detector: PhaseDetector):
        """R1 following bearish R3 -> ACCUMULATION."""
        series = _make_regime_series([
            (3, TrendDirection.BEARISH, 60),
            (1, None, 20),
        ])
        ohlcv = _make_ohlcv(200, trend=-0.001)
        result = detector.detect("TEST", ohlcv, series)
        assert result.phase == PhaseID.ACCUMULATION

    def test_r2_after_bearish_r4(self, detector: PhaseDetector):
        """R2 following bearish R4 -> ACCUMULATION."""
        series = _make_regime_series([
            (4, TrendDirection.BEARISH, 60),
            (2, None, 20),
        ])
        ohlcv = _make_ohlcv(200, trend=-0.002, volatility=0.02)
        result = detector.detect("TEST", ohlcv, series)
        assert result.phase == PhaseID.ACCUMULATION


class TestDistribution:
    def test_r1_after_bullish_r3(self, detector: PhaseDetector):
        """R1 following bullish R3 -> DISTRIBUTION."""
        series = _make_regime_series([
            (3, TrendDirection.BULLISH, 60),
            (1, None, 20),
        ])
        ohlcv = _make_ohlcv(200, trend=0.001)
        result = detector.detect("TEST", ohlcv, series)
        assert result.phase == PhaseID.DISTRIBUTION

    def test_r2_after_bullish_r4(self, detector: PhaseDetector):
        """R2 following bullish R4 -> DISTRIBUTION."""
        series = _make_regime_series([
            (4, TrendDirection.BULLISH, 60),
            (2, None, 20),
        ])
        ohlcv = _make_ohlcv(200, trend=0.002, volatility=0.02)
        result = detector.detect("TEST", ohlcv, series)
        assert result.phase == PhaseID.DISTRIBUTION


class TestAmbiguousMR:
    def test_mr_after_mr_below_sma_is_accumulation(self, detector: PhaseDetector):
        """R1 after R2 with price below SMA and no higher lows -> ACCUMULATION."""
        series = _make_regime_series([
            (2, None, 60),
            (1, None, 20),
        ])
        # Downtrend leaves price below SMA; use very low trend to avoid higher_lows
        ohlcv = _make_ohlcv(200, trend=-0.005, volatility=0.001)
        result = detector.detect("TEST", ohlcv, series)
        assert result.phase == PhaseID.ACCUMULATION

    def test_mr_after_mr_above_sma_is_distribution(self, detector: PhaseDetector):
        """R1 after R2 with price above SMA and no lower highs -> DISTRIBUTION."""
        series = _make_regime_series([
            (2, None, 60),
            (1, None, 20),
        ])
        # Strong uptrend leaves price above SMA; use very low vol to avoid lower_highs
        ohlcv = _make_ohlcv(200, trend=0.005, volatility=0.001)
        result = detector.detect("TEST", ohlcv, series)
        assert result.phase == PhaseID.DISTRIBUTION


class TestConfidence:
    def test_volume_confirms_accumulation(self, detector: PhaseDetector):
        """Declining volume boosts accumulation confidence."""
        series = _make_regime_series([(3, TrendDirection.BEARISH, 60), (1, None, 20)])
        ohlcv_declining = _make_ohlcv(200, trend=-0.001, volume_trend="declining")
        ohlcv_rising = _make_ohlcv(200, trend=-0.001, volume_trend="rising")
        r_declining = detector.detect("TEST", ohlcv_declining, series)
        r_rising = detector.detect("TEST", ohlcv_rising, series)
        assert r_declining.confidence > r_rising.confidence

    def test_confidence_bounded(self, detector: PhaseDetector):
        """Confidence must be in [0.1, 0.95]."""
        series = _make_regime_series([(3, TrendDirection.BULLISH, 100)])
        ohlcv = _make_ohlcv(200, trend=0.003)
        result = detector.detect("TEST", ohlcv, series)
        assert 0.10 <= result.confidence <= 0.95


class TestPhaseAge:
    def test_short_phase_flagged(self, detector: PhaseDetector):
        """Phase under min_phase_days should have contradiction."""
        series = _make_regime_series([
            (3, TrendDirection.BEARISH, 60),
            (1, None, 5),  # Only 5 days
        ])
        ohlcv = _make_ohlcv(200, trend=-0.001)
        result = detector.detect("TEST", ohlcv, series)
        assert any("Short phase" in c for c in result.evidence.contradictions)


class TestFullCycle:
    def test_full_wyckoff_cycle(self, detector: PhaseDetector):
        """Walk through accumulation -> markup -> distribution -> markdown."""
        # Phase 1: Accumulation (R1 after bearish R3)
        series_accum = _make_regime_series([(3, TrendDirection.BEARISH, 60), (1, None, 30)])
        ohlcv = _make_ohlcv(200, trend=-0.001, volume_trend="declining")
        r1 = detector.detect("TEST", ohlcv, series_accum)
        assert r1.phase == PhaseID.ACCUMULATION

        # Phase 2: Markup (bullish R3)
        series_markup = _make_regime_series([(3, TrendDirection.BULLISH, 60)])
        ohlcv = _make_ohlcv(200, trend=0.003)
        r2 = detector.detect("TEST", ohlcv, series_markup)
        assert r2.phase == PhaseID.MARKUP

        # Phase 3: Distribution (R1 after bullish R3)
        series_distrib = _make_regime_series([(3, TrendDirection.BULLISH, 60), (1, None, 30)])
        ohlcv = _make_ohlcv(200, trend=0.001)
        r3 = detector.detect("TEST", ohlcv, series_distrib)
        assert r3.phase == PhaseID.DISTRIBUTION

        # Phase 4: Markdown (bearish R3)
        series_markdown = _make_regime_series([(3, TrendDirection.BEARISH, 60)])
        ohlcv = _make_ohlcv(200, trend=-0.003)
        r4 = detector.detect("TEST", ohlcv, series_markdown)
        assert r4.phase == PhaseID.MARKDOWN


class TestPhaseResult:
    def test_serialization_roundtrip(self, detector: PhaseDetector):
        """PhaseResult should serialize/deserialize cleanly."""
        series = _make_regime_series([(3, TrendDirection.BULLISH, 60)])
        ohlcv = _make_ohlcv(200, trend=0.002)
        result = detector.detect("TEST", ohlcv, series)
        json_str = result.model_dump_json()
        roundtrip = PhaseResult.model_validate_json(json_str)
        assert roundtrip.phase == result.phase
        assert roundtrip.ticker == result.ticker
        assert roundtrip.confidence == result.confidence

    def test_has_transitions(self, detector: PhaseDetector):
        """PhaseResult should include transition probabilities."""
        series = _make_regime_series([(3, TrendDirection.BULLISH, 60)])
        ohlcv = _make_ohlcv(200, trend=0.002)
        result = detector.detect("TEST", ohlcv, series)
        assert len(result.transitions) > 0
        total = sum(t.probability for t in result.transitions)
        assert abs(total - 1.0) < 0.05, f"Transition probs should sum to ~1.0, got {total}"

    def test_has_evidence(self, detector: PhaseDetector):
        """PhaseResult should include evidence."""
        series = _make_regime_series([(3, TrendDirection.BULLISH, 60)])
        ohlcv = _make_ohlcv(200, trend=0.002)
        result = detector.detect("TEST", ohlcv, series)
        assert result.evidence.regime_signal != ""
        assert result.evidence.volume_signal != ""

    def test_strategy_comment_set(self, detector: PhaseDetector):
        """PhaseResult should have a strategy comment."""
        series = _make_regime_series([(3, TrendDirection.BULLISH, 60)])
        ohlcv = _make_ohlcv(200, trend=0.002)
        result = detector.detect("TEST", ohlcv, series)
        assert result.strategy_comment != ""
        assert "LEAP" in result.strategy_comment

    def test_phase_name_matches(self, detector: PhaseDetector):
        """Phase name should match the configured name."""
        series = _make_regime_series([(3, TrendDirection.BULLISH, 60)])
        ohlcv = _make_ohlcv(200, trend=0.002)
        result = detector.detect("TEST", ohlcv, series)
        settings = get_settings()
        expected_name = settings.phases.names[int(result.phase)]
        assert result.phase_name == expected_name


class TestConfig:
    def test_config_loads_phase_settings(self):
        """Phase settings should load from defaults.yaml."""
        reset_settings()
        settings = get_settings()
        assert hasattr(settings, "phases")
        assert settings.phases.swing_lookback == 5
        assert settings.phases.min_phase_days == 10
        assert 1 in settings.phases.names
        assert settings.phases.names[1] == "Accumulation"
        assert 1 in settings.phases.strategies
        assert "LEAP" in settings.phases.strategies[1]

    def test_config_phase_colors(self):
        """Phase colors should be defined."""
        settings = get_settings()
        assert len(settings.phases.colors) == 4
        for i in range(1, 5):
            assert i in settings.phases.colors


class TestEmptyData:
    def test_empty_regime_series(self, detector: PhaseDetector):
        """Empty regime series returns low-confidence result."""
        series = RegimeTimeSeries(ticker="TEST", entries=[])
        ohlcv = _make_ohlcv(200)
        result = detector.detect("TEST", ohlcv, series)
        assert result.confidence == 0.10
