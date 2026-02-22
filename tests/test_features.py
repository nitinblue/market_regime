"""Tests for feature computation pipeline."""

import numpy as np
import pandas as pd
import pytest

from market_regime.features.pipeline import (
    _compute_raw_features,
    _normalize_features,
    _validate_ohlcv,
    compute_features,
    compute_features_with_inspection,
    FEATURE_NAMES,
)
from market_regime.models.features import FeatureConfig, FeatureInspection


class TestValidation:
    def test_missing_columns_raises(self):
        df = pd.DataFrame({"Close": [1, 2]}, index=pd.date_range("2024-01-01", periods=2))
        with pytest.raises(ValueError, match="missing columns"):
            _validate_ohlcv(df)

    def test_non_datetime_index_raises(self):
        df = pd.DataFrame(
            {"Open": [1], "High": [2], "Low": [0.5], "Close": [1.5], "Volume": [100]},
            index=[0],
        )
        with pytest.raises(ValueError, match="DatetimeIndex"):
            _validate_ohlcv(df)

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"],
            index=pd.DatetimeIndex([]),
        )
        with pytest.raises(ValueError, match="empty"):
            _validate_ohlcv(df)

    def test_valid_ohlcv_passes(self, sample_ohlcv_trending):
        _validate_ohlcv(sample_ohlcv_trending)  # Should not raise


class TestLogReturns:
    def test_log_return_1d_correctness(self):
        """Known prices -> expected log returns."""
        prices = [100.0, 105.0, 110.0, 108.0, 112.0]
        dates = pd.bdate_range("2024-01-01", periods=5)
        df = pd.DataFrame(
            {
                "Open": prices,
                "High": [p * 1.01 for p in prices],
                "Low": [p * 0.99 for p in prices],
                "Close": prices,
                "Volume": [1e6] * 5,
            },
            index=dates,
        )
        raw = _compute_raw_features(df, FeatureConfig())
        expected = np.log(np.array(prices[1:]) / np.array(prices[:-1]))
        actual = raw["log_return_1d"].dropna().values
        np.testing.assert_allclose(actual, expected, rtol=1e-10)

    def test_log_return_5d_correctness(self):
        """5-day log return uses shift(5)."""
        prices = list(range(100, 111))  # 11 prices
        dates = pd.bdate_range("2024-01-01", periods=11)
        df = pd.DataFrame(
            {
                "Open": prices,
                "High": [p + 1 for p in prices],
                "Low": [p - 1 for p in prices],
                "Close": [float(p) for p in prices],
                "Volume": [1e6] * 11,
            },
            index=dates,
        )
        raw = _compute_raw_features(df, FeatureConfig())
        # First non-NaN 5d return: log(105/100)
        first_valid = raw["log_return_5d"].dropna().iloc[0]
        assert abs(first_valid - np.log(105 / 100)) < 1e-10


class TestRealizedVol:
    def test_annualization(self, sample_ohlcv_trending):
        """Realized vol is annualized (multiplied by sqrt(252))."""
        config = FeatureConfig()
        raw = _compute_raw_features(sample_ohlcv_trending, config)
        log_ret = raw["log_return_1d"]
        rolling_std = log_ret.rolling(config.realized_vol_window).std()
        expected_annualized = rolling_std * np.sqrt(252)
        # Compare at a point where both are valid
        idx = 50
        np.testing.assert_allclose(
            raw["realized_vol"].iloc[idx],
            expected_annualized.iloc[idx],
            rtol=1e-10,
        )


class TestATRNormalized:
    def test_price_relative(self, sample_ohlcv_trending):
        """ATR normalized should be relative to price level."""
        raw = _compute_raw_features(sample_ohlcv_trending, FeatureConfig())
        # All values should be small fractions (< 1 for reasonable data)
        valid = raw["atr_normalized"].dropna()
        assert (valid > 0).all()
        assert (valid < 1).all()


class TestTrendStrength:
    def test_uptrend_positive(self, sample_ohlcv_trending):
        """Uptrending data should have mostly positive trend strength."""
        raw = _compute_raw_features(sample_ohlcv_trending, FeatureConfig())
        valid = raw["trend_strength"].dropna()
        # Most values should be positive for uptrend
        assert (valid > 0).mean() > 0.6

    def test_choppy_near_zero(self, sample_ohlcv_choppy):
        """Choppy data should have trend strength near zero on average."""
        raw = _compute_raw_features(sample_ohlcv_choppy, FeatureConfig())
        valid = raw["trend_strength"].dropna()
        assert abs(valid.mean()) < 0.01


class TestVolumeAnomaly:
    def test_scaling(self, sample_ohlcv_trending):
        """Volume anomaly should average around 1.0."""
        raw = _compute_raw_features(sample_ohlcv_trending, FeatureConfig())
        valid = raw["volume_anomaly"].dropna()
        assert 0.5 < valid.mean() < 1.5


class TestNormalization:
    def test_nan_warmup_dropped(self, sample_ohlcv_trending):
        """Normalized features should have no NaN rows."""
        features = compute_features(sample_ohlcv_trending)
        assert not features.isna().any().any()

    def test_fewer_rows_than_input(self, sample_ohlcv_trending):
        """Warmup rows are dropped during normalization."""
        features = compute_features(sample_ohlcv_trending)
        assert len(features) < len(sample_ohlcv_trending)

    def test_feature_columns(self, sample_ohlcv_trending):
        """Output has expected feature columns."""
        features = compute_features(sample_ohlcv_trending)
        assert list(features.columns) == FEATURE_NAMES


class TestInspection:
    def test_inspection_model(self, sample_ohlcv_trending):
        """Inspection returns correct metadata."""
        features, inspection = compute_features_with_inspection(
            sample_ohlcv_trending, "TEST"
        )
        assert isinstance(inspection, FeatureInspection)
        assert inspection.ticker == "TEST"
        assert inspection.normalized_row_count == len(features)
        assert inspection.feature_names == FEATURE_NAMES

    def test_inspection_normalization_params(self, sample_ohlcv_trending):
        """Inspection includes normalization means and stds."""
        _, inspection = compute_features_with_inspection(
            sample_ohlcv_trending, "TEST"
        )
        assert len(inspection.normalization_means) == 1
        assert len(inspection.normalization_stds) == 1
        # Each dict should have all feature names as keys
        means_keys = set(inspection.normalization_means[0].keys())
        assert means_keys == set(FEATURE_NAMES)
