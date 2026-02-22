"""Tests for ParquetCache."""

from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from market_analyzer.data.cache.parquet_cache import ParquetCache
from market_analyzer.models.data import CacheMeta, DataType, ProviderType


@pytest.fixture
def cache(tmp_path: Path) -> ParquetCache:
    return ParquetCache(cache_dir=tmp_path, staleness_hours=18.0)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", periods=5, freq="B")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        },
        index=dates,
    )


def _make_meta(cache: ParquetCache, ticker: str = "SPY", first: str = "2025-01-02", last: str = "2025-01-08") -> CacheMeta:
    return CacheMeta(
        ticker=ticker.upper(),
        data_type=DataType.OHLCV,
        provider=ProviderType.YFINANCE,
        first_date=date.fromisoformat(first),
        last_date=date.fromisoformat(last),
        last_fetched=datetime.now(),
        row_count=5,
        file_path=cache._parquet_path(ticker, DataType.OHLCV),
    )


class TestWriteReadRoundtrip:
    def test_write_and_read(self, cache: ParquetCache, sample_df: pd.DataFrame) -> None:
        meta = _make_meta(cache)
        cache.write("SPY", DataType.OHLCV, sample_df, meta)

        result = cache.read("SPY", DataType.OHLCV)
        assert result is not None
        # Parquet doesn't preserve DatetimeIndex frequency, so check_freq=False
        pd.testing.assert_frame_equal(result, sample_df, check_freq=False)

    def test_read_cache_miss(self, cache: ParquetCache) -> None:
        result = cache.read("AAPL", DataType.OHLCV)
        assert result is None

    def test_meta_persisted(self, cache: ParquetCache, sample_df: pd.DataFrame) -> None:
        meta = _make_meta(cache)
        cache.write("SPY", DataType.OHLCV, sample_df, meta)

        loaded = cache.get_meta("SPY", DataType.OHLCV)
        assert loaded is not None
        assert loaded.ticker == "SPY"
        assert loaded.row_count == 5
        assert loaded.first_date == date(2025, 1, 2)

    def test_parquet_file_created(self, cache: ParquetCache, sample_df: pd.DataFrame) -> None:
        meta = _make_meta(cache)
        cache.write("SPY", DataType.OHLCV, sample_df, meta)

        path = cache._parquet_path("SPY", DataType.OHLCV)
        assert path.exists()


class TestStaleness:
    def test_no_cache_is_stale(self, cache: ParquetCache) -> None:
        assert cache.is_stale("SPY", DataType.OHLCV) is True

    def test_fresh_cache(self, cache: ParquetCache, sample_df: pd.DataFrame) -> None:
        meta = _make_meta(cache)
        cache.write("SPY", DataType.OHLCV, sample_df, meta)

        assert cache.is_stale("SPY", DataType.OHLCV) is False

    def test_old_cache_is_stale(self, cache: ParquetCache, sample_df: pd.DataFrame) -> None:
        meta = _make_meta(cache)
        meta.last_fetched = datetime.now() - timedelta(hours=24)
        cache.write("SPY", DataType.OHLCV, sample_df, meta)

        assert cache.is_stale("SPY", DataType.OHLCV) is True

    def test_weekend_freshness_saturday(self, cache: ParquetCache, sample_df: pd.DataFrame) -> None:
        """On Saturday, Friday's data should be fresh even if >18h old."""
        friday = date(2025, 1, 3)  # A Friday
        saturday = date(2025, 1, 4)

        meta = _make_meta(cache, last=friday.isoformat())
        meta.last_fetched = datetime(2025, 1, 3, 16, 0, 0)  # Friday 4pm
        cache.write("SPY", DataType.OHLCV, sample_df, meta)

        with patch("market_analyzer.data.cache.parquet_cache.date") as mock_date:
            mock_date.today.return_value = saturday
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            assert cache.is_stale("SPY", DataType.OHLCV) is False

    def test_weekend_freshness_sunday(self, cache: ParquetCache, sample_df: pd.DataFrame) -> None:
        """On Sunday, Friday's data should be fresh even if >18h old."""
        friday = date(2025, 1, 3)
        sunday = date(2025, 1, 5)

        meta = _make_meta(cache, last=friday.isoformat())
        meta.last_fetched = datetime(2025, 1, 3, 16, 0, 0)
        cache.write("SPY", DataType.OHLCV, sample_df, meta)

        with patch("market_analyzer.data.cache.parquet_cache.date") as mock_date:
            mock_date.today.return_value = sunday
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            assert cache.is_stale("SPY", DataType.OHLCV) is False


class TestDeltaDates:
    def test_fresh_cache_returns_none(self, cache: ParquetCache, sample_df: pd.DataFrame) -> None:
        meta = _make_meta(cache)
        cache.write("SPY", DataType.OHLCV, sample_df, meta)

        result = cache.delta_dates("SPY", DataType.OHLCV, date.today())
        assert result is None

    def test_stale_cache_returns_delta(self, cache: ParquetCache, sample_df: pd.DataFrame) -> None:
        meta = _make_meta(cache, last="2025-01-08")
        meta.last_fetched = datetime.now() - timedelta(hours=24)
        cache.write("SPY", DataType.OHLCV, sample_df, meta)

        end = date(2025, 1, 15)
        result = cache.delta_dates("SPY", DataType.OHLCV, end)
        assert result == (date(2025, 1, 9), date(2025, 1, 15))

    def test_no_cache_returns_none(self, cache: ParquetCache) -> None:
        result = cache.delta_dates("SPY", DataType.OHLCV, date.today())
        assert result is None


class TestInvalidate:
    def test_invalidate_removes_file_and_meta(self, cache: ParquetCache, sample_df: pd.DataFrame) -> None:
        meta = _make_meta(cache)
        cache.write("SPY", DataType.OHLCV, sample_df, meta)

        cache.invalidate("SPY", DataType.OHLCV)

        assert cache.read("SPY", DataType.OHLCV) is None
        assert cache.get_meta("SPY", DataType.OHLCV) is None

    def test_invalidate_all_types(self, cache: ParquetCache, sample_df: pd.DataFrame) -> None:
        meta = _make_meta(cache)
        cache.write("SPY", DataType.OHLCV, sample_df, meta)

        cache.invalidate("SPY")  # No data_type = all types

        assert cache.read("SPY", DataType.OHLCV) is None
        assert cache.get_meta("SPY", DataType.OHLCV) is None

    def test_invalidate_nonexistent_is_noop(self, cache: ParquetCache) -> None:
        cache.invalidate("AAPL", DataType.OHLCV)  # Should not raise
