"""Tests for DataService (cache + provider orchestration)."""

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from market_regime.data.cache.parquet_cache import ParquetCache
from market_regime.data.exceptions import NoProviderError
from market_regime.data.providers.base import DataProvider
from market_regime.data.registry import ProviderRegistry
from market_regime.data.service import DataService
from market_regime.models.data import (
    DataRequest,
    DataType,
    ProviderType,
)


def _make_ohlcv(start: str, periods: int) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="B")
    return pd.DataFrame(
        {
            "Open": range(100, 100 + periods),
            "High": range(101, 101 + periods),
            "Low": range(99, 99 + periods),
            "Close": [100.5 + i for i in range(periods)],
            "Volume": [1000 + i * 100 for i in range(periods)],
        },
        index=dates,
    )


class MockProvider(DataProvider):
    """Mock provider for testing."""

    def __init__(self) -> None:
        self.fetch_calls: list[DataRequest] = []

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.YFINANCE

    @property
    def supported_data_types(self) -> list[DataType]:
        return [DataType.OHLCV]

    def fetch(self, request: DataRequest) -> pd.DataFrame:
        self.fetch_calls.append(request)
        return _make_ohlcv(
            request.start_date.isoformat() if request.start_date else "2025-01-02",
            5,
        )

    def validate_ticker(self, ticker: str) -> bool:
        return True


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture
def service(tmp_path: Path, mock_provider: MockProvider) -> DataService:
    cache = ParquetCache(cache_dir=tmp_path, staleness_hours=18.0)
    registry = ProviderRegistry()
    registry.register(mock_provider)
    return DataService(cache=cache, registry=registry)


class TestCacheMissThenHit:
    def test_first_call_fetches(self, service: DataService, mock_provider: MockProvider) -> None:
        request = DataRequest(
            ticker="SPY",
            data_type=DataType.OHLCV,
            start_date=date(2025, 1, 2),
            end_date=date(2025, 1, 8),
        )
        df, result = service.get(request)

        assert not result.from_cache
        assert len(mock_provider.fetch_calls) == 1
        assert len(df) > 0

    def test_second_call_uses_cache(self, service: DataService, mock_provider: MockProvider) -> None:
        request = DataRequest(
            ticker="SPY",
            data_type=DataType.OHLCV,
            start_date=date(2025, 1, 2),
            end_date=date(2025, 1, 8),
        )
        service.get(request)
        df, result = service.get(request)

        assert result.from_cache
        assert len(mock_provider.fetch_calls) == 1  # Only one fetch


class TestDeltaFetch:
    def test_delta_fetch_merges(self, service: DataService, mock_provider: MockProvider) -> None:
        # First fetch
        request1 = DataRequest(
            ticker="SPY",
            data_type=DataType.OHLCV,
            start_date=date(2025, 1, 2),
            end_date=date(2025, 1, 8),
        )
        service.get(request1)

        # Make cache stale
        meta = service._cache.get_meta("SPY", DataType.OHLCV)
        meta.last_fetched = datetime.now() - timedelta(hours=24)
        all_meta = service._cache._load_meta()
        key = service._cache._meta_key("SPY", DataType.OHLCV)
        all_meta[key] = meta
        service._cache._save_meta(all_meta)

        # Second fetch should delta
        request2 = DataRequest(
            ticker="SPY",
            data_type=DataType.OHLCV,
            start_date=date(2025, 1, 2),
            end_date=date(2025, 1, 15),
        )
        df, result = service.get(request2)

        assert not result.from_cache
        assert len(mock_provider.fetch_calls) == 2
        # Delta fetch should have start = last_date + 1
        delta_call = mock_provider.fetch_calls[1]
        assert delta_call.start_date > date(2025, 1, 2)


class TestConvenienceMethods:
    def test_get_ohlcv(self, service: DataService, mock_provider: MockProvider) -> None:
        df = service.get_ohlcv("SPY", start_date=date(2025, 1, 2), end_date=date(2025, 1, 8))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]


class TestCacheStatus:
    def test_cache_status_empty(self, service: DataService) -> None:
        result = service.cache_status("SPY")
        assert result == []

    def test_cache_status_after_fetch(self, service: DataService) -> None:
        service.get_ohlcv("SPY", start_date=date(2025, 1, 2), end_date=date(2025, 1, 8))
        result = service.cache_status("SPY", DataType.OHLCV)
        assert len(result) == 1
        assert result[0].ticker == "SPY"


class TestInvalidateCache:
    def test_invalidate_forces_refetch(self, service: DataService, mock_provider: MockProvider) -> None:
        request = DataRequest(
            ticker="SPY",
            data_type=DataType.OHLCV,
            start_date=date(2025, 1, 2),
            end_date=date(2025, 1, 8),
        )
        service.get(request)
        service.invalidate_cache("SPY", DataType.OHLCV)
        df, result = service.get(request)

        assert not result.from_cache
        assert len(mock_provider.fetch_calls) == 2


class TestNoProvider:
    def test_missing_provider_raises(self, tmp_path: Path) -> None:
        cache = ParquetCache(cache_dir=tmp_path)
        registry = ProviderRegistry()
        svc = DataService(cache=cache, registry=registry)

        request = DataRequest(ticker="SPY", data_type=DataType.OHLCV)
        with pytest.raises(NoProviderError):
            svc.get(request)
