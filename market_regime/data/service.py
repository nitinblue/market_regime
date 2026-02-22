"""DataService: orchestrates cache + providers for historical data."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from market_regime.data.cache.parquet_cache import ParquetCache
from market_regime.data.exceptions import DataFetchError
from market_regime.data.providers.yfinance import YFinanceProvider
from market_regime.data.registry import ProviderRegistry
from market_regime.models.data import CacheMeta, DataRequest, DataResult, DataType, ProviderType


class DataService:
    """Entry point for all historical data access.

    Cache-first: checks parquet cache, delta-fetches only missing dates.
    """

    def __init__(
        self,
        cache: ParquetCache | None = None,
        registry: ProviderRegistry | None = None,
    ) -> None:
        self._cache = cache or ParquetCache()
        self._registry = registry or self._default_registry()

    @staticmethod
    def _default_registry() -> ProviderRegistry:
        reg = ProviderRegistry()
        reg.register(YFinanceProvider())
        return reg

    def get(self, request: DataRequest) -> tuple[pd.DataFrame, DataResult]:
        """Get data (cache-first, delta-fetch if stale)."""
        provider = self._registry.resolve(request.ticker, request.data_type)
        end_date = request.end_date or date.today()

        # Check cache
        cached_df = self._cache.read(request.ticker, request.data_type)

        if cached_df is not None and not self._cache.is_stale(request.ticker, request.data_type):
            # Fresh cache — apply date filter and return
            df = self._filter_dates(cached_df, request.start_date, end_date)
            result = DataResult(
                ticker=request.ticker,
                data_type=request.data_type,
                provider=provider.provider_type,
                from_cache=True,
                date_range=(df.index[0].date(), df.index[-1].date()),
                row_count=len(df),
            )
            return df, result

        # Need to fetch — determine if delta or full
        delta = self._cache.delta_dates(request.ticker, request.data_type, end_date)

        if cached_df is not None and delta is not None:
            # Delta fetch — only missing dates
            fetch_request = DataRequest(
                ticker=request.ticker,
                data_type=request.data_type,
                start_date=delta[0],
                end_date=delta[1],
            )
            new_df = provider.fetch(fetch_request)
            # Merge: concat + deduplicate + sort
            merged = pd.concat([cached_df, new_df])
            merged = merged[~merged.index.duplicated(keep="last")]
            merged.sort_index(inplace=True)
            df = merged
        else:
            # Full fetch
            fetch_request = DataRequest(
                ticker=request.ticker,
                data_type=request.data_type,
                start_date=request.start_date,
                end_date=end_date,
            )
            df = provider.fetch(fetch_request)

        # Write to cache
        meta = CacheMeta(
            ticker=request.ticker.upper(),
            data_type=request.data_type,
            provider=provider.provider_type,
            first_date=df.index[0].date(),
            last_date=df.index[-1].date(),
            last_fetched=datetime.now(),
            row_count=len(df),
            file_path=self._cache._parquet_path(request.ticker, request.data_type),
        )
        self._cache.write(request.ticker, request.data_type, df, meta)

        # Apply date filter for the returned result
        filtered = self._filter_dates(df, request.start_date, end_date)
        result = DataResult(
            ticker=request.ticker,
            data_type=request.data_type,
            provider=provider.provider_type,
            from_cache=False,
            date_range=(filtered.index[0].date(), filtered.index[-1].date()),
            row_count=len(filtered),
        )
        return filtered, result

    def get_ohlcv(
        self,
        ticker: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Convenience: fetch OHLCV data for a ticker."""
        request = DataRequest(
            ticker=ticker,
            data_type=DataType.OHLCV,
            start_date=start_date,
            end_date=end_date,
        )
        df, _ = self.get(request)
        return df

    def get_options_iv(
        self,
        ticker: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Convenience: fetch options/IV data for a ticker."""
        request = DataRequest(
            ticker=ticker,
            data_type=DataType.OPTIONS_IV,
            start_date=start_date,
            end_date=end_date,
        )
        df, _ = self.get(request)
        return df

    def cache_status(
        self, ticker: str, data_type: DataType | None = None
    ) -> list[CacheMeta]:
        """Check what's cached for a ticker."""
        types = [data_type] if data_type else list(DataType)
        results = []
        for dt in types:
            meta = self._cache.get_meta(ticker, dt)
            if meta is not None:
                results.append(meta)
        return results

    def invalidate_cache(
        self, ticker: str, data_type: DataType | None = None
    ) -> None:
        """Force re-fetch on next request."""
        self._cache.invalidate(ticker, data_type)

    @staticmethod
    def _filter_dates(
        df: pd.DataFrame, start_date: date | None, end_date: date
    ) -> pd.DataFrame:
        """Filter DataFrame to requested date range."""
        if start_date is not None:
            df = df[df.index >= pd.Timestamp(start_date)]
        df = df[df.index <= pd.Timestamp(end_date)]
        return df
