"""DataService: orchestrates cache + providers for historical data."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from market_analyzer.data.cache.parquet_cache import ParquetCache, _SNAPSHOT_DATA_TYPES
from market_analyzer.data.exceptions import DataFetchError
from market_analyzer.data.providers.yfinance import YFinanceProvider
from market_analyzer.data.registry import ProviderRegistry
from market_analyzer.models.data import CacheMeta, DataRequest, DataResult, DataType, ProviderType


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

    # Default lookback when no start_date provided (2 years covers HMM training needs)
    DEFAULT_LOOKBACK_DAYS = 730

    def get(self, request: DataRequest) -> tuple[pd.DataFrame, DataResult]:
        """Get data (cache-first, delta-fetch if stale).

        Snapshot data types (e.g. OPTIONS_CHAIN) skip delta-fetch and date
        filtering — they are fully refreshed when stale.
        """
        provider = self._registry.resolve(request.ticker, request.data_type)
        is_snapshot = request.data_type in _SNAPSHOT_DATA_TYPES
        end_date = request.end_date or date.today()

        # Default to 2-year lookback if no start_date (time-series only)
        if not is_snapshot and request.start_date is None:
            request = DataRequest(
                ticker=request.ticker,
                data_type=request.data_type,
                start_date=end_date - timedelta(days=self.DEFAULT_LOOKBACK_DAYS),
                end_date=request.end_date,
            )

        # Check cache
        cached_df = self._cache.read(request.ticker, request.data_type)

        if cached_df is not None and not self._cache.is_stale(request.ticker, request.data_type):
            # Fresh cache
            if is_snapshot:
                df = cached_df
            else:
                df = self._filter_dates(cached_df, request.start_date, end_date)
            if df.empty:
                raise DataFetchError(
                    str(provider.provider_type), request.ticker,
                    "Cached data exists but no rows match the requested date range",
                )
            result = DataResult(
                ticker=request.ticker,
                data_type=request.data_type,
                provider=provider.provider_type,
                from_cache=True,
                date_range=self._date_range(df, is_snapshot),
                row_count=len(df),
            )
            return df, result

        # Need to fetch
        if not is_snapshot:
            # Time-series: try delta-fetch
            delta = self._cache.delta_dates(request.ticker, request.data_type, end_date)

            if cached_df is not None and delta is not None:
                fetch_request = DataRequest(
                    ticker=request.ticker,
                    data_type=request.data_type,
                    start_date=delta[0],
                    end_date=delta[1],
                )
                new_df = provider.fetch(fetch_request)
                merged = pd.concat([cached_df, new_df])
                merged = merged[~merged.index.duplicated(keep="last")]
                merged.sort_index(inplace=True)
                df = merged
            else:
                fetch_request = DataRequest(
                    ticker=request.ticker,
                    data_type=request.data_type,
                    start_date=request.start_date,
                    end_date=end_date,
                )
                df = provider.fetch(fetch_request)
        else:
            # Snapshot: always full refresh
            df = provider.fetch(request)

        if df.empty:
            raise DataFetchError(
                str(provider.provider_type), request.ticker,
                "Provider returned no data. Verify the ticker is valid and has trading history.",
            )

        # Write to cache
        meta = CacheMeta(
            ticker=request.ticker.upper(),
            data_type=request.data_type,
            provider=provider.provider_type,
            first_date=self._first_date(df, is_snapshot),
            last_date=self._last_date(df, is_snapshot),
            last_fetched=datetime.now(),
            row_count=len(df),
            file_path=self._cache._parquet_path(request.ticker, request.data_type),
        )
        self._cache.write(request.ticker, request.data_type, df, meta)

        # Apply date filter (time-series only)
        if is_snapshot:
            filtered = df
        else:
            filtered = self._filter_dates(df, request.start_date, end_date)
        if filtered.empty:
            raise DataFetchError(
                str(provider.provider_type), request.ticker,
                "Fetched data exists but no rows match the requested date range",
            )
        result = DataResult(
            ticker=request.ticker,
            data_type=request.data_type,
            provider=provider.provider_type,
            from_cache=False,
            date_range=self._date_range(filtered, is_snapshot),
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

    def get_options_chain(self, ticker: str) -> pd.DataFrame:
        """Convenience: fetch full options chain for a ticker (snapshot, cache-first)."""
        request = DataRequest(ticker=ticker, data_type=DataType.OPTIONS_CHAIN)
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
        """Filter DataFrame to requested date range (time-series only)."""
        if start_date is not None:
            df = df[df.index >= pd.Timestamp(start_date)]
        df = df[df.index <= pd.Timestamp(end_date)]
        return df

    @staticmethod
    def _date_range(df: pd.DataFrame, is_snapshot: bool) -> tuple[date, date]:
        """Extract (first, last) date from DataFrame."""
        if is_snapshot:
            today = date.today()
            return (today, today)
        return (df.index[0].date(), df.index[-1].date())

    @staticmethod
    def _first_date(df: pd.DataFrame, is_snapshot: bool) -> date:
        if is_snapshot:
            return date.today()
        return df.index[0].date()

    @staticmethod
    def _last_date(df: pd.DataFrame, is_snapshot: bool) -> date:
        if is_snapshot:
            return date.today()
        return df.index[-1].date()
