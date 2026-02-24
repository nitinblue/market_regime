"""YFinanceProvider: OHLCV data via yfinance."""

from __future__ import annotations

from datetime import timedelta

import yfinance as yf
import pandas as pd

from market_analyzer.data.exceptions import DataFetchError, InvalidTickerError
from market_analyzer.data.providers.base import DataProvider
from market_analyzer.models.data import DataRequest, DataType, ProviderType


class YFinanceProvider(DataProvider):
    """Fetches OHLCV data from Yahoo Finance."""

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.YFINANCE

    @property
    def supported_data_types(self) -> list[DataType]:
        return [DataType.OHLCV]

    def fetch(self, request: DataRequest) -> pd.DataFrame:
        """Fetch OHLCV data from yfinance.

        Returns DataFrame with columns [Open, High, Low, Close, Volume]
        and a DatetimeIndex sorted ascending. Raises DataFetchError on failure.
        """
        try:
            # yfinance end_date is exclusive — add 1 day to include it
            end = request.end_date + timedelta(days=1) if request.end_date else None
            df = yf.download(
                request.ticker,
                start=request.start_date,
                end=end,
                progress=False,
                auto_adjust=True,
            )
        except Exception as e:
            raise DataFetchError("yfinance", request.ticker, str(e)) from e

        if df is None or df.empty:
            raise DataFetchError(
                "yfinance", request.ticker, "No data returned (empty DataFrame)"
            )

        # yfinance may return MultiIndex columns for single ticker — flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure expected columns exist
        expected = {"Open", "High", "Low", "Close", "Volume"}
        missing = expected - set(df.columns)
        if missing:
            raise DataFetchError(
                "yfinance", request.ticker,
                f"Missing columns: {missing}"
            )

        # Keep only OHLCV columns, sorted ascending
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.DatetimeIndex(df.index)
        df.sort_index(inplace=True)

        # Drop rows with NaN in required columns
        df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

        if df.empty:
            raise DataFetchError(
                "yfinance", request.ticker,
                "All rows had NaN values after cleaning"
            )

        return df

    def validate_ticker(self, ticker: str) -> bool:
        """Check if ticker exists on Yahoo Finance."""
        try:
            info = yf.Ticker(ticker).info
            # yf returns a dict with 'regularMarketPrice' for valid tickers
            return info is not None and info.get("regularMarketPrice") is not None
        except Exception:
            return False
