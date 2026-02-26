"""YFinanceProvider: OHLCV data via yfinance."""

from __future__ import annotations

from datetime import timedelta

import yfinance as yf
import pandas as pd

from market_analyzer.data.exceptions import DataFetchError, InvalidTickerError
from market_analyzer.data.providers.base import DataProvider
from market_analyzer.models.data import DataRequest, DataType, ProviderType


# Aliases for tickers whose yfinance symbol differs from the common name.
# Keys: user-facing ticker.  Values: yfinance symbol.
_YFINANCE_ALIASES: dict[str, str] = {
    "SPX":  "^GSPC",   # S&P 500 Index
    "NDX":  "^NDX",    # Nasdaq-100 Index
    "DJX":  "^DJI",    # Dow Jones Industrial Average
    "RUT":  "^RUT",    # Russell 2000 Index
    "COMP": "^IXIC",   # Nasdaq Composite
    "SOX":  "^SOX",    # PHLX Semiconductor Index
    "VIX":  "^VIX",    # CBOE Volatility Index
    "TNX":  "^TNX",    # 10-Year Treasury Yield
    "OEX":  "^OEX",    # S&P 100 Index
    "XSP":  "^GSPC",   # Mini-SPX (same underlying as SPX)
}


class YFinanceProvider(DataProvider):
    """Fetches OHLCV and options chain data from Yahoo Finance."""

    @staticmethod
    def _resolve_ticker(ticker: str) -> str:
        """Translate user-facing ticker to yfinance symbol."""
        return _YFINANCE_ALIASES.get(ticker.upper(), ticker)

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.YFINANCE

    @property
    def supported_data_types(self) -> list[DataType]:
        return [DataType.OHLCV, DataType.OPTIONS_CHAIN]

    def fetch(self, request: DataRequest) -> pd.DataFrame:
        if request.data_type == DataType.OPTIONS_CHAIN:
            return self._fetch_options_chain(request)
        return self._fetch_ohlcv(request)

    def _fetch_options_chain(self, request: DataRequest) -> pd.DataFrame:
        """Fetch full options chain from yfinance.

        Returns DataFrame with columns:
            expiration(date), strike(float), option_type(str: "call"/"put"),
            bid(float), ask(float), last_price(float), volume(int),
            open_interest(int), implied_volatility(float), in_the_money(bool)
        Index: RangeIndex (one row per strike × option_type × expiration).
        """
        try:
            yf_symbol = self._resolve_ticker(request.ticker)
            ticker_obj = yf.Ticker(yf_symbol)
            expirations = ticker_obj.options
        except Exception as e:
            raise DataFetchError("yfinance", request.ticker, f"Failed to get options expirations: {e}") from e

        if not expirations:
            raise DataFetchError("yfinance", request.ticker, "No options expirations available")

        all_rows: list[pd.DataFrame] = []
        for exp_str in expirations:
            try:
                chain = ticker_obj.option_chain(exp_str)
            except Exception:
                continue  # Skip expirations that fail

            for opt_type, df_raw in [("call", chain.calls), ("put", chain.puts)]:
                if df_raw is None or df_raw.empty:
                    continue
                chunk = pd.DataFrame({
                    "expiration": pd.Timestamp(exp_str).date(),
                    "strike": df_raw["strike"].values,
                    "option_type": opt_type,
                    "bid": df_raw["bid"].values,
                    "ask": df_raw["ask"].values,
                    "last_price": df_raw["lastPrice"].values,
                    "volume": df_raw["volume"].fillna(0).astype(int).values,
                    "open_interest": df_raw["openInterest"].fillna(0).astype(int).values,
                    "implied_volatility": df_raw["impliedVolatility"].values,
                    "in_the_money": df_raw["inTheMoney"].values,
                })
                all_rows.append(chunk)

        if not all_rows:
            raise DataFetchError("yfinance", request.ticker, "No options chain data returned")

        result = pd.concat(all_rows, ignore_index=True)
        return result

    def _fetch_ohlcv(self, request: DataRequest) -> pd.DataFrame:
        """Fetch OHLCV data from yfinance.

        Returns DataFrame with columns [Open, High, Low, Close, Volume]
        and a DatetimeIndex sorted ascending. Raises DataFetchError on failure.
        """
        try:
            # yfinance end_date is exclusive — add 1 day to include it
            yf_symbol = self._resolve_ticker(request.ticker)
            end = request.end_date + timedelta(days=1) if request.end_date else None
            df = yf.download(
                yf_symbol,
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
            yf_symbol = self._resolve_ticker(ticker)
            info = yf.Ticker(yf_symbol).info
            # yf returns a dict with 'regularMarketPrice' for valid tickers
            return info is not None and info.get("regularMarketPrice") is not None
        except Exception:
            return False
