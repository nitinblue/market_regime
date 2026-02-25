"""Tests for YFinanceProvider OPTIONS_CHAIN fetch (mocked yfinance)."""

from datetime import date
from unittest.mock import MagicMock, PropertyMock, patch

import pandas as pd
import pytest

from market_analyzer.data.exceptions import DataFetchError
from market_analyzer.data.providers.yfinance import YFinanceProvider
from market_analyzer.models.data import DataRequest, DataType


@pytest.fixture
def provider() -> YFinanceProvider:
    return YFinanceProvider()


@pytest.fixture
def mock_calls_df() -> pd.DataFrame:
    return pd.DataFrame({
        "strike": [580.0, 585.0, 590.0],
        "bid": [12.0, 8.5, 5.0],
        "ask": [12.5, 9.0, 5.5],
        "lastPrice": [12.25, 8.75, 5.25],
        "volume": [1500, 2000, 3000],
        "openInterest": [10000, 15000, 20000],
        "impliedVolatility": [0.18, 0.17, 0.16],
        "inTheMoney": [True, True, False],
    })


@pytest.fixture
def mock_puts_df() -> pd.DataFrame:
    return pd.DataFrame({
        "strike": [575.0, 580.0, 585.0],
        "bid": [4.0, 7.0, 11.0],
        "ask": [4.5, 7.5, 11.5],
        "lastPrice": [4.25, 7.25, 11.25],
        "volume": [800, 1200, 900],
        "openInterest": [5000, 8000, 6000],
        "impliedVolatility": [0.19, 0.18, 0.17],
        "inTheMoney": [False, False, True],
    })


class TestOptionsChainFetch:
    def test_fetch_returns_expected_columns(
        self, provider: YFinanceProvider, mock_calls_df: pd.DataFrame, mock_puts_df: pd.DataFrame
    ) -> None:
        request = DataRequest(ticker="SPY", data_type=DataType.OPTIONS_CHAIN)

        mock_ticker = MagicMock()
        mock_ticker.options = ("2026-03-20",)
        chain = MagicMock()
        chain.calls = mock_calls_df
        chain.puts = mock_puts_df
        mock_ticker.option_chain.return_value = chain

        with patch("market_analyzer.data.providers.yfinance.yf.Ticker", return_value=mock_ticker):
            result = provider.fetch(request)

        expected_cols = [
            "expiration", "strike", "option_type", "bid", "ask",
            "last_price", "volume", "open_interest", "implied_volatility", "in_the_money",
        ]
        assert list(result.columns) == expected_cols
        # 3 calls + 3 puts = 6 rows
        assert len(result) == 6

    def test_fetch_has_correct_option_types(
        self, provider: YFinanceProvider, mock_calls_df: pd.DataFrame, mock_puts_df: pd.DataFrame
    ) -> None:
        request = DataRequest(ticker="SPY", data_type=DataType.OPTIONS_CHAIN)

        mock_ticker = MagicMock()
        mock_ticker.options = ("2026-03-20",)
        chain = MagicMock()
        chain.calls = mock_calls_df
        chain.puts = mock_puts_df
        mock_ticker.option_chain.return_value = chain

        with patch("market_analyzer.data.providers.yfinance.yf.Ticker", return_value=mock_ticker):
            result = provider.fetch(request)

        assert set(result["option_type"].unique()) == {"call", "put"}
        assert len(result[result["option_type"] == "call"]) == 3
        assert len(result[result["option_type"] == "put"]) == 3

    def test_fetch_multiple_expirations(
        self, provider: YFinanceProvider, mock_calls_df: pd.DataFrame, mock_puts_df: pd.DataFrame
    ) -> None:
        request = DataRequest(ticker="SPY", data_type=DataType.OPTIONS_CHAIN)

        mock_ticker = MagicMock()
        mock_ticker.options = ("2026-03-20", "2026-04-17")
        chain = MagicMock()
        chain.calls = mock_calls_df
        chain.puts = mock_puts_df
        mock_ticker.option_chain.return_value = chain

        with patch("market_analyzer.data.providers.yfinance.yf.Ticker", return_value=mock_ticker):
            result = provider.fetch(request)

        # 2 expirations × (3 calls + 3 puts) = 12 rows
        assert len(result) == 12
        assert len(result["expiration"].unique()) == 2

    def test_fetch_no_expirations_raises(self, provider: YFinanceProvider) -> None:
        request = DataRequest(ticker="SPY", data_type=DataType.OPTIONS_CHAIN)

        mock_ticker = MagicMock()
        mock_ticker.options = ()

        with patch("market_analyzer.data.providers.yfinance.yf.Ticker", return_value=mock_ticker):
            with pytest.raises(DataFetchError, match="No options expirations"):
                provider.fetch(request)

    def test_fetch_exception_raises(self, provider: YFinanceProvider) -> None:
        request = DataRequest(ticker="SPY", data_type=DataType.OPTIONS_CHAIN)

        with patch("market_analyzer.data.providers.yfinance.yf.Ticker", side_effect=Exception("network")):
            with pytest.raises(DataFetchError, match="Failed to get options expirations"):
                provider.fetch(request)

    def test_fetch_handles_nan_volume_oi(self, provider: YFinanceProvider) -> None:
        """NaN in volume/openInterest should be filled with 0."""
        request = DataRequest(ticker="SPY", data_type=DataType.OPTIONS_CHAIN)

        calls = pd.DataFrame({
            "strike": [580.0],
            "bid": [12.0],
            "ask": [12.5],
            "lastPrice": [12.25],
            "volume": [float("nan")],
            "openInterest": [float("nan")],
            "impliedVolatility": [0.18],
            "inTheMoney": [True],
        })

        mock_ticker = MagicMock()
        mock_ticker.options = ("2026-03-20",)
        chain = MagicMock()
        chain.calls = calls
        chain.puts = pd.DataFrame()  # Empty puts
        mock_ticker.option_chain.return_value = chain

        with patch("market_analyzer.data.providers.yfinance.yf.Ticker", return_value=mock_ticker):
            result = provider.fetch(request)

        assert result["volume"].iloc[0] == 0
        assert result["open_interest"].iloc[0] == 0

    def test_fetch_skips_failed_expiration(
        self, provider: YFinanceProvider, mock_calls_df: pd.DataFrame, mock_puts_df: pd.DataFrame
    ) -> None:
        """If one expiration fails, others should still be fetched."""
        request = DataRequest(ticker="SPY", data_type=DataType.OPTIONS_CHAIN)

        mock_ticker = MagicMock()
        mock_ticker.options = ("2026-03-20", "2026-04-17")

        chain_ok = MagicMock()
        chain_ok.calls = mock_calls_df
        chain_ok.puts = mock_puts_df

        call_count = 0

        def side_effect(exp_str):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("bad expiration")
            return chain_ok

        mock_ticker.option_chain.side_effect = side_effect

        with patch("market_analyzer.data.providers.yfinance.yf.Ticker", return_value=mock_ticker):
            result = provider.fetch(request)

        # Only second expiration succeeded: 3 calls + 3 puts
        assert len(result) == 6

    def test_ohlcv_still_works(self, provider: YFinanceProvider) -> None:
        """Ensure OHLCV fetch is not broken by OPTIONS_CHAIN addition."""
        dates = pd.date_range("2025-01-02", periods=3, freq="B")
        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [101.0, 102.0, 103.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [100.5, 101.5, 102.5],
                "Volume": [1000, 1100, 1200],
            },
            index=dates,
        )

        request = DataRequest(ticker="SPY", data_type=DataType.OHLCV)
        with patch("market_analyzer.data.providers.yfinance.yf.download", return_value=df):
            result = provider.fetch(request)

        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
