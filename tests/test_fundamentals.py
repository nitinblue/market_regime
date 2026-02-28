"""Tests for stock fundamentals module — mocks yfinance."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from market_analyzer.fundamentals.fetch import (
    _safe_get,
    _build_52week,
    _get_asset_type,
    fetch_fundamentals,
    invalidate_fundamentals_cache,
    _cache,
)
from market_analyzer.models.fundamentals import FundamentalsSnapshot


class TestSafeGet:
    def test_normal_value(self):
        assert _safe_get({"x": 42.0}, "x") == 42.0

    def test_int_value(self):
        assert _safe_get({"x": 10}, "x") == 10.0

    def test_nan_returns_none(self):
        assert _safe_get({"x": float("nan")}, "x") is None

    def test_inf_returns_none(self):
        assert _safe_get({"x": float("inf")}, "x") is None

    def test_neg_inf_returns_none(self):
        assert _safe_get({"x": float("-inf")}, "x") is None

    def test_missing_key_returns_none(self):
        assert _safe_get({}, "x") is None

    def test_none_value_returns_none(self):
        assert _safe_get({"x": None}, "x") is None

    def test_string_value_returns_none(self):
        assert _safe_get({"x": "not a number"}, "x") is None

    def test_string_number_returns_float(self):
        assert _safe_get({"x": "3.14"}, "x") == 3.14


class TestBuild52Week:
    def test_normal_calculation(self):
        info = {"fiftyTwoWeekHigh": 200.0, "fiftyTwoWeekLow": 100.0}
        result = _build_52week(info, 150.0)
        assert result.high == 200.0
        assert result.low == 100.0
        assert result.pct_from_high == pytest.approx(-25.0)
        assert result.pct_from_low == pytest.approx(50.0)

    def test_at_high(self):
        info = {"fiftyTwoWeekHigh": 200.0, "fiftyTwoWeekLow": 100.0}
        result = _build_52week(info, 200.0)
        assert result.pct_from_high == pytest.approx(0.0)
        assert result.pct_from_low == pytest.approx(100.0)

    def test_missing_high_low(self):
        result = _build_52week({}, 150.0)
        assert result.high is None
        assert result.low is None
        assert result.pct_from_high is None
        assert result.pct_from_low is None

    def test_no_price(self):
        info = {"fiftyTwoWeekHigh": 200.0, "fiftyTwoWeekLow": 100.0}
        result = _build_52week(info, None)
        assert result.pct_from_high is None
        assert result.pct_from_low is None


def _mock_info() -> dict:
    """Standard mock info dict mimicking yfinance Ticker.info."""
    return {
        "regularMarketPrice": 150.0,
        "longName": "Test Corp",
        "sector": "Technology",
        "industry": "Software",
        "beta": 1.2,
        "trailingPE": 25.0,
        "forwardPE": 20.0,
        "pegRatio": 1.5,
        "priceToBook": 8.0,
        "priceToSalesTrailing12Months": 6.0,
        "trailingEps": 6.0,
        "forwardEps": 7.5,
        "earningsGrowth": 0.15,
        "marketCap": 2_000_000_000_000,
        "totalRevenue": 400_000_000_000,
        "revenuePerShare": 25.0,
        "revenueGrowth": 0.08,
        "profitMargins": 0.25,
        "grossMargins": 0.45,
        "operatingMargins": 0.30,
        "ebitdaMargins": 0.35,
        "operatingCashflow": 120_000_000_000,
        "freeCashflow": 100_000_000_000,
        "totalCash": 60_000_000_000,
        "totalCashPerShare": 3.75,
        "totalDebt": 110_000_000_000,
        "debtToEquity": 180.0,
        "currentRatio": 1.07,
        "returnOnAssets": 0.28,
        "returnOnEquity": 1.60,
        "dividendYield": 0.005,
        "dividendRate": 1.0,
        "fiftyTwoWeekHigh": 200.0,
        "fiftyTwoWeekLow": 120.0,
    }


def _mock_ticker(info: dict | None = None) -> MagicMock:
    """Create a mock yfinance.Ticker."""
    mock = MagicMock()
    mock.info = info or _mock_info()
    mock.get_earnings_dates.return_value = pd.DataFrame(
        {
            "EPS Estimate": [1.50, 1.40],
            "Reported EPS": [1.55, 1.42],
            "Surprise(%)": [3.3, 1.4],
        },
        index=pd.to_datetime(["2026-01-15", "2025-10-15"]),
    )
    mock.calendar = {"Earnings Date": [pd.Timestamp("2026-04-20")]}
    return mock


class TestFetchFundamentals:
    def setup_method(self):
        """Clear cache before each test."""
        _cache.clear()

    @patch("market_analyzer.fundamentals.fetch.yf.Ticker")
    def test_returns_snapshot(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _mock_ticker()
        result = fetch_fundamentals("AAPL", ttl_minutes=60)

        assert isinstance(result, FundamentalsSnapshot)
        assert result.ticker == "AAPL"
        assert result.business.long_name == "Test Corp"
        assert result.valuation.trailing_pe == 25.0
        assert result.earnings.trailing_eps == 6.0

    @patch("market_analyzer.fundamentals.fetch.yf.Ticker")
    def test_cache_hit(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _mock_ticker()

        # First call
        fetch_fundamentals("AAPL", ttl_minutes=60)
        # Second call should use cache
        fetch_fundamentals("AAPL", ttl_minutes=60)

        # yf.Ticker should only be called once
        assert mock_ticker_cls.call_count == 1

    @patch("market_analyzer.fundamentals.fetch.yf.Ticker")
    def test_cache_invalidation(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _mock_ticker()

        fetch_fundamentals("AAPL", ttl_minutes=60)
        invalidate_fundamentals_cache("AAPL")
        fetch_fundamentals("AAPL", ttl_minutes=60)

        assert mock_ticker_cls.call_count == 2

    @patch("market_analyzer.fundamentals.fetch.yf.Ticker")
    def test_invalidate_all(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _mock_ticker()

        fetch_fundamentals("AAPL", ttl_minutes=60)
        fetch_fundamentals("MSFT", ttl_minutes=60)
        invalidate_fundamentals_cache()  # clear all
        fetch_fundamentals("AAPL", ttl_minutes=60)

        assert mock_ticker_cls.call_count == 3

    @patch("market_analyzer.fundamentals.fetch.yf.Ticker")
    def test_52week_calculations(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _mock_ticker()
        result = fetch_fundamentals("AAPL", ttl_minutes=60)

        assert result.fifty_two_week.high == 200.0
        assert result.fifty_two_week.low == 120.0
        assert result.fifty_two_week.pct_from_high is not None
        assert result.fifty_two_week.pct_from_high < 0  # below 52w high
        assert result.fifty_two_week.pct_from_low is not None
        assert result.fifty_two_week.pct_from_low > 0  # above 52w low

    @patch("market_analyzer.fundamentals.fetch.yf.Ticker")
    def test_handles_missing_fields(self, mock_ticker_cls):
        """Sparse info dict should still produce a valid snapshot."""
        sparse_info = {"regularMarketPrice": 100.0}
        mock_ticker_cls.return_value = _mock_ticker(sparse_info)
        result = fetch_fundamentals("XYZ", ttl_minutes=60)

        assert result.ticker == "XYZ"
        assert result.valuation.trailing_pe is None
        assert result.business.long_name is None
        assert result.fifty_two_week.high is None

    @patch("market_analyzer.fundamentals.fetch.yf.Ticker")
    def test_invalid_ticker_raises(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _mock_ticker({"regularMarketPrice": None})
        with pytest.raises(ValueError, match="No data"):
            fetch_fundamentals("INVALID", ttl_minutes=60)

    @patch("market_analyzer.fundamentals.fetch.yf.Ticker")
    def test_case_normalization(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _mock_ticker()
        result = fetch_fundamentals("aapl", ttl_minutes=60)
        assert result.ticker == "AAPL"

    @patch("market_analyzer.fundamentals.fetch.yf.Ticker")
    def test_recent_earnings_parsed(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _mock_ticker()
        result = fetch_fundamentals("AAPL", ttl_minutes=60)
        assert len(result.recent_earnings) == 2
        assert result.recent_earnings[0].eps_actual == 1.55

    @patch("market_analyzer.fundamentals.fetch.yf.Ticker")
    def test_upcoming_events_parsed(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _mock_ticker()
        result = fetch_fundamentals("AAPL", ttl_minutes=60)
        assert result.upcoming_events.next_earnings_date == date(2026, 4, 20)


# --- Asset type detection ---

class TestGetAssetType:
    def test_equity_from_quote_type(self):
        assert _get_asset_type({"quoteType": "EQUITY"}) == "EQUITY"

    def test_etf_from_quote_type(self):
        assert _get_asset_type({"quoteType": "ETF"}) == "ETF"

    def test_index_from_quote_type(self):
        assert _get_asset_type({"quoteType": "INDEX"}) == "INDEX"

    def test_mutual_fund(self):
        assert _get_asset_type({"quoteType": "MUTUALFUND"}) == "MUTUALFUND"

    def test_crypto(self):
        assert _get_asset_type({"quoteType": "CRYPTOCURRENCY"}) == "CRYPTOCURRENCY"

    def test_case_insensitive(self):
        assert _get_asset_type({"quoteType": "etf"}) == "ETF"

    def test_missing_quote_type_with_sector_is_equity(self):
        assert _get_asset_type({"sector": "Technology"}) == "EQUITY"

    def test_missing_quote_type_with_total_assets_is_etf(self):
        assert _get_asset_type({"totalAssets": 50_000_000}) == "ETF"

    def test_missing_quote_type_empty_info_is_equity(self):
        assert _get_asset_type({}) == "EQUITY"


# --- ETF boundary: no earnings calls ---

def _mock_etf_info() -> dict:
    """ETF info dict — has quoteType=ETF, no sector/industry."""
    return {
        "regularMarketPrice": 185.0,
        "quoteType": "ETF",
        "longName": "SPDR Gold Shares",
        "beta": 0.12,
        "fiftyTwoWeekHigh": 200.0,
        "fiftyTwoWeekLow": 150.0,
        "totalAssets": 50_000_000_000,
        "dividendYield": 0.0,
    }


def _mock_etf_ticker() -> MagicMock:
    """Mock yfinance.Ticker for an ETF — earnings calls should NOT be made."""
    mock = MagicMock()
    mock.info = _mock_etf_info()
    # These should NOT be called for ETFs
    mock.get_earnings_dates.side_effect = AssertionError(
        "get_earnings_dates should not be called for ETFs"
    )
    mock.calendar = None  # sentinel
    return mock


class TestETFBoundary:
    def setup_method(self):
        _cache.clear()

    @patch("market_analyzer.fundamentals.fetch.yf.Ticker")
    def test_etf_skips_earnings_calls(self, mock_ticker_cls):
        """ETF should never call get_earnings_dates() or .calendar."""
        mock = _mock_etf_ticker()
        mock_ticker_cls.return_value = mock

        result = fetch_fundamentals("GLD", ttl_minutes=60)

        assert result.asset_type == "ETF"
        assert result.recent_earnings == []
        assert result.upcoming_events.days_to_earnings is None
        # get_earnings_dates should not have been called
        mock.get_earnings_dates.assert_not_called()

    @patch("market_analyzer.fundamentals.fetch.yf.Ticker")
    def test_etf_has_basic_fields(self, mock_ticker_cls):
        """ETF returns 52-week, beta, dividends — no P/E or EPS."""
        mock_ticker_cls.return_value = _mock_etf_ticker()
        result = fetch_fundamentals("GLD", ttl_minutes=60)

        assert result.business.long_name == "SPDR Gold Shares"
        assert result.business.beta == 0.12
        assert result.fifty_two_week.high == 200.0
        assert result.valuation.trailing_pe is None
        assert result.earnings.trailing_eps is None

    @patch("market_analyzer.fundamentals.fetch.yf.Ticker")
    def test_etf_is_equity_false(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _mock_etf_ticker()
        result = fetch_fundamentals("GLD", ttl_minutes=60)
        assert not result.is_equity
        assert not result.has_earnings

    @patch("market_analyzer.fundamentals.fetch.yf.Ticker")
    def test_equity_is_equity_true(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _mock_ticker()
        result = fetch_fundamentals("AAPL", ttl_minutes=60)
        assert result.is_equity
        assert result.asset_type == "EQUITY"

    @patch("market_analyzer.fundamentals.fetch.yf.Ticker")
    def test_index_skips_earnings(self, mock_ticker_cls):
        """Index tickers skip earnings calls."""
        mock = MagicMock()
        mock.info = {
            "regularMarketPrice": 5800.0,
            "quoteType": "INDEX",
            "longName": "S&P 500",
        }
        mock.get_earnings_dates.side_effect = AssertionError("should not call")
        mock_ticker_cls.return_value = mock

        result = fetch_fundamentals("SPX", ttl_minutes=60)
        assert result.asset_type == "INDEX"
        assert result.recent_earnings == []
        mock.get_earnings_dates.assert_not_called()

    @patch("market_analyzer.fundamentals.fetch.yf.Ticker")
    def test_mutualfund_skips_earnings(self, mock_ticker_cls):
        mock = MagicMock()
        mock.info = {
            "regularMarketPrice": 100.0,
            "quoteType": "MUTUALFUND",
            "longName": "Vanguard 500",
        }
        mock.get_earnings_dates.side_effect = AssertionError("should not call")
        mock_ticker_cls.return_value = mock

        result = fetch_fundamentals("VFIAX", ttl_minutes=60)
        assert result.asset_type == "MUTUALFUND"
        mock.get_earnings_dates.assert_not_called()

    @patch("market_analyzer.fundamentals.fetch.yf.Ticker")
    def test_no_market_price_raises(self, mock_ticker_cls):
        """Tickers with no price at all should raise ValueError."""
        mock = MagicMock()
        mock.info = {"quoteType": "ETF"}  # no regularMarketPrice
        mock_ticker_cls.return_value = mock

        with pytest.raises(ValueError, match="No data"):
            fetch_fundamentals("FAKE", ttl_minutes=60)

    @patch("market_analyzer.fundamentals.fetch.yf.Ticker")
    def test_info_fetch_failure_raises(self, mock_ticker_cls):
        """Network error on ticker.info raises ValueError."""
        mock = MagicMock()
        mock.info.__getitem__ = MagicMock(side_effect=ConnectionError("network"))
        type(mock).info = property(lambda s: (_ for _ in ()).throw(ConnectionError("network")))
        mock_ticker_cls.return_value = mock

        with pytest.raises(ValueError, match="Failed to fetch"):
            fetch_fundamentals("SPY", ttl_minutes=60)
