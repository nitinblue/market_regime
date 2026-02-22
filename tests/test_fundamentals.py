"""Tests for stock fundamentals module — mocks yfinance."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from market_regime.fundamentals.fetch import (
    _safe_get,
    _build_52week,
    fetch_fundamentals,
    invalidate_fundamentals_cache,
    _cache,
)
from market_regime.models.fundamentals import FundamentalsSnapshot


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

    @patch("market_regime.fundamentals.fetch.yf.Ticker")
    def test_returns_snapshot(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _mock_ticker()
        result = fetch_fundamentals("AAPL", ttl_minutes=60)

        assert isinstance(result, FundamentalsSnapshot)
        assert result.ticker == "AAPL"
        assert result.business.long_name == "Test Corp"
        assert result.valuation.trailing_pe == 25.0
        assert result.earnings.trailing_eps == 6.0

    @patch("market_regime.fundamentals.fetch.yf.Ticker")
    def test_cache_hit(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _mock_ticker()

        # First call
        fetch_fundamentals("AAPL", ttl_minutes=60)
        # Second call should use cache
        fetch_fundamentals("AAPL", ttl_minutes=60)

        # yf.Ticker should only be called once
        assert mock_ticker_cls.call_count == 1

    @patch("market_regime.fundamentals.fetch.yf.Ticker")
    def test_cache_invalidation(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _mock_ticker()

        fetch_fundamentals("AAPL", ttl_minutes=60)
        invalidate_fundamentals_cache("AAPL")
        fetch_fundamentals("AAPL", ttl_minutes=60)

        assert mock_ticker_cls.call_count == 2

    @patch("market_regime.fundamentals.fetch.yf.Ticker")
    def test_invalidate_all(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _mock_ticker()

        fetch_fundamentals("AAPL", ttl_minutes=60)
        fetch_fundamentals("MSFT", ttl_minutes=60)
        invalidate_fundamentals_cache()  # clear all
        fetch_fundamentals("AAPL", ttl_minutes=60)

        assert mock_ticker_cls.call_count == 3

    @patch("market_regime.fundamentals.fetch.yf.Ticker")
    def test_52week_calculations(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _mock_ticker()
        result = fetch_fundamentals("AAPL", ttl_minutes=60)

        assert result.fifty_two_week.high == 200.0
        assert result.fifty_two_week.low == 120.0
        assert result.fifty_two_week.pct_from_high is not None
        assert result.fifty_two_week.pct_from_high < 0  # below 52w high
        assert result.fifty_two_week.pct_from_low is not None
        assert result.fifty_two_week.pct_from_low > 0  # above 52w low

    @patch("market_regime.fundamentals.fetch.yf.Ticker")
    def test_handles_missing_fields(self, mock_ticker_cls):
        """Sparse info dict should still produce a valid snapshot."""
        sparse_info = {"regularMarketPrice": 100.0}
        mock_ticker_cls.return_value = _mock_ticker(sparse_info)
        result = fetch_fundamentals("XYZ", ttl_minutes=60)

        assert result.ticker == "XYZ"
        assert result.valuation.trailing_pe is None
        assert result.business.long_name is None
        assert result.fifty_two_week.high is None

    @patch("market_regime.fundamentals.fetch.yf.Ticker")
    def test_invalid_ticker_raises(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _mock_ticker({"regularMarketPrice": None})
        with pytest.raises(ValueError, match="No data"):
            fetch_fundamentals("INVALID", ttl_minutes=60)

    @patch("market_regime.fundamentals.fetch.yf.Ticker")
    def test_case_normalization(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _mock_ticker()
        result = fetch_fundamentals("aapl", ttl_minutes=60)
        assert result.ticker == "AAPL"

    @patch("market_regime.fundamentals.fetch.yf.Ticker")
    def test_recent_earnings_parsed(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _mock_ticker()
        result = fetch_fundamentals("AAPL", ttl_minutes=60)
        assert len(result.recent_earnings) == 2
        assert result.recent_earnings[0].eps_actual == 1.55

    @patch("market_regime.fundamentals.fetch.yf.Ticker")
    def test_upcoming_events_parsed(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _mock_ticker()
        result = fetch_fundamentals("AAPL", ttl_minutes=60)
        assert result.upcoming_events.next_earnings_date == date(2026, 4, 20)
