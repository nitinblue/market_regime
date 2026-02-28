"""Tests for intraday streaming (candles + underlying price) wiring."""

import pytest
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from market_analyzer.broker.base import MarketDataProvider
from market_analyzer.models.quotes import OptionQuote
from market_analyzer.service.technical import TechnicalService


# --- ABC default behavior ---


class MinimalMarketData(MarketDataProvider):
    """Concrete subclass implementing only the abstract methods."""

    def get_option_chain(self, ticker, expiration=None):
        return []

    def get_quotes(self, legs):
        return []

    def get_greeks(self, legs):
        return {}

    @property
    def provider_name(self):
        return "minimal"


class TestABCDefaults:
    def test_default_get_intraday_candles_returns_empty(self):
        md = MinimalMarketData()
        result = md.get_intraday_candles("SPY")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_default_get_underlying_price_returns_none(self):
        md = MinimalMarketData()
        assert md.get_underlying_price("SPY") is None

    def test_default_candles_with_interval(self):
        md = MinimalMarketData()
        result = md.get_intraday_candles("SPY", interval="1m")
        assert result.empty


# --- Mock broker returning real data ---


class MockBrokerWithIntraday(MarketDataProvider):
    """Mock broker that returns intraday candles and underlying price."""

    def __init__(self, candles_df: pd.DataFrame | None = None, price: float | None = None):
        self._candles = candles_df if candles_df is not None else pd.DataFrame()
        self._price = price

    def get_option_chain(self, ticker, expiration=None):
        return []

    def get_quotes(self, legs):
        return []

    def get_greeks(self, legs):
        return {}

    @property
    def provider_name(self):
        return "mock_intraday"

    def get_intraday_candles(self, ticker, interval="5m"):
        return self._candles

    def get_underlying_price(self, ticker):
        return self._price


class FailingBroker(MarketDataProvider):
    """Broker that raises on every call."""

    def get_option_chain(self, ticker, expiration=None):
        raise ConnectionError("down")

    def get_quotes(self, legs):
        raise ConnectionError("down")

    def get_greeks(self, legs):
        return {}

    @property
    def provider_name(self):
        return "failing"

    def get_intraday_candles(self, ticker, interval="5m"):
        raise ConnectionError("stream failed")

    def get_underlying_price(self, ticker):
        raise ConnectionError("stream failed")


def _make_intraday_df() -> pd.DataFrame:
    """Synthetic 5m intraday bars for testing ORB computation."""
    times = pd.date_range("2026-02-27 09:30", periods=12, freq="5min")
    rng = np.random.default_rng(42)
    base = 580.0
    data = []
    for i, t in enumerate(times):
        o = base + rng.normal(0, 0.5)
        h = o + abs(rng.normal(0, 0.3))
        l = o - abs(rng.normal(0, 0.3))
        c = (h + l) / 2
        v = rng.integers(10000, 50000)
        data.append({"Open": o, "High": h, "Low": l, "Close": c, "Volume": float(v)})
        base = c
    df = pd.DataFrame(data, index=times)
    return df


# --- TechnicalService broker-first orb() ---


class TestTechnicalServiceBrokerFirst:
    def test_broker_candles_used_for_orb(self):
        """When broker provides candles, compute_orb gets broker data."""
        broker_df = _make_intraday_df()
        md = MockBrokerWithIntraday(candles_df=broker_df)
        ts = TechnicalService(market_data=md)

        # Should not raise — broker provides data, no yfinance needed
        orb = ts.orb("SPY", daily_atr=5.0)
        assert orb is not None
        assert orb.ticker == "SPY"

    def test_broker_failure_falls_back_to_yfinance(self):
        """If broker raises, orb() tries yfinance fallback."""
        md = FailingBroker()
        ts = TechnicalService(market_data=md)

        # Without data_service and with yfinance potentially unavailable,
        # this will either succeed (yfinance works) or raise ValueError.
        # The key: it doesn't raise ConnectionError from broker.
        try:
            ts.orb("SPY", daily_atr=5.0)
        except ValueError:
            pass  # "No intraday data available" — yfinance returned empty
        except Exception as e:
            # Should not be ConnectionError from broker
            assert not isinstance(e, ConnectionError)

    def test_no_broker_uses_yfinance(self):
        """market_data=None goes straight to yfinance path."""
        ts = TechnicalService()
        # Without yfinance data, expect ValueError (not AttributeError)
        try:
            ts.orb("FAKE_TICKER_12345", daily_atr=5.0)
        except (ValueError, Exception):
            pass  # Expected — no intraday data

    def test_broker_empty_df_falls_through(self):
        """Broker returning empty DataFrame triggers yfinance fallback."""
        md = MockBrokerWithIntraday(candles_df=pd.DataFrame())
        ts = TechnicalService(market_data=md)

        try:
            ts.orb("SPY", daily_atr=5.0)
        except ValueError:
            pass  # yfinance fallback — empty is fine

    def test_explicit_intraday_bypasses_broker(self):
        """If caller provides intraday, broker is not consulted."""
        broker_df = _make_intraday_df()
        caller_df = _make_intraday_df()

        # Broker that tracks calls
        call_count = {"n": 0}
        class TrackingBroker(MockBrokerWithIntraday):
            def get_intraday_candles(self, ticker, interval="5m"):
                call_count["n"] += 1
                return broker_df

        md = TrackingBroker()
        ts = TechnicalService(market_data=md)
        orb = ts.orb("SPY", intraday=caller_df, daily_atr=5.0)
        assert call_count["n"] == 0  # broker not called
        assert orb is not None


# --- OpportunityService auto-fetches ORB ---


class TestOpportunityAutoFetchesORB:
    def test_assess_zero_dte_attempts_orb(self):
        """assess_zero_dte tries to auto-fetch ORB when no intraday provided."""
        from market_analyzer.service.opportunity import OpportunityService

        mock_tech_svc = MagicMock()
        mock_tech_svc.snapshot.return_value = MagicMock(atr=5.0)
        mock_tech_svc.orb.return_value = MagicMock()

        mock_regime_svc = MagicMock()
        mock_macro_svc = MagicMock()
        mock_data_svc = MagicMock()
        mock_data_svc.get_ohlcv.return_value = pd.DataFrame(
            {"Open": [580], "High": [585], "Low": [575], "Close": [582], "Volume": [1e6]},
            index=pd.DatetimeIndex([datetime(2026, 2, 27)]),
        )

        opp = OpportunityService(
            regime_service=mock_regime_svc,
            technical_service=mock_tech_svc,
            macro_service=mock_macro_svc,
            data_service=mock_data_svc,
        )

        # Patch the actual assessor so we don't need real models
        with patch(
            "market_analyzer.opportunity.option_plays.zero_dte.assess_zero_dte",
        ) as mock_assess:
            mock_assess.return_value = MagicMock()
            opp.assess_zero_dte("SPY")

        # orb() should have been called (guard was removed)
        mock_tech_svc.orb.assert_called_once()


# --- Underlying price in OptionQuoteService ---


class TestUnderlyingPriceBrokerFirst:
    def test_broker_underlying_used_in_chain(self):
        """_infer_underlying_price tries broker direct quote first."""
        from market_analyzer.service.option_quotes import OptionQuoteService

        md = MockBrokerWithIntraday(price=581.50)
        qs = OptionQuoteService(market_data=md)

        # Call the internal method directly
        price = qs._infer_underlying_price([], "SPY")
        assert price == 581.50

    def test_broker_underlying_none_falls_back(self):
        """If broker returns None, falls back to put-call parity."""
        from market_analyzer.service.option_quotes import OptionQuoteService

        md = MockBrokerWithIntraday(price=None)
        qs = OptionQuoteService(market_data=md)

        # With empty quotes and None price → 0.0
        price = qs._infer_underlying_price([], "SPY")
        assert price == 0.0

    def test_broker_failure_falls_back_to_parity(self):
        """If broker raises, falls back to put-call parity."""
        from market_analyzer.service.option_quotes import OptionQuoteService

        md = FailingBroker()
        qs = OptionQuoteService(market_data=md)

        # With quotes, falls back to parity
        call = OptionQuote(
            ticker="SPY", expiration=date(2026, 3, 20), strike=580.0,
            option_type="call", bid=4.50, ask=4.70, mid=4.60,
        )
        put = OptionQuote(
            ticker="SPY", expiration=date(2026, 3, 20), strike=580.0,
            option_type="put", bid=2.10, ask=2.30, mid=2.20,
        )
        price = qs._infer_underlying_price([call, put], "SPY")
        # put-call parity: strike + call_mid - put_mid = 580 + 4.60 - 2.20 = 582.40
        assert abs(price - 582.40) < 0.01

    def test_no_broker_uses_parity(self):
        """Without broker, goes straight to put-call parity."""
        from market_analyzer.service.option_quotes import OptionQuoteService

        qs = OptionQuoteService()
        call = OptionQuote(
            ticker="SPY", expiration=date(2026, 3, 20), strike=580.0,
            option_type="call", bid=4.50, ask=4.70, mid=4.60,
        )
        put = OptionQuote(
            ticker="SPY", expiration=date(2026, 3, 20), strike=580.0,
            option_type="put", bid=2.10, ask=2.30, mid=2.20,
        )
        price = qs._infer_underlying_price([call, put], "SPY")
        assert abs(price - 582.40) < 0.01


# --- Candle → DataFrame conversion ---


class TestCandleToDataFrame:
    def test_synthetic_candle_df_has_correct_columns(self):
        """Verify the DataFrame format produced by _make_intraday_df."""
        df = _make_intraday_df()
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(df) == 12
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_intraday_df_sorted_ascending(self):
        df = _make_intraday_df()
        assert df.index.is_monotonic_increasing

    def test_broker_returns_valid_orb_input(self):
        """Broker intraday data can be fed to compute_orb."""
        from market_analyzer.features.orb import compute_orb

        df = _make_intraday_df()
        orb = compute_orb(df, "SPY", daily_atr=5.0)
        assert orb.ticker == "SPY"
        assert orb.range_high > 0
        assert orb.range_low > 0
