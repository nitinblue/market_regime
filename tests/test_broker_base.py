"""Tests for broker ABC contracts."""

import pytest
from datetime import date

from market_analyzer.broker.base import (
    BrokerSession,
    MarketDataProvider,
    MarketMetricsProvider,
)
from market_analyzer.models.quotes import MarketMetrics, OptionQuote


# --- ABCs are truly abstract ---

class TestABCCannotInstantiate:
    def test_broker_session(self):
        with pytest.raises(TypeError, match="abstract"):
            BrokerSession()

    def test_market_data_provider(self):
        with pytest.raises(TypeError, match="abstract"):
            MarketDataProvider()

    def test_market_metrics_provider(self):
        with pytest.raises(TypeError, match="abstract"):
            MarketMetricsProvider()


# --- Minimal mock implementations satisfy ABCs ---

class MockBrokerSession(BrokerSession):
    def __init__(self):
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def broker_name(self) -> str:
        return "mock_broker"


class MockMarketData(MarketDataProvider):
    def get_option_chain(self, ticker, expiration=None):
        return [OptionQuote(
            ticker=ticker, expiration=date(2026, 3, 20), strike=580.0,
            option_type="call", bid=2.50, ask=2.60, mid=2.55,
        )]

    def get_quotes(self, legs):
        return []

    def get_greeks(self, legs):
        return {}

    @property
    def provider_name(self) -> str:
        return "mock_broker"


class MockMetrics(MarketMetricsProvider):
    def get_metrics(self, tickers):
        return {t: MarketMetrics(ticker=t, iv_rank=35.0) for t in tickers}


class TestMockImplementation:
    def test_broker_session_lifecycle(self):
        s = MockBrokerSession()
        assert not s.is_connected
        assert s.connect()
        assert s.is_connected
        assert s.broker_name == "mock_broker"
        s.disconnect()
        assert not s.is_connected

    def test_market_data_provider_chain(self):
        md = MockMarketData()
        chain = md.get_option_chain("SPY")
        assert len(chain) == 1
        assert chain[0].ticker == "SPY"
        assert chain[0].bid == 2.50
        assert md.provider_name == "mock_broker"

    def test_market_data_provider_quotes(self):
        md = MockMarketData()
        assert md.get_quotes([]) == []

    def test_market_data_provider_greeks(self):
        md = MockMarketData()
        assert md.get_greeks([]) == {}

    def test_metrics_provider(self):
        mp = MockMetrics()
        result = mp.get_metrics(["SPY", "GLD"])
        assert "SPY" in result
        assert result["SPY"].iv_rank == 35.0
        assert result["GLD"].ticker == "GLD"

    def test_provider_name_required(self):
        """provider_name is part of the contract."""
        md = MockMarketData()
        assert isinstance(md.provider_name, str)
        assert len(md.provider_name) > 0
