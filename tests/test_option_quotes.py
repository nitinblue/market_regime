"""Tests for OptionQuoteService — all mocked, no network."""

import pytest
from datetime import date, datetime
from unittest.mock import MagicMock, patch

from market_analyzer.broker.base import MarketDataProvider, MarketMetricsProvider
from market_analyzer.models.opportunity import LegAction, LegSpec
from market_analyzer.models.quotes import MarketMetrics, OptionQuote, QuoteSnapshot
from market_analyzer.service.option_quotes import OptionQuoteService


# --- Fixtures ---

def _make_quote(ticker="SPY", strike=580.0, opt_type="put", bid=2.10, ask=2.30):
    return OptionQuote(
        ticker=ticker, expiration=date(2026, 3, 20), strike=strike,
        option_type=opt_type, bid=bid, ask=ask, mid=round((bid + ask) / 2, 4),
        delta=-0.28, gamma=0.01, theta=-0.12, vega=0.15,
        implied_volatility=0.22,
    )


def _make_leg(strike=580.0, opt_type="put"):
    return LegSpec(
        role="short_put", action=LegAction.SELL_TO_OPEN,
        option_type=opt_type, strike=strike,
        strike_label=f"{strike:.0f} {opt_type}",
        expiration=date(2026, 3, 20), days_to_expiry=22,
        atm_iv_at_expiry=0.22,
    )


class MockMarketData(MarketDataProvider):
    def __init__(self):
        self._chain = [
            _make_quote(strike=575, bid=3.50, ask=3.70),
            _make_quote(strike=580, bid=2.10, ask=2.30),
            _make_quote(strike=585, bid=1.20, ask=1.35),
            _make_quote(strike=580, opt_type="call", bid=4.50, ask=4.70),
        ]

    def get_option_chain(self, ticker, expiration=None):
        return self._chain

    def get_quotes(self, legs):
        return [_make_quote(strike=leg.strike, opt_type=leg.option_type) for leg in legs]

    def get_greeks(self, legs):
        return {}

    @property
    def provider_name(self):
        return "mock_broker"


class MockMetrics(MarketMetricsProvider):
    def get_metrics(self, tickers):
        return {
            t: MarketMetrics(
                ticker=t, iv_rank=32.5, iv_percentile=45.2,
                beta=1.05, liquidity_rating=4.5,
            )
            for t in tickers
        }


# --- Tests ---

class TestOptionQuoteServiceBroker:
    def test_source_returns_broker_when_connected(self):
        qs = OptionQuoteService(market_data=MockMarketData())
        assert qs.source == "mock_broker"
        assert qs.has_broker

    def test_source_returns_yfinance_without_broker(self):
        qs = OptionQuoteService()
        assert qs.source == "yfinance"
        assert not qs.has_broker

    def test_get_chain_returns_broker_quotes(self):
        qs = OptionQuoteService(market_data=MockMarketData())
        snap = qs.get_chain("SPY")
        assert snap.source == "mock_broker"
        assert len(snap.quotes) == 4
        assert snap.quotes[0].bid == 3.50

    def test_get_chain_snapshot_has_source(self):
        qs = OptionQuoteService(market_data=MockMarketData())
        snap = qs.get_chain("SPY")
        assert isinstance(snap, QuoteSnapshot)
        assert snap.source == "mock_broker"

    def test_leg_quotes_returns_option_quotes(self):
        qs = OptionQuoteService(market_data=MockMarketData())
        legs = [_make_leg(strike=580), _make_leg(strike=575)]
        quotes = qs.get_leg_quotes(legs, ticker="SPY")
        assert len(quotes) == 2
        assert all(isinstance(q, OptionQuote) for q in quotes)
        assert quotes[0].bid > 0
        assert quotes[0].ask > quotes[0].bid

    def test_metrics_returns_market_metrics(self):
        qs = OptionQuoteService(metrics=MockMetrics())
        m = qs.get_metrics("SPY")
        assert m is not None
        assert m.iv_rank == 32.5
        assert m.iv_percentile == 45.2

    def test_metrics_returns_none_without_provider(self):
        qs = OptionQuoteService()
        assert qs.get_metrics("SPY") is None


class TestOptionQuoteServiceFallback:
    def test_no_broker_still_works(self):
        """OptionQuoteService with no broker returns yfinance source."""
        qs = OptionQuoteService()
        assert qs.source == "yfinance"
        snap = qs.get_chain("SPY")
        assert snap.source in ("yfinance", "none")

    def test_broker_failure_falls_back(self):
        """If broker raises, fallback to yfinance."""
        class FailingMarketData(MarketDataProvider):
            def get_option_chain(self, ticker, expiration=None):
                raise ConnectionError("broker down")
            def get_quotes(self, legs):
                raise ConnectionError("broker down")
            def get_greeks(self, legs):
                return {}
            @property
            def provider_name(self):
                return "failing_broker"

        qs = OptionQuoteService(market_data=FailingMarketData())
        snap = qs.get_chain("SPY")
        # Should not raise — falls back gracefully
        assert snap.source in ("yfinance", "none")


class TestOptionQuoteModel:
    def test_option_quote_roundtrip(self):
        q = _make_quote()
        data = q.model_dump()
        q2 = OptionQuote(**data)
        assert q2.bid == q.bid
        assert q2.delta == q.delta

    def test_market_metrics_roundtrip(self):
        m = MarketMetrics(
            ticker="SPY", iv_rank=32.5, iv_percentile=45.2,
            beta=1.05, earnings_date=date(2026, 4, 15),
        )
        data = m.model_dump()
        m2 = MarketMetrics(**data)
        assert m2.iv_rank == 32.5
        assert m2.earnings_date == date(2026, 4, 15)

    def test_quote_snapshot_roundtrip(self):
        snap = QuoteSnapshot(
            ticker="SPY", as_of=datetime(2026, 2, 26, 10, 30),
            underlying_price=580.0,
            quotes=[_make_quote()],
            source="tastytrade",
        )
        data = snap.model_dump()
        snap2 = QuoteSnapshot(**data)
        assert snap2.source == "tastytrade"
        assert len(snap2.quotes) == 1


class TestAdjustmentUsesRealQuotes:
    def test_adjustment_with_quote_service(self):
        """AdjustmentService with quote_service uses it for P&L."""
        from market_analyzer.service.adjustment import AdjustmentService
        from market_analyzer.models.opportunity import (
            LegAction, LegSpec, OrderSide, StructureType, TradeSpec,
        )
        from market_analyzer.models.regime import RegimeID, RegimeResult
        from market_analyzer.models.technicals import TechnicalSnapshot

        qs = OptionQuoteService(market_data=MockMarketData())
        adj = AdjustmentService(quote_service=qs)

        assert "real quotes" in adj.quote_source

    def test_adjustment_without_broker_no_costs(self):
        from market_analyzer.service.adjustment import AdjustmentService

        adj = AdjustmentService()
        assert "no broker" in adj.quote_source

    def test_adjustment_with_yfinance_only_no_costs(self):
        from market_analyzer.service.adjustment import AdjustmentService

        qs = OptionQuoteService()
        adj = AdjustmentService(quote_service=qs)
        assert "no broker" in adj.quote_source

    def test_leg_quotes_empty_without_broker(self):
        """get_leg_quotes returns empty list without broker (no BS fallback)."""
        qs = OptionQuoteService()
        legs = [_make_leg(strike=580)]
        quotes = qs.get_leg_quotes(legs, ticker="SPY")
        assert quotes == []
