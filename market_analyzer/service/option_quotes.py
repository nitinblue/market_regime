"""Option quote service — works with any broker, yfinance fallback.

Usage::

    # With TastyTrade
    qs = OptionQuoteService(market_data=tt_market_data, metrics=tt_metrics)

    # Without broker (yfinance only)
    qs = OptionQuoteService(data_service=DataService())

    # Both (broker preferred, yfinance as fallback)
    qs = OptionQuoteService(market_data=tt_md, data_service=DataService())
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import TYPE_CHECKING

from market_analyzer.models.quotes import MarketMetrics, OptionQuote, QuoteSnapshot

if TYPE_CHECKING:
    from market_analyzer.broker.base import MarketDataProvider, MarketMetricsProvider
    from market_analyzer.data.service import DataService
    from market_analyzer.models.opportunity import LegSpec

logger = logging.getLogger(__name__)


class OptionQuoteService:
    """Fetches option quotes via broker, with yfinance fallback.

    Callers never check which source is active — the service handles
    broker-first / yfinance-fallback internally.
    """

    def __init__(
        self,
        market_data: MarketDataProvider | None = None,
        metrics: MarketMetricsProvider | None = None,
        data_service: DataService | None = None,
    ) -> None:
        self._market_data = market_data
        self._metrics = metrics
        self._data_service = data_service

    @property
    def source(self) -> str:
        """Broker name if connected, 'yfinance' otherwise."""
        if self._market_data:
            return self._market_data.provider_name
        return "yfinance"

    @property
    def has_broker(self) -> bool:
        return self._market_data is not None

    def get_chain(
        self, ticker: str, expiration: date | None = None,
    ) -> QuoteSnapshot:
        """Full chain with bid/ask/Greeks.

        Tries broker first, falls back to yfinance OPTIONS_CHAIN.
        """
        if self._market_data:
            try:
                quotes = self._market_data.get_option_chain(ticker, expiration)
                # Get underlying price from nearest ATM quote
                underlying = self._infer_underlying_price(quotes, ticker)
                return QuoteSnapshot(
                    ticker=ticker,
                    as_of=datetime.now(),
                    underlying_price=underlying,
                    quotes=quotes,
                    source=self._market_data.provider_name,
                )
            except Exception as e:
                logger.warning("Broker chain failed for %s, falling back to yfinance: %s", ticker, e)

        # yfinance fallback
        return self._yfinance_chain(ticker, expiration)

    def get_leg_quotes(self, legs: list[LegSpec], ticker: str = "") -> list[OptionQuote]:
        """Quotes for specific legs (used by adjustment service).

        Returns broker quotes only. Returns empty list if no broker connected.
        Does NOT fall back to Black-Scholes estimates.
        """
        if self._market_data:
            try:
                return self._market_data.get_quotes(legs)
            except Exception as e:
                logger.warning("Broker leg quotes failed: %s", e)

        return []

    def get_metrics(self, ticker: str) -> MarketMetrics | None:
        """IV rank, percentile, beta. None if no metrics provider."""
        if not self._metrics:
            return None
        try:
            result = self._metrics.get_metrics([ticker])
            return result.get(ticker)
        except Exception as e:
            logger.warning("Metrics fetch failed for %s: %s", ticker, e)
            return None

    # -- yfinance fallback --

    def _yfinance_chain(
        self, ticker: str, expiration: date | None,
    ) -> QuoteSnapshot:
        """Build QuoteSnapshot from yfinance OPTIONS_CHAIN data."""
        if not self._data_service:
            return QuoteSnapshot(
                ticker=ticker,
                as_of=datetime.now(),
                underlying_price=0.0,
                quotes=[],
                source="none",
            )

        from market_analyzer.models.data import DataRequest, DataType

        try:
            df, result = self._data_service.get(DataRequest(
                ticker=ticker, data_type=DataType.OPTIONS_CHAIN,
            ))
        except Exception as e:
            logger.warning("yfinance chain fetch failed for %s: %s", ticker, e)
            return QuoteSnapshot(
                ticker=ticker,
                as_of=datetime.now(),
                underlying_price=0.0,
                quotes=[],
                source="yfinance",
            )

        # Get underlying price from OHLCV cache
        underlying = 0.0
        try:
            ohlcv = self._data_service.get_ohlcv(ticker)
            if not ohlcv.empty:
                underlying = float(ohlcv["Close"].iloc[-1])
        except Exception:
            pass

        quotes: list[OptionQuote] = []
        for _, row in df.iterrows():
            bid = float(row.get("bid", 0) or 0)
            ask = float(row.get("ask", 0) or 0)
            mid = round((bid + ask) / 2, 4) if (bid + ask) > 0 else float(row.get("lastPrice", 0) or 0)

            exp_val = row.get("expiration")
            if exp_val is None:
                continue
            if hasattr(exp_val, "date"):
                exp_date = exp_val.date() if callable(exp_val.date) else exp_val
            else:
                exp_date = exp_val

            if expiration and exp_date != expiration:
                continue

            quotes.append(OptionQuote(
                ticker=ticker,
                expiration=exp_date,
                strike=float(row.get("strike", 0)),
                option_type=str(row.get("option_type", "call")).lower(),
                bid=bid,
                ask=ask,
                mid=mid,
                last=float(row.get("lastPrice", 0) or 0),
                volume=int(row.get("volume", 0) or 0),
                open_interest=int(row.get("openInterest", 0) or 0),
                implied_volatility=float(row.get("impliedVolatility", 0) or 0),
            ))

        return QuoteSnapshot(
            ticker=ticker,
            as_of=datetime.now(),
            underlying_price=underlying,
            quotes=quotes,
            source="yfinance",
        )

    def _infer_underlying_price(
        self, quotes: list[OptionQuote], ticker: str,
    ) -> float:
        """Infer underlying price — try broker direct quote first, then put-call parity."""
        # Try direct broker underlying price (DXLink equity quote)
        if self._market_data is not None:
            try:
                price = self._market_data.get_underlying_price(ticker)
                if price and price > 0:
                    return price
            except Exception:
                pass

        if not quotes:
            return 0.0

        # Find the strike where call and put mid are closest
        calls = {q.strike: q for q in quotes if q.option_type == "call"}
        puts = {q.strike: q for q in quotes if q.option_type == "put"}
        common = set(calls) & set(puts)

        if common:
            # ATM is where call_mid - put_mid is closest to 0
            best = min(common, key=lambda s: abs(calls[s].mid - puts[s].mid))
            # put-call parity: S ~ strike + call - put
            return round(best + calls[best].mid - puts[best].mid, 2)

        # Fallback: just pick midpoint of strikes
        strikes = sorted({q.strike for q in quotes})
        if strikes:
            return strikes[len(strikes) // 2]
        return 0.0
