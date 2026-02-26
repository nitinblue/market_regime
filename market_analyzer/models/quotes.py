"""Broker-agnostic quote and market metrics models."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel


class OptionQuote(BaseModel):
    """Real market quote for a single option contract."""

    ticker: str
    expiration: date
    strike: float
    option_type: str  # "call" | "put"
    bid: float
    ask: float
    mid: float  # (bid + ask) / 2
    last: float | None = None
    volume: int = 0
    open_interest: int = 0
    implied_volatility: float | None = None
    # Greeks (from DXLink or broker)
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None


class QuoteSnapshot(BaseModel):
    """Collection of quotes for an underlying at a point in time."""

    ticker: str
    as_of: datetime
    underlying_price: float
    quotes: list[OptionQuote]
    source: str  # "tastytrade" | "schwab" | "yfinance" | etc.


class MarketMetrics(BaseModel):
    """Market-level metrics for an underlying (IV rank, beta, etc.)."""

    ticker: str
    iv_rank: float | None = None
    iv_percentile: float | None = None
    iv_index: float | None = None
    iv_30_day: float | None = None
    hv_30_day: float | None = None
    hv_60_day: float | None = None
    beta: float | None = None
    corr_spy: float | None = None
    liquidity_rating: float | None = None
    earnings_date: date | None = None
