"""Pydantic models for volatility surface analysis."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class VolSurfacePoint(BaseModel):
    """Single point on the vol surface (one strike × one expiration)."""

    expiration: date
    strike: float
    moneyness: float        # strike / underlying_price
    option_type: str         # "call" or "put"
    implied_vol: float
    bid: float
    ask: float
    mid_iv: float            # midpoint approximation
    open_interest: int
    volume: int


class TermStructurePoint(BaseModel):
    """ATM implied volatility for a single expiration."""

    expiration: date
    days_to_expiry: int
    atm_iv: float            # IV at nearest-to-ATM strike
    atm_strike: float


class SkewSlice(BaseModel):
    """Volatility skew for a single expiration."""

    expiration: date
    days_to_expiry: int
    atm_iv: float
    otm_put_iv: float        # ~5% OTM put IV
    otm_call_iv: float       # ~5% OTM call IV
    put_skew: float           # otm_put_iv - atm_iv
    call_skew: float          # otm_call_iv - atm_iv
    skew_ratio: float         # put_skew / call_skew (>1 = put-heavy)


class VolatilitySurface(BaseModel):
    """Complete volatility surface for one ticker."""

    ticker: str
    as_of_date: date
    underlying_price: float
    expirations: list[date]

    # Term structure
    term_structure: list[TermStructurePoint]
    front_iv: float           # Nearest expiration ATM IV
    back_iv: float            # ~30-60 DTE ATM IV
    term_slope: float         # (back_iv - front_iv) / front_iv — positive = contango
    is_contango: bool         # back_iv > front_iv (normal)
    is_backwardation: bool    # front_iv > back_iv (event-driven, earnings)

    # Skew
    skew_by_expiry: list[SkewSlice]

    # Calendar-specific metrics
    calendar_edge_score: float    # 0-1: how favorable for calendar spreads
    best_calendar_expiries: tuple[date, date] | None  # (sell_front, buy_back)
    iv_differential_pct: float    # (front_iv - back_iv) / back_iv as pct

    # Data quality
    total_contracts: int
    avg_bid_ask_spread_pct: float
    data_quality: str             # "good" / "fair" / "poor"

    summary: str
