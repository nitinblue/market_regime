"""Core logic for fetching stock fundamentals via yfinance."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any

import yfinance as yf

from market_regime.models.fundamentals import (
    BusinessInfo,
    CashMetrics,
    DebtMetrics,
    DividendMetrics,
    EarningsEvent,
    EarningsMetrics,
    FiftyTwoWeek,
    FundamentalsSnapshot,
    MarginMetrics,
    ReturnMetrics,
    RevenueMetrics,
    UpcomingEvents,
    ValuationMetrics,
)

# Module-level cache: ticker -> (snapshot, expiry_time)
_cache: dict[str, tuple[FundamentalsSnapshot, datetime]] = {}


def _safe_get(info: dict[str, Any], key: str) -> float | None:
    """Safely extract a numeric value from yfinance info dict.

    Handles NaN, Inf, missing keys, and non-numeric values.
    """
    val = info.get(key)
    if val is None:
        return None
    try:
        fval = float(val)
        if math.isnan(fval) or math.isinf(fval):
            return None
        return fval
    except (TypeError, ValueError):
        return None


def _safe_get_int(info: dict[str, Any], key: str) -> int | None:
    """Safely extract an integer value from yfinance info dict."""
    val = _safe_get(info, key)
    if val is None:
        return None
    return int(val)


def _build_52week(info: dict[str, Any], current_price: float | None) -> FiftyTwoWeek:
    """Build 52-week range with pct_from_high/low calculations."""
    high = _safe_get(info, "fiftyTwoWeekHigh")
    low = _safe_get(info, "fiftyTwoWeekLow")

    pct_from_high: float | None = None
    pct_from_low: float | None = None

    if current_price is not None and high is not None and high != 0:
        pct_from_high = (current_price - high) / high * 100
    if current_price is not None and low is not None and low != 0:
        pct_from_low = (current_price - low) / low * 100

    return FiftyTwoWeek(
        high=high,
        low=low,
        pct_from_high=pct_from_high,
        pct_from_low=pct_from_low,
    )


def _parse_earnings_dates(ticker_obj: yf.Ticker) -> list[EarningsEvent]:
    """Parse recent earnings from yfinance get_earnings_dates()."""
    try:
        df = ticker_obj.get_earnings_dates(limit=8)
    except Exception:
        return []

    if df is None or df.empty:
        return []

    events: list[EarningsEvent] = []
    for idx, row in df.iterrows():
        try:
            dt = idx.date() if hasattr(idx, "date") else idx
        except Exception:
            continue

        def _get_val(col: str) -> float | None:
            if col not in row:
                return None
            v = row[col]
            try:
                fv = float(v)
                return None if (math.isnan(fv) or math.isinf(fv)) else fv
            except (TypeError, ValueError):
                return None

        events.append(
            EarningsEvent(
                date=dt,
                eps_estimate=_get_val("EPS Estimate"),
                eps_actual=_get_val("Reported EPS"),
                eps_difference=_get_val("Surprise(%)"),  # yfinance labels vary
                surprise_pct=_get_val("Surprise(%)"),
            )
        )
    return events


def _parse_upcoming_events(ticker_obj: yf.Ticker, info: dict[str, Any]) -> UpcomingEvents:
    """Parse upcoming events (earnings date, dividends) from yfinance."""
    today = date.today()

    # Next earnings date
    next_earnings: date | None = None
    days_to_earnings: int | None = None

    try:
        cal = ticker_obj.calendar
        if cal is not None:
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed is not None:
                    if isinstance(ed, list) and len(ed) > 0:
                        next_earnings = ed[0].date() if hasattr(ed[0], "date") else ed[0]
                    elif hasattr(ed, "date"):
                        next_earnings = ed.date()
            # pandas DataFrame case
            elif hasattr(cal, "loc"):
                if "Earnings Date" in cal.index:
                    ed = cal.loc["Earnings Date"]
                    if hasattr(ed, "iloc"):
                        ed = ed.iloc[0]
                    if hasattr(ed, "date"):
                        next_earnings = ed.date()
    except Exception:
        pass

    if next_earnings is not None:
        days_to_earnings = (next_earnings - today).days

    # Dividend dates from info
    ex_div_raw = info.get("exDividendDate")
    ex_div: date | None = None
    if ex_div_raw is not None:
        try:
            if isinstance(ex_div_raw, (int, float)) and not math.isnan(ex_div_raw):
                ex_div = datetime.fromtimestamp(int(ex_div_raw)).date()
            elif hasattr(ex_div_raw, "date"):
                ex_div = ex_div_raw.date()
        except Exception:
            pass

    div_date_raw = info.get("dividendDate")
    div_date: date | None = None
    if div_date_raw is not None:
        try:
            if isinstance(div_date_raw, (int, float)) and not math.isnan(div_date_raw):
                div_date = datetime.fromtimestamp(int(div_date_raw)).date()
            elif hasattr(div_date_raw, "date"):
                div_date = div_date_raw.date()
        except Exception:
            pass

    return UpcomingEvents(
        next_earnings_date=next_earnings,
        days_to_earnings=days_to_earnings,
        ex_dividend_date=ex_div,
        dividend_date=div_date,
    )


def fetch_fundamentals(
    ticker: str,
    ttl_minutes: int | None = None,
) -> FundamentalsSnapshot:
    """Fetch stock fundamentals for a ticker via yfinance.

    Results are cached in-memory with a configurable TTL.

    Args:
        ticker: Stock ticker symbol (case-insensitive).
        ttl_minutes: Cache TTL in minutes. None = use config default.

    Returns:
        FundamentalsSnapshot with all available data.

    Raises:
        ValueError: If ticker is invalid or returns no data.
    """
    ticker = ticker.upper()

    if ttl_minutes is None:
        from market_regime.config import get_settings
        ttl_minutes = get_settings().fundamentals.cache_ttl_minutes

    # Check cache
    now = datetime.now()
    if ticker in _cache:
        snapshot, expiry = _cache[ticker]
        if now < expiry:
            return snapshot

    # Fetch from yfinance
    t = yf.Ticker(ticker)
    info = t.info

    if not info or info.get("regularMarketPrice") is None:
        raise ValueError(f"No data returned for ticker '{ticker}'. Verify it's a valid symbol.")

    current_price = _safe_get(info, "regularMarketPrice") or _safe_get(info, "currentPrice")

    snapshot = FundamentalsSnapshot(
        ticker=ticker,
        as_of=now,
        business=BusinessInfo(
            long_name=info.get("longName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            beta=_safe_get(info, "beta"),
        ),
        valuation=ValuationMetrics(
            trailing_pe=_safe_get(info, "trailingPE"),
            forward_pe=_safe_get(info, "forwardPE"),
            peg_ratio=_safe_get(info, "pegRatio"),
            price_to_book=_safe_get(info, "priceToBook"),
            price_to_sales=_safe_get(info, "priceToSalesTrailing12Months"),
        ),
        earnings=EarningsMetrics(
            trailing_eps=_safe_get(info, "trailingEps"),
            forward_eps=_safe_get(info, "forwardEps"),
            earnings_growth=_safe_get(info, "earningsGrowth"),
        ),
        revenue=RevenueMetrics(
            market_cap=_safe_get_int(info, "marketCap"),
            total_revenue=_safe_get_int(info, "totalRevenue"),
            revenue_per_share=_safe_get(info, "revenuePerShare"),
            revenue_growth=_safe_get(info, "revenueGrowth"),
        ),
        margins=MarginMetrics(
            profit_margins=_safe_get(info, "profitMargins"),
            gross_margins=_safe_get(info, "grossMargins"),
            operating_margins=_safe_get(info, "operatingMargins"),
            ebitda_margins=_safe_get(info, "ebitdaMargins"),
        ),
        cash=CashMetrics(
            operating_cashflow=_safe_get_int(info, "operatingCashflow"),
            free_cashflow=_safe_get_int(info, "freeCashflow"),
            total_cash=_safe_get_int(info, "totalCash"),
            total_cash_per_share=_safe_get(info, "totalCashPerShare"),
        ),
        debt=DebtMetrics(
            total_debt=_safe_get_int(info, "totalDebt"),
            debt_to_equity=_safe_get(info, "debtToEquity"),
            current_ratio=_safe_get(info, "currentRatio"),
        ),
        returns=ReturnMetrics(
            return_on_assets=_safe_get(info, "returnOnAssets"),
            return_on_equity=_safe_get(info, "returnOnEquity"),
        ),
        dividends=DividendMetrics(
            dividend_yield=_safe_get(info, "dividendYield"),
            dividend_rate=_safe_get(info, "dividendRate"),
        ),
        fifty_two_week=_build_52week(info, current_price),
        recent_earnings=_parse_earnings_dates(t),
        upcoming_events=_parse_upcoming_events(t, info),
    )

    _cache[ticker] = (snapshot, now + timedelta(minutes=ttl_minutes))
    return snapshot


def invalidate_fundamentals_cache(ticker: str | None = None) -> None:
    """Clear fundamentals cache.

    Args:
        ticker: Clear cache for this ticker only. None = clear all.
    """
    if ticker is None:
        _cache.clear()
    else:
        _cache.pop(ticker.upper(), None)
