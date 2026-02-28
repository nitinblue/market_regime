"""Pydantic models for stock fundamentals data."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel


class ValuationMetrics(BaseModel):
    trailing_pe: float | None = None
    forward_pe: float | None = None
    peg_ratio: float | None = None
    price_to_book: float | None = None
    price_to_sales: float | None = None


class EarningsMetrics(BaseModel):
    trailing_eps: float | None = None
    forward_eps: float | None = None
    earnings_growth: float | None = None


class RevenueMetrics(BaseModel):
    market_cap: int | None = None
    total_revenue: int | None = None
    revenue_per_share: float | None = None
    revenue_growth: float | None = None


class MarginMetrics(BaseModel):
    profit_margins: float | None = None
    gross_margins: float | None = None
    operating_margins: float | None = None
    ebitda_margins: float | None = None


class CashMetrics(BaseModel):
    operating_cashflow: int | None = None
    free_cashflow: int | None = None
    total_cash: int | None = None
    total_cash_per_share: float | None = None


class DebtMetrics(BaseModel):
    total_debt: int | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None


class ReturnMetrics(BaseModel):
    return_on_assets: float | None = None
    return_on_equity: float | None = None


class DividendMetrics(BaseModel):
    dividend_yield: float | None = None
    dividend_rate: float | None = None


class BusinessInfo(BaseModel):
    long_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    beta: float | None = None


class FiftyTwoWeek(BaseModel):
    high: float | None = None
    low: float | None = None
    pct_from_high: float | None = None
    pct_from_low: float | None = None


class EarningsEvent(BaseModel):
    date: date
    eps_estimate: float | None = None
    eps_actual: float | None = None
    eps_difference: float | None = None
    surprise_pct: float | None = None


class UpcomingEvents(BaseModel):
    next_earnings_date: date | None = None
    days_to_earnings: int | None = None
    ex_dividend_date: date | None = None
    dividend_date: date | None = None


class FundamentalsSnapshot(BaseModel):
    """Complete fundamentals snapshot for a single ticker."""

    ticker: str
    as_of: datetime
    asset_type: str = "EQUITY"  # EQUITY, ETF, INDEX, MUTUALFUND, etc.
    business: BusinessInfo
    valuation: ValuationMetrics
    earnings: EarningsMetrics
    revenue: RevenueMetrics
    margins: MarginMetrics
    cash: CashMetrics
    debt: DebtMetrics
    returns: ReturnMetrics
    dividends: DividendMetrics
    fifty_two_week: FiftyTwoWeek
    recent_earnings: list[EarningsEvent]
    upcoming_events: UpcomingEvents

    @property
    def is_equity(self) -> bool:
        return self.asset_type == "EQUITY"

    @property
    def has_earnings(self) -> bool:
        return self.is_equity and (
            self.upcoming_events.days_to_earnings is not None
            or len(self.recent_earnings) > 0
        )
