"""Algorithmic generation of recurring economic release dates.

CPI, NFP, PCE, GDP follow predictable monthly/quarterly patterns.
Dates are approximate — actual BLS/BEA release dates may shift by +-1 day.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the nth occurrence of a weekday in a given month.

    Args:
        year: Calendar year.
        month: Calendar month (1-12).
        weekday: 0=Monday, 4=Friday, etc.
        n: 1-based occurrence (1=first, 2=second, ...).
    """
    first_day = date(year, month, 1)
    # Days until first occurrence of target weekday
    days_ahead = (weekday - first_day.weekday()) % 7
    first_occurrence = first_day + timedelta(days=days_ahead)
    return first_occurrence + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last occurrence of a weekday in a given month."""
    last_day = date(year, month, calendar.monthrange(year, month)[1])
    days_back = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=days_back)


def generate_nfp_dates(start_year: int, end_year: int) -> list[date]:
    """Non-Farm Payrolls: 1st Friday of each month."""
    dates: list[date] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            dates.append(_nth_weekday(year, month, 4, 1))  # 4=Friday
    return sorted(dates)


def generate_cpi_dates(start_year: int, end_year: int) -> list[date]:
    """CPI: ~2nd Tuesday of each month (BLS pattern)."""
    dates: list[date] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            dates.append(_nth_weekday(year, month, 1, 2))  # 1=Tuesday, 2nd
    return sorted(dates)


def generate_pce_dates(start_year: int, end_year: int) -> list[date]:
    """PCE Price Index: ~last Friday of each month."""
    dates: list[date] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            dates.append(_last_weekday(year, month, 4))  # 4=Friday
    return sorted(dates)


def generate_gdp_dates(start_year: int, end_year: int) -> list[date]:
    """GDP advance estimate: ~4 weeks after quarter end (last Thursday of month after quarter)."""
    dates: list[date] = []
    for year in range(start_year, end_year + 1):
        # GDP released in Jan (Q4), Apr (Q1), Jul (Q2), Oct (Q3)
        for month in [1, 4, 7, 10]:
            dates.append(_last_weekday(year, month, 3))  # 3=Thursday
    return sorted(dates)
