"""Options expiry calendar — OpEx, VIX settlement, quad witching, weekly expiries.

Reuses _nth_weekday from _econ_schedule.py for date math.
"""

from __future__ import annotations

from datetime import date, timedelta
from enum import StrEnum

from pydantic import BaseModel

from market_analyzer.macro._econ_schedule import _nth_weekday

_QUARTERLY_MONTHS = {3, 6, 9, 12}


class ExpiryType(StrEnum):
    MONTHLY_OPEX = "monthly_opex"
    WEEKLY_OPEX = "weekly_opex"
    VIX_SETTLEMENT = "vix_settlement"
    QUARTERLY_FUTURES = "quarterly_futures"
    QUAD_WITCHING = "quad_witching"


class ExpiryEvent(BaseModel):
    """A single options-expiry event."""

    date: date
    expiry_type: ExpiryType
    label: str


def monthly_opex(year: int, month: int) -> date:
    """3rd Friday of the given month."""
    return _nth_weekday(year, month, 4, 3)  # 4=Friday, 3rd occurrence


def vix_settlement(year: int, month: int) -> date:
    """Wednesday before 3rd Friday (VIX settlement day)."""
    opex = monthly_opex(year, month)
    return opex - timedelta(days=2)  # Friday - 2 = Wednesday


def is_quad_witching(d: date) -> bool:
    """True if d is the 3rd Friday of Mar/Jun/Sep/Dec."""
    if d.month not in _QUARTERLY_MONTHS:
        return False
    return d == monthly_opex(d.year, d.month)


def weekly_opex_fridays(start: date, end: date) -> list[date]:
    """All Fridays between start and end (inclusive) — weekly OpEx days."""
    fridays: list[date] = []
    # Advance to first Friday on or after start
    d = start + timedelta(days=(4 - start.weekday()) % 7)
    while d <= end:
        fridays.append(d)
        d += timedelta(weeks=1)
    return fridays


def get_expiry_calendar(start: date, end: date) -> list[ExpiryEvent]:
    """All expiry events in [start, end], sorted by date.

    Produces monthly OpEx, VIX settlement, quad witching, and quarterly futures.
    Weekly OpEx is NOT included (too noisy) — use weekly_opex_fridays() separately.
    """
    events: list[ExpiryEvent] = []

    # Iterate months in range
    y, m = start.year, start.month
    end_y, end_m = end.year, end.month

    while (y, m) <= (end_y, end_m):
        opex = monthly_opex(y, m)

        if start <= opex <= end:
            # Quad witching check
            if m in _QUARTERLY_MONTHS:
                events.append(ExpiryEvent(
                    date=opex,
                    expiry_type=ExpiryType.QUAD_WITCHING,
                    label=f"{_month_name(m)} Quad Witching",
                ))
                events.append(ExpiryEvent(
                    date=opex,
                    expiry_type=ExpiryType.QUARTERLY_FUTURES,
                    label=f"{_month_name(m)} Quarterly Futures Expiry",
                ))
            else:
                events.append(ExpiryEvent(
                    date=opex,
                    expiry_type=ExpiryType.MONTHLY_OPEX,
                    label=f"{_month_name(m)} Monthly OpEx",
                ))

        # VIX settlement
        vix = vix_settlement(y, m)
        if start <= vix <= end:
            events.append(ExpiryEvent(
                date=vix,
                expiry_type=ExpiryType.VIX_SETTLEMENT,
                label=f"{_month_name(m)} VIX Settlement",
            ))

        # Advance month
        m += 1
        if m > 12:
            m = 1
            y += 1

    events.sort(key=lambda e: e.date)
    return events


def upcoming_expiries(as_of: date | None = None, days_ahead: int = 7) -> list[ExpiryEvent]:
    """Expiry events in [as_of, as_of + days_ahead]."""
    today = as_of or date.today()
    return get_expiry_calendar(today, today + timedelta(days=days_ahead))


def _month_name(month: int) -> str:
    """Short month name."""
    names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return names[month]
