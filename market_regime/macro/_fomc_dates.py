"""Hardcoded FOMC announcement dates (2nd day of each meeting).

Source: federalreserve.gov/monetarypolicy/fomccalendars.htm
"""

from __future__ import annotations

from datetime import date

# (announcement_date, has_summary_of_economic_projections)
# SEP at quarterly meetings: Mar, Jun, Sep, Dec
FOMC_DATES: list[tuple[date, bool]] = [
    # 2025
    (date(2025, 1, 29), False),
    (date(2025, 3, 19), True),
    (date(2025, 5, 7), False),
    (date(2025, 6, 18), True),
    (date(2025, 7, 30), False),
    (date(2025, 9, 17), True),
    (date(2025, 10, 29), False),
    (date(2025, 12, 10), True),
    # 2026
    (date(2026, 1, 28), False),
    (date(2026, 3, 18), True),
    (date(2026, 4, 29), False),
    (date(2026, 6, 17), True),
    (date(2026, 7, 29), False),
    (date(2026, 9, 16), True),
    (date(2026, 10, 28), False),
    (date(2026, 12, 9), True),
]
