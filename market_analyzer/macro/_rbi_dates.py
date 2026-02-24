"""Hardcoded RBI monetary policy announcement dates.

Source: rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx
Reserve Bank of India MPC (Monetary Policy Committee) meeting schedule.
"""

from __future__ import annotations

from datetime import date

# (announcement_date, is_bi_monthly_review)
# RBI MPC meets 6 times a year (roughly every 2 months)
RBI_MPC_DATES: list[tuple[date, bool]] = [
    # 2025
    (date(2025, 2, 7), True),
    (date(2025, 4, 9), True),
    (date(2025, 6, 6), True),
    (date(2025, 8, 8), True),
    (date(2025, 10, 8), True),
    (date(2025, 12, 5), True),
    # 2026
    (date(2026, 2, 6), True),
    (date(2026, 4, 8), True),
    (date(2026, 6, 5), True),
    (date(2026, 8, 7), True),
    (date(2026, 10, 7), True),
    (date(2026, 12, 4), True),
]
