"""Tests for macro economic calendar — all offline, no mocks needed."""

from datetime import date

import pytest

from market_regime.macro._econ_schedule import (
    generate_cpi_dates,
    generate_gdp_dates,
    generate_nfp_dates,
    generate_pce_dates,
)
from market_regime.macro._fomc_dates import FOMC_DATES
from market_regime.macro.calendar import get_macro_calendar, get_next_event
from market_regime.models.macro import MacroEventType


class TestDateGeneration:
    def test_nfp_dates_are_fridays(self):
        dates = generate_nfp_dates(2025, 2026)
        for d in dates:
            assert d.weekday() == 4, f"{d} is not a Friday"

    def test_nfp_12_per_year(self):
        dates = generate_nfp_dates(2026, 2026)
        assert len(dates) == 12

    def test_cpi_in_second_week(self):
        dates = generate_cpi_dates(2026, 2026)
        for d in dates:
            assert d.weekday() == 1, f"{d} is not a Tuesday"
            assert 8 <= d.day <= 14, f"CPI date {d} not in 2nd week"

    def test_pce_last_week(self):
        dates = generate_pce_dates(2026, 2026)
        for d in dates:
            assert d.weekday() == 4, f"{d} is not a Friday"
            assert d.day >= 22, f"PCE date {d} not in last week"

    def test_gdp_4_per_year(self):
        dates = generate_gdp_dates(2026, 2026)
        assert len(dates) == 4

    def test_all_dates_sorted(self):
        for gen in [generate_nfp_dates, generate_cpi_dates, generate_pce_dates, generate_gdp_dates]:
            dates = gen(2025, 2026)
            assert dates == sorted(dates)


class TestFOMCDates:
    def test_sorted(self):
        dates_only = [d for d, _ in FOMC_DATES]
        assert dates_only == sorted(dates_only)

    def test_8_per_year(self):
        for year in [2025, 2026]:
            count = sum(1 for d, _ in FOMC_DATES if d.year == year)
            assert count == 8, f"Expected 8 FOMC dates in {year}, got {count}"

    def test_4_seps_per_year(self):
        for year in [2025, 2026]:
            sep_count = sum(1 for d, has_sep in FOMC_DATES if d.year == year and has_sep)
            assert sep_count == 4, f"Expected 4 SEPs in {year}, got {sep_count}"


class TestMacroCalendar:
    def test_returns_calendar(self):
        cal = get_macro_calendar(as_of=date(2026, 2, 21), lookahead_days=60)
        assert cal is not None
        assert len(cal.events) > 0

    def test_has_next_event(self):
        cal = get_macro_calendar(as_of=date(2026, 2, 21), lookahead_days=60)
        assert cal.next_event is not None
        assert cal.days_to_next is not None
        assert cal.days_to_next >= 0

    def test_has_next_fomc(self):
        cal = get_macro_calendar(as_of=date(2026, 2, 21), lookahead_days=60)
        assert cal.next_fomc is not None
        assert cal.next_fomc.event_type == MacroEventType.FOMC
        assert cal.days_to_next_fomc is not None
        assert cal.days_to_next_fomc >= 0

    def test_events_sorted(self):
        cal = get_macro_calendar(as_of=date(2026, 2, 21), lookahead_days=90)
        dates = [e.date for e in cal.events]
        assert dates == sorted(dates)

    def test_lookahead_filter(self):
        cal_short = get_macro_calendar(as_of=date(2026, 2, 21), lookahead_days=7)
        cal_long = get_macro_calendar(as_of=date(2026, 2, 21), lookahead_days=90)
        assert len(cal_long.events) >= len(cal_short.events)

    def test_events_next_7_and_30(self):
        cal = get_macro_calendar(as_of=date(2026, 2, 21), lookahead_days=60)
        assert len(cal.events_next_30_days) >= len(cal.events_next_7_days)
        for e in cal.events_next_7_days:
            assert (e.date - date(2026, 2, 21)).days <= 7
        for e in cal.events_next_30_days:
            assert (e.date - date(2026, 2, 21)).days <= 30


class TestGetNextEvent:
    def test_next_fomc_after_feb_21_2026(self):
        event = get_next_event(MacroEventType.FOMC, as_of=date(2026, 2, 21))
        assert event is not None
        assert event.date == date(2026, 3, 18)

    def test_next_nfp_after_feb_21_2026(self):
        event = get_next_event(MacroEventType.NFP, as_of=date(2026, 2, 21))
        assert event is not None
        assert event.date == date(2026, 3, 6)

    def test_next_any_event(self):
        event = get_next_event(as_of=date(2026, 2, 21))
        assert event is not None
        assert event.date >= date(2026, 2, 21)

    def test_returns_none_past_all_dates(self):
        event = get_next_event(as_of=date(2030, 1, 1))
        assert event is None
