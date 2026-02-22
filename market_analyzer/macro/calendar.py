"""Query functions for the macro economic calendar."""

from __future__ import annotations

from datetime import date, timedelta

from market_analyzer.macro._econ_schedule import (
    generate_cpi_dates,
    generate_gdp_dates,
    generate_nfp_dates,
    generate_pce_dates,
)
from market_analyzer.macro._fomc_dates import FOMC_DATES
from market_analyzer.models.macro import (
    MacroCalendar,
    MacroEvent,
    MacroEventImpact,
    MacroEventType,
)

# Year range for algorithmic date generation
_GEN_START = 2025
_GEN_END = 2027


def _build_all_events() -> list[MacroEvent]:
    """Build the full list of macro events from all sources."""
    events: list[MacroEvent] = []

    # FOMC
    for dt, has_sep in FOMC_DATES:
        sep_note = " (with SEP/dot plot)" if has_sep else ""
        events.append(
            MacroEvent(
                event_type=MacroEventType.FOMC,
                date=dt,
                name=f"FOMC Rate Decision{sep_note}",
                impact=MacroEventImpact.HIGH,
                description="Federal Reserve interest rate decision and policy statement.",
                options_impact="IV crush after announcement. Straddle sellers target this date.",
            )
        )

    # NFP
    for dt in generate_nfp_dates(_GEN_START, _GEN_END):
        events.append(
            MacroEvent(
                event_type=MacroEventType.NFP,
                date=dt,
                name="Non-Farm Payrolls",
                impact=MacroEventImpact.HIGH,
                description="Monthly employment report from BLS.",
                options_impact="Pre-market volatility spike. Short-dated options repriced.",
            )
        )

    # CPI
    for dt in generate_cpi_dates(_GEN_START, _GEN_END):
        events.append(
            MacroEvent(
                event_type=MacroEventType.CPI,
                date=dt,
                name="CPI Report",
                impact=MacroEventImpact.HIGH,
                description="Consumer Price Index — inflation gauge.",
                options_impact="Major IV event. Impacts rate expectations and equity vol.",
            )
        )

    # PCE
    for dt in generate_pce_dates(_GEN_START, _GEN_END):
        events.append(
            MacroEvent(
                event_type=MacroEventType.PCE,
                date=dt,
                name="PCE Price Index",
                impact=MacroEventImpact.MEDIUM,
                description="Fed's preferred inflation measure.",
                options_impact="Moderate IV impact. Watch for surprises vs expectations.",
            )
        )

    # GDP
    for dt in generate_gdp_dates(_GEN_START, _GEN_END):
        events.append(
            MacroEvent(
                event_type=MacroEventType.GDP,
                date=dt,
                name="GDP Report",
                impact=MacroEventImpact.MEDIUM,
                description="Quarterly GDP advance estimate.",
                options_impact="Moderate impact. Sector rotation possible on surprises.",
            )
        )

    return sorted(events, key=lambda e: e.date)


# Module-level cache — built once
_ALL_EVENTS: list[MacroEvent] | None = None


def _get_all_events() -> list[MacroEvent]:
    global _ALL_EVENTS
    if _ALL_EVENTS is None:
        _ALL_EVENTS = _build_all_events()
    return _ALL_EVENTS


def get_macro_calendar(
    as_of: date | None = None,
    lookback_days: int = 7,
    lookahead_days: int | None = None,
) -> MacroCalendar:
    """Build a macro calendar centered on as_of date.

    Args:
        as_of: Reference date (default: today).
        lookback_days: Include past events within this many days.
        lookahead_days: Include future events within this many days.
            If None, uses config setting.
    """
    if as_of is None:
        as_of = date.today()
    if lookahead_days is None:
        from market_analyzer.config import get_settings
        lookahead_days = get_settings().macro.lookahead_days

    all_events = _get_all_events()
    start = as_of - timedelta(days=lookback_days)
    end = as_of + timedelta(days=lookahead_days)

    events = [e for e in all_events if start <= e.date <= end]

    # Future events only (for next_event / next_fomc)
    future = [e for e in all_events if e.date >= as_of]

    next_event = future[0] if future else None
    days_to_next = (next_event.date - as_of).days if next_event else None

    fomc_future = [e for e in future if e.event_type == MacroEventType.FOMC]
    next_fomc = fomc_future[0] if fomc_future else None
    days_to_next_fomc = (next_fomc.date - as_of).days if next_fomc else None

    events_7 = [e for e in all_events if as_of <= e.date <= as_of + timedelta(days=7)]
    events_30 = [e for e in all_events if as_of <= e.date <= as_of + timedelta(days=30)]

    return MacroCalendar(
        events=events,
        next_event=next_event,
        days_to_next=days_to_next,
        next_fomc=next_fomc,
        days_to_next_fomc=days_to_next_fomc,
        events_next_7_days=events_7,
        events_next_30_days=events_30,
    )


def get_next_event(
    event_type: MacroEventType | None = None,
    as_of: date | None = None,
) -> MacroEvent | None:
    """Get the next macro event of a given type (or any type if None).

    Args:
        event_type: Filter by event type. None = any.
        as_of: Reference date (default: today).
    """
    if as_of is None:
        as_of = date.today()

    all_events = _get_all_events()
    for event in all_events:
        if event.date < as_of:
            continue
        if event_type is not None and event.event_type != event_type:
            continue
        return event
    return None
