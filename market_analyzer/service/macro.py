"""MacroService: economic calendar."""

from __future__ import annotations

from datetime import date

from market_analyzer.models.macro import MacroCalendar


class MacroService:
    """Access macro economic calendar (FOMC, CPI, NFP, PCE, GDP)."""

    def calendar(
        self,
        as_of: date | None = None,
        lookahead_days: int | None = None,
    ) -> MacroCalendar:
        """Get macro economic calendar."""
        from market_analyzer.macro.calendar import get_macro_calendar

        return get_macro_calendar(as_of=as_of, lookahead_days=lookahead_days)
