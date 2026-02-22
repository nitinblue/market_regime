"""OpportunityService: 0DTE, LEAP, breakout, and momentum assessment."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from market_analyzer.models.opportunity import (
    BreakoutOpportunity,
    LEAPOpportunity,
    MomentumOpportunity,
    ZeroDTEOpportunity,
)

if TYPE_CHECKING:
    from market_analyzer.data.service import DataService
    from market_analyzer.service.fundamental import FundamentalService
    from market_analyzer.service.macro import MacroService
    from market_analyzer.service.phase import PhaseService
    from market_analyzer.service.regime import RegimeService
    from market_analyzer.service.technical import TechnicalService


class OpportunityService:
    """Assess trading opportunities across multiple time horizons."""

    def __init__(
        self,
        regime_service: RegimeService | None = None,
        technical_service: TechnicalService | None = None,
        phase_service: PhaseService | None = None,
        fundamental_service: FundamentalService | None = None,
        macro_service: MacroService | None = None,
        data_service: DataService | None = None,
    ) -> None:
        self.regime_service = regime_service
        self.technical_service = technical_service
        self.phase_service = phase_service
        self.fundamental_service = fundamental_service
        self.macro_service = macro_service
        self.data_service = data_service

    def _get_ohlcv(self, ticker: str, ohlcv: pd.DataFrame | None) -> pd.DataFrame:
        if ohlcv is not None:
            return ohlcv
        if self.data_service is None:
            raise ValueError(
                "Either provide ohlcv DataFrame or initialize OpportunityService with a DataService"
            )
        return self.data_service.get_ohlcv(ticker)

    def _get_fundamentals(self, ticker: str):
        """Best-effort fundamentals fetch (None on failure)."""
        if self.fundamental_service is None:
            return None
        try:
            return self.fundamental_service.get(ticker)
        except Exception:
            return None

    def assess_zero_dte(
        self,
        ticker: str,
        ohlcv: pd.DataFrame | None = None,
        intraday: pd.DataFrame | None = None,
        as_of: date | None = None,
    ) -> ZeroDTEOpportunity:
        """Assess 0DTE opportunity for a single instrument."""
        from market_analyzer.opportunity.zero_dte import assess_zero_dte as _assess

        if self.regime_service is None or self.technical_service is None:
            raise ValueError("OpportunityService requires regime and technical services")
        if self.macro_service is None:
            raise ValueError("OpportunityService requires macro service")

        df = self._get_ohlcv(ticker, ohlcv)
        regime = self.regime_service.detect(ticker, df)
        technicals = self.technical_service.snapshot(ticker, df)
        macro = self.macro_service.calendar(as_of=as_of)

        orb = None
        if intraday is not None and self.technical_service is not None:
            orb = self.technical_service.orb(
                ticker, intraday=intraday, daily_atr=technicals.atr
            )

        fundamentals = self._get_fundamentals(ticker)

        return _assess(
            ticker=ticker,
            regime=regime,
            technicals=technicals,
            macro=macro,
            fundamentals=fundamentals,
            orb=orb,
            as_of=as_of,
        )

    def assess_leap(
        self,
        ticker: str,
        ohlcv: pd.DataFrame | None = None,
        as_of: date | None = None,
    ) -> LEAPOpportunity:
        """Assess LEAP opportunity for a single instrument."""
        from market_analyzer.opportunity.leap import assess_leap as _assess

        if self.regime_service is None or self.technical_service is None:
            raise ValueError("OpportunityService requires regime and technical services")
        if self.phase_service is None or self.macro_service is None:
            raise ValueError("OpportunityService requires phase and macro services")

        df = self._get_ohlcv(ticker, ohlcv)
        regime = self.regime_service.detect(ticker, df)
        technicals = self.technical_service.snapshot(ticker, df)
        phase = self.phase_service.detect(ticker, df)
        macro = self.macro_service.calendar(as_of=as_of)
        fundamentals = self._get_fundamentals(ticker)

        return _assess(
            ticker=ticker,
            regime=regime,
            technicals=technicals,
            phase=phase,
            macro=macro,
            fundamentals=fundamentals,
            as_of=as_of,
        )

    def assess_breakout(
        self,
        ticker: str,
        ohlcv: pd.DataFrame | None = None,
        as_of: date | None = None,
    ) -> BreakoutOpportunity:
        """Assess breakout opportunity for a single instrument."""
        from market_analyzer.opportunity.breakout import assess_breakout as _assess

        if self.regime_service is None or self.technical_service is None:
            raise ValueError("OpportunityService requires regime and technical services")
        if self.phase_service is None or self.macro_service is None:
            raise ValueError("OpportunityService requires phase and macro services")

        df = self._get_ohlcv(ticker, ohlcv)
        regime = self.regime_service.detect(ticker, df)
        technicals = self.technical_service.snapshot(ticker, df)
        phase = self.phase_service.detect(ticker, df)
        macro = self.macro_service.calendar(as_of=as_of)
        fundamentals = self._get_fundamentals(ticker)

        return _assess(
            ticker=ticker,
            regime=regime,
            technicals=technicals,
            phase=phase,
            macro=macro,
            fundamentals=fundamentals,
            as_of=as_of,
        )

    def assess_momentum(
        self,
        ticker: str,
        ohlcv: pd.DataFrame | None = None,
        as_of: date | None = None,
    ) -> MomentumOpportunity:
        """Assess momentum opportunity for a single instrument."""
        from market_analyzer.opportunity.momentum import assess_momentum as _assess

        if self.regime_service is None or self.technical_service is None:
            raise ValueError("OpportunityService requires regime and technical services")
        if self.phase_service is None or self.macro_service is None:
            raise ValueError("OpportunityService requires phase and macro services")

        df = self._get_ohlcv(ticker, ohlcv)
        regime = self.regime_service.detect(ticker, df)
        technicals = self.technical_service.snapshot(ticker, df)
        phase = self.phase_service.detect(ticker, df)
        macro = self.macro_service.calendar(as_of=as_of)
        fundamentals = self._get_fundamentals(ticker)

        return _assess(
            ticker=ticker,
            regime=regime,
            technicals=technicals,
            phase=phase,
            macro=macro,
            fundamentals=fundamentals,
            as_of=as_of,
        )
