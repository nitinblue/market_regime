"""OpportunityService: option plays + setup assessment."""

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
    from market_analyzer.service.vol_surface import VolSurfaceService


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
        vol_surface_service: VolSurfaceService | None = None,
    ) -> None:
        self.regime_service = regime_service
        self.technical_service = technical_service
        self.phase_service = phase_service
        self.fundamental_service = fundamental_service
        self.macro_service = macro_service
        self.data_service = data_service
        self.vol_surface_service = vol_surface_service

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
        from market_analyzer.opportunity.option_plays.zero_dte import assess_zero_dte as _assess

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
        from market_analyzer.opportunity.option_plays.leap import assess_leap as _assess

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
        from market_analyzer.opportunity.setups.breakout import assess_breakout as _assess

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
        from market_analyzer.opportunity.setups.momentum import assess_momentum as _assess

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

    def assess_orb(
        self,
        ticker: str,
        intraday: pd.DataFrame,
        ohlcv: pd.DataFrame | None = None,
        as_of: date | None = None,
    ):
        """Assess ORB setup opportunity (requires intraday data)."""
        from market_analyzer.opportunity.setups.orb import assess_orb as _assess

        if self.regime_service is None or self.technical_service is None:
            raise ValueError("OpportunityService requires regime and technical services")

        df = self._get_ohlcv(ticker, ohlcv)
        regime = self.regime_service.detect(ticker, df)
        technicals = self.technical_service.snapshot(ticker, df)

        orb = self.technical_service.orb(
            ticker, intraday=intraday, daily_atr=technicals.atr
        )

        phase = None
        if self.phase_service is not None:
            phase = self.phase_service.detect(ticker, df)

        macro = None
        if self.macro_service is not None:
            macro = self.macro_service.calendar(as_of=as_of)

        fundamentals = self._get_fundamentals(ticker)

        return _assess(
            ticker=ticker,
            regime=regime,
            technicals=technicals,
            orb=orb,
            phase=phase,
            macro=macro,
            fundamentals=fundamentals,
            as_of=as_of,
        )

    # --- Vol-surface-dependent option plays ---

    def _get_vol_surface(self, ticker: str):
        """Best-effort vol surface fetch (None on failure)."""
        if self.vol_surface_service is None:
            return None
        try:
            return self.vol_surface_service.surface(ticker)
        except Exception:
            return None

    def assess_calendar(
        self,
        ticker: str,
        ohlcv: pd.DataFrame | None = None,
        as_of: date | None = None,
    ):
        """Assess calendar spread opportunity."""
        from market_analyzer.opportunity.option_plays.calendar import assess_calendar as _assess

        if self.regime_service is None or self.technical_service is None:
            raise ValueError("OpportunityService requires regime and technical services")

        df = self._get_ohlcv(ticker, ohlcv)
        regime = self.regime_service.detect(ticker, df)
        technicals = self.technical_service.snapshot(ticker, df)
        fundamentals = self._get_fundamentals(ticker)
        vol_surface = self._get_vol_surface(ticker)

        return _assess(
            ticker=ticker, regime=regime, technicals=technicals,
            vol_surface=vol_surface, fundamentals=fundamentals, as_of=as_of,
        )

    def assess_diagonal(
        self,
        ticker: str,
        ohlcv: pd.DataFrame | None = None,
        as_of: date | None = None,
    ):
        """Assess diagonal spread opportunity."""
        from market_analyzer.opportunity.option_plays.diagonal import assess_diagonal as _assess

        if self.regime_service is None or self.technical_service is None:
            raise ValueError("OpportunityService requires regime and technical services")

        df = self._get_ohlcv(ticker, ohlcv)
        regime = self.regime_service.detect(ticker, df)
        technicals = self.technical_service.snapshot(ticker, df)
        phase = self.phase_service.detect(ticker, df) if self.phase_service else None
        fundamentals = self._get_fundamentals(ticker)
        vol_surface = self._get_vol_surface(ticker)

        return _assess(
            ticker=ticker, regime=regime, technicals=technicals,
            vol_surface=vol_surface, phase=phase, fundamentals=fundamentals, as_of=as_of,
        )

    def assess_iron_condor(
        self,
        ticker: str,
        ohlcv: pd.DataFrame | None = None,
        as_of: date | None = None,
    ):
        """Assess iron condor opportunity — the #1 income strategy."""
        from market_analyzer.opportunity.option_plays.iron_condor import assess_iron_condor as _assess

        if self.regime_service is None or self.technical_service is None:
            raise ValueError("OpportunityService requires regime and technical services")

        df = self._get_ohlcv(ticker, ohlcv)
        regime = self.regime_service.detect(ticker, df)
        technicals = self.technical_service.snapshot(ticker, df)
        fundamentals = self._get_fundamentals(ticker)
        vol_surface = self._get_vol_surface(ticker)

        return _assess(
            ticker=ticker, regime=regime, technicals=technicals,
            vol_surface=vol_surface, fundamentals=fundamentals, as_of=as_of,
        )

    def assess_iron_butterfly(
        self,
        ticker: str,
        ohlcv: pd.DataFrame | None = None,
        as_of: date | None = None,
    ):
        """Assess iron butterfly opportunity."""
        from market_analyzer.opportunity.option_plays.iron_butterfly import assess_iron_butterfly as _assess

        if self.regime_service is None or self.technical_service is None:
            raise ValueError("OpportunityService requires regime and technical services")

        df = self._get_ohlcv(ticker, ohlcv)
        regime = self.regime_service.detect(ticker, df)
        technicals = self.technical_service.snapshot(ticker, df)
        fundamentals = self._get_fundamentals(ticker)
        vol_surface = self._get_vol_surface(ticker)

        return _assess(
            ticker=ticker, regime=regime, technicals=technicals,
            vol_surface=vol_surface, fundamentals=fundamentals, as_of=as_of,
        )

    def assess_ratio_spread(
        self,
        ticker: str,
        ohlcv: pd.DataFrame | None = None,
        as_of: date | None = None,
    ):
        """Assess ratio spread opportunity."""
        from market_analyzer.opportunity.option_plays.ratio_spread import assess_ratio_spread as _assess

        if self.regime_service is None or self.technical_service is None:
            raise ValueError("OpportunityService requires regime and technical services")

        df = self._get_ohlcv(ticker, ohlcv)
        regime = self.regime_service.detect(ticker, df)
        technicals = self.technical_service.snapshot(ticker, df)
        phase = self.phase_service.detect(ticker, df) if self.phase_service else None
        fundamentals = self._get_fundamentals(ticker)
        vol_surface = self._get_vol_surface(ticker)

        return _assess(
            ticker=ticker, regime=regime, technicals=technicals,
            vol_surface=vol_surface, phase=phase, fundamentals=fundamentals, as_of=as_of,
        )
