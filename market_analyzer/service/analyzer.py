"""MarketAnalyzer: top-level facade composing all services."""

from __future__ import annotations

from typing import TYPE_CHECKING

from market_analyzer.models.regime import RegimeConfig
from market_analyzer.service.fundamental import FundamentalService
from market_analyzer.service.macro import MacroService
from market_analyzer.service.opportunity import OpportunityService
from market_analyzer.service.phase import PhaseService
from market_analyzer.service.regime import RegimeService
from market_analyzer.service.technical import TechnicalService

if TYPE_CHECKING:
    from market_analyzer.data.service import DataService


class MarketAnalyzer:
    """Top-level facade composing all market analysis services.

    Usage::

        from market_analyzer import MarketAnalyzer, DataService

        ma = MarketAnalyzer(data_service=DataService())

        regime = ma.regime.detect("SPY")
        tech = ma.technicals.snapshot("SPY")
        phase = ma.phase.detect("SPY")
        fund = ma.fundamentals.get("SPY")
        macro = ma.macro.calendar()
        bo = ma.opportunity.assess_breakout("SPY")
    """

    def __init__(
        self,
        data_service: DataService | None = None,
        config: RegimeConfig = RegimeConfig(),
    ) -> None:
        self.data = data_service
        self.regime = RegimeService(config=config, data_service=data_service)
        self.technicals = TechnicalService(data_service=data_service)
        self.phase = PhaseService(
            regime_service=self.regime, data_service=data_service
        )
        self.fundamentals = FundamentalService()
        self.macro = MacroService()
        self.opportunity = OpportunityService(
            regime_service=self.regime,
            technical_service=self.technicals,
            phase_service=self.phase,
            fundamental_service=self.fundamentals,
            macro_service=self.macro,
            data_service=data_service,
        )
