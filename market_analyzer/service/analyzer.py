"""MarketAnalyzer: top-level facade composing all services."""

from __future__ import annotations

from typing import TYPE_CHECKING

from market_analyzer.models.regime import RegimeConfig
from market_analyzer.service.fundamental import FundamentalService
from market_analyzer.service.levels import LevelsService
from market_analyzer.service.macro import MacroService
from market_analyzer.service.opportunity import OpportunityService
from market_analyzer.service.phase import PhaseService
from market_analyzer.service.black_swan import BlackSwanService
from market_analyzer.service.ranking import TradeRankingService
from market_analyzer.service.regime import RegimeService
from market_analyzer.service.technical import TechnicalService
from market_analyzer.service.context import MarketContextService
from market_analyzer.service.instrument import InstrumentAnalysisService
from market_analyzer.service.screening import ScreeningService
from market_analyzer.service.entry import EntryService
from market_analyzer.service.strategy import StrategyService
from market_analyzer.service.exit import ExitService
from market_analyzer.service.vol_surface import VolSurfaceService
from market_analyzer.service.adjustment import AdjustmentService
from market_analyzer.service.option_quotes import OptionQuoteService
from market_analyzer.service.trading_plan import TradingPlanService

if TYPE_CHECKING:
    from market_analyzer.broker.base import MarketDataProvider, MarketMetricsProvider
    from market_analyzer.data.service import DataService


class MarketAnalyzer:
    """Top-level facade composing all market analysis services.

    Usage::

        from market_analyzer import MarketAnalyzer, DataService

        ma = MarketAnalyzer(data_service=DataService())

        # --- Existing APIs ---
        regime = ma.regime.detect("SPY")
        tech = ma.technicals.snapshot("SPY")
        phase = ma.phase.detect("SPY")
        fund = ma.fundamentals.get("SPY")
        macro = ma.macro.calendar()
        bo = ma.opportunity.assess_breakout("SPY")

        # --- NEW workflow APIs ---
        ctx = ma.context.assess()                     # Q1a: Environment safe?
        analysis = ma.instrument.analyze("SPY")       # Q1b: What's the ticker doing?
        candidates = ma.screening.scan(["SPY","GLD"]) # Q1c: Where are setups?
        entry = ma.entry.confirm("SPY", EntryTriggerType.BREAKOUT_CONFIRMED)  # Q2
        params = ma.strategy.select("SPY", regime=r, technicals=t)            # Q3
        exit_plan = ma.exit.plan("SPY", params, entry_price=580.0,            # Q4
                                 regime=r, technicals=t, levels=l)
    """

    def __init__(
        self,
        data_service: DataService | None = None,
        config: RegimeConfig = RegimeConfig(),
        market: str | None = None,
        market_data: MarketDataProvider | None = None,
        market_metrics: MarketMetricsProvider | None = None,
    ) -> None:
        self.data = data_service

        # --- Existing services (unchanged) ---
        self.regime = RegimeService(config=config, data_service=data_service)
        self.technicals = TechnicalService(
            data_service=data_service, market_data=market_data,
        )
        self.phase = PhaseService(
            regime_service=self.regime, data_service=data_service
        )
        self.fundamentals = FundamentalService()
        self.macro = MacroService()
        self.levels = LevelsService(
            technical_service=self.technicals,
            regime_service=self.regime,
            data_service=data_service,
        )
        self.vol_surface = VolSurfaceService(data_service=data_service)
        self.opportunity = OpportunityService(
            regime_service=self.regime,
            technical_service=self.technicals,
            phase_service=self.phase,
            fundamental_service=self.fundamentals,
            macro_service=self.macro,
            data_service=data_service,
            vol_surface_service=self.vol_surface,
        )
        self.black_swan = BlackSwanService(data_service=data_service)
        self.ranking = TradeRankingService(
            opportunity_service=self.opportunity,
            levels_service=self.levels,
            black_swan_service=self.black_swan,
            data_service=data_service,
        )

        # --- NEW workflow services ---
        self.context = MarketContextService(
            regime_service=self.regime,
            macro_service=self.macro,
            black_swan_service=self.black_swan,
            market=market,
        )
        self.instrument = InstrumentAnalysisService(
            regime_service=self.regime,
            technical_service=self.technicals,
            phase_service=self.phase,
            levels_service=self.levels,
            fundamental_service=self.fundamentals,
            opportunity_service=self.opportunity,
            data_service=data_service,
        )
        self.screening = ScreeningService(
            regime_service=self.regime,
            technical_service=self.technicals,
            phase_service=self.phase,
            data_service=data_service,
        )
        self.entry = EntryService(
            technical_service=self.technicals,
            levels_service=self.levels,
            data_service=data_service,
        )
        self.strategy = StrategyService()
        self.exit = ExitService(
            levels_service=self.levels,
            regime_service=self.regime,
        )
        self.quotes = OptionQuoteService(
            market_data=market_data,
            metrics=market_metrics,
            data_service=data_service,
        )
        self.adjustment = AdjustmentService(quote_service=self.quotes)
        self.plan = TradingPlanService(analyzer=self)
