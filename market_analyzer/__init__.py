"""Market analysis toolkit: regime detection, technicals, phase detection, and opportunity assessment."""

# Config
from market_analyzer.config import (
    CalendarSettings,
    DiagonalSettings,
    ExitSettings,
    IronButterflySettings,
    IronCondorSettings,
    MarketDef,
    MarketSettings,
    RatioSpreadSettings,
    ScreeningSettings,
    Settings,
    StrategySettings,
    get_settings,
)

# Models
from market_analyzer.models.regime import (
    CrossTickerEntry,
    FeatureZScore,
    HMMModelInfo,
    LabelAlignmentDetail,
    RegimeConfig,
    RegimeDistributionEntry,
    RegimeExplanation,
    RegimeHistoryDay,
    RegimeID,
    RegimeResult,
    RegimeTimeSeries,
    RegimeTimeSeriesEntry,
    ResearchReport,
    StateMeansRow,
    TickerResearch,
    TransitionRow,
    TrendDirection,
)
from market_analyzer.models.phase import PhaseID, PhaseResult
from market_analyzer.models.data import DataType, ProviderType, DataRequest, DataResult
from market_analyzer.models.features import FeatureConfig, FeatureInspection
from market_analyzer.models.technicals import TechnicalSnapshot, TechnicalSignal

# New workflow models
from market_analyzer.models.context import IntermarketDashboard, IntermarketEntry, MarketContext
from market_analyzer.models.instrument import InstrumentAnalysis
from market_analyzer.models.entry import EntryConfirmation, EntryCondition, EntryTriggerType
from market_analyzer.models.strategy import (
    OptionStructure,
    OptionStructureType,
    PositionSize,
    StrategyParameters,
)
from market_analyzer.models.adjustment import (
    AdjustmentAnalysis,
    AdjustmentOption,
    AdjustmentType,
    PositionStatus,
    TestedSide,
)
from market_analyzer.models.exit_plan import (
    AdjustmentTrigger,
    AdjustmentTriggerType,
    ExitPlan,
    ExitReason,
    ExitTarget,
)

# Services
from market_analyzer.service.analyzer import MarketAnalyzer
from market_analyzer.service.regime_service import RegimeService
from market_analyzer.service.technical import TechnicalService
from market_analyzer.service.phase import PhaseService
from market_analyzer.service.fundamental import FundamentalService
from market_analyzer.service.macro import MacroService
from market_analyzer.service.levels import LevelsService
from market_analyzer.service.opportunity import OpportunityService
from market_analyzer.service.black_swan import BlackSwanService
from market_analyzer.service.ranking import TradeRankingService
from market_analyzer.data.service import DataService

# New workflow services
from market_analyzer.service.context import MarketContextService
from market_analyzer.service.instrument import InstrumentAnalysisService
from market_analyzer.service.screening import ScreeningService, ScreenCandidate, ScreeningResult
from market_analyzer.service.entry import EntryService
from market_analyzer.service.strategy import StrategyService
from market_analyzer.service.exit import ExitService
from market_analyzer.service.adjustment import AdjustmentService

# Phase detection
from market_analyzer.phases.detector import PhaseDetector

# Fundamentals
from market_analyzer.models.fundamentals import FundamentalsSnapshot
from market_analyzer.fundamentals.fetch import fetch_fundamentals

# Macro
from market_analyzer.models.macro import MacroCalendar, MacroEvent, MacroEventType
from market_analyzer.macro.calendar import get_macro_calendar

# Vol surface
from market_analyzer.models.vol_surface import (
    SkewSlice,
    TermStructurePoint,
    VolatilitySurface,
)
from market_analyzer.service.vol_surface import VolSurfaceService
from market_analyzer.features.vol_surface import compute_vol_surface

# Trading plan
from market_analyzer.models.trading_plan import (
    DailyTradingPlan,
    DayVerdict,
    PlanHorizon,
    PlanTrade,
    RiskBudget,
)
from market_analyzer.macro.expiry import ExpiryEvent, ExpiryType
from market_analyzer.service.trading_plan import TradingPlanService
from market_analyzer.config import TradingPlanSettings

# Opportunity assessment
from market_analyzer.models.levels import (
    LevelRole,
    LevelSource,
    LevelsAnalysis,
    PriceLevel,
    StopLoss,
    Target,
    TradeDirection,
)
from market_analyzer.models.opportunity import (
    BreakoutOpportunity,
    LEAPOpportunity,
    LegAction,
    LegSpec,
    MomentumOpportunity,
    ORBDecision,
    OrderSide,
    RiskProfile,
    StructureProfile,
    StructureType,
    TradeSpec,
    Verdict,
    ZeroDTEOpportunity,
    ZeroDTEStrategy,
    get_structure_profile,
)
from market_analyzer.opportunity.option_plays.earnings import EarningsOpportunity
from market_analyzer.opportunity.setups.mean_reversion import MeanReversionOpportunity
from market_analyzer.opportunity.option_plays.zero_dte import assess_zero_dte
from market_analyzer.opportunity.option_plays.leap import assess_leap
from market_analyzer.opportunity.option_plays.earnings import assess_earnings_play
from market_analyzer.opportunity.option_plays.calendar import (
    CalendarOpportunity, CalendarStrategy, assess_calendar,
)
from market_analyzer.opportunity.option_plays.diagonal import (
    DiagonalOpportunity, DiagonalStrategy, assess_diagonal,
)
from market_analyzer.opportunity.option_plays.iron_condor import (
    IronCondorOpportunity, IronCondorStrategy, assess_iron_condor,
)
from market_analyzer.opportunity.option_plays.iron_butterfly import (
    IronButterflyOpportunity, IronButterflyStrategy, assess_iron_butterfly,
)
from market_analyzer.opportunity.option_plays.ratio_spread import (
    RatioSpreadOpportunity, RatioSpreadStrategy, assess_ratio_spread,
)
from market_analyzer.opportunity.setups.breakout import assess_breakout
from market_analyzer.opportunity.setups.momentum import assess_momentum
from market_analyzer.opportunity.setups.mean_reversion import assess_mean_reversion
from market_analyzer.opportunity.setups.orb import (
    ORBSetupOpportunity, ORBStrategy, assess_orb,
)

# Black Swan / Tail-Risk
from market_analyzer.models.black_swan import (
    AlertLevel,
    BlackSwanAlert,
    CircuitBreaker,
    IndicatorStatus,
    StressIndicator,
)
from market_analyzer.models.ranking import (
    RankedEntry,
    RankingFeedback,
    ScoreBreakdown,
    StrategyType,
    TradeRankingResult,
)
from market_analyzer.features.black_swan import compute_black_swan_alert

# Features
from market_analyzer.features.pipeline import compute_features
from market_analyzer.features.technicals import compute_technicals

__all__ = [
    # Config
    "Settings",
    "get_settings",
    "MarketDef",
    "MarketSettings",
    "ScreeningSettings",
    "StrategySettings",
    "ExitSettings",
    # Services (facade)
    "MarketAnalyzer",
    # Services (individual — existing)
    "RegimeService",
    "TechnicalService",
    "PhaseService",
    "FundamentalService",
    "MacroService",
    "LevelsService",
    "OpportunityService",
    "BlackSwanService",
    "TradeRankingService",
    "DataService",
    # Services (individual — new workflow)
    "MarketContextService",
    "InstrumentAnalysisService",
    "ScreeningService",
    "EntryService",
    "StrategyService",
    "ExitService",
    "AdjustmentService",
    # Regime models
    "RegimeID",
    "RegimeResult",
    "RegimeConfig",
    "RegimeExplanation",
    "HMMModelInfo",
    "RegimeTimeSeries",
    "RegimeTimeSeriesEntry",
    "TrendDirection",
    # Phase models
    "PhaseID",
    "PhaseResult",
    "PhaseDetector",
    # Research models
    "TickerResearch",
    "CrossTickerEntry",
    "ResearchReport",
    "TransitionRow",
    "StateMeansRow",
    "LabelAlignmentDetail",
    "FeatureZScore",
    "RegimeHistoryDay",
    "RegimeDistributionEntry",
    # Data models
    "DataType",
    "ProviderType",
    "DataRequest",
    "DataResult",
    # Feature models
    "FeatureConfig",
    "FeatureInspection",
    # Technical models
    "TechnicalSnapshot",
    "TechnicalSignal",
    # Context models (new)
    "MarketContext",
    "IntermarketDashboard",
    "IntermarketEntry",
    # Instrument models (new)
    "InstrumentAnalysis",
    # Entry models (new)
    "EntryTriggerType",
    "EntryCondition",
    "EntryConfirmation",
    # Strategy models (new)
    "OptionStructureType",
    "OptionStructure",
    "StrategyParameters",
    "PositionSize",
    # Adjustment models (new)
    "AdjustmentAnalysis",
    "AdjustmentOption",
    "AdjustmentType",
    "PositionStatus",
    "TestedSide",
    # Exit models (new)
    "ExitPlan",
    "ExitTarget",
    "ExitReason",
    "AdjustmentTrigger",
    "AdjustmentTriggerType",
    # Screening models (new)
    "ScreenCandidate",
    "ScreeningResult",
    # Fundamentals
    "FundamentalsSnapshot",
    "fetch_fundamentals",
    # Macro
    "MacroCalendar",
    "MacroEvent",
    "MacroEventType",
    "get_macro_calendar",
    # Levels models
    "LevelRole",
    "LevelSource",
    "LevelsAnalysis",
    "PriceLevel",
    "StopLoss",
    "Target",
    "TradeDirection",
    # Vol surface
    "VolSurfaceService",
    "VolatilitySurface",
    "TermStructurePoint",
    "SkewSlice",
    "compute_vol_surface",
    # Opportunity models
    "LegAction",
    "LegSpec",
    "StructureType",
    "OrderSide",
    "TradeSpec",
    "Verdict",
    "RiskProfile",
    "StructureProfile",
    "get_structure_profile",
    "EarningsOpportunity",
    "MeanReversionOpportunity",
    "ZeroDTEOpportunity",
    "ZeroDTEStrategy",
    "ORBDecision",
    "LEAPOpportunity",
    "BreakoutOpportunity",
    "MomentumOpportunity",
    "CalendarOpportunity",
    "CalendarStrategy",
    "DiagonalOpportunity",
    "DiagonalStrategy",
    "IronCondorOpportunity",
    "IronCondorStrategy",
    "IronButterflyOpportunity",
    "IronButterflyStrategy",
    "RatioSpreadOpportunity",
    "RatioSpreadStrategy",
    "assess_zero_dte",
    "assess_leap",
    "assess_breakout",
    "assess_momentum",
    "assess_mean_reversion",
    "assess_orb",
    "ORBSetupOpportunity",
    "ORBStrategy",
    "assess_earnings_play",
    "assess_calendar",
    "assess_diagonal",
    "assess_iron_condor",
    "assess_iron_butterfly",
    "assess_ratio_spread",
    # Black Swan
    "AlertLevel",
    "BlackSwanAlert",
    "CircuitBreaker",
    "IndicatorStatus",
    "StressIndicator",
    "compute_black_swan_alert",
    # Ranking
    "StrategyType",
    "ScoreBreakdown",
    "RankedEntry",
    "TradeRankingResult",
    "RankingFeedback",
    # Trading plan
    "DailyTradingPlan",
    "DayVerdict",
    "PlanHorizon",
    "PlanTrade",
    "RiskBudget",
    "ExpiryEvent",
    "ExpiryType",
    "TradingPlanService",
    "TradingPlanSettings",
    # Functions
    "compute_features",
    "compute_technicals",
]
