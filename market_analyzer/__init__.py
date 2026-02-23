"""Market analysis toolkit: regime detection, technicals, phase detection, and opportunity assessment."""

# Config
from market_analyzer.config import Settings, get_settings

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

# Services
from market_analyzer.service.analyzer import MarketAnalyzer
from market_analyzer.service.regime_service import RegimeService
from market_analyzer.service.technical import TechnicalService
from market_analyzer.service.phase import PhaseService
from market_analyzer.service.fundamental import FundamentalService
from market_analyzer.service.macro import MacroService
from market_analyzer.service.levels import LevelsService
from market_analyzer.service.opportunity import OpportunityService
from market_analyzer.data.service import DataService

# Phase detection
from market_analyzer.phases.detector import PhaseDetector

# Fundamentals
from market_analyzer.models.fundamentals import FundamentalsSnapshot
from market_analyzer.fundamentals.fetch import fetch_fundamentals

# Macro
from market_analyzer.models.macro import MacroCalendar, MacroEvent, MacroEventType
from market_analyzer.macro.calendar import get_macro_calendar

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
    MomentumOpportunity,
    Verdict,
    ZeroDTEOpportunity,
)
from market_analyzer.opportunity.zero_dte import assess_zero_dte
from market_analyzer.opportunity.leap import assess_leap
from market_analyzer.opportunity.breakout import assess_breakout
from market_analyzer.opportunity.momentum import assess_momentum

# Features
from market_analyzer.features.pipeline import compute_features
from market_analyzer.features.technicals import compute_technicals

__all__ = [
    # Config
    "Settings",
    "get_settings",
    # Services (facade)
    "MarketAnalyzer",
    # Services (individual)
    "RegimeService",
    "TechnicalService",
    "PhaseService",
    "FundamentalService",
    "MacroService",
    "LevelsService",
    "OpportunityService",
    "DataService",
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
    # Opportunity models
    "Verdict",
    "ZeroDTEOpportunity",
    "LEAPOpportunity",
    "BreakoutOpportunity",
    "MomentumOpportunity",
    "assess_zero_dte",
    "assess_leap",
    "assess_breakout",
    "assess_momentum",
    # Functions
    "compute_features",
    "compute_technicals",
]
