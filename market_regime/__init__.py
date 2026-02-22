"""Historical market data service and HMM-based regime detection for options trading."""

# Config
from market_regime.config import Settings, get_settings

# Models
from market_regime.models.regime import (
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
from market_regime.models.phase import PhaseID, PhaseResult
from market_regime.models.data import DataType, ProviderType, DataRequest, DataResult
from market_regime.models.features import FeatureConfig, FeatureInspection
from market_regime.models.technicals import TechnicalSnapshot, TechnicalSignal

# Services
from market_regime.service.regime_service import RegimeService
from market_regime.data.service import DataService

# Phase detection
from market_regime.phases.detector import PhaseDetector

# Features
from market_regime.features.pipeline import compute_features
from market_regime.features.technicals import compute_technicals

__all__ = [
    # Config
    "Settings",
    "get_settings",
    # Services
    "RegimeService",
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
    # Functions
    "compute_features",
    "compute_technicals",
]
