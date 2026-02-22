"""Regime detection data models."""

from __future__ import annotations

from datetime import date
from enum import IntEnum, StrEnum
from typing import Any

from pydantic import BaseModel, model_validator

from market_regime.models.features import FeatureInspection
from market_regime.models.phase import PhaseID, PhaseResult


class TrendDirection(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"


class RegimeID(IntEnum):
    R1_LOW_VOL_MR = 1
    R2_HIGH_VOL_MR = 2
    R3_LOW_VOL_TREND = 3
    R4_HIGH_VOL_TREND = 4

    @property
    def is_trending(self) -> bool:
        return self in (RegimeID.R3_LOW_VOL_TREND, RegimeID.R4_HIGH_VOL_TREND)

    @property
    def is_mean_reverting(self) -> bool:
        return self in (RegimeID.R1_LOW_VOL_MR, RegimeID.R2_HIGH_VOL_MR)

    @property
    def is_high_vol(self) -> bool:
        return self in (RegimeID.R2_HIGH_VOL_MR, RegimeID.R4_HIGH_VOL_TREND)

    @property
    def is_low_vol(self) -> bool:
        return self in (RegimeID.R1_LOW_VOL_MR, RegimeID.R3_LOW_VOL_TREND)


class RegimeResult(BaseModel):
    ticker: str
    regime: RegimeID
    confidence: float
    regime_probabilities: dict[int, float]
    trend_direction: TrendDirection | None = None
    as_of_date: date
    model_version: str


class RegimeConfig(BaseModel):
    n_states: int | None = None
    training_lookback_years: float | None = None
    feature_lookback_days: int | None = None
    refit_frequency_days: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _apply_defaults(cls, data: Any) -> Any:
        from market_regime.config import get_settings

        s = get_settings().regime
        defaults = {
            "n_states": s.n_states,
            "training_lookback_years": s.training_lookback_years,
            "feature_lookback_days": s.feature_lookback_days,
            "refit_frequency_days": s.refit_frequency_days,
        }
        if isinstance(data, dict):
            for key, default in defaults.items():
                if data.get(key) is None:
                    data[key] = default
        return data


class LabelAlignment(BaseModel):
    """Maps raw HMM states to RegimeID using vol/trend 2x2 grid."""

    state_to_regime: dict[int, int]
    per_state_vol_mean: dict[int, float]
    per_state_trend_mean: dict[int, float]
    vol_threshold: float
    trend_threshold: float


class HMMModelInfo(BaseModel):
    """Full inspection of a fitted HMM, indexed by RegimeID."""

    state_means: dict[int, list[float]]
    state_covariances: dict[int, list[list[float]]]
    transition_matrix: list[list[float]]
    initial_probabilities: dict[int, float]
    label_alignment: LabelAlignment
    feature_names: list[str]
    training_rows: int
    training_date_range: tuple[date, date] | None = None


class RegimeTimeSeriesEntry(BaseModel):
    """Single row in regime time series."""

    date: date
    regime: RegimeID
    confidence: float
    probabilities: dict[int, float]
    trend_direction: TrendDirection | None = None


class RegimeTimeSeries(BaseModel):
    """Regime classification for every date in a window."""

    ticker: str
    entries: list[RegimeTimeSeriesEntry]


class RegimeExplanation(BaseModel):
    """Master 'show everything' inspection model."""

    regime_result: RegimeResult
    feature_inspection: FeatureInspection
    model_info: HMMModelInfo
    regime_series: RegimeTimeSeries
    explanation_text: str


# --- Research API models ---


class TransitionRow(BaseModel):
    """Single row of the transition matrix with interpretation."""

    from_regime: RegimeID
    to_probabilities: dict[int, float]
    stay_probability: float
    stability: str  # "very sticky" | "sticky" | "moderately stable" | "unstable"
    likely_transition_target: RegimeID | None = None


class StateMeansRow(BaseModel):
    """Feature means for a single regime with vol/trend character."""

    regime: RegimeID
    feature_means: dict[str, float]
    vol_character: str  # "high-vol" | "low-vol"
    trend_character: str  # "trending" | "mean-rev"


class LabelAlignmentDetail(BaseModel):
    """Label alignment detail for a single regime."""

    regime: RegimeID
    vol_mean: float
    trend_mean: float
    vol_side: str  # "low" | "high"
    trend_side: str  # "MR" | "trend"
    vol_threshold: float
    trend_threshold: float


class FeatureZScore(BaseModel):
    """Z-score for a single feature with semantic comment."""

    feature: str
    z_score: float
    comment: str


class RegimeHistoryDay(BaseModel):
    """Single day in regime history with change detection."""

    date: date
    regime: RegimeID
    trend_direction: TrendDirection | None = None
    confidence: float
    changed_from: RegimeID | None = None


class RegimeDistributionEntry(BaseModel):
    """Distribution entry for a single regime over the full window."""

    regime: RegimeID
    name: str
    days: int
    percentage: float
    is_dominant: bool
    is_rare: bool


class TickerResearch(BaseModel):
    """Full interpreted regime research for a single ticker."""

    ticker: str
    regime_result: RegimeResult
    explanation_text: str
    transition_matrix: list[TransitionRow]
    state_means: list[StateMeansRow]
    label_alignment: list[LabelAlignmentDetail]
    current_features: list[FeatureZScore]
    recent_history: list[RegimeHistoryDay]
    regime_distribution: list[RegimeDistributionEntry]
    strategy_comment: str
    model_info: HMMModelInfo
    phase_result: PhaseResult | None = None


class CrossTickerEntry(BaseModel):
    """Cross-ticker comparison entry."""

    ticker: str
    regime: RegimeID
    trend_direction: TrendDirection | None = None
    confidence: float
    regime_probabilities: dict[int, float]
    strategy_comment: str
    phase: PhaseID | None = None
    phase_name: str | None = None


class ResearchReport(BaseModel):
    """Full research output for one or more tickers."""

    tickers: list[TickerResearch]
    comparison: list[CrossTickerEntry] | None = None
