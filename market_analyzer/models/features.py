"""Feature engineering data models."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, model_validator


class FeatureConfig(BaseModel):
    log_return_windows: list[int] | None = None
    realized_vol_window: int | None = None
    atr_window: int | None = None
    trend_window: int | None = None
    volume_window: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _apply_defaults(cls, data: Any) -> Any:
        from market_analyzer.config import get_settings

        s = get_settings().features
        defaults = {
            "log_return_windows": s.log_return_windows,
            "realized_vol_window": s.realized_vol_window,
            "atr_window": s.atr_window,
            "trend_window": s.trend_window,
            "volume_window": s.volume_window,
        }
        if isinstance(data, dict):
            for key, default in defaults.items():
                if data.get(key) is None:
                    data[key] = default
        return data


class FeatureVector(BaseModel):
    log_return_1d: float
    log_return_5d: float
    realized_vol: float
    atr_normalized: float
    trend_strength: float
    volume_anomaly: float | None = None


class FeatureInspection(BaseModel):
    """Full inspection of feature computation for a ticker."""

    ticker: str
    date_range: tuple[date, date]
    raw_row_count: int
    normalized_row_count: int
    feature_names: list[str]
    raw_features: list[dict]
    normalized_features: list[dict]
    normalization_means: list[dict]
    normalization_stds: list[dict]
