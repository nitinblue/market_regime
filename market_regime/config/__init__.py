"""Central configuration — loaded from YAML, overridable per-field."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# --- Settings models ---


class FeaturesSettings(BaseModel):
    log_return_windows: list[int] = [1, 5]
    realized_vol_window: int = 20
    atr_window: int = 14
    trend_window: int = 20
    volume_window: int = 20
    annualization_factor: int = 252


class RegimeSettings(BaseModel):
    n_states: int = 4
    training_lookback_years: float = 2.0
    feature_lookback_days: int = 60
    refit_frequency_days: int = 30


class HMMSettings(BaseModel):
    covariance_type: str = "full"
    n_iter: int = 100
    random_state: int = 42


class CacheSettings(BaseModel):
    staleness_hours: float = 18.0
    cache_dir: str | None = None  # None = ~/.market_regime/cache
    model_dir: str | None = None  # None = ~/.market_regime/models


class ZScoreThresholds(BaseModel):
    normal: float = 0.5
    mild: float = 1.0
    elevated: float = 2.0


class StabilityThresholds(BaseModel):
    very_sticky: float = 0.95
    sticky: float = 0.90
    moderately_stable: float = 0.80


class InterpretationSettings(BaseModel):
    zscore_thresholds: ZScoreThresholds = Field(default_factory=ZScoreThresholds)
    stability_thresholds: StabilityThresholds = Field(default_factory=StabilityThresholds)
    trend_strength_boundary: float = 0.3
    rare_regime_pct: float = 10.0
    recent_history_days: int = 20


class RegimeDefinitionSettings(BaseModel):
    names: dict[int, str] = Field(default_factory=lambda: {
        1: "Low-Vol Mean Reverting",
        2: "High-Vol Mean Reverting",
        3: "Low-Vol Trending",
        4: "High-Vol Trending",
    })
    strategies: dict[int, str] = Field(default_factory=lambda: {
        1: "Primary: theta (IC, strangles). Avoid directional.",
        2: "Selective: theta (wider wings). Avoid directional.",
        3: "Primary: directional spreads. Light theta.",
        4: "Selective: directional (risk-defined). Long vega.",
    })
    colors: dict[int, str] = Field(default_factory=lambda: {
        1: "#4CAF50",
        2: "#FF9800",
        3: "#2196F3",
        4: "#F44336",
    })
    labels: dict[int, str] = Field(default_factory=lambda: {
        1: "R1: Low-Vol MR",
        2: "R2: High-Vol MR",
        3: "R3: Low-Vol Trend",
        4: "R4: High-Vol Trend",
    })


class PlotSettings(BaseModel):
    figure_size: list[int] = Field(default_factory=lambda: [14, 7])
    height_ratios: list[int] = Field(default_factory=lambda: [3, 1])
    font_size: int = 8
    legend_alpha: float = 0.9
    xaxis_rotation: int = 30
    month_interval: int = 2


class DisplaySettings(BaseModel):
    default_tickers: list[str] = Field(default_factory=lambda: ["SPY", "GLD", "QQQ", "TLT"])
    confidence_cap: float = 99.9
    plot: PlotSettings = Field(default_factory=PlotSettings)


class Settings(BaseModel):
    """Central config — loaded from YAML, overridable per-field."""

    features: FeaturesSettings = Field(default_factory=FeaturesSettings)
    regime: RegimeSettings = Field(default_factory=RegimeSettings)
    hmm: HMMSettings = Field(default_factory=HMMSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    interpretation: InterpretationSettings = Field(default_factory=InterpretationSettings)
    regimes: RegimeDefinitionSettings = Field(default_factory=RegimeDefinitionSettings)
    display: DisplaySettings = Field(default_factory=DisplaySettings)


# --- Loading ---

_DEFAULTS_PATH = Path(__file__).parent / "defaults.yaml"
_USER_CONFIG_PATH = Path.home() / ".market_regime" / "config.yaml"

_cached_settings: Settings | None = None


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Returns new dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(
    user_config_path: Path | None = None,
    _force_reload: bool = False,
) -> Settings:
    """Load defaults.yaml, merge ~/.market_regime/config.yaml if present.

    Args:
        user_config_path: Override path for user config file.
        _force_reload: Bypass cache (for testing).

    Returns:
        Merged Settings instance.
    """
    global _cached_settings
    if _cached_settings is not None and not _force_reload:
        return _cached_settings

    # Layer 1: package defaults
    with open(_DEFAULTS_PATH) as f:
        defaults = yaml.safe_load(f)

    # Layer 2: user overrides
    user_path = user_config_path or _USER_CONFIG_PATH
    if user_path.exists():
        with open(user_path) as f:
            user = yaml.safe_load(f) or {}
        merged = _deep_merge(defaults, user)
    else:
        merged = defaults

    _cached_settings = Settings(**merged)
    return _cached_settings


def get_settings() -> Settings:
    """Get cached settings (singleton). Loads on first call."""
    return load_settings()


def reset_settings() -> None:
    """Clear cached settings. Next get_settings() will reload from YAML."""
    global _cached_settings
    _cached_settings = None
