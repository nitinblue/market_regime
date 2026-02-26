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
    cache_dir: str | None = None  # None = ~/.market_analyzer/cache
    model_dir: str | None = None  # None = ~/.market_analyzer/models


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
    figure_size: list[float] = Field(default_factory=lambda: [14, 10])
    height_ratios: list[float] = Field(default_factory=lambda: [5, 1.5, 1.5, 1])
    font_size: int = 8
    legend_alpha: float = 0.9
    xaxis_rotation: int = 30
    month_interval: int = 2


class TechnicalsSettings(BaseModel):
    sma_windows: list[int] = Field(default_factory=lambda: [20, 50, 200])
    ema_windows: list[int] = Field(default_factory=lambda: [9, 21])
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    vwma_window: int = 20
    atr_period: int = 14
    stochastic_k: int = 14
    stochastic_d: int = 3
    stochastic_overbought: float = 80.0
    stochastic_oversold: float = 20.0
    vcp_lookback_days: int = 120
    vcp_min_contractions: int = 2
    vcp_min_contraction_pct: float = 3.0
    vcp_tightening_ratio: float = 0.7
    vcp_ready_range_pct: float = 5.0
    vcp_pivot_proximity_pct: float = 3.0
    ob_lookback_days: int = 60
    ob_impulse_atr_multiple: float = 2.0
    ob_min_volume_factor: float = 1.3
    ob_max_blocks: int = 5
    fvg_lookback_days: int = 60
    fvg_min_gap_pct: float = 0.3
    fvg_max_gaps: int = 5


class PhaseSettings(BaseModel):
    swing_lookback: int = 5
    swing_threshold_pct: float = 1.5
    min_phase_days: int = 10
    volume_trend_window: int = 20
    volume_decline_threshold: float = 0.8
    range_analysis_window: int = 30
    sma_period: int = 50
    min_regime_run_days: int = 5
    names: dict[int, str] = Field(default_factory=lambda: {
        1: "Accumulation",
        2: "Markup",
        3: "Distribution",
        4: "Markdown",
    })
    colors: dict[int, str] = Field(default_factory=lambda: {
        1: "#8BC34A",
        2: "#4CAF50",
        3: "#FF5722",
        4: "#D32F2F",
    })
    strategies: dict[int, str] = Field(default_factory=lambda: {
        1: "LEAP entry zone: buy calls on dips to support. Low IV = cheap options.",
        2: "LEAP hold zone: ride existing LEAPs. Trail stops. Add on pullbacks.",
        3: "LEAP exit zone: take profits, tighten stops. Rotate to protective puts.",
        4: "LEAP avoid zone: close bullish LEAPs. Consider bear put LEAPs if R4.",
    })


class FundamentalsSettings(BaseModel):
    cache_ttl_minutes: int = 60


class MacroSettings(BaseModel):
    lookahead_days: int = 60


class ZeroDTESettings(BaseModel):
    earnings_blackout_days: int = 1
    min_atr_pct: float = 0.3
    max_atr_pct: float = 3.0
    r4_confidence_threshold: float = 0.7
    go_threshold: float = 0.55
    caution_threshold: float = 0.35
    atr_sweet_low: float = 0.5
    atr_sweet_high: float = 1.5
    regime_multipliers: dict[int, float] = Field(default_factory=lambda: {
        1: 1.0, 2: 0.85, 3: 0.6, 4: 0.3,
    })


class LEAPSettings(BaseModel):
    earnings_blackout_days: int = 5
    r4_confidence_threshold: float = 0.7
    distribution_confidence_threshold: float = 0.65
    markdown_confidence_threshold: float = 0.65
    min_fundamental_score: float = 0.2
    go_threshold: float = 0.50
    caution_threshold: float = 0.30
    earnings_growth_strong: float = 0.15
    earnings_growth_moderate: float = 0.05
    revenue_growth_strong: float = 0.15
    revenue_growth_moderate: float = 0.05
    margin_strong: float = 0.20
    margin_moderate: float = 0.10
    debt_low: float = 50.0
    debt_moderate: float = 100.0
    debt_high: float = 200.0
    pe_cheap: float = 15.0
    pe_fair: float = 25.0
    pe_expensive: float = 40.0
    bull_entry_52wk_pct: float = 40.0


class BreakoutSettings(BaseModel):
    r4_confidence_threshold: float = 0.70
    r2_confidence_threshold: float = 0.80
    earnings_blackout_days: int = 2
    min_base_days: int = 10
    go_threshold: float = 0.55
    caution_threshold: float = 0.35
    vcp_ready_min_score: float = 0.6
    vcp_breakout_confirmation: float = 0.8
    bollinger_squeeze_bandwidth: float = 0.04
    range_compression_threshold: float = 0.3
    volume_declining_threshold: float = 0.8
    resistance_proximity_pct: float = 2.0
    atr_low_baseline_pct: float = 0.5
    regime_multipliers: dict[int, float] = Field(default_factory=lambda: {
        1: 0.85, 2: 0.6, 3: 1.0, 4: 0.3,
    })


class MomentumSettings(BaseModel):
    r1_confidence_threshold: float = 0.70
    earnings_blackout_days: int = 3
    rsi_extreme_overbought: float = 85.0
    rsi_extreme_oversold: float = 15.0
    go_threshold: float = 0.55
    caution_threshold: float = 0.35
    rsi_healthy_bull_low: float = 50.0
    rsi_healthy_bull_high: float = 70.0
    rsi_healthy_bear_low: float = 30.0
    rsi_healthy_bear_high: float = 50.0
    volume_confirmation_factor: float = 1.2
    pullback_to_ma_pct: float = 1.5
    regime_multipliers: dict[int, float] = Field(default_factory=lambda: {
        1: 0.3, 2: 0.6, 3: 1.0, 4: 0.75,
    })


class CalendarSettings(BaseModel):
    r4_confidence_threshold: float = 0.70
    earnings_blackout_days: int = 5
    max_bid_ask_spread_pct: float = 2.0
    atm_iv_high: float = 0.25
    atm_iv_moderate: float = 0.15
    go_threshold: float = 0.55
    caution_threshold: float = 0.35
    regime_multipliers: dict[int, float] = Field(default_factory=lambda: {
        1: 1.0, 2: 0.90, 3: 0.65, 4: 0.3,
    })


class DiagonalSettings(BaseModel):
    r4_confidence_threshold: float = 0.70
    earnings_blackout_days: int = 5
    max_skew_ratio: float = 4.0
    go_threshold: float = 0.55
    caution_threshold: float = 0.35
    regime_multipliers: dict[int, float] = Field(default_factory=lambda: {
        1: 0.80, 2: 0.55, 3: 1.0, 4: 0.3,
    })


class IronButterflySettings(BaseModel):
    trending_confidence_threshold: float = 0.70
    earnings_blackout_days: int = 3
    min_atm_iv: float = 0.15
    atm_iv_excellent: float = 0.30
    atm_iv_good: float = 0.20
    go_threshold: float = 0.55
    caution_threshold: float = 0.35
    regime_multipliers: dict[int, float] = Field(default_factory=lambda: {
        1: 0.90, 2: 1.0, 3: 0.4, 4: 0.2,
    })


class IronCondorSettings(BaseModel):
    r4_confidence_threshold: float = 0.70
    r3_confidence_threshold: float = 0.75
    earnings_blackout_days: int = 3
    min_iv: float = 0.10
    iv_excellent: float = 0.25
    iv_good: float = 0.18
    go_threshold: float = 0.55
    caution_threshold: float = 0.35
    regime_multipliers: dict[int, float] = Field(default_factory=lambda: {
        1: 1.0, 2: 0.85, 3: 0.4, 4: 0.2,
    })


class RatioSpreadSettings(BaseModel):
    r2_confidence_threshold: float = 0.75
    earnings_blackout_days: int = 5
    min_skew_pct: float = 0.02
    margin_warning_threshold: str = "$50K+"
    go_threshold: float = 0.55
    caution_threshold: float = 0.35
    regime_multipliers: dict[int, float] = Field(default_factory=lambda: {
        1: 1.0, 2: 0.55, 3: 0.85, 4: 0.2,
    })


class OpportunitySettings(BaseModel):
    zero_dte: ZeroDTESettings = Field(default_factory=ZeroDTESettings)
    leap: LEAPSettings = Field(default_factory=LEAPSettings)
    breakout: BreakoutSettings = Field(default_factory=BreakoutSettings)
    momentum: MomentumSettings = Field(default_factory=MomentumSettings)
    calendar: CalendarSettings = Field(default_factory=CalendarSettings)
    diagonal: DiagonalSettings = Field(default_factory=DiagonalSettings)
    iron_condor: IronCondorSettings = Field(default_factory=IronCondorSettings)
    iron_butterfly: IronButterflySettings = Field(default_factory=IronButterflySettings)
    ratio_spread: RatioSpreadSettings = Field(default_factory=RatioSpreadSettings)


class LevelsSettings(BaseModel):
    confluence_proximity_pct: float = 0.5
    min_stop_distance_pct: float = 0.3
    max_stop_distance_pct: float = 5.0
    atr_stop_buffer_multiple: float = 0.5
    atr_fallback_multiple: float = 2.0
    min_target_distance_pct: float = 0.5
    min_risk_reward: float = 1.5
    max_targets: int = 3
    max_strength_denominator: float = 3.0
    source_weights: dict[str, float] = Field(default_factory=lambda: {
        "swing_support": 1.0,
        "swing_resistance": 1.0,
        "order_block_high": 0.9,
        "order_block_low": 0.9,
        "vcp_pivot": 0.85,
        "sma_200": 0.8,
        "sma_50": 0.7,
        "fvg_high": 0.7,
        "fvg_low": 0.7,
        "sma_20": 0.5,
        "ema_21": 0.5,
        "bollinger_upper": 0.5,
        "bollinger_middle": 0.5,
        "bollinger_lower": 0.5,
        "vwma_20": 0.5,
        "ema_9": 0.4,
        "orb_level": 0.6,
    })


class ORBSettings(BaseModel):
    opening_minutes: int = 30
    extensions: list[float] = Field(default_factory=lambda: [1.0, 1.5, 2.0])
    market_open_hour: int = 9
    market_open_minute: int = 30


class RankingWeightsSettings(BaseModel):
    verdict: float = 0.25
    confidence: float = 0.25
    regime_alignment: float = 0.15
    risk_reward: float = 0.15
    technical_quality: float = 0.10
    phase_alignment: float = 0.10


class RankingSettings(BaseModel):
    weights: RankingWeightsSettings = Field(default_factory=RankingWeightsSettings)
    income_bias_boost: float = 0.05
    macro_penalty_per_event: float = 0.02
    macro_penalty_max: float = 0.10
    earnings_penalty: float = 0.10
    earnings_proximity_days: int = 3
    risk_reward_excellent: float = 3.0
    risk_reward_good: float = 2.0
    risk_reward_fair: float = 1.0


class BlackSwanSettings(BaseModel):
    vix_warning: float = 20.0
    vix_danger: float = 30.0
    vix_critical: float = 40.0
    term_structure_warning: float = 0.95
    term_structure_danger: float = 1.05
    term_structure_critical: float = 1.20
    credit_warning_pct: float = -0.5
    credit_danger_pct: float = -1.5
    credit_critical_pct: float = -3.0
    spy_warning_pct: float = -1.0
    spy_danger_pct: float = -2.0
    spy_critical_pct: float = -4.0
    rv_iv_warning: float = 0.0
    rv_iv_danger: float = 5.0
    rv_iv_critical: float = 15.0
    tlt_warning_pct: float = 1.0
    tlt_danger_pct: float = 2.0
    tlt_critical_pct: float = 3.0
    em_warning_pct: float = -1.0
    em_danger_pct: float = -3.0
    em_critical_pct: float = -5.0
    yield_curve_warning_bps: float = 50.0
    yield_curve_danger_bps: float = 0.0
    yield_curve_critical_bps: float = -25.0
    put_call_warning: float = 0.80
    put_call_danger: float = 1.00
    put_call_critical: float = 1.30
    alert_elevated: float = 0.25
    alert_high: float = 0.50
    alert_critical: float = 0.75
    lookback_days: int = 20


class DisplaySettings(BaseModel):
    default_tickers: list[str] = Field(default_factory=lambda: ["SPX", "GLD", "QQQ", "TLT"])
    confidence_cap: float = 99.9
    plot: PlotSettings = Field(default_factory=PlotSettings)


class MarketDef(BaseModel):
    """Definition for a single market (US, India, etc.)."""

    name: str
    suffix: str = ""                    # "" for US, ".NS" for NSE, ".BO" for BSE
    currency: str = "USD"
    exchange: str = ""
    timezone: str = "America/New_York"
    market_open: str = "09:30"
    market_close: str = "16:00"
    reference_tickers: list[str] = Field(default_factory=list)
    stress_vix_ticker: str = "^VIX"     # Market-specific VIX equivalent


class MarketSettings(BaseModel):
    """Multi-market configuration."""

    default_market: str = "US"
    markets: dict[str, MarketDef] = Field(default_factory=lambda: {
        "US": MarketDef(
            name="US",
            suffix="",
            currency="USD",
            exchange="NYSE/NASDAQ",
            timezone="America/New_York",
            market_open="09:30",
            market_close="16:00",
            reference_tickers=["SPY", "QQQ", "TLT", "GLD", "HYG"],
            stress_vix_ticker="^VIX",
        ),
        "India": MarketDef(
            name="India",
            suffix=".NS",
            currency="INR",
            exchange="NSE",
            timezone="Asia/Kolkata",
            market_open="09:15",
            market_close="15:30",
            reference_tickers=["^NSEI", "^NSEBANK", "GOLDBEES.NS"],
            stress_vix_ticker="^INDIAVIX",
        ),
    })


class ScreeningSettings(BaseModel):
    """Settings for universe screening."""

    min_volume_20d_avg: int = 500_000
    min_price: float = 5.0
    max_price: float = 10_000.0
    breakout_proximity_pct: float = 3.0
    mean_reversion_rsi_low: float = 25.0
    mean_reversion_rsi_high: float = 75.0
    income_regime_preference: list[int] = Field(default_factory=lambda: [1, 2])
    momentum_min_rsi: float = 50.0


class StrategySettings(BaseModel):
    """Settings for strategy selection and sizing."""

    default_account_size: float = 50_000.0
    ira_account_size: float = 200_000.0
    max_position_pct: float = 0.05          # 5% of account per position
    max_portfolio_risk_pct: float = 0.20    # 20% total portfolio risk
    default_dte_range: list[int] = Field(default_factory=lambda: [30, 45])
    income_delta_range: list[float] = Field(default_factory=lambda: [0.15, 0.30])
    directional_delta_range: list[float] = Field(default_factory=lambda: [0.30, 0.50])


class ExitSettings(BaseModel):
    """Settings for exit planning."""

    profit_target_pcts: list[float] = Field(default_factory=lambda: [50.0, 75.0])
    stop_loss_pct: float = 200.0            # 2x credit received for credit spreads
    time_exit_dte: int = 7                  # Close at 7 DTE
    theta_decay_exit_pct: float = 50.0      # Close at 50% max profit (income)
    regime_change_review: bool = True       # Review on regime change


class TradingPlanSettings(BaseModel):
    """Settings for daily trading plan generation."""

    default_tickers: list[str] = Field(default_factory=lambda: ["SPX", "QQQ", "GLD", "TLT", "IWM"])
    max_trades_per_plan: int = 10
    daily_risk_pct: float = 0.02            # 2% of account per day
    max_new_positions_normal: int = 3
    max_new_positions_light: int = 1
    fill_slippage_pct: float = 0.20         # 20% slippage tolerance for cutoff
    include_0dte: bool = True
    include_leaps: bool = True


class Settings(BaseModel):
    """Central config — loaded from YAML, overridable per-field."""

    features: FeaturesSettings = Field(default_factory=FeaturesSettings)
    regime: RegimeSettings = Field(default_factory=RegimeSettings)
    hmm: HMMSettings = Field(default_factory=HMMSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    interpretation: InterpretationSettings = Field(default_factory=InterpretationSettings)
    regimes: RegimeDefinitionSettings = Field(default_factory=RegimeDefinitionSettings)
    phases: PhaseSettings = Field(default_factory=PhaseSettings)
    technicals: TechnicalsSettings = Field(default_factory=TechnicalsSettings)
    fundamentals: FundamentalsSettings = Field(default_factory=FundamentalsSettings)
    macro: MacroSettings = Field(default_factory=MacroSettings)
    opportunity: OpportunitySettings = Field(default_factory=OpportunitySettings)
    levels: LevelsSettings = Field(default_factory=LevelsSettings)
    orb: ORBSettings = Field(default_factory=ORBSettings)
    ranking: RankingSettings = Field(default_factory=RankingSettings)
    black_swan: BlackSwanSettings = Field(default_factory=BlackSwanSettings)
    display: DisplaySettings = Field(default_factory=DisplaySettings)
    markets: MarketSettings = Field(default_factory=MarketSettings)
    screening: ScreeningSettings = Field(default_factory=ScreeningSettings)
    strategy: StrategySettings = Field(default_factory=StrategySettings)
    exit: ExitSettings = Field(default_factory=ExitSettings)
    trading_plan: TradingPlanSettings = Field(default_factory=TradingPlanSettings)


# --- Loading ---

_DEFAULTS_PATH = Path(__file__).parent / "defaults.yaml"
_USER_CONFIG_PATH = Path.home() / ".market_analyzer" / "config.yaml"
_LEGACY_CONFIG_PATH = Path.home() / ".market_regime" / "config.yaml"

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
    """Load defaults.yaml, merge ~/.market_analyzer/config.yaml if present.

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

    # Layer 2: user overrides (check new path, fall back to legacy)
    user_path = user_config_path or _USER_CONFIG_PATH
    if not user_path.exists() and user_config_path is None and _LEGACY_CONFIG_PATH.exists():
        user_path = _LEGACY_CONFIG_PATH
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
