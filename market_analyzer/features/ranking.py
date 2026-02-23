"""Pure scoring functions for trade idea ranking.

All functions are stateless — they accept data and config, return scores.
No I/O, no service dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from market_analyzer.config import RankingSettings, get_settings
from market_analyzer.models.ranking import ScoreBreakdown, StrategyType
from market_analyzer.models.opportunity import Verdict
from market_analyzer.models.technicals import TechnicalSnapshot
from market_analyzer.models.levels import LevelsAnalysis


# --- Alignment matrices ---

# Regime x Strategy alignment (R1=1, R2=2, R3=3, R4=4)
REGIME_STRATEGY_ALIGNMENT: dict[tuple[int, StrategyType], float] = {
    (1, StrategyType.ZERO_DTE): 1.0,
    (1, StrategyType.LEAP): 0.3,
    (1, StrategyType.BREAKOUT): 0.4,
    (1, StrategyType.MOMENTUM): 0.2,
    (2, StrategyType.ZERO_DTE): 0.6,
    (2, StrategyType.LEAP): 0.2,
    (2, StrategyType.BREAKOUT): 0.3,
    (2, StrategyType.MOMENTUM): 0.3,
    (3, StrategyType.ZERO_DTE): 0.5,
    (3, StrategyType.LEAP): 1.0,
    (3, StrategyType.BREAKOUT): 0.8,
    (3, StrategyType.MOMENTUM): 1.0,
    (4, StrategyType.ZERO_DTE): 0.3,
    (4, StrategyType.LEAP): 0.5,
    (4, StrategyType.BREAKOUT): 1.0,
    (4, StrategyType.MOMENTUM): 0.8,
}

# Phase x Strategy alignment (P1=1, P2=2, P3=3, P4=4)
PHASE_STRATEGY_ALIGNMENT: dict[tuple[int, StrategyType], float] = {
    (1, StrategyType.ZERO_DTE): 0.7,
    (1, StrategyType.LEAP): 0.9,
    (1, StrategyType.BREAKOUT): 1.0,
    (1, StrategyType.MOMENTUM): 0.3,
    (2, StrategyType.ZERO_DTE): 0.8,
    (2, StrategyType.LEAP): 0.7,
    (2, StrategyType.BREAKOUT): 0.5,
    (2, StrategyType.MOMENTUM): 1.0,
    (3, StrategyType.ZERO_DTE): 0.7,
    (3, StrategyType.LEAP): 0.2,
    (3, StrategyType.BREAKOUT): 0.3,
    (3, StrategyType.MOMENTUM): 0.4,
    (4, StrategyType.ZERO_DTE): 0.4,
    (4, StrategyType.LEAP): 0.1,
    (4, StrategyType.BREAKOUT): 0.2,
    (4, StrategyType.MOMENTUM): 0.6,
}


def get_regime_alignment(regime_id: int, strategy: StrategyType) -> float:
    """Look up regime x strategy alignment score."""
    return REGIME_STRATEGY_ALIGNMENT.get((regime_id, strategy), 0.5)


def get_phase_alignment(phase_id: int, strategy: StrategyType) -> float:
    """Look up phase x strategy alignment score."""
    return PHASE_STRATEGY_ALIGNMENT.get((phase_id, strategy), 0.5)


def compute_verdict_score(verdict: Verdict) -> float:
    """Convert verdict to a 0-1 score."""
    if verdict == Verdict.GO:
        return 1.0
    elif verdict == Verdict.CAUTION:
        return 0.5
    return 0.0


def compute_technical_quality(technicals: TechnicalSnapshot) -> float:
    """Compute a 0-1 composite technical quality score.

    Components:
    - RSI in healthy zone (30-70): 0.3
    - MACD bullish crossover: 0.2, bearish: 0.0, neither: 0.1
    - Price above SMA-50: 0.2
    - Price above SMA-200: 0.15
    - Stochastic not in extreme: 0.15
    """
    score = 0.0

    # RSI: healthy zone = 30-70
    rsi = technicals.rsi.value
    if 30 <= rsi <= 70:
        score += 0.3
    elif 20 <= rsi < 30 or 70 < rsi <= 80:
        score += 0.15  # mildly extreme
    # else: extreme — no points

    # MACD crossover
    if technicals.macd.is_bullish_crossover:
        score += 0.2
    elif technicals.macd.is_bearish_crossover:
        score += 0.0
    else:
        score += 0.1

    # Price vs SMA-50
    if technicals.moving_averages.price_vs_sma_50_pct > 0:
        score += 0.2

    # Price vs SMA-200
    if technicals.moving_averages.price_vs_sma_200_pct > 0:
        score += 0.15

    # Stochastic: not in extreme (20-80)
    if not technicals.stochastic.is_overbought and not technicals.stochastic.is_oversold:
        score += 0.15

    return min(score, 1.0)


def compute_risk_reward_score(
    levels: LevelsAnalysis | None,
    cfg: RankingSettings | None = None,
) -> float:
    """Convert risk/reward ratio to a 0-1 score.

    R:R >= 3.0: 1.0
    R:R 2.0-3.0: 0.7
    R:R 1.0-2.0: 0.4
    R:R < 1.0 or unavailable: 0.1
    """
    if cfg is None:
        cfg = get_settings().ranking

    if levels is None or levels.best_target is None:
        return 0.1

    rr = levels.best_target.risk_reward_ratio
    if rr >= cfg.risk_reward_excellent:
        return 1.0
    elif rr >= cfg.risk_reward_good:
        return 0.7
    elif rr >= cfg.risk_reward_fair:
        return 0.4
    return 0.1


def compute_income_bias_boost(
    strategy: StrategyType,
    regime_id: int,
    cfg: RankingSettings | None = None,
) -> float:
    """Income-first bias: boost theta strategies in R1/R2."""
    if cfg is None:
        cfg = get_settings().ranking

    if strategy == StrategyType.ZERO_DTE and regime_id in (1, 2):
        return cfg.income_bias_boost
    return 0.0


def compute_macro_penalty(
    events_next_7_days: int,
    cfg: RankingSettings | None = None,
) -> float:
    """Penalty for nearby macro events. -0.02 per event, max -0.10."""
    if cfg is None:
        cfg = get_settings().ranking
    return min(events_next_7_days * cfg.macro_penalty_per_event, cfg.macro_penalty_max)


def compute_earnings_penalty(
    days_to_earnings: int | None,
    cfg: RankingSettings | None = None,
) -> float:
    """Penalty if earnings are within proximity threshold."""
    if cfg is None:
        cfg = get_settings().ranking
    if days_to_earnings is not None and 0 <= days_to_earnings <= cfg.earnings_proximity_days:
        return cfg.earnings_penalty
    return 0.0


def compute_composite_score(
    verdict: Verdict,
    confidence: float,
    regime_id: int,
    phase_id: int,
    strategy: StrategyType,
    technicals: TechnicalSnapshot,
    levels: LevelsAnalysis | None,
    black_swan_score: float,
    events_next_7_days: int,
    days_to_earnings: int | None,
    weights: dict[str, float] | None = None,
    cfg: RankingSettings | None = None,
) -> ScoreBreakdown:
    """Compute the full composite score breakdown for a single ticker x strategy.

    Args:
        verdict: GO / CAUTION / NO_GO from opportunity assessment.
        confidence: 0-1 confidence from opportunity assessment.
        regime_id: Current regime (1-4).
        phase_id: Current phase (1-4).
        strategy: Strategy type being scored.
        technicals: Technical snapshot for the ticker.
        levels: Price levels analysis (may be None).
        black_swan_score: 0-1 composite stress score from BlackSwanAlert.
        events_next_7_days: Count of macro events in next 7 days.
        days_to_earnings: Days until next earnings (None if unknown).
        weights: Override component weights (for WeightProvider).
        cfg: Override ranking config.

    Returns:
        ScoreBreakdown with all component scores and final composite.
    """
    if cfg is None:
        cfg = get_settings().ranking
    if weights is None:
        w = cfg.weights
        weights = {
            "verdict": w.verdict,
            "confidence": w.confidence,
            "regime_alignment": w.regime_alignment,
            "risk_reward": w.risk_reward,
            "technical_quality": w.technical_quality,
            "phase_alignment": w.phase_alignment,
        }

    # Component scores
    verdict_score = compute_verdict_score(verdict)
    confidence_score = min(max(confidence, 0.0), 1.0)
    regime_alignment = get_regime_alignment(regime_id, strategy)
    risk_reward = compute_risk_reward_score(levels, cfg)
    technical_quality = compute_technical_quality(technicals)
    phase_alignment = get_phase_alignment(phase_id, strategy)
    income_bias = compute_income_bias_boost(strategy, regime_id, cfg)
    macro_pen = compute_macro_penalty(events_next_7_days, cfg)
    earnings_pen = compute_earnings_penalty(days_to_earnings, cfg)
    black_swan_penalty = black_swan_score  # stored for transparency

    # Weighted base score
    base = (
        weights["verdict"] * verdict_score
        + weights["confidence"] * confidence_score
        + weights["regime_alignment"] * regime_alignment
        + weights["risk_reward"] * risk_reward
        + weights["technical_quality"] * technical_quality
        + weights["phase_alignment"] * phase_alignment
    )

    # Apply adjustments
    base += income_bias
    base -= macro_pen
    base -= earnings_pen

    # Black swan: multiplicative penalty
    base *= (1.0 - black_swan_score)

    # Clamp to [0, 1]
    base = min(max(base, 0.0), 1.0)

    return ScoreBreakdown(
        verdict_score=verdict_score,
        confidence_score=confidence_score,
        regime_alignment=regime_alignment,
        risk_reward=risk_reward,
        technical_quality=technical_quality,
        phase_alignment=phase_alignment,
        income_bias_boost=income_bias,
        black_swan_penalty=black_swan_penalty,
        macro_penalty=macro_pen,
        earnings_penalty=earnings_pen,
    )


def composite_from_breakdown(breakdown: ScoreBreakdown, weights: dict[str, float] | None = None) -> float:
    """Recompute the final composite score from a ScoreBreakdown.

    Useful when you need the scalar separately from the breakdown.
    """
    if weights is None:
        cfg = get_settings().ranking
        w = cfg.weights
        weights = {
            "verdict": w.verdict,
            "confidence": w.confidence,
            "regime_alignment": w.regime_alignment,
            "risk_reward": w.risk_reward,
            "technical_quality": w.technical_quality,
            "phase_alignment": w.phase_alignment,
        }

    base = (
        weights["verdict"] * breakdown.verdict_score
        + weights["confidence"] * breakdown.confidence_score
        + weights["regime_alignment"] * breakdown.regime_alignment
        + weights["risk_reward"] * breakdown.risk_reward
        + weights["technical_quality"] * breakdown.technical_quality
        + weights["phase_alignment"] * breakdown.phase_alignment
    )
    base += breakdown.income_bias_boost
    base -= breakdown.macro_penalty
    base -= breakdown.earnings_penalty
    base *= (1.0 - breakdown.black_swan_penalty)
    return min(max(base, 0.0), 1.0)


# --- WeightProvider ABC (ML hook) ---


class WeightProvider(ABC):
    """Abstract base for weight providers. Default is config-based."""

    @abstractmethod
    def get_weights(self, ticker: str, strategy: StrategyType) -> dict[str, float]:
        """Return component weights for scoring."""
        ...


class ConfigWeightProvider(WeightProvider):
    """Default: reads weights from config/defaults.yaml."""

    def __init__(self, cfg: RankingSettings | None = None) -> None:
        if cfg is None:
            cfg = get_settings().ranking
        self._weights = {
            "verdict": cfg.weights.verdict,
            "confidence": cfg.weights.confidence,
            "regime_alignment": cfg.weights.regime_alignment,
            "risk_reward": cfg.weights.risk_reward,
            "technical_quality": cfg.weights.technical_quality,
            "phase_alignment": cfg.weights.phase_alignment,
        }

    def get_weights(self, ticker: str, strategy: StrategyType) -> dict[str, float]:
        return self._weights.copy()
