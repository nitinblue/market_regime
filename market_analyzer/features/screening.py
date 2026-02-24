"""Universe screening functions — pure computation, no data fetching."""

from __future__ import annotations

from typing import TYPE_CHECKING

from market_analyzer.config import get_settings
from market_analyzer.models.regime import RegimeID

if TYPE_CHECKING:
    from market_analyzer.models.phase import PhaseID, PhaseResult
    from market_analyzer.models.regime import RegimeResult
    from market_analyzer.models.technicals import TechnicalSnapshot


def passes_volume_filter(technicals: TechnicalSnapshot) -> bool:
    """Check if ticker has sufficient average volume."""
    cfg = get_settings().screening
    # Use VWMA as a proxy for average volume activity
    return technicals.current_price >= cfg.min_price and technicals.current_price <= cfg.max_price


def screen_breakout(
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
) -> tuple[bool, float, str]:
    """Screen for breakout candidates.

    Returns:
        (passes, score 0-1, reason)
    """
    cfg = get_settings().screening
    score = 0.0
    reasons: list[str] = []

    # VCP setup present
    if technicals.vcp is not None and technicals.vcp.score >= 0.5:
        score += 0.3
        reasons.append(f"VCP score {technicals.vcp.score:.2f}")

    # Bollinger squeeze
    if technicals.bollinger.bandwidth < 0.05:
        score += 0.2
        reasons.append("Bollinger squeeze")

    # Near resistance (within proximity %)
    sr = technicals.support_resistance
    if sr.resistance is not None and sr.price_vs_resistance_pct is not None:
        if abs(sr.price_vs_resistance_pct) <= cfg.breakout_proximity_pct:
            score += 0.2
            reasons.append(f"Near resistance ({sr.price_vs_resistance_pct:+.1f}%)")

    # Trending regime favors breakouts
    if regime.regime in (RegimeID.R3_LOW_VOL_TREND,):
        score += 0.2
        reasons.append("R3 trending regime")
    elif regime.regime == RegimeID.R1_LOW_VOL_MR:
        score += 0.1
        reasons.append("R1 low-vol (base building)")

    reason = "; ".join(reasons) if reasons else "No breakout signals"
    return score >= 0.3, min(score, 1.0), reason


def screen_momentum(
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
) -> tuple[bool, float, str]:
    """Screen for momentum continuation candidates.

    Returns:
        (passes, score 0-1, reason)
    """
    cfg = get_settings().screening
    score = 0.0
    reasons: list[str] = []

    # RSI in healthy zone
    rsi = technicals.rsi.value
    if cfg.momentum_min_rsi <= rsi <= 70:
        score += 0.25
        reasons.append(f"RSI {rsi:.0f} (healthy)")
    elif 70 < rsi <= 80:
        score += 0.15
        reasons.append(f"RSI {rsi:.0f} (strong but extended)")

    # MACD bullish
    if technicals.macd.histogram > 0:
        score += 0.2
        reasons.append("MACD histogram positive")
    if technicals.macd.is_bullish_crossover:
        score += 0.15
        reasons.append("MACD bullish crossover")

    # Above key MAs
    ma = technicals.moving_averages
    if ma.price_vs_sma_50_pct > 0 and ma.price_vs_sma_200_pct > 0:
        score += 0.2
        reasons.append("Above SMA50 & SMA200")
    elif ma.price_vs_sma_50_pct > 0:
        score += 0.1
        reasons.append("Above SMA50")

    # Trending regime
    if regime.regime.is_trending:
        score += 0.2
        reasons.append(f"R{regime.regime} trending")

    reason = "; ".join(reasons) if reasons else "No momentum signals"
    return score >= 0.3, min(score, 1.0), reason


def screen_mean_reversion(
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
) -> tuple[bool, float, str]:
    """Screen for mean-reversion / oversold bounce candidates.

    Returns:
        (passes, score 0-1, reason)
    """
    cfg = get_settings().screening
    score = 0.0
    reasons: list[str] = []

    rsi = technicals.rsi.value

    # RSI oversold
    if rsi <= cfg.mean_reversion_rsi_low:
        score += 0.35
        reasons.append(f"RSI {rsi:.0f} (oversold)")
    elif rsi <= 35:
        score += 0.2
        reasons.append(f"RSI {rsi:.0f} (approaching oversold)")

    # RSI overbought (short side mean reversion)
    if rsi >= cfg.mean_reversion_rsi_high:
        score += 0.3
        reasons.append(f"RSI {rsi:.0f} (overbought)")

    # Bollinger %B extreme
    bb = technicals.bollinger
    if bb.percent_b <= 0.0:
        score += 0.25
        reasons.append("Below lower Bollinger band")
    elif bb.percent_b >= 1.0:
        score += 0.25
        reasons.append("Above upper Bollinger band")

    # Mean-reverting regime
    if regime.regime.is_mean_reverting:
        score += 0.2
        reasons.append(f"R{regime.regime} mean-reverting")

    # Stochastic oversold
    if technicals.stochastic.is_oversold:
        score += 0.1
        reasons.append("Stochastic oversold")
    elif technicals.stochastic.is_overbought:
        score += 0.1
        reasons.append("Stochastic overbought")

    reason = "; ".join(reasons) if reasons else "No mean-reversion signals"
    return score >= 0.3, min(score, 1.0), reason


def screen_income(
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
) -> tuple[bool, float, str]:
    """Screen for income/theta-harvesting candidates.

    Returns:
        (passes, score 0-1, reason)
    """
    cfg = get_settings().screening
    score = 0.0
    reasons: list[str] = []

    # Prefer R1/R2 (mean-reverting regimes)
    if regime.regime == RegimeID.R1_LOW_VOL_MR:
        score += 0.35
        reasons.append("R1 ideal for income")
    elif regime.regime == RegimeID.R2_HIGH_VOL_MR:
        score += 0.25
        reasons.append("R2 selective income")

    # RSI neutral zone (range-bound)
    rsi = technicals.rsi.value
    if 35 <= rsi <= 65:
        score += 0.2
        reasons.append(f"RSI {rsi:.0f} (neutral range)")

    # Low ATR% (manageable moves)
    if technicals.atr_pct < 1.5:
        score += 0.15
        reasons.append(f"ATR% {technicals.atr_pct:.2f} (low)")
    elif technicals.atr_pct < 2.5:
        score += 0.1
        reasons.append(f"ATR% {technicals.atr_pct:.2f} (moderate)")

    # High confidence in regime
    if regime.confidence >= 0.7:
        score += 0.15
        reasons.append(f"High regime confidence ({regime.confidence:.0%})")

    # Phase indicator: accumulation or distribution (range-bound)
    phase = technicals.phase
    if phase.phase.value in ("accumulation", "distribution"):
        score += 0.15
        reasons.append(f"Phase: {phase.phase.value}")

    reason = "; ".join(reasons) if reasons else "Not ideal for income"
    return score >= 0.3, min(score, 1.0), reason
