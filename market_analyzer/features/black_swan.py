"""Pure computation for tail-risk / black swan alert scoring.

All functions are stateless — they accept raw values and config,
and return model objects.  No network, no side effects.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from market_analyzer.config import BlackSwanSettings
from market_analyzer.models.black_swan import (
    AlertLevel,
    BlackSwanAlert,
    CircuitBreaker,
    IndicatorStatus,
    StressIndicator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interpolate(value: float, lo: float, hi: float) -> float:
    """Linear interpolation clamped to [0, 1]."""
    if hi == lo:
        return 1.0 if value >= hi else 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def _status_from_score(score: float) -> IndicatorStatus:
    if score >= 0.85:
        return IndicatorStatus.CRITICAL
    if score >= 0.5:
        return IndicatorStatus.DANGER
    if score >= 0.2:
        return IndicatorStatus.WARNING
    return IndicatorStatus.NORMAL


# ---------------------------------------------------------------------------
# Individual indicator scorers
# ---------------------------------------------------------------------------

def score_vix_level(
    vix: float | None,
    cfg: BlackSwanSettings,
) -> StressIndicator:
    if vix is None:
        return StressIndicator(
            name="VIX Level", value=None, score=0.0,
            status=IndicatorStatus.UNAVAILABLE, weight=0.20,
            description="VIX data unavailable",
        )
    if vix >= cfg.vix_critical:
        score = 1.0
    elif vix >= cfg.vix_danger:
        score = 0.5 + _interpolate(vix, cfg.vix_danger, cfg.vix_critical) * 0.5
    elif vix >= cfg.vix_warning:
        score = 0.2 + _interpolate(vix, cfg.vix_warning, cfg.vix_danger) * 0.3
    else:
        score = _interpolate(vix, 0.0, cfg.vix_warning) * 0.2
    return StressIndicator(
        name="VIX Level", value=vix, score=round(score, 4),
        status=_status_from_score(score), weight=0.20,
        description=f"VIX at {vix:.1f}",
    )


def score_vix_term_structure(
    ratio: float | None,
    cfg: BlackSwanSettings,
) -> StressIndicator:
    if ratio is None:
        return StressIndicator(
            name="VIX Term Structure", value=None, score=0.0,
            status=IndicatorStatus.UNAVAILABLE, weight=0.12,
            description="VIX term structure data unavailable",
        )
    if ratio >= cfg.term_structure_critical:
        score = 1.0
    elif ratio >= cfg.term_structure_danger:
        score = 0.6 + _interpolate(ratio, cfg.term_structure_danger, cfg.term_structure_critical) * 0.4
    elif ratio >= cfg.term_structure_warning:
        score = 0.3 + _interpolate(ratio, cfg.term_structure_warning, cfg.term_structure_danger) * 0.3
    else:
        score = _interpolate(ratio, 0.0, cfg.term_structure_warning) * 0.3
    state = "backwardation" if ratio > 1.0 else "contango"
    return StressIndicator(
        name="VIX Term Structure", value=ratio, score=round(score, 4),
        status=_status_from_score(score), weight=0.12,
        description=f"VIX/VIX3M ratio {ratio:.3f} ({state})",
    )


def score_credit_stress(
    pct_change: float | None,
    cfg: BlackSwanSettings,
) -> StressIndicator:
    if pct_change is None:
        return StressIndicator(
            name="Credit Stress", value=None, score=0.0,
            status=IndicatorStatus.UNAVAILABLE, weight=0.15,
            description="Credit data unavailable",
        )
    # pct_change is negative when stress increases (HYG/LQD drops)
    if pct_change <= cfg.credit_critical_pct:
        score = 1.0
    elif pct_change <= cfg.credit_danger_pct:
        score = 0.6 + _interpolate(-pct_change, -cfg.credit_danger_pct, -cfg.credit_critical_pct) * 0.4
    elif pct_change <= cfg.credit_warning_pct:
        score = 0.3 + _interpolate(-pct_change, -cfg.credit_warning_pct, -cfg.credit_danger_pct) * 0.3
    else:
        score = 0.0
    return StressIndicator(
        name="Credit Stress", value=pct_change, score=round(score, 4),
        status=_status_from_score(score), weight=0.15,
        description=f"HYG/LQD ratio change {pct_change:+.2f}% from 20d avg",
    )


def score_spy_drawdown(
    daily_return_pct: float | None,
    cfg: BlackSwanSettings,
) -> StressIndicator:
    if daily_return_pct is None:
        return StressIndicator(
            name="SPY Drawdown", value=None, score=0.0,
            status=IndicatorStatus.UNAVAILABLE, weight=0.13,
            description="SPY data unavailable",
        )
    if daily_return_pct <= cfg.spy_critical_pct:
        score = 1.0
    elif daily_return_pct <= cfg.spy_danger_pct:
        score = 0.6 + _interpolate(-daily_return_pct, -cfg.spy_danger_pct, -cfg.spy_critical_pct) * 0.4
    elif daily_return_pct <= cfg.spy_warning_pct:
        score = 0.3 + _interpolate(-daily_return_pct, -cfg.spy_warning_pct, -cfg.spy_danger_pct) * 0.3
    else:
        score = 0.0
    return StressIndicator(
        name="SPY Drawdown", value=daily_return_pct, score=round(score, 4),
        status=_status_from_score(score), weight=0.13,
        description=f"SPY 1-day return {daily_return_pct:+.2f}%",
    )


def score_rv_iv_gap(
    gap: float | None,
    cfg: BlackSwanSettings,
) -> StressIndicator:
    if gap is None:
        return StressIndicator(
            name="RV vs IV", value=None, score=0.0,
            status=IndicatorStatus.UNAVAILABLE, weight=0.10,
            description="RV/IV data unavailable",
        )
    if gap >= cfg.rv_iv_critical:
        score = 1.0
    elif gap >= cfg.rv_iv_danger:
        score = 0.6 + _interpolate(gap, cfg.rv_iv_danger, cfg.rv_iv_critical) * 0.4
    elif gap >= cfg.rv_iv_warning:
        score = 0.3 + _interpolate(gap, cfg.rv_iv_warning, cfg.rv_iv_danger) * 0.3
    else:
        score = 0.0
    return StressIndicator(
        name="RV vs IV", value=gap, score=round(score, 4),
        status=_status_from_score(score), weight=0.10,
        description=f"20d RV minus VIX: {gap:+.1f} pts",
    )


def score_treasury_stress(
    abs_return_pct: float | None,
    cfg: BlackSwanSettings,
) -> StressIndicator:
    if abs_return_pct is None:
        return StressIndicator(
            name="Treasury Stress", value=None, score=0.0,
            status=IndicatorStatus.UNAVAILABLE, weight=0.08,
            description="TLT data unavailable",
        )
    if abs_return_pct >= cfg.tlt_critical_pct:
        score = 1.0
    elif abs_return_pct >= cfg.tlt_danger_pct:
        score = 0.6 + _interpolate(abs_return_pct, cfg.tlt_danger_pct, cfg.tlt_critical_pct) * 0.4
    elif abs_return_pct >= cfg.tlt_warning_pct:
        score = 0.3 + _interpolate(abs_return_pct, cfg.tlt_warning_pct, cfg.tlt_danger_pct) * 0.3
    else:
        score = 0.0
    return StressIndicator(
        name="Treasury Stress", value=abs_return_pct, score=round(score, 4),
        status=_status_from_score(score), weight=0.08,
        description=f"TLT absolute 1-day return {abs_return_pct:.2f}%",
    )


def score_em_contagion(
    pct_change: float | None,
    cfg: BlackSwanSettings,
) -> StressIndicator:
    if pct_change is None:
        return StressIndicator(
            name="EM Contagion", value=None, score=0.0,
            status=IndicatorStatus.UNAVAILABLE, weight=0.08,
            description="EEM/SPY data unavailable",
        )
    if pct_change <= cfg.em_critical_pct:
        score = 1.0
    elif pct_change <= cfg.em_danger_pct:
        score = 0.6 + _interpolate(-pct_change, -cfg.em_danger_pct, -cfg.em_critical_pct) * 0.4
    elif pct_change <= cfg.em_warning_pct:
        score = 0.3 + _interpolate(-pct_change, -cfg.em_warning_pct, -cfg.em_danger_pct) * 0.3
    else:
        score = 0.0
    return StressIndicator(
        name="EM Contagion", value=pct_change, score=round(score, 4),
        status=_status_from_score(score), weight=0.08,
        description=f"EEM/SPY ratio change {pct_change:+.2f}% from 20d avg",
    )


def score_yield_curve(
    spread_bps: float | None,
    cfg: BlackSwanSettings,
) -> StressIndicator:
    if spread_bps is None:
        return StressIndicator(
            name="Yield Curve", value=None, score=0.0,
            status=IndicatorStatus.UNAVAILABLE, weight=0.07,
            description="Yield curve data unavailable (FRED)",
        )
    # Inverted = lower spread = more stress. Thresholds decrease.
    if spread_bps <= cfg.yield_curve_critical_bps:
        score = 0.8
    elif spread_bps <= cfg.yield_curve_danger_bps:
        score = 0.5 + _interpolate(-spread_bps, -cfg.yield_curve_danger_bps, -cfg.yield_curve_critical_bps) * 0.3
    elif spread_bps <= cfg.yield_curve_warning_bps:
        score = 0.2 + _interpolate(-spread_bps + cfg.yield_curve_warning_bps, 0, cfg.yield_curve_warning_bps - cfg.yield_curve_danger_bps) * 0.3
    else:
        score = 0.0
    return StressIndicator(
        name="Yield Curve", value=spread_bps, score=round(score, 4),
        status=_status_from_score(score), weight=0.07,
        description=f"2Y-10Y spread {spread_bps:+.0f} bps",
    )


def score_put_call_ratio(
    ratio: float | None,
    cfg: BlackSwanSettings,
) -> StressIndicator:
    if ratio is None:
        return StressIndicator(
            name="Put/Call Ratio", value=None, score=0.0,
            status=IndicatorStatus.UNAVAILABLE, weight=0.07,
            description="Put/call data unavailable (FRED)",
        )
    if ratio >= cfg.put_call_critical:
        score = 1.0
    elif ratio >= cfg.put_call_danger:
        score = 0.6 + _interpolate(ratio, cfg.put_call_danger, cfg.put_call_critical) * 0.4
    elif ratio >= cfg.put_call_warning:
        score = 0.3 + _interpolate(ratio, cfg.put_call_warning, cfg.put_call_danger) * 0.3
    else:
        score = 0.0
    return StressIndicator(
        name="Put/Call Ratio", value=ratio, score=round(score, 4),
        status=_status_from_score(score), weight=0.07,
        description=f"CBOE equity P/C ratio {ratio:.2f}",
    )


# ---------------------------------------------------------------------------
# Circuit breakers
# ---------------------------------------------------------------------------

def check_circuit_breakers(
    vix: float | None,
    vix_ratio: float | None,
    spy_return_pct: float | None,
    credit_daily_drop_pct: float | None,
    cfg: BlackSwanSettings,
) -> list[CircuitBreaker]:
    breakers: list[CircuitBreaker] = []

    breakers.append(CircuitBreaker(
        name="vix_extreme",
        triggered=vix is not None and vix > cfg.vix_critical,
        value=vix,
        threshold=cfg.vix_critical,
        description=f"VIX > {cfg.vix_critical:.0f}",
    ))

    breakers.append(CircuitBreaker(
        name="vix_backwardation",
        triggered=vix_ratio is not None and vix_ratio > cfg.term_structure_critical,
        value=vix_ratio,
        threshold=cfg.term_structure_critical,
        description=f"VIX/VIX3M > {cfg.term_structure_critical:.2f}",
    ))

    breakers.append(CircuitBreaker(
        name="spy_crash",
        triggered=spy_return_pct is not None and spy_return_pct < cfg.spy_critical_pct,
        value=spy_return_pct,
        threshold=cfg.spy_critical_pct,
        description=f"SPY 1-day return < {cfg.spy_critical_pct:.0f}%",
    ))

    breakers.append(CircuitBreaker(
        name="credit_collapse",
        triggered=credit_daily_drop_pct is not None and credit_daily_drop_pct < cfg.credit_critical_pct,
        value=credit_daily_drop_pct,
        threshold=cfg.credit_critical_pct,
        description=f"HYG/LQD 1-day drop > {abs(cfg.credit_critical_pct):.0f}%",
    ))

    return breakers


# ---------------------------------------------------------------------------
# Action text
# ---------------------------------------------------------------------------

_ACTION_TEXT = {
    AlertLevel.NORMAL: "Business as usual. No unusual tail-risk detected.",
    AlertLevel.ELEVATED: "Reduce new position sizes. Tighten stops on existing positions.",
    AlertLevel.HIGH: "Flatten directional exposure. Scale into hedges. No new theta.",
    AlertLevel.CRITICAL: "HALT all new trades. Unwind leveraged positions immediately.",
}


def _action_for_level(level: AlertLevel) -> str:
    return _ACTION_TEXT[level]


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def compute_black_swan_alert(
    *,
    vix: float | None = None,
    vix_ratio: float | None = None,
    credit_pct_change: float | None = None,
    credit_daily_drop_pct: float | None = None,
    spy_daily_return_pct: float | None = None,
    rv_iv_gap: float | None = None,
    tlt_abs_return_pct: float | None = None,
    em_pct_change: float | None = None,
    yield_curve_bps: float | None = None,
    put_call_ratio: float | None = None,
    as_of_date: date | None = None,
    cfg: BlackSwanSettings | None = None,
) -> BlackSwanAlert:
    """Compute the full tail-risk alert from raw indicator values.

    Pure function — no network, no side effects.
    """
    from market_analyzer.config import get_settings

    cfg = cfg or get_settings().black_swan
    as_of_date = as_of_date or date.today()

    # Score each indicator
    indicators = [
        score_vix_level(vix, cfg),
        score_vix_term_structure(vix_ratio, cfg),
        score_credit_stress(credit_pct_change, cfg),
        score_spy_drawdown(spy_daily_return_pct, cfg),
        score_rv_iv_gap(rv_iv_gap, cfg),
        score_treasury_stress(tlt_abs_return_pct, cfg),
        score_em_contagion(em_pct_change, cfg),
        score_yield_curve(yield_curve_bps, cfg),
        score_put_call_ratio(put_call_ratio, cfg),
    ]

    # Circuit breakers
    breakers = check_circuit_breakers(
        vix=vix,
        vix_ratio=vix_ratio,
        spy_return_pct=spy_daily_return_pct,
        credit_daily_drop_pct=credit_daily_drop_pct,
        cfg=cfg,
    )
    triggered_count = sum(1 for b in breakers if b.triggered)

    # Composite score — re-normalize weights for available indicators only
    available = [ind for ind in indicators if ind.status != IndicatorStatus.UNAVAILABLE]
    if available:
        total_weight = sum(ind.weight for ind in available)
        composite = sum(ind.score * ind.weight for ind in available) / total_weight
    else:
        composite = 0.0

    composite = round(float(np.clip(composite, 0.0, 1.0)), 4)

    # Alert level — circuit breakers override
    if triggered_count > 0:
        level = AlertLevel.CRITICAL
    elif composite >= cfg.alert_critical:
        level = AlertLevel.CRITICAL
    elif composite >= cfg.alert_high:
        level = AlertLevel.HIGH
    elif composite >= cfg.alert_elevated:
        level = AlertLevel.ELEVATED
    else:
        level = AlertLevel.NORMAL

    action = _action_for_level(level)

    # Summary
    breaker_names = [b.name for b in breakers if b.triggered]
    if breaker_names:
        summary = (
            f"CRITICAL: {triggered_count} circuit breaker(s) triggered "
            f"({', '.join(breaker_names)}). Composite stress {composite:.0%}. "
            f"{action}"
        )
    else:
        n_danger = sum(1 for ind in indicators if ind.status in (IndicatorStatus.DANGER, IndicatorStatus.CRITICAL))
        summary = (
            f"{level.value.upper()}: Composite stress {composite:.0%}. "
            f"{n_danger} indicator(s) at danger/critical. "
            f"{action}"
        )

    return BlackSwanAlert(
        as_of_date=as_of_date,
        alert_level=level,
        composite_score=composite,
        circuit_breakers=breakers,
        indicators=indicators,
        triggered_breakers=triggered_count,
        action=action,
        summary=summary,
    )
