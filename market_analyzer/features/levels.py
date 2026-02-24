"""Unified price level extraction, confluence clustering, and R:R computation.

Pure functions — no data fetching. Accepts TechnicalSnapshot + optional
RegimeResult/ORBData and returns LevelsAnalysis.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from market_analyzer.config import get_settings
from market_analyzer.models.levels import (
    LevelRole,
    LevelSource,
    LevelsAnalysis,
    PriceLevel,
    StopLoss,
    Target,
    TradeDirection,
)
from market_analyzer.models.technicals import (
    MarketPhase,
    TechnicalSnapshot,
)

if TYPE_CHECKING:
    from market_analyzer.models.regime import RegimeResult
    from market_analyzer.models.technicals import ORBData


# ---------------------------------------------------------------------------
# Source quality weights (default; overridable via config)
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: dict[str, float] = {
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
}


# ---------------------------------------------------------------------------
# Stage 1: Extract raw (price, source) tuples from TechnicalSnapshot
# ---------------------------------------------------------------------------

def _extract_raw_levels(
    snap: TechnicalSnapshot,
    orb: ORBData | None = None,
) -> list[tuple[float, LevelSource]]:
    """Walk TechnicalSnapshot fields and emit (price, LevelSource) pairs."""
    levels: list[tuple[float, LevelSource]] = []

    # Support / Resistance
    sr = snap.support_resistance
    if sr.support is not None and sr.support > 0:
        levels.append((sr.support, LevelSource.SWING_SUPPORT))
    if sr.resistance is not None and sr.resistance > 0:
        levels.append((sr.resistance, LevelSource.SWING_RESISTANCE))

    # Moving averages
    ma = snap.moving_averages
    if ma.sma_20 > 0:
        levels.append((ma.sma_20, LevelSource.SMA_20))
    if ma.sma_50 > 0:
        levels.append((ma.sma_50, LevelSource.SMA_50))
    if ma.sma_200 > 0:
        levels.append((ma.sma_200, LevelSource.SMA_200))
    if ma.ema_9 > 0:
        levels.append((ma.ema_9, LevelSource.EMA_9))
    if ma.ema_21 > 0:
        levels.append((ma.ema_21, LevelSource.EMA_21))

    # Bollinger Bands
    bb = snap.bollinger
    levels.append((bb.upper, LevelSource.BOLLINGER_UPPER))
    levels.append((bb.middle, LevelSource.BOLLINGER_MIDDLE))
    levels.append((bb.lower, LevelSource.BOLLINGER_LOWER))

    # VWMA
    if snap.vwma_20 > 0:
        levels.append((snap.vwma_20, LevelSource.VWMA_20))

    # VCP pivot
    if snap.vcp and snap.vcp.pivot_price is not None and snap.vcp.pivot_price > 0:
        levels.append((snap.vcp.pivot_price, LevelSource.VCP_PIVOT))

    # Order Blocks (smart money)
    if snap.smart_money:
        for ob in snap.smart_money.order_blocks:
            if ob.is_broken:
                continue
            if ob.type.value == "bullish":
                levels.append((ob.high, LevelSource.ORDER_BLOCK_HIGH))
                levels.append((ob.low, LevelSource.ORDER_BLOCK_LOW))
            else:
                levels.append((ob.high, LevelSource.ORDER_BLOCK_HIGH))
                levels.append((ob.low, LevelSource.ORDER_BLOCK_LOW))

    # Fair Value Gaps
    if snap.smart_money:
        for fvg in snap.smart_money.fair_value_gaps:
            if fvg.is_filled:
                continue
            levels.append((fvg.high, LevelSource.FVG_HIGH))
            levels.append((fvg.low, LevelSource.FVG_LOW))

    # ORB levels (optional, intraday)
    if orb:
        for orb_level in orb.levels:
            levels.append((orb_level.price, LevelSource.ORB_LEVEL))

    return levels


# ---------------------------------------------------------------------------
# Stage 2: Cluster nearby levels (confluence)
# ---------------------------------------------------------------------------

def _cluster_levels(
    raw: list[tuple[float, LevelSource]],
    proximity_pct: float,
    source_weights: dict[str, float],
    max_strength_denom: float,
) -> list[tuple[float, list[LevelSource], float]]:
    """Merge levels within proximity_pct into clusters.

    Returns list of (price, sources, strength) sorted by price ascending.
    """
    if not raw:
        return []

    # Sort by price
    sorted_levels = sorted(raw, key=lambda x: x[0])

    clusters: list[list[tuple[float, LevelSource]]] = []
    current_cluster: list[tuple[float, LevelSource]] = [sorted_levels[0]]

    for price, source in sorted_levels[1:]:
        cluster_avg = sum(p for p, _ in current_cluster) / len(current_cluster)
        if cluster_avg > 0 and abs(price - cluster_avg) / cluster_avg * 100 <= proximity_pct:
            current_cluster.append((price, source))
        else:
            clusters.append(current_cluster)
            current_cluster = [(price, source)]
    clusters.append(current_cluster)

    result: list[tuple[float, list[LevelSource], float]] = []
    for cluster in clusters:
        # Weighted average price
        weights = [source_weights.get(s.value, 0.5) for _, s in cluster]
        total_w = sum(weights)
        if total_w == 0:
            avg_price = sum(p for p, _ in cluster) / len(cluster)
        else:
            avg_price = sum(p * w for (p, _), w in zip(cluster, weights)) / total_w

        # Deduplicate sources
        sources = list(dict.fromkeys(s for _, s in cluster))
        # Strength = sum of weights / denominator, capped at 1.0
        strength = min(1.0, sum(source_weights.get(s.value, 0.5) for s in sources) / max_strength_denom)
        result.append((avg_price, sources, strength))

    return sorted(result, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Stage 3: Classify as support/resistance relative to entry
# ---------------------------------------------------------------------------

def _classify_levels(
    clustered: list[tuple[float, list[LevelSource], float]],
    entry_price: float,
) -> tuple[list[PriceLevel], list[PriceLevel]]:
    """Split clustered levels into support (below) and resistance (above)."""
    supports: list[PriceLevel] = []
    resistances: list[PriceLevel] = []

    for price, sources, strength in clustered:
        dist_pct = (price - entry_price) / entry_price * 100
        level = PriceLevel(
            price=round(price, 2),
            role=LevelRole.SUPPORT if price < entry_price else LevelRole.RESISTANCE,
            sources=sources,
            confluence_score=len(sources),
            strength=round(strength, 3),
            distance_pct=round(dist_pct, 2),
            description=_level_description(sources, price, entry_price),
        )
        if price < entry_price:
            supports.append(level)
        else:
            resistances.append(level)

    # Supports: nearest first (descending)
    supports.sort(key=lambda l: l.price, reverse=True)
    # Resistances: nearest first (ascending)
    resistances.sort(key=lambda l: l.price)

    return supports, resistances


def _level_description(sources: list[LevelSource], price: float, entry: float) -> str:
    src_names = ", ".join(s.value.replace("_", " ").title() for s in sources[:3])
    suffix = f" (+{len(sources) - 3} more)" if len(sources) > 3 else ""
    return f"${price:.2f} — {src_names}{suffix}"


# ---------------------------------------------------------------------------
# Stage 4: Stop loss
# ---------------------------------------------------------------------------

def _compute_stop(
    direction: TradeDirection,
    entry_price: float,
    levels: list[PriceLevel],
    atr: float,
    min_dist_pct: float,
    max_dist_pct: float,
    atr_buffer_mult: float,
    atr_fallback_mult: float,
) -> StopLoss | None:
    """Compute stop loss from support/resistance levels + ATR buffer."""
    buffer = atr * atr_buffer_mult

    if direction == TradeDirection.LONG:
        # Look at support levels (sorted nearest first = descending)
        candidates = [
            l for l in levels
            if abs(l.distance_pct) >= min_dist_pct
            and abs(l.distance_pct) <= max_dist_pct
        ]
        if candidates:
            level = candidates[0]  # nearest that meets min distance
            stop_price = level.price - buffer
        else:
            # ATR fallback
            stop_price = entry_price - atr * atr_fallback_mult
            level = PriceLevel(
                price=round(stop_price + buffer, 2),
                role=LevelRole.SUPPORT,
                sources=[],
                confluence_score=0,
                strength=0.0,
                distance_pct=round((stop_price + buffer - entry_price) / entry_price * 100, 2),
                description=f"ATR fallback ({atr_fallback_mult}x ATR)",
            )
    else:
        # Short: resistance levels (sorted nearest first = ascending)
        candidates = [
            l for l in levels
            if abs(l.distance_pct) >= min_dist_pct
            and abs(l.distance_pct) <= max_dist_pct
        ]
        if candidates:
            level = candidates[0]
            stop_price = level.price + buffer
        else:
            stop_price = entry_price + atr * atr_fallback_mult
            level = PriceLevel(
                price=round(stop_price - buffer, 2),
                role=LevelRole.RESISTANCE,
                sources=[],
                confluence_score=0,
                strength=0.0,
                distance_pct=round((stop_price - buffer - entry_price) / entry_price * 100, 2),
                description=f"ATR fallback ({atr_fallback_mult}x ATR)",
            )

    dist_pct = abs(stop_price - entry_price) / entry_price * 100
    dollar_risk = abs(entry_price - stop_price)

    return StopLoss(
        price=round(stop_price, 2),
        distance_pct=round(dist_pct, 2),
        dollar_risk_per_share=round(dollar_risk, 2),
        level=level,
        atr_buffer=round(buffer, 2),
        description=f"Stop ${stop_price:.2f} ({dist_pct:.1f}% from entry)",
    )


# ---------------------------------------------------------------------------
# Stage 5: Targets & R:R
# ---------------------------------------------------------------------------

def _compute_targets(
    direction: TradeDirection,
    entry_price: float,
    levels: list[PriceLevel],
    stop: StopLoss | None,
    min_dist_pct: float,
    max_targets: int,
    min_rr: float,
) -> tuple[list[Target], Target | None]:
    """Compute targets from resistance (long) or support (short) levels."""
    if stop is None:
        return [], None

    risk = stop.dollar_risk_per_share
    if risk <= 0:
        return [], None

    candidates = [
        l for l in levels
        if abs(l.distance_pct) >= min_dist_pct
    ][:max_targets]

    targets: list[Target] = []
    for lvl in candidates:
        reward = abs(lvl.price - entry_price)
        rr = reward / risk if risk > 0 else 0.0
        targets.append(Target(
            price=lvl.price,
            distance_pct=round(abs(lvl.distance_pct), 2),
            dollar_reward_per_share=round(reward, 2),
            risk_reward_ratio=round(rr, 2),
            level=lvl,
            description=f"T ${lvl.price:.2f} R:R={rr:.1f}",
        ))

    # Best target: highest R:R above threshold
    best = None
    qualifying = [t for t in targets if t.risk_reward_ratio >= min_rr]
    if qualifying:
        best = max(qualifying, key=lambda t: t.risk_reward_ratio)

    return targets, best


# ---------------------------------------------------------------------------
# Stage 6: Direction auto-detection
# ---------------------------------------------------------------------------

def _detect_direction(
    snap: TechnicalSnapshot,
    regime: RegimeResult | None = None,
) -> tuple[TradeDirection, bool]:
    """Auto-detect trade direction from regime, phase, then MA fallback.

    Returns (direction, auto_detected). auto_detected=True means the caller
    did not specify a direction.
    """
    # Priority 1: Regime trend_direction
    if regime and regime.trend_direction is not None:
        if regime.trend_direction.value == "bullish":
            return TradeDirection.LONG, True
        return TradeDirection.SHORT, True

    # Priority 2: Phase indicator
    phase = snap.phase
    if phase.phase in (MarketPhase.MARKUP, MarketPhase.ACCUMULATION):
        return TradeDirection.LONG, True
    if phase.phase in (MarketPhase.MARKDOWN, MarketPhase.DISTRIBUTION):
        return TradeDirection.SHORT, True

    # Priority 3: MA fallback — price vs SMA 50
    if snap.current_price > snap.moving_averages.sma_50:
        return TradeDirection.LONG, True
    return TradeDirection.SHORT, True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_levels(
    snap: TechnicalSnapshot,
    regime: RegimeResult | None = None,
    orb: ORBData | None = None,
    direction: str | None = None,
    entry_price: float | None = None,
) -> LevelsAnalysis:
    """Compute unified price levels with confluence, stop, targets, and R:R.

    Args:
        snap: TechnicalSnapshot from TechnicalService.snapshot().
        regime: Optional RegimeResult for direction auto-detection.
        orb: Optional ORBData for intraday level inclusion.
        direction: Caller override ("long" or "short"). Auto-detected if None.
        entry_price: Explicit entry price. Defaults to snap.current_price.

    Returns:
        LevelsAnalysis with ranked levels, stop loss, targets, and R:R.
    """
    cfg = get_settings().levels

    # Entry price
    entry = entry_price if entry_price is not None else snap.current_price

    # Direction
    if direction is not None:
        trade_dir = TradeDirection(direction)
        auto_detected = False
    else:
        trade_dir, auto_detected = _detect_direction(snap, regime)

    # Stage 1: extract raw levels
    raw = _extract_raw_levels(snap, orb)

    # Stage 2: cluster
    weights = cfg.source_weights
    clustered = _cluster_levels(raw, cfg.confluence_proximity_pct, weights, cfg.max_strength_denominator)

    # Stage 3: classify
    supports, resistances = _classify_levels(clustered, entry)

    # Stage 4: stop loss
    stop_levels = supports if trade_dir == TradeDirection.LONG else resistances
    stop = _compute_stop(
        trade_dir, entry, stop_levels,
        snap.atr,
        cfg.min_stop_distance_pct,
        cfg.max_stop_distance_pct,
        cfg.atr_stop_buffer_multiple,
        cfg.atr_fallback_multiple,
    )

    # Stage 5: targets
    target_levels = resistances if trade_dir == TradeDirection.LONG else supports
    targets, best = _compute_targets(
        trade_dir, entry, target_levels, stop,
        cfg.min_target_distance_pct,
        cfg.max_targets,
        cfg.min_risk_reward,
    )

    # Summary
    summary_parts = [f"{snap.ticker} {trade_dir.value.upper()}"]
    summary_parts.append(f"Entry ${entry:.2f}")
    if stop:
        summary_parts.append(f"Stop ${stop.price:.2f} ({stop.distance_pct:.1f}%)")
    if best:
        summary_parts.append(f"Best target ${best.price:.2f} R:R={best.risk_reward_ratio:.1f}")
    summary_parts.append(f"{len(supports)}S/{len(resistances)}R levels")
    summary = " | ".join(summary_parts)

    return LevelsAnalysis(
        ticker=snap.ticker,
        as_of_date=snap.as_of_date,
        entry_price=round(entry, 2),
        direction=trade_dir,
        direction_auto_detected=auto_detected,
        current_price=snap.current_price,
        atr=snap.atr,
        atr_pct=snap.atr_pct,
        support_levels=supports,
        resistance_levels=resistances,
        stop_loss=stop,
        targets=targets,
        best_target=best,
        summary=summary,
    )
