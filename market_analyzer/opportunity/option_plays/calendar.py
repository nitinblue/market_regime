"""Calendar spread opportunity assessment — go/no-go + strategy recommendation.

Calendar spreads sell front-month, buy back-month at same strike.
Edge comes from: (1) front IV > back IV at same strike, (2) time decay differential,
(3) mean-reverting underlying.
Best in R1/R2 (range-bound, front IV elevated from near-term uncertainty).
Worst in R4 (directional moves blow through strike, calendar loses).
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from market_analyzer.config import get_settings
from market_analyzer.models.opportunity import (
    HardStop,
    OpportunitySignal,
    StrategyRecommendation,
    TradeSpec,
    Verdict,
)
from market_analyzer.models.regime import RegimeID

if TYPE_CHECKING:
    from market_analyzer.models.fundamentals import FundamentalsSnapshot
    from market_analyzer.models.regime import RegimeResult
    from market_analyzer.models.technicals import TechnicalSnapshot
    from market_analyzer.models.vol_surface import VolatilitySurface


# --- Strategy enum ---

class CalendarStrategy(str):
    ATM_CALENDAR = "atm_calendar"
    OTM_CALL_CALENDAR = "otm_call_calendar"
    OTM_PUT_CALENDAR = "otm_put_calendar"
    DOUBLE_CALENDAR = "double_calendar"
    NO_TRADE = "no_trade"


# --- Opportunity model ---

from pydantic import BaseModel


class CalendarOpportunity(BaseModel):
    """Calendar spread opportunity assessment."""

    ticker: str
    as_of_date: date
    verdict: Verdict
    confidence: float
    hard_stops: list[HardStop]
    signals: list[OpportunitySignal]
    strategy: StrategyRecommendation
    calendar_strategy: str
    regime_id: int
    regime_confidence: float
    front_iv: float
    back_iv: float
    term_slope: float
    calendar_edge_score: float
    days_to_earnings: int | None
    trade_spec: TradeSpec | None = None
    summary: str


# --- Public API ---


def assess_calendar(
    ticker: str,
    regime: RegimeResult,
    technicals: TechnicalSnapshot,
    vol_surface: VolatilitySurface | None = None,
    fundamentals: FundamentalsSnapshot | None = None,
    as_of: date | None = None,
) -> CalendarOpportunity:
    """Assess calendar spread opportunity for a single instrument.

    Pure function — consumes pre-computed analysis, no data fetching.
    """
    cfg = get_settings().opportunity.calendar
    today = as_of or date.today()

    days_to_earnings: int | None = None
    if fundamentals is not None:
        days_to_earnings = fundamentals.upcoming_events.days_to_earnings

    # --- Hard stops ---
    hard_stops = _check_hard_stops(regime, vol_surface, days_to_earnings, cfg)

    # Defaults when no vol surface
    front_iv = vol_surface.front_iv if vol_surface else 0.0
    back_iv = vol_surface.back_iv if vol_surface else 0.0
    term_slope = vol_surface.term_slope if vol_surface else 0.0
    cal_edge = vol_surface.calendar_edge_score if vol_surface else 0.0

    if hard_stops:
        return CalendarOpportunity(
            ticker=ticker,
            as_of_date=today,
            verdict=Verdict.NO_GO,
            confidence=0.0,
            hard_stops=hard_stops,
            signals=[],
            strategy=_no_trade_rec(),
            calendar_strategy=CalendarStrategy.NO_TRADE,
            regime_id=int(regime.regime),
            regime_confidence=regime.confidence,
            front_iv=front_iv,
            back_iv=back_iv,
            term_slope=term_slope,
            calendar_edge_score=cal_edge,
            days_to_earnings=days_to_earnings,
            summary=f"NO_GO: {hard_stops[0].description}",
        )

    assert vol_surface is not None  # Guarded by hard stop

    # --- Signals ---
    signals = _score_signals(regime, technicals, vol_surface, days_to_earnings, cfg)

    # --- Confidence ---
    raw = sum(s.weight for s in signals if s.favorable)
    regime_mult = cfg.regime_multipliers.get(int(regime.regime), 0.5)
    confidence = min(1.0, raw * regime_mult)

    # --- Verdict ---
    if confidence >= cfg.go_threshold:
        verdict = Verdict.GO
    elif confidence >= cfg.caution_threshold:
        verdict = Verdict.CAUTION
    else:
        verdict = Verdict.NO_GO

    # --- Strategy ---
    cal_strat, strat_rec = _select_strategy(regime, technicals, vol_surface, confidence)

    # --- Trade spec ---
    trade_spec = _compute_trade_spec(ticker, technicals, vol_surface, cal_strat) if verdict != Verdict.NO_GO else None

    summary = _build_summary(ticker, verdict, confidence, cal_strat, front_iv, back_iv, term_slope)

    return CalendarOpportunity(
        ticker=ticker,
        as_of_date=today,
        verdict=verdict,
        confidence=confidence,
        hard_stops=hard_stops,
        signals=signals,
        strategy=strat_rec,
        calendar_strategy=cal_strat,
        regime_id=int(regime.regime),
        regime_confidence=regime.confidence,
        front_iv=front_iv,
        back_iv=back_iv,
        term_slope=term_slope,
        calendar_edge_score=cal_edge,
        days_to_earnings=days_to_earnings,
        trade_spec=trade_spec,
        summary=summary,
    )


# --- Internal helpers ---


def _check_hard_stops(regime, vol_surface, days_to_earnings, cfg) -> list[HardStop]:
    stops: list[HardStop] = []

    # R4 at high confidence
    if regime.regime == RegimeID.R4_HIGH_VOL_TREND and regime.confidence >= cfg.r4_confidence_threshold:
        stops.append(HardStop(
            name="R4 trending",
            description="R4 (high-vol trending) with high confidence — directional moves destroy calendars",
        ))

    # No vol surface data
    if vol_surface is None:
        stops.append(HardStop(
            name="No vol surface",
            description="No options chain/vol surface data — cannot assess calendar",
        ))
        return stops

    # Term structure flat with no edge
    if vol_surface.data_quality == "poor":
        stops.append(HardStop(
            name="Poor data quality",
            description="Options chain data quality too poor for calendar assessment",
        ))

    # Bid-ask too wide
    if vol_surface.avg_bid_ask_spread_pct > cfg.max_bid_ask_spread_pct:
        stops.append(HardStop(
            name="Wide bid-ask",
            description=f"Average bid-ask spread {vol_surface.avg_bid_ask_spread_pct:.1f}% too wide",
        ))

    # Earnings between front and back expiry
    if (
        days_to_earnings is not None
        and vol_surface.best_calendar_expiries is not None
        and 0 < days_to_earnings < cfg.earnings_blackout_days
    ):
        stops.append(HardStop(
            name="Earnings imminent",
            description=f"Earnings in {days_to_earnings} days — IV crush asymmetry risk",
        ))

    return stops


def _score_signals(regime, technicals, vol_surface, days_to_earnings, cfg) -> list[OpportunitySignal]:
    signals: list[OpportunitySignal] = []

    # 1. Term structure contango (0.20)
    if vol_surface.is_backwardation:
        signals.append(OpportunitySignal(
            name="Term structure backwardation",
            favorable=True,
            weight=0.20,
            description=f"Front IV ({vol_surface.front_iv:.1%}) > Back IV ({vol_surface.back_iv:.1%}) — ideal for selling front",
        ))
    elif vol_surface.is_contango and vol_surface.term_slope < 0.10:
        signals.append(OpportunitySignal(
            name="Mild contango",
            favorable=True,
            weight=0.10,
            description=f"Mild contango (slope {vol_surface.term_slope:.1%}) — acceptable for calendars",
        ))
    else:
        signals.append(OpportunitySignal(
            name="Steep contango",
            favorable=False,
            weight=0.20,
            description=f"Steep contango (slope {vol_surface.term_slope:.1%}) — back month expensive",
        ))

    # 2. Regime favorable (0.20)
    regime_id = int(regime.regime)
    if regime_id in (1, 2):
        signals.append(OpportunitySignal(
            name="Mean-reverting regime",
            favorable=True,
            weight=0.20,
            description=f"R{regime_id} (mean-reverting) — ideal for calendar spreads",
        ))
    elif regime_id == 3:
        signals.append(OpportunitySignal(
            name="Mild trend regime",
            favorable=True,
            weight=0.10,
            description="R3 (low-vol trending) — moderate calendar environment",
        ))
    else:
        signals.append(OpportunitySignal(
            name="Trending regime",
            favorable=False,
            weight=0.20,
            description="R4 (high-vol trending) — poor calendar environment",
        ))

    # 3. ATM IV level (0.15)
    avg_iv = (vol_surface.front_iv + vol_surface.back_iv) / 2
    if avg_iv >= cfg.atm_iv_high:
        signals.append(OpportunitySignal(
            name="High ATM IV",
            favorable=True,
            weight=0.15,
            description=f"ATM IV {avg_iv:.1%} — good premium collection",
        ))
    elif avg_iv >= cfg.atm_iv_moderate:
        signals.append(OpportunitySignal(
            name="Moderate ATM IV",
            favorable=True,
            weight=0.08,
            description=f"ATM IV {avg_iv:.1%} — adequate premium",
        ))
    else:
        signals.append(OpportunitySignal(
            name="Low ATM IV",
            favorable=False,
            weight=0.15,
            description=f"ATM IV {avg_iv:.1%} — thin premium",
        ))

    # 4. Mean-reversion indicators (0.10)
    rsi = technicals.rsi.value if technicals.rsi else 50.0
    if 40 <= rsi <= 60:
        signals.append(OpportunitySignal(
            name="Neutral RSI",
            favorable=True,
            weight=0.10,
            description=f"RSI {rsi:.0f} — neutral/centered, good for calendars",
        ))
    else:
        signals.append(OpportunitySignal(
            name="Directional RSI",
            favorable=False,
            weight=0.10,
            description=f"RSI {rsi:.0f} — directional bias less ideal",
        ))

    # 5. Skew favorable (0.10)
    if vol_surface.skew_by_expiry:
        front_skew = vol_surface.skew_by_expiry[0]
        if abs(front_skew.skew_ratio) < 2.0:
            signals.append(OpportunitySignal(
                name="Neutral skew",
                favorable=True,
                weight=0.10,
                description=f"Skew ratio {front_skew.skew_ratio:.1f} — balanced",
            ))
        else:
            signals.append(OpportunitySignal(
                name="Steep skew",
                favorable=False,
                weight=0.10,
                description=f"Skew ratio {front_skew.skew_ratio:.1f} — asymmetric risk",
            ))

    # 6. Calendar edge score (0.10)
    if vol_surface.calendar_edge_score >= 0.5:
        signals.append(OpportunitySignal(
            name="Strong calendar edge",
            favorable=True,
            weight=0.10,
            description=f"Calendar edge score {vol_surface.calendar_edge_score:.2f} — favorable",
        ))
    elif vol_surface.calendar_edge_score >= 0.25:
        signals.append(OpportunitySignal(
            name="Moderate calendar edge",
            favorable=True,
            weight=0.05,
            description=f"Calendar edge score {vol_surface.calendar_edge_score:.2f} — acceptable",
        ))

    # 7. No earnings between legs (0.10)
    if days_to_earnings is None or days_to_earnings > 45:
        signals.append(OpportunitySignal(
            name="Clear earnings window",
            favorable=True,
            weight=0.10,
            description="No earnings in near-term calendar window",
        ))
    elif days_to_earnings > cfg.earnings_blackout_days:
        signals.append(OpportunitySignal(
            name="Earnings proximity",
            favorable=False,
            weight=0.10,
            description=f"Earnings in {days_to_earnings} days — asymmetric IV risk",
        ))

    # 8. Liquidity (0.05)
    if vol_surface.data_quality == "good":
        signals.append(OpportunitySignal(
            name="Good liquidity",
            favorable=True,
            weight=0.05,
            description="Good options chain liquidity and data quality",
        ))

    return signals


def _select_strategy(regime, technicals, vol_surface, confidence):
    """Select the appropriate calendar strategy variant."""
    regime_id = int(regime.regime)
    rsi = technicals.rsi.value if technicals.rsi else 50.0

    if confidence < 0.30:
        return CalendarStrategy.NO_TRADE, _no_trade_rec()

    # R1 + neutral → ATM calendar
    if regime_id == 1:
        return CalendarStrategy.ATM_CALENDAR, StrategyRecommendation(
            name="ATM Calendar Spread",
            direction="neutral",
            structure="Sell front-month ATM, buy back-month ATM (same strike)",
            rationale="R1 mean-reverting + neutral RSI — pure theta play",
            risk_notes=["Max loss = net debit paid", "Danger if price moves sharply away from strike"],
        )

    # R2 + wide range → double calendar
    if regime_id == 2:
        return CalendarStrategy.DOUBLE_CALENDAR, StrategyRecommendation(
            name="Double Calendar Spread",
            direction="neutral",
            structure="Two calendar spreads bracketing current price",
            rationale="R2 high-vol MR — wide range, double calendar covers more ground",
            risk_notes=["Higher capital outlay", "Still loses if price breaks range"],
        )

    # R3 + bullish bias → OTM call calendar
    if regime_id == 3 and rsi >= 50:
        return CalendarStrategy.OTM_CALL_CALENDAR, StrategyRecommendation(
            name="OTM Call Calendar",
            direction="bullish",
            structure="Sell front-month OTM call, buy back-month same strike call",
            rationale="R3 mild uptrend — OTM call calendar captures upside bias",
            risk_notes=["Loses if trend reverses", "Requires price to reach strike by back expiry"],
        )

    # R3 + bearish bias → OTM put calendar
    if regime_id == 3 and rsi < 50:
        return CalendarStrategy.OTM_PUT_CALENDAR, StrategyRecommendation(
            name="OTM Put Calendar",
            direction="bearish",
            structure="Sell front-month OTM put, buy back-month same strike put",
            rationale="R3 mild downtrend — OTM put calendar captures downside bias",
            risk_notes=["Loses if trend reverses", "Requires price to reach strike by back expiry"],
        )

    # Default: ATM calendar
    return CalendarStrategy.ATM_CALENDAR, StrategyRecommendation(
        name="ATM Calendar Spread",
        direction="neutral",
        structure="Sell front-month ATM, buy back-month ATM (same strike)",
        rationale="Calendar spread for theta collection",
        risk_notes=["Max loss = net debit paid"],
    )


def _compute_trade_spec(ticker, technicals, vol_surface, cal_strat) -> TradeSpec | None:
    """Compute actionable trade parameters for calendar spread."""
    from market_analyzer.opportunity.option_plays._trade_spec_helpers import build_dual_expiry_trade_spec

    return build_dual_expiry_trade_spec(
        ticker=ticker,
        price=technicals.current_price,
        atr=technicals.atr,
        vol_surface=vol_surface,
        structure_type="calendar",
        strategy_type=cal_strat,
        front_dte_min=20,
        front_dte_max=30,
        back_dte_min=50,
        back_dte_max=70,
    )


def _no_trade_rec() -> StrategyRecommendation:
    return StrategyRecommendation(
        name="No Trade",
        direction="neutral",
        structure="No position",
        rationale="Conditions not favorable for calendar spread",
        risk_notes=[],
    )


def _build_summary(ticker, verdict, confidence, cal_strat, front_iv, back_iv, term_slope) -> str:
    parts = [f"{verdict.upper()}: {ticker}"]
    parts.append(f"Calendar: {cal_strat}")
    parts.append(f"Front IV: {front_iv:.1%}")
    parts.append(f"Back IV: {back_iv:.1%}")
    parts.append(f"Slope: {term_slope:+.1%}")
    parts.append(f"Score: {confidence:.0%}")
    return " | ".join(parts)
