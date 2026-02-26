"""TradingPlanService: daily orchestration layer — 'what should I trade today?'"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, timedelta
from typing import TYPE_CHECKING

from market_analyzer.config import get_settings
from market_analyzer.macro.expiry import (
    ExpiryEvent,
    ExpiryType,
    get_expiry_calendar,
    upcoming_expiries,
)
from market_analyzer.models.black_swan import AlertLevel
from market_analyzer.models.macro import MacroEventType
from market_analyzer.models.opportunity import Verdict
from market_analyzer.models.ranking import RankedEntry, StrategyType
from market_analyzer.models.trading_plan import (
    DailyTradingPlan,
    DayVerdict,
    PlanHorizon,
    PlanTrade,
    RiskBudget,
)
from market_analyzer.opportunity.option_plays._trade_spec_helpers import (
    compute_max_entry_price,
)

if TYPE_CHECKING:
    from market_analyzer.models.context import MarketContext
    from market_analyzer.service.analyzer import MarketAnalyzer

logger = logging.getLogger(__name__)


def _dte_to_horizon(dte: int | None) -> PlanHorizon:
    """Map DTE to plan horizon bucket."""
    if dte is None or dte <= 0:
        return PlanHorizon.ZERO_DTE
    if dte <= 7:
        return PlanHorizon.WEEKLY
    if dte <= 60:
        return PlanHorizon.MONTHLY
    return PlanHorizon.LEAP


def _expiry_note_for_trade(trade_spec, expiry_events: list[ExpiryEvent]) -> str | None:
    """Tag trade if its expiration coincides with an expiry event."""
    if trade_spec is None:
        return None
    expiry_dates = {e.date: e for e in expiry_events}
    # Check target expiration
    if trade_spec.target_expiration in expiry_dates:
        return expiry_dates[trade_spec.target_expiration].label
    # Check front expiration (calendar/diagonal)
    if trade_spec.front_expiration and trade_spec.front_expiration in expiry_dates:
        return f"Front leg: {expiry_dates[trade_spec.front_expiration].label}"
    return None


class TradingPlanService:
    """Generate a complete daily trading plan.

    Composes context, ranking, expiry calendar, and fill-price estimation
    into a single actionable artifact for cotrader.
    """

    def __init__(self, analyzer: MarketAnalyzer) -> None:
        self.analyzer = analyzer

    def generate(
        self,
        tickers: list[str] | None = None,
        plan_date: date | None = None,
        strategies: list[StrategyType] | None = None,
    ) -> DailyTradingPlan:
        """Generate a complete daily trading plan."""
        cfg = get_settings().trading_plan
        today = date.today()
        plan_for = plan_date or today
        tickers = tickers or cfg.default_tickers

        # 1. Market context (black swan gate, environment, position_size_factor)
        context = self.analyzer.context.assess(as_of=plan_for)

        # 2. Expiry events
        today_events = get_expiry_calendar(plan_for, plan_for)
        upcoming = upcoming_expiries(as_of=plan_for, days_ahead=7)
        # All events within 30 days for tagging trades
        all_near_expiries = get_expiry_calendar(plan_for, plan_for + timedelta(days=30))

        # 3. Day verdict
        day_verdict, verdict_reasons = self._compute_day_verdict(context, today_events)

        # 4. Risk budget
        account_size = get_settings().strategy.default_account_size
        max_positions = (
            cfg.max_new_positions_normal
            if day_verdict == DayVerdict.TRADE
            else cfg.max_new_positions_light
            if day_verdict == DayVerdict.TRADE_LIGHT
            else 0
        )
        risk_budget = RiskBudget(
            max_new_positions=max_positions,
            max_daily_risk_dollars=round(account_size * cfg.daily_risk_pct, 2),
            position_size_factor=context.position_size_factor,
        )

        # 5. If NO_TRADE or AVOID, return empty plan
        if day_verdict in (DayVerdict.NO_TRADE, DayVerdict.AVOID):
            return DailyTradingPlan(
                as_of_date=today,
                plan_for_date=plan_for,
                day_verdict=day_verdict,
                day_verdict_reasons=verdict_reasons,
                risk_budget=risk_budget,
                expiry_events=today_events,
                upcoming_expiries=upcoming,
                trades_by_horizon={h: [] for h in PlanHorizon},
                all_trades=[],
                total_trades=0,
                summary=f"Day verdict: {day_verdict.value}. {'; '.join(verdict_reasons)}",
            )

        # 6. Filter strategies based on config
        if strategies is None:
            strategies = list(StrategyType)
            if not cfg.include_0dte:
                strategies = [s for s in strategies if s != StrategyType.ZERO_DTE]
            if not cfg.include_leaps:
                strategies = [s for s in strategies if s != StrategyType.LEAP]

        # 7. Rank trades
        ranking_result = self.analyzer.ranking.rank(
            tickers, strategies=strategies, as_of=plan_for,
        )

        # 8. Convert ranked entries to PlanTrades
        plan_trades: list[PlanTrade] = []
        slippage = cfg.fill_slippage_pct

        for entry in ranking_result.top_trades:
            # Only include GO and CAUTION
            if entry.verdict == Verdict.NO_GO:
                continue

            # Determine horizon
            dte = entry.trade_spec.target_dte if entry.trade_spec else None
            horizon = self._strategy_to_horizon(entry.strategy_type, dte)

            # Compute max entry price
            max_price = None
            if entry.trade_spec is not None:
                max_price = compute_max_entry_price(entry.trade_spec, slippage)

            # Expiry note
            expiry_note = _expiry_note_for_trade(entry.trade_spec, all_near_expiries)

            plan_trades.append(PlanTrade(
                rank=0,  # Re-ranked after filtering
                ticker=entry.ticker,
                strategy_type=entry.strategy_type,
                horizon=horizon,
                verdict=entry.verdict,
                composite_score=entry.composite_score,
                direction=entry.direction,
                trade_spec=entry.trade_spec,
                max_entry_price=max_price,
                rationale=entry.rationale,
                risk_notes=entry.risk_notes,
                expiry_note=expiry_note,
            ))

        # 9. Cap at max trades
        plan_trades = plan_trades[:cfg.max_trades_per_plan]

        # 10. Assign ranks
        for i, pt in enumerate(plan_trades):
            pt.rank = i + 1

        # 11. Bucket by horizon
        by_horizon: dict[PlanHorizon, list[PlanTrade]] = defaultdict(list)
        for pt in plan_trades:
            by_horizon[pt.horizon].append(pt)
        # Ensure all horizons present
        for h in PlanHorizon:
            if h not in by_horizon:
                by_horizon[h] = []

        # 12. Summary
        summary = self._build_summary(
            day_verdict, plan_trades, risk_budget, today_events, ranking_result.black_swan_level,
        )

        return DailyTradingPlan(
            as_of_date=today,
            plan_for_date=plan_for,
            day_verdict=day_verdict,
            day_verdict_reasons=verdict_reasons,
            risk_budget=risk_budget,
            expiry_events=today_events,
            upcoming_expiries=upcoming,
            trades_by_horizon=dict(by_horizon),
            all_trades=plan_trades,
            total_trades=len(plan_trades),
            summary=summary,
        )

    def _compute_day_verdict(
        self,
        context: MarketContext,
        today_events: list[ExpiryEvent],
    ) -> tuple[DayVerdict, list[str]]:
        """Determine the day's trading verdict."""
        reasons: list[str] = []

        # NO_TRADE: black swan CRITICAL
        if context.black_swan.alert_level == AlertLevel.CRITICAL:
            reasons.append(f"Black swan CRITICAL (score={context.black_swan.composite_score:.2f})")
            return DayVerdict.NO_TRADE, reasons

        # Check macro events for today
        macro_events_today = context.macro.events_next_7_days
        fomc_today = any(
            e.event_type == MacroEventType.FOMC and e.date == context.as_of_date
            for e in macro_events_today
        )
        has_quad_witching = any(
            e.expiry_type == ExpiryType.QUAD_WITCHING for e in today_events
        )
        has_high_impact_today = any(
            e.event_type in (MacroEventType.CPI, MacroEventType.NFP, MacroEventType.PCE)
            and e.date == context.as_of_date
            for e in macro_events_today
        )
        has_monthly_opex = any(
            e.expiry_type == ExpiryType.MONTHLY_OPEX for e in today_events
        )
        has_vix_settlement = any(
            e.expiry_type == ExpiryType.VIX_SETTLEMENT for e in today_events
        )

        # AVOID: FOMC day, quad witching, or black swan ELEVATED+
        if fomc_today:
            reasons.append("FOMC announcement day")
        if has_quad_witching:
            reasons.append("Quad witching day")
        if context.black_swan.alert_level == AlertLevel.HIGH:
            reasons.append(f"Black swan HIGH (score={context.black_swan.composite_score:.2f})")
        if context.black_swan.alert_level == AlertLevel.ELEVATED:
            reasons.append(f"Black swan ELEVATED (score={context.black_swan.composite_score:.2f})")

        if fomc_today or has_quad_witching or context.black_swan.alert_level == AlertLevel.HIGH:
            return DayVerdict.AVOID, reasons

        # TRADE_LIGHT: high-impact macro today, monthly OpEx, VIX settlement, elevated BS
        if has_high_impact_today:
            reasons.append("High-impact macro event today (CPI/NFP/PCE)")
        if has_monthly_opex:
            reasons.append("Monthly OpEx day")
        if has_vix_settlement:
            reasons.append("VIX settlement day")
        if context.black_swan.alert_level == AlertLevel.ELEVATED:
            # already added reason above
            pass

        if has_high_impact_today or has_monthly_opex or has_vix_settlement or \
                context.black_swan.alert_level == AlertLevel.ELEVATED:
            return DayVerdict.TRADE_LIGHT, reasons

        # TRADE: normal conditions
        reasons.append("Normal conditions")
        return DayVerdict.TRADE, reasons

    @staticmethod
    def _strategy_to_horizon(strategy_type: StrategyType, dte: int | None) -> PlanHorizon:
        """Determine horizon from strategy type and DTE."""
        # Strategy-type overrides
        if strategy_type == StrategyType.ZERO_DTE:
            return PlanHorizon.ZERO_DTE
        if strategy_type == StrategyType.LEAP:
            return PlanHorizon.LEAP
        # Otherwise use DTE
        return _dte_to_horizon(dte)

    @staticmethod
    def _build_summary(
        verdict: DayVerdict,
        trades: list[PlanTrade],
        budget: RiskBudget,
        today_events: list[ExpiryEvent],
        bs_level: str,
    ) -> str:
        parts = [f"Day: {verdict.value.upper()}"]
        parts.append(
            f"Budget: {budget.max_new_positions} positions, "
            f"${budget.max_daily_risk_dollars:,.0f} risk, "
            f"{budget.position_size_factor:.0%} sizing"
        )
        if trades:
            go = sum(1 for t in trades if t.verdict == Verdict.GO)
            caution = sum(1 for t in trades if t.verdict == Verdict.CAUTION)
            parts.append(f"Trades: {len(trades)} ({go} GO, {caution} CAUTION)")
            best = trades[0]
            parts.append(f"Top: {best.ticker} {best.strategy_type} (score={best.composite_score:.2f})")
        else:
            parts.append("No trades")
        if today_events:
            labels = [e.label for e in today_events[:3]]
            parts.append(f"Expiry: {', '.join(labels)}")
        if bs_level != "normal":
            parts.append(f"Alert: {bs_level}")
        return " | ".join(parts)
