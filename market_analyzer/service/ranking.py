"""TradeRankingService: rank trade ideas across tickers and strategies."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from market_analyzer.config import get_settings
from market_analyzer.features.ranking import (
    ConfigWeightProvider,
    WeightProvider,
    composite_from_breakdown,
    compute_composite_score,
)
from market_analyzer.models.black_swan import AlertLevel
from market_analyzer.models.opportunity import (
    BreakoutOpportunity,
    LEAPOpportunity,
    MomentumOpportunity,
    Verdict,
    ZeroDTEOpportunity,
)
from market_analyzer.models.ranking import (
    RankedEntry,
    RankingFeedback,
    StrategyType,
    TradeRankingResult,
)

if TYPE_CHECKING:
    from market_analyzer.data.service import DataService
    from market_analyzer.service.black_swan import BlackSwanService
    from market_analyzer.service.levels import LevelsService
    from market_analyzer.service.opportunity import OpportunityService

logger = logging.getLogger(__name__)

# Type alias for any opportunity result
OpportunityResult = Union[
    ZeroDTEOpportunity, LEAPOpportunity, BreakoutOpportunity, MomentumOpportunity
]

# Map StrategyType -> OpportunityService method name
_ASSESS_METHODS: dict[StrategyType, str] = {
    StrategyType.ZERO_DTE: "assess_zero_dte",
    StrategyType.LEAP: "assess_leap",
    StrategyType.BREAKOUT: "assess_breakout",
    StrategyType.MOMENTUM: "assess_momentum",
}


def _extract_regime_id(result: OpportunityResult) -> int:
    """Extract regime_id from any opportunity result."""
    return result.regime_id


def _extract_phase_id(result: OpportunityResult) -> int:
    """Extract phase_id from opportunity result (default 1 for 0DTE which lacks it)."""
    return getattr(result, "phase_id", 1)


def _extract_days_to_earnings(result: OpportunityResult) -> int | None:
    return result.days_to_earnings


def _extract_strategy_name(result: OpportunityResult) -> str:
    """Extract specific strategy name (e.g. 'iron_condor')."""
    if isinstance(result, ZeroDTEOpportunity):
        return result.zero_dte_strategy.value
    elif isinstance(result, LEAPOpportunity):
        return result.leap_strategy.value
    elif isinstance(result, BreakoutOpportunity):
        return result.breakout_strategy.value
    elif isinstance(result, MomentumOpportunity):
        return result.momentum_strategy.value
    return "unknown"


def _extract_macro_events_7d(result: OpportunityResult) -> int:
    """Extract macro event count. Available directly on some results."""
    if isinstance(result, ZeroDTEOpportunity):
        return 1 if result.has_macro_event_today else 0
    elif isinstance(result, LEAPOpportunity):
        return result.macro_events_next_30_days  # best proxy
    return 0


class TradeRankingService:
    """Rank trade ideas across tickers and strategy types.

    Runs opportunity assessments, scores each ticker x strategy pair,
    and returns ranked results with grouping by ticker and by strategy.
    """

    def __init__(
        self,
        opportunity_service: OpportunityService,
        levels_service: LevelsService,
        black_swan_service: BlackSwanService,
        data_service: DataService | None = None,
        weight_provider: WeightProvider | None = None,
    ) -> None:
        self.opportunity = opportunity_service
        self.levels = levels_service
        self.black_swan = black_swan_service
        self.data_service = data_service
        self.weight_provider = weight_provider or ConfigWeightProvider()

    def rank(
        self,
        tickers: list[str],
        strategies: list[StrategyType] | None = None,
        as_of: date | None = None,
    ) -> TradeRankingResult:
        """Run all assessments, score, and rank.

        Args:
            tickers: List of tickers to assess.
            strategies: Strategy types to evaluate. None = all 4.
            as_of: Override assessment date.

        Returns:
            TradeRankingResult with ranked entries.
        """
        as_of = as_of or date.today()
        strategies = strategies or list(StrategyType)

        # 1. Black swan gate check
        black_swan_level = AlertLevel.NORMAL
        black_swan_score = 0.0
        black_swan_gate = False

        try:
            bs_alert = self.black_swan.alert(as_of_date=as_of)
            black_swan_level = bs_alert.alert_level
            black_swan_score = bs_alert.composite_score
            black_swan_gate = bs_alert.alert_level == AlertLevel.CRITICAL
        except Exception:
            logger.warning("Black swan check failed, proceeding without gate", exc_info=True)

        # If CRITICAL, return empty ranking with gate=True
        if black_swan_gate:
            return TradeRankingResult(
                as_of_date=as_of,
                tickers=tickers,
                top_trades=[],
                by_ticker={t: [] for t in tickers},
                by_strategy={s: [] for s in strategies},
                black_swan_level=black_swan_level.value,
                black_swan_gate=True,
                total_assessed=0,
                total_actionable=0,
                summary=f"CRITICAL black swan alert (score={black_swan_score:.2f}). All trading halted.",
            )

        # 2. For each ticker x strategy, run assessment
        entries: list[RankedEntry] = []
        total_assessed = 0

        for ticker in tickers:
            # Get technicals once per ticker (used in scoring)
            technicals = None
            try:
                technicals = self.opportunity.technical_service.snapshot(ticker)
            except Exception:
                logger.warning("Failed to get technicals for %s", ticker, exc_info=True)
                continue

            # Get levels once per ticker (for R:R)
            levels_result = None
            try:
                levels_result = self.levels.analyze(ticker)
            except Exception:
                logger.debug("Levels analysis failed for %s, R:R will default", ticker)

            # Get macro event count (from the opportunity service's macro)
            macro_events_7d = 0
            try:
                macro = self.opportunity.macro_service.calendar(as_of=as_of)
                macro_events_7d = len(macro.events_next_7_days)
            except Exception:
                pass

            for strategy in strategies:
                total_assessed += 1
                method_name = _ASSESS_METHODS[strategy]
                assess_fn = getattr(self.opportunity, method_name, None)
                if assess_fn is None:
                    continue

                try:
                    result: OpportunityResult = assess_fn(ticker, as_of=as_of)
                except Exception:
                    logger.warning(
                        "Assessment failed for %s/%s", ticker, strategy, exc_info=True
                    )
                    continue

                regime_id = _extract_regime_id(result)
                phase_id = _extract_phase_id(result)
                days_to_earnings = _extract_days_to_earnings(result)

                # Get weights from provider
                weights = self.weight_provider.get_weights(ticker, strategy)

                # Compute score
                breakdown = compute_composite_score(
                    verdict=result.verdict,
                    confidence=result.confidence,
                    regime_id=regime_id,
                    phase_id=phase_id,
                    strategy=strategy,
                    technicals=technicals,
                    levels=levels_result,
                    black_swan_score=black_swan_score,
                    events_next_7_days=macro_events_7d,
                    days_to_earnings=days_to_earnings,
                    weights=weights,
                )
                composite = composite_from_breakdown(breakdown, weights)

                entries.append(
                    RankedEntry(
                        rank=0,  # assigned after sorting
                        ticker=ticker,
                        strategy_type=strategy,
                        verdict=result.verdict,
                        composite_score=round(composite, 4),
                        breakdown=breakdown,
                        strategy_name=_extract_strategy_name(result),
                        direction=result.strategy.direction,
                        rationale=result.strategy.rationale,
                        risk_notes=result.strategy.risk_notes,
                    )
                )

        # 3. Sort by composite_score descending, assign ranks
        entries.sort(key=lambda e: e.composite_score, reverse=True)
        for i, entry in enumerate(entries):
            entry.rank = i + 1

        # 4. Group by ticker and by strategy
        by_ticker: dict[str, list[RankedEntry]] = defaultdict(list)
        by_strategy: dict[StrategyType, list[RankedEntry]] = defaultdict(list)

        for entry in entries:
            by_ticker[entry.ticker].append(entry)
            by_strategy[entry.strategy_type].append(entry)

        # Ensure all tickers/strategies appear even if empty
        for t in tickers:
            if t not in by_ticker:
                by_ticker[t] = []
        for s in strategies:
            if s not in by_strategy:
                by_strategy[s] = []

        total_actionable = sum(1 for e in entries if e.verdict != Verdict.NO_GO)

        # Generate summary
        summary = self._build_summary(entries, total_assessed, total_actionable, black_swan_level)

        return TradeRankingResult(
            as_of_date=as_of,
            tickers=tickers,
            top_trades=entries,
            by_ticker=dict(by_ticker),
            by_strategy=dict(by_strategy),
            black_swan_level=black_swan_level.value,
            black_swan_gate=False,
            total_assessed=total_assessed,
            total_actionable=total_actionable,
            summary=summary,
        )

    def record_feedback(self, feedback: RankingFeedback) -> None:
        """Store feedback for future RL training. Appends to parquet."""
        try:
            import pandas as pd

            feedback_dir = Path.home() / ".market_analyzer" / "feedback"
            feedback_dir.mkdir(parents=True, exist_ok=True)
            path = feedback_dir / "ranking_feedback.parquet"

            row = feedback.model_dump()
            df_new = pd.DataFrame([row])

            if path.exists():
                df_existing = pd.read_parquet(path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_combined = df_new

            df_combined.to_parquet(path, index=False)
            logger.info("Recorded ranking feedback: %s/%s", feedback.ticker, feedback.strategy_type)
        except Exception:
            logger.warning("Failed to record ranking feedback", exc_info=True)

    @staticmethod
    def _build_summary(
        entries: list[RankedEntry],
        total_assessed: int,
        total_actionable: int,
        alert_level: AlertLevel,
    ) -> str:
        if not entries:
            return f"No trades assessed. Alert level: {alert_level.value}."

        best = entries[0]
        go_count = sum(1 for e in entries if e.verdict == Verdict.GO)

        parts = [
            f"Ranked {total_assessed} ticker/strategy pairs.",
            f"{total_actionable} actionable ({go_count} GO, {total_actionable - go_count} CAUTION).",
        ]
        if best.composite_score > 0:
            parts.append(
                f"Top: {best.ticker} {best.strategy_type} "
                f"(score={best.composite_score:.2f}, {best.verdict})."
            )
        if alert_level != AlertLevel.NORMAL:
            parts.append(f"Alert: {alert_level.value}.")

        return " ".join(parts)
