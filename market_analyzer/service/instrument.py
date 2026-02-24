"""InstrumentAnalysisService: unified per-ticker report."""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from market_analyzer.models.instrument import InstrumentAnalysis
from market_analyzer.models.regime import RegimeID

if TYPE_CHECKING:
    from market_analyzer.data.service import DataService
    from market_analyzer.service.fundamental import FundamentalService
    from market_analyzer.service.levels import LevelsService
    from market_analyzer.service.opportunity import OpportunityService
    from market_analyzer.service.phase import PhaseService
    from market_analyzer.service.regime import RegimeService
    from market_analyzer.service.technical import TechnicalService

logger = logging.getLogger(__name__)


class InstrumentAnalysisService:
    """Produce a unified analysis report for a single ticker.

    Composes regime, phase, technicals, levels, fundamentals,
    and optional opportunity assessments into one InstrumentAnalysis.
    """

    def __init__(
        self,
        regime_service: RegimeService | None = None,
        technical_service: TechnicalService | None = None,
        phase_service: PhaseService | None = None,
        levels_service: LevelsService | None = None,
        fundamental_service: FundamentalService | None = None,
        opportunity_service: OpportunityService | None = None,
        data_service: DataService | None = None,
    ) -> None:
        self.regime_service = regime_service
        self.technical_service = technical_service
        self.phase_service = phase_service
        self.levels_service = levels_service
        self.fundamental_service = fundamental_service
        self.opportunity_service = opportunity_service
        self.data_service = data_service

    def _get_ohlcv(self, ticker: str, ohlcv: pd.DataFrame | None) -> pd.DataFrame:
        if ohlcv is not None:
            return ohlcv
        if self.data_service is None:
            raise ValueError(
                "Either provide ohlcv DataFrame or initialize with a DataService"
            )
        return self.data_service.get_ohlcv(ticker)

    def analyze(
        self,
        ticker: str,
        include_opportunities: bool = False,
        ohlcv: pd.DataFrame | None = None,
    ) -> InstrumentAnalysis:
        """Produce unified analysis for a single ticker.

        Args:
            ticker: Instrument ticker.
            include_opportunities: If True, run breakout/momentum/LEAP/0DTE assessments.
            ohlcv: Optional pre-fetched OHLCV DataFrame.
        """
        if self.regime_service is None or self.technical_service is None or self.phase_service is None:
            raise ValueError(
                "InstrumentAnalysisService requires regime, technical, and phase services"
            )

        df = self._get_ohlcv(ticker, ohlcv)
        today = date.today()

        # Core analysis
        regime = self.regime_service.detect(ticker, df)
        technicals = self.technical_service.snapshot(ticker, df)
        phase = self.phase_service.detect(ticker, df)

        # Optional: levels
        levels = None
        if self.levels_service is not None:
            try:
                levels = self.levels_service.analyze(ticker, ohlcv=df)
            except Exception as exc:
                logger.warning("Levels analysis failed for %s: %s", ticker, exc)

        # Optional: fundamentals
        fundamentals = None
        if self.fundamental_service is not None:
            try:
                fundamentals = self.fundamental_service.get(ticker)
            except Exception as exc:
                logger.warning("Fundamentals fetch failed for %s: %s", ticker, exc)

        # Optional: opportunity assessments
        breakout = momentum = leap = zero_dte = None
        if include_opportunities and self.opportunity_service is not None:
            try:
                breakout = self.opportunity_service.assess_breakout(ticker, df)
            except Exception as exc:
                logger.debug("Breakout assessment failed for %s: %s", ticker, exc)
            try:
                momentum = self.opportunity_service.assess_momentum(ticker, df)
            except Exception as exc:
                logger.debug("Momentum assessment failed for %s: %s", ticker, exc)
            try:
                leap = self.opportunity_service.assess_leap(ticker, df)
            except Exception as exc:
                logger.debug("LEAP assessment failed for %s: %s", ticker, exc)
            try:
                zero_dte = self.opportunity_service.assess_zero_dte(ticker, df)
            except Exception as exc:
                logger.debug("0DTE assessment failed for %s: %s", ticker, exc)

        # Derived fields
        trend_bias = self._determine_trend_bias(regime, technicals)
        vol_label = "high" if regime.regime.is_high_vol else "low"

        actionable: list[str] = []
        from market_analyzer.models.opportunity import Verdict
        if breakout and breakout.verdict != Verdict.NO_GO:
            actionable.append("breakout")
        if momentum and momentum.verdict != Verdict.NO_GO:
            actionable.append("momentum")
        if leap and leap.verdict != Verdict.NO_GO:
            actionable.append("leap")
        if zero_dte and zero_dte.verdict != Verdict.NO_GO:
            actionable.append("zero_dte")

        # Summary
        parts = [
            f"R{regime.regime} ({regime.confidence:.0%})",
            f"Phase: {phase.phase_name}",
            f"RSI: {technicals.rsi.value:.0f}",
            f"ATR%: {technicals.atr_pct:.2f}",
            f"Bias: {trend_bias}",
        ]
        if actionable:
            parts.append(f"Setups: {', '.join(actionable)}")

        return InstrumentAnalysis(
            ticker=ticker,
            as_of_date=today,
            regime=regime,
            phase=phase,
            technicals=technicals,
            levels=levels,
            fundamentals=fundamentals,
            breakout=breakout,
            momentum=momentum,
            leap=leap,
            zero_dte=zero_dte,
            regime_id=regime.regime,
            phase_id=phase.phase,
            trend_bias=trend_bias,
            volatility_label=vol_label,
            actionable_setups=actionable,
            summary=" | ".join(parts),
        )

    def analyze_batch(
        self,
        tickers: list[str],
        include_opportunities: bool = False,
    ) -> dict[str, InstrumentAnalysis]:
        """Analyze multiple tickers. Returns dict keyed by ticker."""
        results: dict[str, InstrumentAnalysis] = {}
        for ticker in tickers:
            try:
                results[ticker] = self.analyze(ticker, include_opportunities=include_opportunities)
            except Exception as exc:
                logger.warning("Analysis failed for %s: %s", ticker, exc)
        return results

    @staticmethod
    def _determine_trend_bias(regime, technicals) -> str:
        """Derive trend bias from regime + technicals."""
        ma = technicals.moving_averages
        # Strong directional signals
        if ma.price_vs_sma_50_pct > 2 and ma.price_vs_sma_200_pct > 5:
            return "bullish"
        if ma.price_vs_sma_50_pct < -2 and ma.price_vs_sma_200_pct < -5:
            return "bearish"
        # Regime direction
        if regime.trend_direction is not None:
            return regime.trend_direction.value
        # Mild signals
        if ma.price_vs_sma_50_pct > 0:
            return "bullish"
        if ma.price_vs_sma_50_pct < 0:
            return "bearish"
        return "neutral"
