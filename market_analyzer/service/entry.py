"""EntryService: confirm entry signals before executing trades."""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from market_analyzer.models.entry import (
    EntryCondition,
    EntryConfirmation,
    EntryTriggerType,
)
from market_analyzer.models.technicals import (
    SignalDirection,
    SignalStrength,
    TechnicalSignal,
)

if TYPE_CHECKING:
    from market_analyzer.data.service import DataService
    from market_analyzer.service.levels import LevelsService
    from market_analyzer.service.technical import TechnicalService

logger = logging.getLogger(__name__)


class EntryService:
    """Confirm entry signals for specific trigger types.

    Checks technical conditions, support/resistance proximity,
    and volume to validate an entry before execution.
    """

    def __init__(
        self,
        technical_service: TechnicalService | None = None,
        levels_service: LevelsService | None = None,
        data_service: DataService | None = None,
    ) -> None:
        self.technical_service = technical_service
        self.levels_service = levels_service
        self.data_service = data_service

    def _get_ohlcv(self, ticker: str, ohlcv: pd.DataFrame | None) -> pd.DataFrame:
        if ohlcv is not None:
            return ohlcv
        if self.data_service is None:
            raise ValueError("Either provide ohlcv DataFrame or initialize with a DataService")
        return self.data_service.get_ohlcv(ticker)

    def confirm(
        self,
        ticker: str,
        trigger_type: EntryTriggerType,
        ohlcv: pd.DataFrame | None = None,
    ) -> EntryConfirmation:
        """Check entry conditions for a given trigger type.

        Args:
            ticker: Instrument ticker.
            trigger_type: The type of entry to confirm.
            ohlcv: Optional pre-fetched OHLCV data.
        """
        if self.technical_service is None:
            raise ValueError("EntryService requires a TechnicalService")

        df = self._get_ohlcv(ticker, ohlcv)
        today = date.today()
        technicals = self.technical_service.snapshot(ticker, df)

        # Get levels if available
        levels = None
        if self.levels_service is not None:
            try:
                levels = self.levels_service.analyze(ticker, ohlcv=df)
            except Exception:
                pass

        # Run conditions for the trigger type
        conditions = self._check_conditions(trigger_type, technicals, levels)
        conditions_met = sum(1 for c in conditions if c.met)
        total = len(conditions)

        # Weighted confidence
        total_weight = sum(c.weight for c in conditions)
        met_weight = sum(c.weight for c in conditions if c.met)
        confidence = met_weight / total_weight if total_weight > 0 else 0.0

        confirmed = confidence >= 0.6

        # Build signals
        signals: list[TechnicalSignal] = []
        if confirmed:
            signals.append(TechnicalSignal(
                name=f"{trigger_type.value} entry",
                direction=SignalDirection.BULLISH if trigger_type in (
                    EntryTriggerType.BREAKOUT_CONFIRMED,
                    EntryTriggerType.MOMENTUM_CONTINUATION,
                    EntryTriggerType.PULLBACK_TO_SUPPORT,
                ) else SignalDirection.NEUTRAL,
                strength=SignalStrength.STRONG if confidence >= 0.8 else SignalStrength.MODERATE,
                description=f"Entry confirmed ({conditions_met}/{total} conditions met)",
            ))

        # Entry/stop suggestions
        entry_price = technicals.current_price
        stop_price = None
        risk = None
        if levels is not None and levels.stop_loss is not None:
            stop_price = levels.stop_loss.price
            risk = abs(entry_price - stop_price)

        summary_parts = [
            "CONFIRMED" if confirmed else "NOT CONFIRMED",
            f"{trigger_type.value}",
            f"{conditions_met}/{total} conditions met",
            f"Confidence: {confidence:.0%}",
        ]

        return EntryConfirmation(
            ticker=ticker,
            as_of_date=today,
            trigger_type=trigger_type,
            confirmed=confirmed,
            confidence=confidence,
            conditions=conditions,
            conditions_met=conditions_met,
            conditions_total=total,
            signals=signals,
            suggested_entry_price=entry_price,
            suggested_stop_price=stop_price,
            risk_per_share=risk,
            summary=" | ".join(summary_parts),
        )

    def _check_conditions(self, trigger_type, technicals, levels) -> list[EntryCondition]:
        """Build condition list for the trigger type."""
        dispatch = {
            EntryTriggerType.BREAKOUT_CONFIRMED: self._breakout_conditions,
            EntryTriggerType.PULLBACK_TO_SUPPORT: self._pullback_conditions,
            EntryTriggerType.MOMENTUM_CONTINUATION: self._momentum_conditions,
            EntryTriggerType.MEAN_REVERSION_EXTREME: self._mean_reversion_conditions,
            EntryTriggerType.ORB_BREAKOUT: self._orb_conditions,
        }
        fn = dispatch.get(trigger_type, self._breakout_conditions)
        return fn(technicals, levels)

    @staticmethod
    def _breakout_conditions(technicals, levels) -> list[EntryCondition]:
        conditions: list[EntryCondition] = []

        # Price near/above resistance
        sr = technicals.support_resistance
        near_resistance = (
            sr.resistance is not None
            and sr.price_vs_resistance_pct is not None
            and sr.price_vs_resistance_pct >= -2.0
        )
        conditions.append(EntryCondition(
            name="Near resistance",
            met=near_resistance,
            weight=0.25,
            description=f"Price vs resistance: {sr.price_vs_resistance_pct or 0:+.1f}%",
        ))

        # VCP ready or breakout
        vcp_ok = technicals.vcp is not None and technicals.vcp.score >= 0.6
        conditions.append(EntryCondition(
            name="VCP setup",
            met=vcp_ok,
            weight=0.2,
            description=f"VCP score: {technicals.vcp.score if technicals.vcp else 0:.2f}",
        ))

        # RSI not overbought
        rsi_ok = technicals.rsi.value <= 75
        conditions.append(EntryCondition(
            name="RSI not extreme",
            met=rsi_ok,
            weight=0.15,
            description=f"RSI: {technicals.rsi.value:.0f}",
        ))

        # Above SMA 50
        above_sma50 = technicals.moving_averages.price_vs_sma_50_pct > 0
        conditions.append(EntryCondition(
            name="Above SMA50",
            met=above_sma50,
            weight=0.2,
            description=f"Price vs SMA50: {technicals.moving_averages.price_vs_sma_50_pct:+.1f}%",
        ))

        # MACD positive
        macd_ok = technicals.macd.histogram > 0
        conditions.append(EntryCondition(
            name="MACD positive",
            met=macd_ok,
            weight=0.2,
            description=f"MACD histogram: {technicals.macd.histogram:+.4f}",
        ))

        return conditions

    @staticmethod
    def _pullback_conditions(technicals, levels) -> list[EntryCondition]:
        conditions: list[EntryCondition] = []

        # Price near support
        sr = technicals.support_resistance
        near_support = (
            sr.support is not None
            and sr.price_vs_support_pct is not None
            and sr.price_vs_support_pct <= 3.0
        )
        conditions.append(EntryCondition(
            name="Near support",
            met=near_support,
            weight=0.3,
            description=f"Price vs support: {sr.price_vs_support_pct or 0:+.1f}%",
        ))

        # RSI oversold or near oversold
        rsi_ok = technicals.rsi.value <= 40
        conditions.append(EntryCondition(
            name="RSI oversold zone",
            met=rsi_ok,
            weight=0.25,
            description=f"RSI: {technicals.rsi.value:.0f}",
        ))

        # Above SMA 200
        above_sma200 = technicals.moving_averages.price_vs_sma_200_pct > 0
        conditions.append(EntryCondition(
            name="Above SMA200 (uptrend intact)",
            met=above_sma200,
            weight=0.25,
            description=f"Price vs SMA200: {technicals.moving_averages.price_vs_sma_200_pct:+.1f}%",
        ))

        # Stochastic oversold
        stoch_ok = technicals.stochastic.k <= 30
        conditions.append(EntryCondition(
            name="Stochastic oversold",
            met=stoch_ok,
            weight=0.2,
            description=f"Stochastic K: {technicals.stochastic.k:.0f}",
        ))

        return conditions

    @staticmethod
    def _momentum_conditions(technicals, levels) -> list[EntryCondition]:
        conditions: list[EntryCondition] = []

        # MACD bullish
        macd_bull = technicals.macd.histogram > 0
        conditions.append(EntryCondition(
            name="MACD positive momentum",
            met=macd_bull,
            weight=0.25,
            description=f"MACD histogram: {technicals.macd.histogram:+.4f}",
        ))

        # RSI 50-70 (healthy uptrend)
        rsi_ok = 50 <= technicals.rsi.value <= 70
        conditions.append(EntryCondition(
            name="RSI healthy range",
            met=rsi_ok,
            weight=0.25,
            description=f"RSI: {technicals.rsi.value:.0f}",
        ))

        # Above all key MAs
        ma = technicals.moving_averages
        above_all = ma.price_vs_sma_20_pct > 0 and ma.price_vs_sma_50_pct > 0
        conditions.append(EntryCondition(
            name="Above SMA20 & SMA50",
            met=above_all,
            weight=0.25,
            description=f"SMA20: {ma.price_vs_sma_20_pct:+.1f}%, SMA50: {ma.price_vs_sma_50_pct:+.1f}%",
        ))

        # Bullish MACD crossover (bonus)
        crossover = technicals.macd.is_bullish_crossover
        conditions.append(EntryCondition(
            name="MACD bullish crossover",
            met=crossover,
            weight=0.25,
            description="Recent MACD bullish crossover" if crossover else "No recent crossover",
        ))

        return conditions

    @staticmethod
    def _mean_reversion_conditions(technicals, levels) -> list[EntryCondition]:
        conditions: list[EntryCondition] = []

        # RSI extreme
        rsi_extreme = technicals.rsi.value <= 25 or technicals.rsi.value >= 75
        conditions.append(EntryCondition(
            name="RSI extreme",
            met=rsi_extreme,
            weight=0.3,
            description=f"RSI: {technicals.rsi.value:.0f}",
        ))

        # Bollinger band extreme
        bb_extreme = technicals.bollinger.percent_b <= 0.0 or technicals.bollinger.percent_b >= 1.0
        conditions.append(EntryCondition(
            name="Bollinger extreme",
            met=bb_extreme,
            weight=0.3,
            description=f"Bollinger %B: {technicals.bollinger.percent_b:.2f}",
        ))

        # Stochastic confirmation
        stoch_extreme = technicals.stochastic.is_oversold or technicals.stochastic.is_overbought
        conditions.append(EntryCondition(
            name="Stochastic extreme",
            met=stoch_extreme,
            weight=0.2,
            description=f"Stochastic K: {technicals.stochastic.k:.0f}",
        ))

        # Not in R4 trending
        # (We don't have regime here, so check price structure)
        phase_ok = technicals.phase.phase.value in ("accumulation", "distribution")
        conditions.append(EntryCondition(
            name="Range-bound phase",
            met=phase_ok,
            weight=0.2,
            description=f"Phase: {technicals.phase.phase.value}",
        ))

        return conditions

    @staticmethod
    def _orb_conditions(technicals, levels) -> list[EntryCondition]:
        """ORB conditions — simplified since we may not have intraday data."""
        conditions: list[EntryCondition] = []

        # Above SMA 20
        above_sma20 = technicals.moving_averages.price_vs_sma_20_pct > 0
        conditions.append(EntryCondition(
            name="Above SMA20",
            met=above_sma20,
            weight=0.25,
            description=f"Price vs SMA20: {technicals.moving_averages.price_vs_sma_20_pct:+.1f}%",
        ))

        # MACD positive
        macd_ok = technicals.macd.histogram > 0
        conditions.append(EntryCondition(
            name="MACD positive",
            met=macd_ok,
            weight=0.25,
            description=f"MACD histogram: {technicals.macd.histogram:+.4f}",
        ))

        # RSI in healthy zone
        rsi_ok = 40 <= technicals.rsi.value <= 70
        conditions.append(EntryCondition(
            name="RSI healthy",
            met=rsi_ok,
            weight=0.25,
            description=f"RSI: {technicals.rsi.value:.0f}",
        ))

        # ATR% in tradeable range
        atr_ok = 0.3 <= technicals.atr_pct <= 3.0
        conditions.append(EntryCondition(
            name="ATR% tradeable",
            met=atr_ok,
            weight=0.25,
            description=f"ATR%: {technicals.atr_pct:.2f}",
        ))

        return conditions
