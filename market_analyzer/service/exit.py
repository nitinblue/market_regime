"""ExitService: generate exit plans for open positions."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from market_analyzer.config import get_settings
from market_analyzer.models.exit_plan import (
    AdjustmentTrigger,
    AdjustmentTriggerType,
    ExitPlan,
    ExitReason,
    ExitTarget,
)
from market_analyzer.models.regime import RegimeID
from market_analyzer.models.strategy import OptionStructureType

if TYPE_CHECKING:
    from market_analyzer.models.levels import LevelsAnalysis
    from market_analyzer.models.regime import RegimeResult
    from market_analyzer.models.strategy import StrategyParameters
    from market_analyzer.models.technicals import TechnicalSnapshot
    from market_analyzer.service.levels import LevelsService
    from market_analyzer.service.regime import RegimeService


class ExitService:
    """Generate exit plans for open or planned positions.

    Combines level-based targets, regime-change triggers,
    and time-decay rules into a complete ExitPlan.
    """

    def __init__(
        self,
        levels_service: LevelsService | None = None,
        regime_service: RegimeService | None = None,
    ) -> None:
        self.levels_service = levels_service
        self.regime_service = regime_service

    def plan(
        self,
        ticker: str,
        strategy: StrategyParameters,
        entry_price: float,
        regime: RegimeResult,
        technicals: TechnicalSnapshot,
        levels: LevelsAnalysis | None = None,
    ) -> ExitPlan:
        """Generate exit plan for a position.

        Args:
            ticker: Instrument ticker.
            strategy: Strategy parameters (structure type).
            entry_price: Trade entry price.
            regime: Current regime.
            technicals: Current technicals.
            levels: Optional levels analysis (for target/stop prices).
        """
        today = date.today()
        cfg = get_settings().exit
        structure = strategy.primary_structure.structure_type

        # --- Profit targets ---
        profit_targets = self._build_profit_targets(
            entry_price, structure, levels, cfg.profit_target_pcts
        )

        # --- Stop loss ---
        stop_loss = self._build_stop_loss(entry_price, structure, levels, cfg.stop_loss_pct)

        # --- Trailing stop ---
        trailing_stop = self._build_trailing_stop(entry_price, technicals)

        # --- Time-based exits ---
        dte_exit = cfg.time_exit_dte
        theta_exit = cfg.theta_decay_exit_pct if self._is_income_strategy(structure) else None

        # --- Adjustment triggers ---
        adjustments = self._build_adjustments(regime, structure, technicals)

        # --- Regime change action ---
        regime_action = self._regime_change_action(regime, structure)

        # --- Max profit/loss estimates ---
        max_loss = None
        max_profit = None
        rr = None
        if stop_loss is not None and profit_targets:
            max_loss = abs(entry_price - stop_loss.price) if stop_loss.price != entry_price else None
            best_target = profit_targets[-1]  # Furthest target
            max_profit = abs(best_target.price - entry_price) if best_target.price != entry_price else None
            if max_loss and max_profit:
                rr = max_profit / max_loss

        # Summary
        parts = [f"{structure.value} @ {entry_price:.2f}"]
        if profit_targets:
            parts.append(f"Targets: {', '.join(f'{t.price:.2f}' for t in profit_targets)}")
        if stop_loss:
            parts.append(f"Stop: {stop_loss.price:.2f}")
        if rr:
            parts.append(f"R:R {rr:.1f}")

        return ExitPlan(
            ticker=ticker,
            as_of_date=today,
            entry_price=entry_price,
            strategy_type=structure.value,
            profit_targets=profit_targets,
            stop_loss=stop_loss,
            trailing_stop=trailing_stop,
            dte_exit_threshold=dte_exit,
            theta_decay_exit_pct=theta_exit,
            adjustments=adjustments,
            regime_change_action=regime_action,
            max_loss_dollars=max_loss,
            max_profit_dollars=max_profit,
            risk_reward_ratio=rr,
            summary=" | ".join(parts),
        )

    @staticmethod
    def _build_profit_targets(
        entry_price: float,
        structure: OptionStructureType,
        levels: LevelsAnalysis | None,
        target_pcts: list[float],
    ) -> list[ExitTarget]:
        """Build profit-taking targets."""
        targets: list[ExitTarget] = []

        # Level-based targets if available
        if levels is not None and levels.targets:
            for i, t in enumerate(levels.targets[:2]):
                pct = ((t.price - entry_price) / entry_price) * 100
                action = "close 50%" if i == 0 else "close remaining"
                targets.append(ExitTarget(
                    price=t.price,
                    pct_from_entry=pct,
                    reason=ExitReason.PROFIT_TARGET,
                    action=action,
                    description=f"Target {i+1}: {t.description}",
                ))
        else:
            # Percentage-based targets
            for i, pct in enumerate(target_pcts):
                direction = 1.0  # Assume long
                target_price = entry_price * (1 + pct / 100)
                action = "close 50%" if i == 0 else "close remaining"
                targets.append(ExitTarget(
                    price=round(target_price, 2),
                    pct_from_entry=pct,
                    reason=ExitReason.PROFIT_TARGET,
                    action=action,
                    description=f"Profit target at +{pct:.0f}% of max profit",
                ))

        return targets

    @staticmethod
    def _build_stop_loss(
        entry_price: float,
        structure: OptionStructureType,
        levels: LevelsAnalysis | None,
        stop_loss_pct: float,
    ) -> ExitTarget | None:
        """Build stop loss level."""
        if levels is not None and levels.stop_loss is not None:
            sl = levels.stop_loss
            pct = ((sl.price - entry_price) / entry_price) * 100
            return ExitTarget(
                price=sl.price,
                pct_from_entry=pct,
                reason=ExitReason.STOP_LOSS,
                action="close all",
                description=f"Stop loss: {sl.description}",
            )

        # Default: percentage-based stop
        stop_price = round(entry_price * (1 - stop_loss_pct / 100 * 0.01), 2)
        return ExitTarget(
            price=stop_price,
            pct_from_entry=-stop_loss_pct * 0.01,
            reason=ExitReason.STOP_LOSS,
            action="close all",
            description=f"Stop at {stop_loss_pct:.0f}% of max loss",
        )

    @staticmethod
    def _build_trailing_stop(
        entry_price: float,
        technicals: TechnicalSnapshot,
    ) -> ExitTarget | None:
        """Build trailing stop based on ATR."""
        atr = technicals.atr
        trail_price = round(entry_price - 2 * atr, 2)
        pct = ((trail_price - entry_price) / entry_price) * 100
        return ExitTarget(
            price=trail_price,
            pct_from_entry=pct,
            reason=ExitReason.STOP_LOSS,
            action="trail stop",
            description=f"Trailing stop: 2x ATR (${atr:.2f}) below entry",
        )

    @staticmethod
    def _build_adjustments(
        regime: RegimeResult,
        structure: OptionStructureType,
        technicals: TechnicalSnapshot,
    ) -> list[AdjustmentTrigger]:
        """Build adjustment triggers."""
        adjustments: list[AdjustmentTrigger] = []

        # Regime change trigger
        adjustments.append(AdjustmentTrigger(
            trigger_type=AdjustmentTriggerType.CLOSE_FULL,
            condition=f"Regime changes from R{regime.regime}",
            action="Review position; close if new regime conflicts with strategy",
            urgency="next_session",
            description="Regime change detected — reassess position validity",
        ))

        # Theta-decay roll for income strategies
        if structure in (
            OptionStructureType.IRON_CONDOR,
            OptionStructureType.IRON_BUTTERFLY,
            OptionStructureType.CREDIT_SPREAD,
        ):
            adjustments.append(AdjustmentTrigger(
                trigger_type=AdjustmentTriggerType.ROLL_OUT,
                condition="DTE < 14 and position is profitable",
                action="Roll to next monthly expiration",
                urgency="next_session",
                description="Roll out for continued theta collection",
            ))
            adjustments.append(AdjustmentTrigger(
                trigger_type=AdjustmentTriggerType.WIDEN_WINGS,
                condition="Short strike tested (price within 1 ATR)",
                action="Widen the tested side or roll away",
                urgency="immediate",
                description="Defend short strike by adjusting",
            ))

        # Directional position adjustments
        if structure in (
            OptionStructureType.DEBIT_SPREAD,
            OptionStructureType.LONG_CALL,
            OptionStructureType.LONG_PUT,
        ):
            adjustments.append(AdjustmentTrigger(
                trigger_type=AdjustmentTriggerType.CLOSE_PARTIAL,
                condition="Position up 50%+ of max profit",
                action="Close half, trail stop on remainder",
                urgency="next_session",
                description="Take partial profits on strong move",
            ))

        return adjustments

    @staticmethod
    def _regime_change_action(regime, structure) -> str:
        """What to do if regime changes."""
        r = regime.regime
        if r in (RegimeID.R1_LOW_VOL_MR, RegimeID.R2_HIGH_VOL_MR):
            return "If regime shifts to R3/R4: close income positions, switch to directional or hedge."
        elif r == RegimeID.R3_LOW_VOL_TREND:
            return "If regime shifts to R4: tighten stops, reduce size. If R1/R2: take profits."
        else:
            return "If regime normalizes to R1/R2: close defensive positions, resume income strategies."

    @staticmethod
    def _is_income_strategy(structure: OptionStructureType) -> bool:
        """Check if this is a theta/income strategy."""
        return structure in (
            OptionStructureType.IRON_CONDOR,
            OptionStructureType.IRON_BUTTERFLY,
            OptionStructureType.CREDIT_SPREAD,
            OptionStructureType.BULL_PUT_SPREAD,
            OptionStructureType.BEAR_CALL_SPREAD,
            OptionStructureType.STRANGLE,
            OptionStructureType.STRADDLE,
            OptionStructureType.CALENDAR_SPREAD,
        )
