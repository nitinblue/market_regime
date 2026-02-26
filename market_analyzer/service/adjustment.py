"""Trade adjustment analyzer — ranks post-entry alternatives by cost efficiency."""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import TYPE_CHECKING

from market_analyzer.models.adjustment import (
    AdjustmentAnalysis,
    AdjustmentOption,
    AdjustmentType,
    PositionStatus,
    TestedSide,
)
from market_analyzer.models.opportunity import (
    LegAction,
    LegSpec,
    StructureType,
    TradeSpec,
)
from market_analyzer.models.regime import RegimeResult
from market_analyzer.models.technicals import TechnicalSnapshot
from market_analyzer.opportunity.option_plays._trade_spec_helpers import (
    _bs_price,
    compute_otm_strike,
    estimate_trade_price,
    snap_strike,
)

if TYPE_CHECKING:
    from market_analyzer.models.vol_surface import VolatilitySurface


# Urgency by regime
_REGIME_URGENCY: dict[int, str] = {
    1: "none",       # R1: low vol MR — structures work as designed
    2: "monitor",    # R2: high vol MR — swings test but revert
    3: "soon",       # R3: trending — tested side needs attention
    4: "immediate",  # R4: explosive — close > adjust
}


class AdjustmentService:
    """Analyzes open positions and ranks adjustment alternatives."""

    def analyze(
        self,
        trade_spec: TradeSpec,
        regime: RegimeResult,
        technicals: TechnicalSnapshot,
        vol_surface: VolatilitySurface | None = None,
    ) -> AdjustmentAnalysis:
        """Analyze an open trade and rank all adjustment options.

        Args:
            trade_spec: The original trade as entered.
            regime: Current regime detection result.
            technicals: Current technical snapshot (provides price, ATR).
            vol_surface: Optional vol surface for roll pricing.

        Returns:
            AdjustmentAnalysis with ranked adjustments (best-first).
        """
        price = technicals.current_price
        atr = technicals.atr

        status, tested_side, dist_put, dist_call = self._assess_status(
            trade_spec, price, atr,
        )

        pnl = self._estimate_pnl(trade_spec, price)

        remaining_dte = max(
            (trade_spec.target_expiration - date.today()).days, 0,
        )

        adjustments = self._generate_adjustments(
            trade_spec, status, tested_side, regime.regime, price, atr,
            vol_surface, remaining_dte,
        )
        adjustments = self._rank(adjustments)

        recommendation = self._build_recommendation(
            adjustments, status, tested_side, regime.regime,
        )

        summary = (
            f"{trade_spec.ticker} {trade_spec.structure_type or 'position'}: "
            f"{status.value.upper()} "
            f"({'no side tested' if tested_side == TestedSide.NONE else tested_side.value + ' side tested'}) | "
            f"P&L: ${pnl:+.2f} | "
            f"{remaining_dte} DTE | R{regime.regime} → {recommendation}"
        )

        return AdjustmentAnalysis(
            ticker=trade_spec.ticker,
            as_of_date=date.today(),
            original_trade=trade_spec,
            current_price=price,
            position_status=status,
            tested_side=tested_side,
            distance_to_short_put_pct=dist_put,
            distance_to_short_call_pct=dist_call,
            pnl_estimate=pnl,
            remaining_dte=remaining_dte,
            regime_id=regime.regime,
            adjustments=adjustments,
            recommendation=recommendation,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Status assessment
    # ------------------------------------------------------------------

    def _assess_status(
        self,
        trade_spec: TradeSpec,
        price: float,
        atr: float,
    ) -> tuple[PositionStatus, TestedSide, float | None, float | None]:
        """Determine position status, tested side, and distances to short strikes."""
        short_puts = [l for l in trade_spec.legs
                      if l.option_type == "put" and l.action == LegAction.SELL_TO_OPEN]
        short_calls = [l for l in trade_spec.legs
                       if l.option_type == "call" and l.action == LegAction.SELL_TO_OPEN]
        long_puts = [l for l in trade_spec.legs
                     if l.option_type == "put" and l.action == LegAction.BUY_TO_OPEN]
        long_calls = [l for l in trade_spec.legs
                      if l.option_type == "call" and l.action == LegAction.BUY_TO_OPEN]

        # Distances to short strikes as pct of price
        dist_put: float | None = None
        dist_call: float | None = None

        if short_puts:
            highest_short_put = max(l.strike for l in short_puts)
            dist_put = (price - highest_short_put) / price * 100
        if short_calls:
            lowest_short_call = min(l.strike for l in short_calls)
            dist_call = (lowest_short_call - price) / price * 100

        # Determine tested side
        put_tested = dist_put is not None and dist_put < (atr / price * 100)
        call_tested = dist_call is not None and dist_call < (atr / price * 100)

        if put_tested and call_tested:
            tested_side = TestedSide.BOTH
        elif put_tested:
            tested_side = TestedSide.PUT
        elif call_tested:
            tested_side = TestedSide.CALL
        else:
            tested_side = TestedSide.NONE

        # Determine status using ATR thresholds
        # Use the closer side for status
        distances_atr: list[float] = []
        if dist_put is not None:
            distances_atr.append(dist_put / (atr / price * 100))  # in ATR multiples
        if dist_call is not None:
            distances_atr.append(dist_call / (atr / price * 100))

        if not distances_atr:
            # No short strikes (debit spread / long option) — check vs long strikes
            return PositionStatus.SAFE, TestedSide.NONE, dist_put, dist_call

        min_dist_atr = min(distances_atr)

        # Check for breached / max_loss
        put_breached = dist_put is not None and dist_put < 0
        call_breached = dist_call is not None and dist_call < 0

        if put_breached or call_breached:
            # Check max_loss: past the protective wing
            if put_breached and long_puts:
                lowest_long_put = min(l.strike for l in long_puts)
                if price < lowest_long_put:
                    return PositionStatus.MAX_LOSS, tested_side, dist_put, dist_call
            if call_breached and long_calls:
                highest_long_call = max(l.strike for l in long_calls)
                if price > highest_long_call:
                    return PositionStatus.MAX_LOSS, tested_side, dist_put, dist_call
            return PositionStatus.BREACHED, tested_side, dist_put, dist_call

        if min_dist_atr > 1.0:
            return PositionStatus.SAFE, tested_side, dist_put, dist_call

        return PositionStatus.TESTED, tested_side, dist_put, dist_call

    # ------------------------------------------------------------------
    # P&L estimation
    # ------------------------------------------------------------------

    def _estimate_pnl(self, trade_spec: TradeSpec, current_price: float) -> float:
        """Estimate current P&L by repricing all legs at current price."""
        remaining_dte = max(
            (trade_spec.target_expiration - date.today()).days, 1,
        )
        dte_years = remaining_dte / 365.0

        entry_price = estimate_trade_price(trade_spec)
        if entry_price is None:
            return 0.0

        # Reprice at current underlying
        current_value = 0.0
        for leg in trade_spec.legs:
            leg_price = _bs_price(
                current_price, leg.strike, dte_years,
                leg.atm_iv_at_expiry, leg.option_type,
            )
            if leg.action == LegAction.SELL_TO_OPEN:
                current_value += leg_price * leg.quantity
            else:
                current_value -= leg_price * leg.quantity

        # P&L = current mark-to-market vs entry
        # entry_price: positive = credit received, negative = debit paid
        # current_value: positive = net credit to close, negative = net debit to close
        # For a credit trade: P&L = entry_credit - cost_to_close
        # cost_to_close = -current_value (flip sign: what we'd pay to unwind)
        return round(entry_price + current_value, 2)

    # ------------------------------------------------------------------
    # Adjustment generation
    # ------------------------------------------------------------------

    def _generate_adjustments(
        self,
        trade_spec: TradeSpec,
        status: PositionStatus,
        tested_side: TestedSide,
        regime_id: int,
        price: float,
        atr: float,
        vol_surface: VolatilitySurface | None,
        remaining_dte: int,
    ) -> list[AdjustmentOption]:
        """Generate all applicable adjustments for the structure type."""
        urgency = _REGIME_URGENCY.get(regime_id, "monitor")
        st = trade_spec.structure_type

        adjustments: list[AdjustmentOption] = []

        # DO_NOTHING is always an option
        adjustments.append(self._do_nothing(status, regime_id, urgency))

        # CLOSE_FULL is always an option
        pnl = self._estimate_pnl(trade_spec, price)
        adjustments.append(self._close_full(trade_spec, pnl, status, urgency))

        # Structure-specific adjustments
        if st in (StructureType.IRON_CONDOR, StructureType.IRON_MAN):
            adjustments.extend(self._ic_adjustments(
                trade_spec, status, tested_side, regime_id, price, atr,
                urgency, remaining_dte,
            ))
        elif st == StructureType.IRON_BUTTERFLY:
            adjustments.extend(self._ic_adjustments(
                trade_spec, status, tested_side, regime_id, price, atr,
                urgency, remaining_dte,
            ))
        elif st == StructureType.CREDIT_SPREAD:
            adjustments.extend(self._credit_spread_adjustments(
                trade_spec, status, tested_side, regime_id, price, atr,
                urgency, remaining_dte,
            ))
        elif st in (StructureType.CALENDAR, StructureType.DOUBLE_CALENDAR):
            adjustments.extend(self._calendar_adjustments(
                trade_spec, status, price, atr, urgency, remaining_dte,
            ))
        elif st == StructureType.RATIO_SPREAD:
            adjustments.extend(self._ratio_adjustments(
                trade_spec, status, tested_side, price, atr, urgency,
            ))
        elif st == StructureType.DEBIT_SPREAD:
            adjustments.extend(self._debit_spread_adjustments(
                trade_spec, status, pnl, urgency,
            ))
        elif st in (StructureType.STRADDLE, StructureType.STRANGLE):
            adjustments.extend(self._straddle_adjustments(
                trade_spec, status, tested_side, price, atr, urgency,
            ))

        return adjustments

    def _do_nothing(
        self, status: PositionStatus, regime_id: int, urgency: str,
    ) -> AdjustmentOption:
        regime_desc = {
            1: "R1 low-vol mean-reverting — theta decay working",
            2: "R2 high-vol mean-reverting — swings likely revert",
            3: "R3 trending — monitor closely",
            4: "R4 explosive — position at high risk",
        }
        rationale = regime_desc.get(regime_id, f"R{regime_id}")
        if status == PositionStatus.SAFE:
            rationale = f"Position safe, {rationale}"
            urgency = "none"
        elif status == PositionStatus.TESTED:
            rationale = f"Strike tested but not breached, {rationale}"

        return AdjustmentOption(
            adjustment_type=AdjustmentType.DO_NOTHING,
            description="Hold position — no adjustment needed",
            new_legs=[],
            close_legs=[],
            estimated_cost=0.0,
            risk_change=0.0,
            efficiency=None,
            urgency=urgency,
            rationale=rationale,
        )

    def _close_full(
        self,
        trade_spec: TradeSpec,
        pnl: float,
        status: PositionStatus,
        urgency: str,
    ) -> AdjustmentOption:
        # Closing cost = negative of current P&L (pay to exit if losing)
        cost = -pnl
        # Risk change: all risk removed
        risk_removed = -(trade_spec.wing_width_points or 5.0) * 100
        eff = abs(risk_removed) / abs(cost) if cost > 0 else None

        close_urgency = "immediate" if status == PositionStatus.MAX_LOSS else urgency

        return AdjustmentOption(
            adjustment_type=AdjustmentType.CLOSE_FULL,
            description=f"Close entire position at ${pnl:+.2f} P&L",
            new_legs=[],
            close_legs=list(trade_spec.legs),
            estimated_cost=round(cost, 2),
            risk_change=round(risk_removed, 2),
            efficiency=round(eff, 2) if eff is not None else None,
            urgency=close_urgency,
            rationale="Exit all risk — realized P&L locked in",
        )

    def _ic_adjustments(
        self,
        trade_spec: TradeSpec,
        status: PositionStatus,
        tested_side: TestedSide,
        regime_id: int,
        price: float,
        atr: float,
        urgency: str,
        remaining_dte: int,
    ) -> list[AdjustmentOption]:
        """Adjustments for iron condors and iron butterflies."""
        adjustments: list[AdjustmentOption] = []
        exp = trade_spec.target_expiration
        avg_iv = self._avg_iv(trade_spec)

        if tested_side in (TestedSide.PUT, TestedSide.BOTH):
            # Roll tested put side away (further OTM)
            short_puts = [l for l in trade_spec.legs
                          if l.option_type == "put" and l.action == LegAction.SELL_TO_OPEN]
            if short_puts:
                old_strike = max(l.strike for l in short_puts)
                new_strike = snap_strike(old_strike - 0.5 * atr, price)
                roll_credit = self._roll_credit(old_strike, new_strike, "put", price, avg_iv, remaining_dte)
                adjustments.append(AdjustmentOption(
                    adjustment_type=AdjustmentType.ROLL_AWAY,
                    description=f"Roll short put {old_strike:.0f}→{new_strike:.0f}",
                    new_legs=[self._make_leg("short_put", LegAction.SELL_TO_OPEN, "put",
                                             new_strike, exp, remaining_dte, avg_iv)],
                    close_legs=[l for l in short_puts],
                    estimated_cost=round(roll_credit, 2),
                    risk_change=-round(abs(old_strike - new_strike) * 100, 2),
                    efficiency=None if roll_credit <= 0 else round(
                        abs(old_strike - new_strike) * 100 / roll_credit, 2),
                    urgency=urgency,
                    rationale=f"Move tested put further OTM — gives price room",
                ))

        if tested_side in (TestedSide.CALL, TestedSide.BOTH):
            # Roll tested call side away
            short_calls = [l for l in trade_spec.legs
                           if l.option_type == "call" and l.action == LegAction.SELL_TO_OPEN]
            if short_calls:
                old_strike = min(l.strike for l in short_calls)
                new_strike = snap_strike(old_strike + 0.5 * atr, price)
                roll_credit = self._roll_credit(old_strike, new_strike, "call", price, avg_iv, remaining_dte)
                adjustments.append(AdjustmentOption(
                    adjustment_type=AdjustmentType.ROLL_AWAY,
                    description=f"Roll short call {old_strike:.0f}→{new_strike:.0f}",
                    new_legs=[self._make_leg("short_call", LegAction.SELL_TO_OPEN, "call",
                                             new_strike, exp, remaining_dte, avg_iv)],
                    close_legs=[l for l in short_calls],
                    estimated_cost=round(roll_credit, 2),
                    risk_change=-round(abs(new_strike - old_strike) * 100, 2),
                    efficiency=None if roll_credit <= 0 else round(
                        abs(new_strike - old_strike) * 100 / roll_credit, 2),
                    urgency=urgency,
                    rationale=f"Move tested call further OTM — gives price room",
                ))

        # Narrow untested side for credit
        if tested_side == TestedSide.PUT:
            short_calls = [l for l in trade_spec.legs
                           if l.option_type == "call" and l.action == LegAction.SELL_TO_OPEN]
            if short_calls:
                old_call = min(l.strike for l in short_calls)
                new_call = snap_strike(old_call - 0.5 * atr, price)
                if new_call > price:  # Must stay OTM
                    credit = self._narrow_credit(old_call, new_call, "call", price, avg_iv, remaining_dte)
                    adjustments.append(AdjustmentOption(
                        adjustment_type=AdjustmentType.NARROW_UNTESTED,
                        description=f"Narrow call {old_call:.0f}→{new_call:.0f} for ${abs(credit):.2f} credit",
                        new_legs=[self._make_leg("short_call", LegAction.SELL_TO_OPEN, "call",
                                                 new_call, exp, remaining_dte, avg_iv)],
                        close_legs=[l for l in short_calls],
                        estimated_cost=round(credit, 2),
                        risk_change=-round(abs(old_call - new_call) * 100, 2),
                        efficiency=None,  # Free credit
                        urgency="monitor",
                        rationale="Collect credit from untested side to offset tested side risk",
                    ))

        elif tested_side == TestedSide.CALL:
            short_puts = [l for l in trade_spec.legs
                          if l.option_type == "put" and l.action == LegAction.SELL_TO_OPEN]
            if short_puts:
                old_put = max(l.strike for l in short_puts)
                new_put = snap_strike(old_put + 0.5 * atr, price)
                if new_put < price:  # Must stay OTM
                    credit = self._narrow_credit(old_put, new_put, "put", price, avg_iv, remaining_dte)
                    adjustments.append(AdjustmentOption(
                        adjustment_type=AdjustmentType.NARROW_UNTESTED,
                        description=f"Narrow put {old_put:.0f}→{new_put:.0f} for ${abs(credit):.2f} credit",
                        new_legs=[self._make_leg("short_put", LegAction.SELL_TO_OPEN, "put",
                                                 new_put, exp, remaining_dte, avg_iv)],
                        close_legs=[l for l in short_puts],
                        estimated_cost=round(credit, 2),
                        risk_change=-round(abs(new_put - old_put) * 100, 2),
                        efficiency=None,
                        urgency="monitor",
                        rationale="Collect credit from untested side to offset tested side risk",
                    ))

        # Convert to butterfly (collapse into tested side)
        if status in (PositionStatus.TESTED, PositionStatus.BREACHED):
            # Compute credit from closing untested side
            untested_type = "call" if tested_side == TestedSide.PUT else "put"
            untested_legs = [l for l in trade_spec.legs if l.option_type == untested_type]
            convert_credit = 0.0
            for leg in untested_legs:
                leg_price = _bs_price(price, leg.strike,
                                      max(remaining_dte, 1) / 365.0,
                                      avg_iv, leg.option_type)
                if leg.action == LegAction.SELL_TO_OPEN:
                    # Buy back short = pay
                    convert_credit += leg_price * leg.quantity
                else:
                    # Sell long = receive
                    convert_credit -= leg_price * leg.quantity
            # Negative = net credit from closing untested side
            convert_credit = round(-convert_credit, 2)

            adjustments.append(AdjustmentOption(
                adjustment_type=AdjustmentType.CONVERT,
                description="Convert to butterfly — collapse untested side to tested",
                new_legs=[],
                close_legs=untested_legs,
                estimated_cost=convert_credit,
                risk_change=-round((trade_spec.wing_width_points or 5.0) * 50, 2),
                efficiency=None,
                urgency=urgency,
                rationale="Convert IC to butterfly reduces risk zone width. "
                          "Close untested side for credit, apply to tested side.",
            ))

        # Roll out in time
        if remaining_dte < 21 and status == PositionStatus.TESTED:
            new_exp = exp + timedelta(days=7)
            new_dte = remaining_dte + 7
            # Compute roll credit: buy back at current DTE, sell at new DTE
            roll_cost = self._roll_out_cost(trade_spec, tested_side, price, avg_iv,
                                            remaining_dte, new_dte)
            adjustments.append(AdjustmentOption(
                adjustment_type=AdjustmentType.ROLL_OUT,
                description=f"Roll tested side out 7 days to {new_exp}",
                new_legs=[],
                close_legs=[],
                estimated_cost=roll_cost,
                risk_change=0.0,
                efficiency=None,
                urgency=urgency,
                rationale="More time = more theta. Roll tested side for credit when possible.",
            ))

        return adjustments

    def _credit_spread_adjustments(
        self,
        trade_spec: TradeSpec,
        status: PositionStatus,
        tested_side: TestedSide,
        regime_id: int,
        price: float,
        atr: float,
        urgency: str,
        remaining_dte: int,
    ) -> list[AdjustmentOption]:
        """Adjustments for credit spreads (2-leg)."""
        adjustments: list[AdjustmentOption] = []
        exp = trade_spec.target_expiration
        avg_iv = self._avg_iv(trade_spec)

        short_legs = [l for l in trade_spec.legs if l.action == LegAction.SELL_TO_OPEN]
        if short_legs:
            short = short_legs[0]
            # Roll away
            if short.option_type == "put":
                new_strike = snap_strike(short.strike - 0.5 * atr, price)
            else:
                new_strike = snap_strike(short.strike + 0.5 * atr, price)
            roll_credit = self._roll_credit(
                short.strike, new_strike, short.option_type, price, avg_iv, remaining_dte,
            )
            adjustments.append(AdjustmentOption(
                adjustment_type=AdjustmentType.ROLL_AWAY,
                description=f"Roll short {short.option_type} {short.strike:.0f}→{new_strike:.0f}",
                new_legs=[self._make_leg(
                    f"short_{short.option_type}", LegAction.SELL_TO_OPEN,
                    short.option_type, new_strike, exp, remaining_dte, avg_iv,
                )],
                close_legs=[short],
                estimated_cost=round(roll_credit, 2),
                risk_change=-round(abs(new_strike - short.strike) * 100, 2),
                efficiency=None if roll_credit <= 0 else round(
                    abs(new_strike - short.strike) * 100 / roll_credit, 2),
                urgency=urgency,
                rationale="Move short strike further OTM for more room",
            ))

            # Roll out
            if remaining_dte < 21:
                new_exp = exp + timedelta(days=7)
                new_dte = remaining_dte + 7
                roll_cost = self._roll_out_cost(
                    trade_spec, tested_side, price, avg_iv,
                    remaining_dte, new_dte,
                )
                adjustments.append(AdjustmentOption(
                    adjustment_type=AdjustmentType.ROLL_OUT,
                    description=f"Roll out 7 days to {new_exp}",
                    new_legs=[],
                    close_legs=[],
                    estimated_cost=roll_cost,
                    risk_change=0.0,
                    efficiency=None,
                    urgency=urgency,
                    rationale="More time = more theta. Roll for credit when possible.",
                ))

        return adjustments

    def _calendar_adjustments(
        self,
        trade_spec: TradeSpec,
        status: PositionStatus,
        price: float,
        atr: float,
        urgency: str,
        remaining_dte: int,
    ) -> list[AdjustmentOption]:
        """Adjustments for calendar and double calendar spreads."""
        adjustments: list[AdjustmentOption] = []
        avg_iv = self._avg_iv(trade_spec)

        # Roll front leg
        if remaining_dte < 10:
            # Compute cost: buy back front at current DTE, sell new front at +30 DTE
            front_legs = [l for l in trade_spec.legs
                          if l.action == LegAction.SELL_TO_OPEN]
            roll_cost = 0.0
            for leg in front_legs:
                old_price = _bs_price(price, leg.strike,
                                      max(remaining_dte, 1) / 365.0,
                                      avg_iv, leg.option_type)
                new_price = _bs_price(price, leg.strike,
                                      (remaining_dte + 30) / 365.0,
                                      avg_iv, leg.option_type)
                # Buy back old (pay), sell new (receive)
                roll_cost += old_price - new_price
            roll_cost = round(roll_cost, 2)

            adjustments.append(AdjustmentOption(
                adjustment_type=AdjustmentType.ROLL_OUT,
                description="Roll front leg to next monthly expiry",
                new_legs=[],
                close_legs=[],
                estimated_cost=roll_cost,
                risk_change=0.0,
                efficiency=None,
                urgency=urgency,
                rationale="Front leg approaching expiry — roll for recurring income",
            ))

        return adjustments

    def _ratio_adjustments(
        self,
        trade_spec: TradeSpec,
        status: PositionStatus,
        tested_side: TestedSide,
        price: float,
        atr: float,
        urgency: str,
    ) -> list[AdjustmentOption]:
        """Adjustments for ratio spreads (naked risk)."""
        adjustments: list[AdjustmentOption] = []
        avg_iv = self._avg_iv(trade_spec)
        exp = trade_spec.target_expiration
        dte = max((exp - date.today()).days, 1)

        # ADD_WING to close naked risk — priority adjustment
        short_legs = [l for l in trade_spec.legs if l.action == LegAction.SELL_TO_OPEN]
        if short_legs:
            short = short_legs[0]
            if short.option_type == "call":
                wing_strike = snap_strike(short.strike + 0.5 * atr, price)
            else:
                wing_strike = snap_strike(short.strike - 0.5 * atr, price)

            wing_cost = _bs_price(price, wing_strike, dte / 365.0, avg_iv, short.option_type)

            adjustments.append(AdjustmentOption(
                adjustment_type=AdjustmentType.ADD_WING,
                description=f"Buy {wing_strike:.0f}{short.option_type[0].upper()} to cap naked risk",
                new_legs=[self._make_leg(
                    f"long_{short.option_type}", LegAction.BUY_TO_OPEN,
                    short.option_type, wing_strike, exp, dte, avg_iv,
                )],
                close_legs=[],
                estimated_cost=round(wing_cost, 2),
                risk_change=-round(abs(wing_strike - short.strike) * 100, 2),
                efficiency=round(abs(wing_strike - short.strike) * 100 / wing_cost, 2) if wing_cost > 0 else None,
                urgency="soon",
                rationale="Naked leg has UNLIMITED risk — adding wing converts to defined risk",
            ))

        return adjustments

    def _debit_spread_adjustments(
        self,
        trade_spec: TradeSpec,
        status: PositionStatus,
        pnl: float,
        urgency: str,
    ) -> list[AdjustmentOption]:
        """Adjustments for debit spreads."""
        adjustments: list[AdjustmentOption] = []

        # Close at target if profitable
        if pnl > 0:
            adjustments.append(AdjustmentOption(
                adjustment_type=AdjustmentType.CLOSE_FULL,
                description=f"Take profit at ${pnl:+.2f}",
                new_legs=[],
                close_legs=list(trade_spec.legs),
                estimated_cost=round(-pnl, 2),
                risk_change=-round((trade_spec.wing_width_points or 5.0) * 100, 2),
                efficiency=None,
                urgency="monitor",
                rationale="Debit spread in profit — consider taking gains",
            ))

        return adjustments

    def _straddle_adjustments(
        self,
        trade_spec: TradeSpec,
        status: PositionStatus,
        tested_side: TestedSide,
        price: float,
        atr: float,
        urgency: str,
    ) -> list[AdjustmentOption]:
        """Adjustments for straddles and strangles."""
        adjustments: list[AdjustmentOption] = []
        avg_iv = self._avg_iv(trade_spec)
        exp = trade_spec.target_expiration
        dte = max((exp - date.today()).days, 1)

        # ADD_WING to define risk
        if trade_spec.order_side == "credit":
            # Short straddle/strangle — undefined risk
            for opt_type in ("put", "call"):
                short_legs = [l for l in trade_spec.legs
                              if l.option_type == opt_type and l.action == LegAction.SELL_TO_OPEN]
                if short_legs:
                    short = short_legs[0]
                    if opt_type == "call":
                        wing_strike = snap_strike(short.strike + atr, price)
                    else:
                        wing_strike = snap_strike(short.strike - atr, price)
                    wing_cost = _bs_price(price, wing_strike, dte / 365.0, avg_iv, opt_type)
                    adjustments.append(AdjustmentOption(
                        adjustment_type=AdjustmentType.ADD_WING,
                        description=f"Buy {wing_strike:.0f}{opt_type[0].upper()} wing to define risk",
                        new_legs=[self._make_leg(
                            f"long_{opt_type}", LegAction.BUY_TO_OPEN,
                            opt_type, wing_strike, exp, dte, avg_iv,
                        )],
                        close_legs=[],
                        estimated_cost=round(wing_cost, 2),
                        risk_change=-round(abs(wing_strike - short.strike) * 100, 2),
                        efficiency=round(abs(wing_strike - short.strike) * 100 / wing_cost, 2) if wing_cost > 0 else None,
                        urgency="soon",
                        rationale=f"Define {opt_type} side risk with protective wing",
                    ))

        # Close tested side
        if tested_side in (TestedSide.PUT, TestedSide.CALL):
            side = "put" if tested_side == TestedSide.PUT else "call"
            tested_legs = [l for l in trade_spec.legs if l.option_type == side]
            if tested_legs:
                # Compute cost to close tested side
                close_cost = 0.0
                for leg in tested_legs:
                    leg_price = _bs_price(price, leg.strike, dte / 365.0,
                                          avg_iv, leg.option_type)
                    if leg.action == LegAction.SELL_TO_OPEN:
                        close_cost += leg_price * leg.quantity  # Buy back short
                    else:
                        close_cost -= leg_price * leg.quantity  # Sell long
                close_cost = round(close_cost, 2)

                adjustments.append(AdjustmentOption(
                    adjustment_type=AdjustmentType.CLOSE_FULL,
                    description=f"Close tested {side} side only",
                    new_legs=[],
                    close_legs=tested_legs,
                    estimated_cost=close_cost,
                    risk_change=-round((trade_spec.wing_width_points or 5.0) * 50, 2),
                    efficiency=None,
                    urgency=urgency,
                    rationale=f"Remove {side} side risk, keep untested side for theta",
                ))

        return adjustments

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _rank(self, adjustments: list[AdjustmentOption]) -> list[AdjustmentOption]:
        """Sort adjustments: best efficiency first, DO_NOTHING and credit adjustments high."""
        def sort_key(adj: AdjustmentOption) -> tuple[int, float]:
            # Priority buckets:
            # 0 = DO_NOTHING (always first — hold is the baseline)
            # 1 = credit-generating adjustments (cost <= 0) with risk reduction
            # 2 = paid adjustments sorted by efficiency (higher = better)
            # 3 = adjustments without efficiency
            # 4 = CLOSE_FULL (always last unless urgency is immediate)
            if adj.adjustment_type == AdjustmentType.DO_NOTHING:
                return (0, 0.0)
            if adj.adjustment_type == AdjustmentType.CLOSE_FULL and adj.urgency != "immediate":
                return (4, adj.estimated_cost)
            if adj.estimated_cost <= 0 and adj.risk_change < 0:
                return (1, adj.estimated_cost)
            if adj.efficiency is not None and adj.efficiency > 0:
                return (2, -adj.efficiency)
            return (3, adj.estimated_cost)

        return sorted(adjustments, key=sort_key)

    def _build_recommendation(
        self,
        adjustments: list[AdjustmentOption],
        status: PositionStatus,
        tested_side: TestedSide,
        regime_id: int,
    ) -> str:
        """Build one-line recommendation from top adjustment."""
        if not adjustments:
            return "No adjustments available"
        top = adjustments[0]
        if top.adjustment_type == AdjustmentType.DO_NOTHING:
            return "Hold — no adjustment needed"
        return f"{top.adjustment_type.value}: {top.description}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _avg_iv(self, trade_spec: TradeSpec) -> float:
        """Average IV across all legs.

        Returns 0.0 if no legs — callers must handle zero IV
        (which causes _bs_price to return intrinsic only).
        """
        if not trade_spec.legs:
            return 0.0
        return sum(l.atm_iv_at_expiry for l in trade_spec.legs) / len(trade_spec.legs)

    def _make_leg(
        self,
        role: str,
        action: LegAction,
        option_type: str,
        strike: float,
        expiration: date,
        dte: int,
        iv: float,
    ) -> LegSpec:
        return LegSpec(
            role=role, action=action, option_type=option_type,
            strike=strike, strike_label=f"{strike:.0f} {option_type}",
            expiration=expiration, days_to_expiry=dte,
            atm_iv_at_expiry=iv,
        )

    def _roll_out_cost(
        self,
        trade_spec: TradeSpec,
        tested_side: TestedSide,
        price: float,
        iv: float,
        old_dte: int,
        new_dte: int,
    ) -> float:
        """Compute cost of rolling short legs out in time.

        Reprices tested-side short legs at old DTE (buy back) and new DTE (sell).
        Returns positive = net debit, negative = net credit.
        """
        side_filter = {
            TestedSide.PUT: "put",
            TestedSide.CALL: "call",
        }
        target_type = side_filter.get(tested_side)

        short_legs = [
            l for l in trade_spec.legs
            if l.action == LegAction.SELL_TO_OPEN
            and (target_type is None or l.option_type == target_type)
        ]
        if not short_legs:
            return 0.0

        total = 0.0
        old_years = max(old_dte, 1) / 365.0
        new_years = max(new_dte, 1) / 365.0
        for leg in short_legs:
            old_price = _bs_price(price, leg.strike, old_years, iv, leg.option_type)
            new_price = _bs_price(price, leg.strike, new_years, iv, leg.option_type)
            # Buy back at old DTE (pay), sell at new DTE (receive)
            total += (old_price - new_price) * leg.quantity
        return round(total, 2)

    def _roll_credit(
        self,
        old_strike: float,
        new_strike: float,
        option_type: str,
        price: float,
        iv: float,
        dte: int,
    ) -> float:
        """Estimate net credit/debit of rolling a short strike.

        Returns positive = pay (debit), negative = receive (credit).
        """
        dte_years = max(dte, 1) / 365.0
        # Buy back old, sell new
        old_price = _bs_price(price, old_strike, dte_years, iv, option_type)
        new_price = _bs_price(price, new_strike, dte_years, iv, option_type)
        # Cost to roll: pay for old (buy back) - receive for new (sell)
        return round(old_price - new_price, 2)

    def _narrow_credit(
        self,
        old_strike: float,
        new_strike: float,
        option_type: str,
        price: float,
        iv: float,
        dte: int,
    ) -> float:
        """Estimate credit from narrowing untested side.

        Returns negative = credit received.
        """
        dte_years = max(dte, 1) / 365.0
        old_price = _bs_price(price, old_strike, dte_years, iv, option_type)
        new_price = _bs_price(price, new_strike, dte_years, iv, option_type)
        # Buy back old (further OTM, cheaper) - sell new (closer, more expensive)
        return round(old_price - new_price, 2)
