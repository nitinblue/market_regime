"""StrategyService: select option structure and size positions."""

from __future__ import annotations

import math
from datetime import date
from typing import TYPE_CHECKING

from market_analyzer.config import get_settings
from market_analyzer.models.regime import RegimeID
from market_analyzer.models.strategy import (
    OptionStructure,
    OptionStructureType,
    PositionSize,
    StrategyParameters,
)

if TYPE_CHECKING:
    from market_analyzer.models.regime import RegimeResult
    from market_analyzer.models.technicals import TechnicalSnapshot


class StrategyService:
    """Select option structures and size positions based on regime + technicals.

    Implements income-first bias: default to theta-harvesting,
    directional only when regime permits.
    """

    def __init__(
        self,
        account_size: float | None = None,
        max_position_pct: float | None = None,
        account_type: str = "taxable",
    ) -> None:
        cfg = get_settings().strategy
        self.account_size = account_size or cfg.default_account_size
        self.max_position_pct = max_position_pct or cfg.max_position_pct
        self.account_type = account_type

    def select(
        self,
        ticker: str,
        regime: RegimeResult,
        technicals: TechnicalSnapshot,
        setup_type: str | None = None,
    ) -> StrategyParameters:
        """Select appropriate option structure for the regime and setup.

        Args:
            ticker: Instrument ticker.
            regime: Current regime result.
            technicals: Current technical snapshot.
            setup_type: Optional setup context (e.g. "breakout", "income").
        """
        today = date.today()
        cfg = get_settings().strategy

        # Select primary structure based on regime
        primary, alternatives = self._select_structures(regime, technicals, setup_type)

        # DTE and delta ranges
        dte_range, delta_range = self._suggest_params(regime, setup_type)

        # Wing width
        wing = self._suggest_wing_width(technicals)

        # Rationale
        regime_rationale = self._regime_rationale(regime, primary)

        summary_parts = [
            f"{primary.structure_type.value}",
            f"Direction: {primary.direction}",
            f"DTE: {dte_range[0]}-{dte_range[1]}",
            f"Delta: {delta_range[0]:.0%}-{delta_range[1]:.0%}",
        ]
        if wing:
            summary_parts.append(f"Width: {wing}")

        return StrategyParameters(
            ticker=ticker,
            as_of_date=today,
            primary_structure=primary,
            alternative_structures=alternatives,
            regime_rationale=regime_rationale,
            setup_type=setup_type or "",
            suggested_dte_range=dte_range,
            suggested_delta_range=delta_range,
            wing_width_suggestion=wing,
            summary=" | ".join(summary_parts),
        )

    def size(
        self,
        strategy: StrategyParameters,
        current_price: float,
        account_size: float | None = None,
    ) -> PositionSize:
        """Calculate position size for a strategy.

        Args:
            strategy: Strategy parameters.
            current_price: Current underlying price.
            account_size: Override account size.
        """
        acct = account_size or self.account_size
        max_risk = acct * self.max_position_pct

        structure = strategy.primary_structure.structure_type

        # Estimate per-contract risk based on structure type
        per_contract_risk = self._estimate_per_contract_risk(
            structure, current_price, strategy.wing_width_suggestion
        )

        if per_contract_risk <= 0:
            suggested = 1
            max_contracts = 1
        else:
            max_contracts = max(1, int(max_risk / per_contract_risk))
            # Conservative: start with half max
            suggested = max(1, max_contracts // 2)

        margin_est = per_contract_risk * suggested
        bp_usage = (margin_est / acct * 100) if acct > 0 else None

        return PositionSize(
            ticker=strategy.ticker,
            strategy=structure,
            account_size=acct,
            max_risk_dollars=max_risk,
            max_risk_pct=self.max_position_pct * 100,
            suggested_contracts=suggested,
            max_contracts=max_contracts,
            margin_estimate=margin_est,
            buying_power_usage_pct=bp_usage,
            rationale=f"Max risk ${max_risk:.0f} ({self.max_position_pct:.0%} of ${acct:,.0f}). "
                      f"Est. {per_contract_risk:.0f}/contract → {suggested} contracts suggested.",
        )

    def _select_structures(self, regime, technicals, setup_type):
        """Select primary + alternative option structures."""
        r = regime.regime
        primary: OptionStructure
        alts: list[OptionStructure] = []

        if r == RegimeID.R1_LOW_VOL_MR:
            # Income-first: iron condors, strangles
            primary = OptionStructure(
                structure_type=OptionStructureType.IRON_CONDOR,
                direction="neutral",
                max_loss="defined",
                theta_exposure="positive",
                vega_exposure="short",
                rationale="R1 ideal for income: low vol + mean-reverting = stable range",
                risk_notes=["Avoid if earnings within 7 days"],
            )
            alts = [
                OptionStructure(
                    structure_type=OptionStructureType.CREDIT_SPREAD,
                    direction="neutral",
                    max_loss="defined",
                    theta_exposure="positive",
                    vega_exposure="short",
                    rationale="Single-side credit spread for directional lean",
                ),
                OptionStructure(
                    structure_type=OptionStructureType.BUTTERFLY,
                    direction="neutral",
                    max_loss="defined",
                    theta_exposure="positive",
                    vega_exposure="short",
                    rationale="Butterfly for pinning expectation",
                ),
            ]

        elif r == RegimeID.R2_HIGH_VOL_MR:
            # Selective income with wider wings
            primary = OptionStructure(
                structure_type=OptionStructureType.IRON_CONDOR,
                direction="neutral",
                max_loss="defined",
                theta_exposure="positive",
                vega_exposure="short",
                rationale="R2 high-vol MR: wider wings for safety, still range-bound",
                risk_notes=["Use wider wings", "Reduce size vs R1"],
            )
            alts = [
                OptionStructure(
                    structure_type=OptionStructureType.IRON_BUTTERFLY,
                    direction="neutral",
                    max_loss="defined",
                    theta_exposure="positive",
                    vega_exposure="short",
                    rationale="Iron butterfly for max premium collection in high IV",
                ),
            ]

        elif r == RegimeID.R3_LOW_VOL_TREND:
            if setup_type == "breakout":
                primary = OptionStructure(
                    structure_type=OptionStructureType.DEBIT_SPREAD,
                    direction="bullish" if technicals.moving_averages.price_vs_sma_50_pct > 0 else "bearish",
                    max_loss="defined",
                    theta_exposure="negative",
                    vega_exposure="long",
                    rationale="R3 trending + breakout: directional debit spread",
                )
            else:
                direction = "bullish" if technicals.moving_averages.price_vs_sma_50_pct > 0 else "bearish"
                primary = OptionStructure(
                    structure_type=OptionStructureType.DEBIT_SPREAD,
                    direction=direction,
                    max_loss="defined",
                    theta_exposure="negative",
                    vega_exposure="neutral",
                    rationale="R3 trending: directional spread with defined risk",
                )
            alts = [
                OptionStructure(
                    structure_type=OptionStructureType.DIAGONAL_SPREAD,
                    direction="bullish",
                    max_loss="defined",
                    theta_exposure="positive",
                    vega_exposure="neutral",
                    rationale="Diagonal for trend + income hybrid",
                ),
                OptionStructure(
                    structure_type=OptionStructureType.PMCC,
                    direction="bullish",
                    max_loss="defined",
                    theta_exposure="positive",
                    vega_exposure="long",
                    rationale="Poor man's covered call — leveraged income in uptrend",
                ),
            ]

        else:  # R4_HIGH_VOL_TREND
            primary = OptionStructure(
                structure_type=OptionStructureType.DEBIT_SPREAD,
                direction="bearish" if technicals.moving_averages.price_vs_sma_50_pct < 0 else "bullish",
                max_loss="defined",
                theta_exposure="negative",
                vega_exposure="long",
                rationale="R4 explosive: defined risk only, long vega",
                risk_notes=["Risk-defined ONLY", "Reduce position size", "No naked exposure"],
            )
            alts = [
                OptionStructure(
                    structure_type=OptionStructureType.PROTECTIVE_PUT,
                    direction="bearish",
                    max_loss="defined",
                    theta_exposure="negative",
                    vega_exposure="long",
                    rationale="Protective put for existing long positions",
                ),
                OptionStructure(
                    structure_type=OptionStructureType.LONG_PUT,
                    direction="bearish",
                    max_loss="defined",
                    theta_exposure="negative",
                    vega_exposure="long",
                    rationale="Long put for crash protection",
                ),
            ]

        return primary, alts

    @staticmethod
    def _suggest_params(regime, setup_type):
        """Suggest DTE and delta ranges."""
        cfg = get_settings().strategy
        r = regime.regime

        if r in (RegimeID.R1_LOW_VOL_MR, RegimeID.R2_HIGH_VOL_MR):
            # Income: shorter DTE, lower delta
            dte = (30, 45)
            delta = (cfg.income_delta_range[0], cfg.income_delta_range[1])
        elif r == RegimeID.R3_LOW_VOL_TREND:
            dte = (30, 60)
            delta = (cfg.directional_delta_range[0], cfg.directional_delta_range[1])
        else:
            # R4: shorter DTE for risk control
            dte = (21, 45)
            delta = (cfg.directional_delta_range[0], cfg.directional_delta_range[1])

        return dte, delta

    @staticmethod
    def _suggest_wing_width(technicals) -> str:
        """Suggest spread width based on ATR."""
        atr_pct = technicals.atr_pct
        if atr_pct < 1.0:
            return "5-wide"
        elif atr_pct < 2.0:
            return "10-wide"
        elif atr_pct < 3.0:
            return "15-wide"
        else:
            return "20-wide"

    @staticmethod
    def _regime_rationale(regime, structure) -> str:
        """Generate regime-specific rationale."""
        r = regime.regime
        names = {
            RegimeID.R1_LOW_VOL_MR: "Low-Vol Mean Reverting",
            RegimeID.R2_HIGH_VOL_MR: "High-Vol Mean Reverting",
            RegimeID.R3_LOW_VOL_TREND: "Low-Vol Trending",
            RegimeID.R4_HIGH_VOL_TREND: "High-Vol Trending",
        }
        return (
            f"Regime R{r} ({names.get(r, 'Unknown')}) at {regime.confidence:.0%} confidence. "
            f"Selected {structure.structure_type.value} ({structure.direction}). "
            f"{'Income-first bias applied.' if r in (RegimeID.R1_LOW_VOL_MR, RegimeID.R2_HIGH_VOL_MR) else 'Directional permitted by regime.'}"
        )

    @staticmethod
    def _estimate_per_contract_risk(structure, current_price, wing_width_str) -> float:
        """Rough per-contract max risk estimate."""
        # Parse wing width
        width = 5.0  # Default
        if wing_width_str:
            try:
                width = float(wing_width_str.split("-")[0])
            except (ValueError, IndexError):
                width = 5.0

        if structure in (
            OptionStructureType.IRON_CONDOR,
            OptionStructureType.IRON_BUTTERFLY,
        ):
            # Max loss ≈ wing width × 100
            return width * 100
        elif structure in (
            OptionStructureType.CREDIT_SPREAD,
            OptionStructureType.DEBIT_SPREAD,
            OptionStructureType.BULL_CALL_SPREAD,
            OptionStructureType.BEAR_PUT_SPREAD,
            OptionStructureType.BULL_PUT_SPREAD,
            OptionStructureType.BEAR_CALL_SPREAD,
        ):
            return width * 100
        elif structure in (
            OptionStructureType.BUTTERFLY,
        ):
            return width * 50  # Butterfly risk is roughly half the width
        elif structure in (
            OptionStructureType.LONG_CALL,
            OptionStructureType.LONG_PUT,
            OptionStructureType.PROTECTIVE_PUT,
        ):
            # Debit = premium, estimate ~2% of underlying
            return current_price * 0.02 * 100
        elif structure in (
            OptionStructureType.PMCC,
            OptionStructureType.DIAGONAL_SPREAD,
            OptionStructureType.CALENDAR_SPREAD,
        ):
            return current_price * 0.03 * 100
        else:
            return width * 100
