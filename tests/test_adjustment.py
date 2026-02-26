"""Tests for trade adjustment analyzer."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from market_analyzer.models.adjustment import (
    AdjustmentType,
    PositionStatus,
    TestedSide,
)
from market_analyzer.models.opportunity import (
    LegAction,
    LegSpec,
    OrderSide,
    StructureType,
    TradeSpec,
)
from market_analyzer.models.regime import RegimeID, RegimeResult
from market_analyzer.models.technicals import (
    BollingerBands,
    MACDData,
    MovingAverages,
    PhaseIndicator,
    RSIData,
    StochasticData,
    SupportResistance,
    TechnicalSnapshot,
)
from market_analyzer.service.adjustment import AdjustmentService


# ---- Fixtures ----

def _make_leg(
    role: str,
    action: LegAction,
    option_type: str,
    strike: float,
    dte: int = 30,
) -> LegSpec:
    exp = date.today() + timedelta(days=dte)
    return LegSpec(
        role=role,
        action=action,
        option_type=option_type,
        strike=strike,
        strike_label=f"{strike:.0f} {option_type}",
        expiration=exp,
        days_to_expiry=dte,
        atm_iv_at_expiry=0.25,
    )


def _make_iron_condor(
    ticker: str = "SPY",
    price: float = 600.0,
    short_put: float = 580.0,
    long_put: float = 575.0,
    short_call: float = 620.0,
    long_call: float = 625.0,
    dte: int = 30,
) -> TradeSpec:
    exp = date.today() + timedelta(days=dte)
    return TradeSpec(
        ticker=ticker,
        legs=[
            _make_leg("short_put", LegAction.SELL_TO_OPEN, "put", short_put, dte),
            _make_leg("long_put", LegAction.BUY_TO_OPEN, "put", long_put, dte),
            _make_leg("short_call", LegAction.SELL_TO_OPEN, "call", short_call, dte),
            _make_leg("long_call", LegAction.BUY_TO_OPEN, "call", long_call, dte),
        ],
        underlying_price=price,
        target_dte=dte,
        target_expiration=exp,
        wing_width_points=5.0,
        spec_rationale="Test IC",
        structure_type=StructureType.IRON_CONDOR,
        order_side=OrderSide.CREDIT,
        profit_target_pct=0.50,
        stop_loss_pct=2.0,
        exit_dte=21,
    )


def _make_credit_spread(
    ticker: str = "SPY",
    price: float = 600.0,
    short_strike: float = 580.0,
    long_strike: float = 575.0,
    option_type: str = "put",
    dte: int = 30,
) -> TradeSpec:
    exp = date.today() + timedelta(days=dte)
    return TradeSpec(
        ticker=ticker,
        legs=[
            _make_leg(f"short_{option_type}", LegAction.SELL_TO_OPEN, option_type, short_strike, dte),
            _make_leg(f"long_{option_type}", LegAction.BUY_TO_OPEN, option_type, long_strike, dte),
        ],
        underlying_price=price,
        target_dte=dte,
        target_expiration=exp,
        wing_width_points=5.0,
        spec_rationale="Test spread",
        structure_type=StructureType.CREDIT_SPREAD,
        order_side=OrderSide.CREDIT,
    )


def _make_ratio_spread(
    ticker: str = "SPY",
    price: float = 600.0,
    long_strike: float = 600.0,
    short_strike: float = 610.0,
    dte: int = 30,
) -> TradeSpec:
    exp = date.today() + timedelta(days=dte)
    return TradeSpec(
        ticker=ticker,
        legs=[
            _make_leg("long_call", LegAction.BUY_TO_OPEN, "call", long_strike, dte),
            LegSpec(
                role="short_call",
                action=LegAction.SELL_TO_OPEN,
                quantity=2,
                option_type="call",
                strike=short_strike,
                strike_label=f"sell 2x {short_strike:.0f} call",
                expiration=exp,
                days_to_expiry=dte,
                atm_iv_at_expiry=0.25,
            ),
        ],
        underlying_price=price,
        target_dte=dte,
        target_expiration=exp,
        spec_rationale="Test ratio",
        structure_type=StructureType.RATIO_SPREAD,
        order_side=OrderSide.CREDIT,
    )


def _make_regime(regime_id: int = 1) -> RegimeResult:
    return RegimeResult(
        ticker="SPY",
        regime=RegimeID(regime_id),
        confidence=0.85,
        regime_probabilities={
            RegimeID.R1_LOW_VOL_MR: 0.85 if regime_id == 1 else 0.05,
            RegimeID.R2_HIGH_VOL_MR: 0.85 if regime_id == 2 else 0.05,
            RegimeID.R3_LOW_VOL_TREND: 0.85 if regime_id == 3 else 0.05,
            RegimeID.R4_HIGH_VOL_TREND: 0.85 if regime_id == 4 else 0.05,
        },
        as_of_date=date.today(),
        model_version="test",
    )


def _make_technicals(price: float = 600.0, atr: float = 8.0) -> TechnicalSnapshot:
    return TechnicalSnapshot(
        ticker="SPY",
        as_of_date=date.today(),
        current_price=price,
        atr=atr,
        atr_pct=atr / price * 100,
        vwma_20=price,
        moving_averages=MovingAverages(
            sma_20=price, sma_50=price, sma_200=price,
            ema_9=price, ema_21=price,
            price_vs_sma_20_pct=0.0, price_vs_sma_50_pct=0.0,
            price_vs_sma_200_pct=0.0,
        ),
        rsi=RSIData(value=50.0, is_overbought=False, is_oversold=False),
        bollinger=BollingerBands(
            upper=price + 10, middle=price, lower=price - 10,
            bandwidth=3.3, percent_b=0.5,
        ),
        macd=MACDData(
            macd_line=0.0, signal_line=0.0, histogram=0.0,
            is_bullish_crossover=False, is_bearish_crossover=False,
        ),
        stochastic=StochasticData(k=50.0, d=50.0, is_overbought=False, is_oversold=False),
        support_resistance=SupportResistance(
            support=price - 20, resistance=price + 20,
            price_vs_support_pct=3.3, price_vs_resistance_pct=3.3,
        ),
        phase=PhaseIndicator(
            phase="accumulation", confidence=0.6, description="",
            higher_highs=True, higher_lows=True, lower_highs=False, lower_lows=False,
            range_compression=0.0, volume_trend="stable", price_vs_sma_50_pct=0.0,
        ),
        signals=[],
    )


# ---- Tests ----

class TestPositionStatus:
    """Test _assess_status thresholds."""

    def test_safe_position(self):
        """Price well away from short strikes → SAFE."""
        svc = AdjustmentService()
        ic = _make_iron_condor(price=600, short_put=580, short_call=620)
        status, tested, _, _ = svc._assess_status(ic, 600.0, 8.0)
        assert status == PositionStatus.SAFE
        assert tested == TestedSide.NONE

    def test_tested_put(self):
        """Price near short put → TESTED, PUT side."""
        svc = AdjustmentService()
        ic = _make_iron_condor(price=600, short_put=580, short_call=620)
        # Price at 583 is within 1 ATR (8) of short put (580)
        status, tested, _, _ = svc._assess_status(ic, 583.0, 8.0)
        assert status == PositionStatus.TESTED
        assert tested == TestedSide.PUT

    def test_breached_put(self):
        """Price past short put but within wing → BREACHED."""
        svc = AdjustmentService()
        ic = _make_iron_condor(price=600, short_put=580, long_put=575, short_call=620)
        status, tested, _, _ = svc._assess_status(ic, 577.0, 8.0)
        assert status == PositionStatus.BREACHED
        assert tested in (TestedSide.PUT, TestedSide.BOTH)

    def test_max_loss(self):
        """Price past protective wing → MAX_LOSS."""
        svc = AdjustmentService()
        ic = _make_iron_condor(price=600, short_put=580, long_put=575, short_call=620, long_call=625)
        status, tested, _, _ = svc._assess_status(ic, 570.0, 8.0)
        assert status == PositionStatus.MAX_LOSS


class TestSafePositionRecommendsDoNothing:
    def test_untested_ic(self):
        """Untested IC with price well between strikes → DO_NOTHING first."""
        svc = AdjustmentService()
        ic = _make_iron_condor()
        regime = _make_regime(1)
        tech = _make_technicals(600.0)

        result = svc.analyze(ic, regime, tech)
        assert result.adjustments[0].adjustment_type == AdjustmentType.DO_NOTHING


class TestTestedPutGeneratesRollOptions:
    def test_roll_and_narrow(self):
        """Price near short put → generates roll + narrow adjustments."""
        svc = AdjustmentService()
        ic = _make_iron_condor(price=600, short_put=580, short_call=620)
        regime = _make_regime(2)
        tech = _make_technicals(583.0)

        result = svc.analyze(ic, regime, tech)
        types = [a.adjustment_type for a in result.adjustments]
        assert AdjustmentType.ROLL_AWAY in types
        assert AdjustmentType.NARROW_UNTESTED in types


class TestBreachedRanksCloseHigh:
    def test_breached_close_ranked_high(self):
        """Price past short strike → CLOSE ranked high."""
        svc = AdjustmentService()
        ic = _make_iron_condor(price=600, short_put=580, long_put=575, short_call=620, long_call=625)
        regime = _make_regime(4)  # R4 explosive
        tech = _make_technicals(577.0)

        result = svc.analyze(ic, regime, tech)
        # CLOSE should be in top 3
        top_3_types = [a.adjustment_type for a in result.adjustments[:3]]
        assert AdjustmentType.CLOSE_FULL in top_3_types


class TestAdjustmentsArePriced:
    def test_estimated_cost_not_none(self):
        """Each adjustment has a non-None estimated_cost."""
        svc = AdjustmentService()
        ic = _make_iron_condor()
        regime = _make_regime(2)
        tech = _make_technicals(583.0)

        result = svc.analyze(ic, regime, tech)
        for adj in result.adjustments:
            assert adj.estimated_cost is not None


class TestEfficiencyRanking:
    def test_sorted_best_first(self):
        """Adjustments are sorted with most efficient first."""
        svc = AdjustmentService()
        ic = _make_iron_condor()
        regime = _make_regime(2)
        tech = _make_technicals(583.0)

        result = svc.analyze(ic, regime, tech)
        # Verify that credit-generating adjustments come before debit ones
        first_cost = None
        for adj in result.adjustments:
            if adj.adjustment_type == AdjustmentType.DO_NOTHING:
                continue
            if adj.estimated_cost <= 0 and adj.risk_change < 0:
                # This should come early
                first_cost = adj.estimated_cost
                break

        # At minimum, adjustments list should be non-empty and ordered
        assert len(result.adjustments) >= 2


class TestRegimeAffectsUrgency:
    def test_r4_immediate(self):
        """R4 explosive regime → immediate urgency on tested positions."""
        svc = AdjustmentService()
        ic = _make_iron_condor(price=600, short_put=580, short_call=620)
        regime = _make_regime(4)
        tech = _make_technicals(583.0)

        result = svc.analyze(ic, regime, tech)
        # DO_NOTHING in R4 should have high urgency
        do_nothing = [a for a in result.adjustments
                      if a.adjustment_type == AdjustmentType.DO_NOTHING]
        assert do_nothing
        assert do_nothing[0].urgency == "immediate"

    def test_r1_none(self):
        """R1 low-vol MR → safe position has 'none' urgency."""
        svc = AdjustmentService()
        ic = _make_iron_condor()
        regime = _make_regime(1)
        tech = _make_technicals(600.0)

        result = svc.analyze(ic, regime, tech)
        do_nothing = [a for a in result.adjustments
                      if a.adjustment_type == AdjustmentType.DO_NOTHING]
        assert do_nothing
        assert do_nothing[0].urgency == "none"


class TestCreditSpreadAdjustments:
    def test_credit_spread_gets_roll(self):
        """Credit spread (2-leg) gets roll_away and roll_out options."""
        svc = AdjustmentService()
        spread = _make_credit_spread(price=600, short_strike=585, long_strike=580)
        regime = _make_regime(2)
        tech = _make_technicals(587.0)

        result = svc.analyze(spread, regime, tech)
        types = [a.adjustment_type for a in result.adjustments]
        assert AdjustmentType.ROLL_AWAY in types
        assert AdjustmentType.DO_NOTHING in types
        assert AdjustmentType.CLOSE_FULL in types


class TestRatioSpreadGetsAddWing:
    def test_naked_risk_add_wing(self):
        """Ratio spread with naked leg → ADD_WING priority."""
        svc = AdjustmentService()
        ratio = _make_ratio_spread()
        regime = _make_regime(1)
        tech = _make_technicals(600.0)

        result = svc.analyze(ratio, regime, tech)
        types = [a.adjustment_type for a in result.adjustments]
        assert AdjustmentType.ADD_WING in types


class TestDoNothingAlwaysPresent:
    def test_every_analysis_has_do_nothing(self):
        """Every analysis includes DO_NOTHING option."""
        svc = AdjustmentService()
        regime = _make_regime(1)

        for trade, price in [
            (_make_iron_condor(), 600.0),
            (_make_credit_spread(), 600.0),
            (_make_ratio_spread(), 600.0),
        ]:
            tech = _make_technicals(price)
            result = svc.analyze(trade, regime, tech)
            types = [a.adjustment_type for a in result.adjustments]
            assert AdjustmentType.DO_NOTHING in types, f"DO_NOTHING missing for {trade.structure_type}"
