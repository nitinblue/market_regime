"""Tests for TradeSpec helpers and integration with assessors."""

from datetime import date, timedelta

import pytest

from market_analyzer.models.opportunity import LegAction, LegSpec, TradeSpec
from market_analyzer.models.vol_surface import TermStructurePoint, SkewSlice, VolatilitySurface
from market_analyzer.opportunity.option_plays._trade_spec_helpers import (
    snap_strike,
    find_best_expiration,
    compute_atm_strike,
    compute_otm_strike,
    build_single_expiry_trade_spec,
    build_dual_expiry_trade_spec,
)


# --- Fixtures ---

def _term_structure(base_date: date | None = None) -> list[TermStructurePoint]:
    today = base_date or date(2026, 3, 1)
    return [
        TermStructurePoint(expiration=today + timedelta(days=7), days_to_expiry=7, atm_iv=0.25, atm_strike=580.0),
        TermStructurePoint(expiration=today + timedelta(days=21), days_to_expiry=21, atm_iv=0.23, atm_strike=580.0),
        TermStructurePoint(expiration=today + timedelta(days=35), days_to_expiry=35, atm_iv=0.22, atm_strike=580.0),
        TermStructurePoint(expiration=today + timedelta(days=63), days_to_expiry=63, atm_iv=0.21, atm_strike=580.0),
    ]


def _vol_surface(price: float = 580.0) -> VolatilitySurface:
    today = date(2026, 3, 1)
    ts = _term_structure(today)
    exps = [pt.expiration for pt in ts]
    skew = SkewSlice(
        expiration=exps[0], days_to_expiry=7, atm_iv=0.25,
        otm_put_iv=0.28, otm_call_iv=0.26, put_skew=0.03, call_skew=0.01, skew_ratio=3.0,
    )
    return VolatilitySurface(
        ticker="SPY", as_of_date=today, underlying_price=price,
        expirations=exps, term_structure=ts,
        front_iv=0.25, back_iv=0.21, term_slope=-0.16,
        is_contango=False, is_backwardation=True,
        skew_by_expiry=[skew],
        calendar_edge_score=0.6, best_calendar_expiries=(exps[0], exps[2]),
        iv_differential_pct=19.0,
        total_contracts=500, avg_bid_ask_spread_pct=0.3,
        data_quality="good", summary="test",
    )


# --- snap_strike tests ---

class TestSnapStrike:
    def test_under_50_half_dollar_ticks(self) -> None:
        assert snap_strike(32.37, 35.0) == 32.5
        assert snap_strike(32.12, 35.0) == 32.0
        assert snap_strike(32.75, 35.0) == 33.0

    def test_50_to_200_dollar_ticks(self) -> None:
        assert snap_strike(100.3, 100.0) == 100.0
        assert snap_strike(100.6, 100.0) == 101.0
        assert snap_strike(150.7, 150.0) == 151.0

    def test_above_200_five_dollar_ticks(self) -> None:
        assert snap_strike(582.0, 580.0) == 580.0
        assert snap_strike(583.0, 580.0) == 585.0
        assert snap_strike(587.5, 580.0) == 590.0

    def test_exact_tick_unchanged(self) -> None:
        assert snap_strike(580.0, 580.0) == 580.0
        assert snap_strike(100.0, 100.0) == 100.0
        assert snap_strike(32.5, 35.0) == 32.5


# --- find_best_expiration tests ---

class TestFindBestExpiration:
    def test_exact_match_in_range(self) -> None:
        ts = _term_structure()
        pt = find_best_expiration(ts, 30, 40)
        assert pt is not None
        assert pt.days_to_expiry == 35

    def test_closest_in_range(self) -> None:
        ts = _term_structure()
        pt = find_best_expiration(ts, 20, 40)
        # 21 and 35 are in range; midpoint is 30, 35 is closer
        assert pt is not None
        assert pt.days_to_expiry in (21, 35)

    def test_fallback_outside_range(self) -> None:
        ts = _term_structure()
        pt = find_best_expiration(ts, 40, 50)
        assert pt is not None
        # Closest to midpoint 45 is 35 or 63
        assert pt.days_to_expiry in (35, 63)

    def test_empty_term_structure(self) -> None:
        assert find_best_expiration([], 30, 45) is None


# --- compute_otm_strike tests ---

class TestComputeOTMStrike:
    def test_put_below_price(self) -> None:
        strike = compute_otm_strike(580.0, 5.8, 1.0, "put", 580.0)
        # 580 - 5.8 = 574.2 -> snap to 575
        assert strike == 575.0

    def test_call_above_price(self) -> None:
        strike = compute_otm_strike(580.0, 5.8, 1.0, "call", 580.0)
        # 580 + 5.8 = 585.8 -> snap to 585
        assert strike == 585.0

    def test_multiplier_scales_distance(self) -> None:
        s1 = compute_otm_strike(580.0, 5.8, 1.0, "call", 580.0)
        s2 = compute_otm_strike(580.0, 5.8, 1.5, "call", 580.0)
        assert s2 > s1


# --- compute_atm_strike tests ---

class TestComputeATMStrike:
    def test_atm_snapped(self) -> None:
        assert compute_atm_strike(581.3) == 580.0
        assert compute_atm_strike(583.5) == 585.0
        assert compute_atm_strike(100.4) == 100.0


# --- build_single_expiry_trade_spec tests ---

class TestBuildSingleExpiryTradeSpec:
    def test_iron_condor_produces_4_legs(self) -> None:
        vs = _vol_surface()
        spec = build_single_expiry_trade_spec(
            ticker="SPY", price=580.0, atr=5.8, regime_id=1, vol_surface=vs,
            structure_type="iron_condor",
        )
        assert spec is not None
        assert len(spec.legs) == 4
        roles = {leg.role for leg in spec.legs}
        assert roles == {"short_put", "long_put", "short_call", "long_call"}

    def test_iron_butterfly_produces_4_legs(self) -> None:
        vs = _vol_surface()
        spec = build_single_expiry_trade_spec(
            ticker="SPY", price=580.0, atr=5.8, regime_id=2, vol_surface=vs,
            structure_type="iron_butterfly",
        )
        assert spec is not None
        assert len(spec.legs) == 4
        # Short put + call should be at same ATM strike
        short_legs = [l for l in spec.legs if l.role.startswith("short")]
        assert short_legs[0].strike == short_legs[1].strike

    def test_ratio_spread_produces_2_legs_with_quantity(self) -> None:
        vs = _vol_surface()
        spec = build_single_expiry_trade_spec(
            ticker="SPY", price=580.0, atr=5.8, regime_id=1, vol_surface=vs,
            structure_type="ratio_spread", direction="bullish",
        )
        assert spec is not None
        assert len(spec.legs) == 2
        short_leg = [l for l in spec.legs if l.action == LegAction.SELL_TO_OPEN][0]
        assert short_leg.quantity == 2

    def test_wing_width_populated_for_ic(self) -> None:
        vs = _vol_surface()
        spec = build_single_expiry_trade_spec(
            ticker="SPY", price=580.0, atr=5.8, regime_id=1, vol_surface=vs,
            structure_type="iron_condor",
        )
        assert spec is not None
        assert spec.wing_width_points is not None
        assert spec.wing_width_points > 0

    def test_ticker_propagated(self) -> None:
        vs = _vol_surface()
        spec = build_single_expiry_trade_spec(
            ticker="AAPL", price=580.0, atr=5.8, regime_id=1, vol_surface=vs,
            structure_type="iron_condor",
        )
        assert spec is not None
        assert spec.ticker == "AAPL"

    def test_empty_term_structure_returns_none(self) -> None:
        vs = _vol_surface()
        vs_empty = vs.model_copy(update={"term_structure": []})
        spec = build_single_expiry_trade_spec(
            ticker="SPY", price=580.0, atr=5.8, regime_id=1, vol_surface=vs_empty,
            structure_type="iron_condor",
        )
        assert spec is None


# --- build_dual_expiry_trade_spec tests ---

class TestBuildDualExpiryTradeSpec:
    def test_calendar_produces_2_legs(self) -> None:
        vs = _vol_surface()
        spec = build_dual_expiry_trade_spec(
            ticker="SPY", price=580.0, atr=5.8, vol_surface=vs,
            structure_type="calendar", strategy_type="atm_calendar",
        )
        assert spec is not None
        assert len(spec.legs) == 2
        assert spec.front_expiration is not None
        assert spec.back_expiration is not None
        assert spec.front_dte < spec.back_dte

    def test_calendar_has_iv_differential(self) -> None:
        vs = _vol_surface()
        spec = build_dual_expiry_trade_spec(
            ticker="SPY", price=580.0, atr=5.8, vol_surface=vs,
            structure_type="calendar", strategy_type="atm_calendar",
        )
        assert spec is not None
        assert spec.iv_differential_pct is not None
        assert spec.iv_at_front is not None
        assert spec.iv_at_back is not None

    def test_diagonal_produces_2_legs(self) -> None:
        vs = _vol_surface()
        spec = build_dual_expiry_trade_spec(
            ticker="SPY", price=580.0, atr=5.8, vol_surface=vs,
            structure_type="diagonal", strategy_type="bull_call_diagonal",
            trend_direction="bullish",
        )
        assert spec is not None
        assert len(spec.legs) == 2
        # Front and back should have different strikes (diagonal)
        assert spec.legs[0].strike != spec.legs[1].strike

    def test_single_expiry_term_structure_returns_none(self) -> None:
        """Can't build dual-expiry with only 1 expiration."""
        vs = _vol_surface()
        vs_single = vs.model_copy(update={"term_structure": [vs.term_structure[0]]})
        spec = build_dual_expiry_trade_spec(
            ticker="SPY", price=580.0, atr=5.8, vol_surface=vs_single,
            structure_type="calendar", strategy_type="atm_calendar",
        )
        assert spec is None


# --- LegSpec properties ---

class TestLegSpecProperties:
    def test_short_code(self) -> None:
        leg = LegSpec(
            role="short_put", action=LegAction.SELL_TO_OPEN, option_type="put", strike=570.0,
            strike_label="test", expiration=date(2026, 3, 27),
            days_to_expiry=31, atm_iv_at_expiry=0.22,
        )
        assert leg.short_code == "STO 1x 570P 3/27/26"

    def test_short_code_half_dollar_strike(self) -> None:
        leg = LegSpec(
            role="short_put", action=LegAction.SELL_TO_OPEN, option_type="call", strike=32.5,
            strike_label="test", expiration=date(2026, 4, 17),
            days_to_expiry=47, atm_iv_at_expiry=0.30,
        )
        assert leg.short_code == "STO 1x 32.5C 4/17/26"

    def test_short_code_quantity_2(self) -> None:
        leg = LegSpec(
            role="short_call", action=LegAction.SELL_TO_OPEN, quantity=2,
            option_type="call", strike=590.0,
            strike_label="test", expiration=date(2026, 3, 27),
            days_to_expiry=31, atm_iv_at_expiry=0.22,
        )
        assert leg.short_code == "STO 2x 590C 3/27/26"

    def test_osi_symbol(self) -> None:
        leg = LegSpec(
            role="short_put", action=LegAction.SELL_TO_OPEN, option_type="put", strike=570.0,
            strike_label="test", expiration=date(2026, 3, 27),
            days_to_expiry=31, atm_iv_at_expiry=0.22,
        )
        assert leg.osi_symbol == "260327P00570000"

    def test_osi_symbol_call(self) -> None:
        leg = LegSpec(
            role="long_call", action=LegAction.BUY_TO_OPEN, option_type="call", strike=590.0,
            strike_label="test", expiration=date(2026, 3, 27),
            days_to_expiry=31, atm_iv_at_expiry=0.22,
        )
        assert leg.osi_symbol == "260327C00590000"


# --- TradeSpec properties ---

class TestTradeSpecProperties:
    def test_leg_codes(self) -> None:
        spec = TradeSpec(
            ticker="SPY",
            legs=[
                LegSpec(role="short_put", action=LegAction.SELL_TO_OPEN,
                        option_type="put", strike=570.0,
                        strike_label="", expiration=date(2026, 3, 27),
                        days_to_expiry=31, atm_iv_at_expiry=0.22),
                LegSpec(role="short_call", action=LegAction.SELL_TO_OPEN,
                        option_type="call", strike=590.0,
                        strike_label="", expiration=date(2026, 3, 27),
                        days_to_expiry=31, atm_iv_at_expiry=0.22),
            ],
            underlying_price=580.0,
            target_dte=31,
            target_expiration=date(2026, 3, 27),
            spec_rationale="test",
        )
        codes = spec.leg_codes
        assert codes == ["STO 1x SPY P570 3/27/26", "STO 1x SPY C590 3/27/26"]

    def test_streamer_symbols(self) -> None:
        spec = TradeSpec(
            ticker="SPY",
            legs=[
                LegSpec(role="short_put", action=LegAction.SELL_TO_OPEN,
                        option_type="put", strike=570.0,
                        strike_label="", expiration=date(2026, 3, 27),
                        days_to_expiry=31, atm_iv_at_expiry=0.22),
            ],
            underlying_price=580.0,
            target_dte=31,
            target_expiration=date(2026, 3, 27),
            spec_rationale="test",
        )
        syms = spec.streamer_symbols
        assert len(syms) == 1
        assert syms[0] == "SPY   260327P00570000"

    def test_order_data(self) -> None:
        spec = TradeSpec(
            ticker="SPY",
            legs=[
                LegSpec(role="short_put", action=LegAction.SELL_TO_OPEN,
                        option_type="put", strike=570.0,
                        strike_label="", expiration=date(2026, 3, 27),
                        days_to_expiry=31, atm_iv_at_expiry=0.22),
            ],
            underlying_price=580.0,
            target_dte=31,
            target_expiration=date(2026, 3, 27),
            spec_rationale="test",
        )
        od = spec.order_data
        assert len(od) == 1
        assert od[0]["action"] == "STO"
        assert od[0]["quantity"] == 1
        assert od[0]["symbol"] == "SPY"
        assert od[0]["strike"] == 570.0
        assert od[0]["option_type"] == "put"
        assert "260327P00570000" in od[0]["osi_symbol"]


# --- Inverse Iron Condor (Iron Man) ---

class TestBuildInverseIronCondorLegs:
    """Tests for the Iron Man (inverse IC) leg builder."""

    def test_produces_4_legs(self) -> None:
        from market_analyzer.opportunity.option_plays._trade_spec_helpers import (
            build_inverse_iron_condor_legs,
        )
        # Use price=$100 (tick=$1) with atr=$5 for clear strike separation
        legs, wing = build_inverse_iron_condor_legs(
            price=100.0, atr=5.0, regime_id=1,
            expiration=date(2026, 3, 1), dte=0, atm_iv=0.22,
        )
        assert len(legs) == 4
        assert wing > 0

    def test_long_strikes_inside_short_strikes(self) -> None:
        """Inner (long) strikes should be closer to price than outer (short) strikes."""
        from market_analyzer.opportunity.option_plays._trade_spec_helpers import (
            build_inverse_iron_condor_legs,
        )
        legs, _ = build_inverse_iron_condor_legs(
            price=100.0, atr=5.0, regime_id=1,
            expiration=date(2026, 3, 1), dte=0, atm_iv=0.22,
        )
        long_put = next(l for l in legs if l.role == "long_put")
        short_put = next(l for l in legs if l.role == "short_put")
        long_call = next(l for l in legs if l.role == "long_call")
        short_call = next(l for l in legs if l.role == "short_call")

        # Long put closer to price (higher) than short put
        assert long_put.strike > short_put.strike
        # Long call closer to price (lower) than short call
        assert long_call.strike < short_call.strike

    def test_actions_correct(self) -> None:
        """BTO on inner legs, STO on outer legs."""
        from market_analyzer.opportunity.option_plays._trade_spec_helpers import (
            build_inverse_iron_condor_legs,
        )
        legs, _ = build_inverse_iron_condor_legs(
            price=100.0, atr=5.0, regime_id=1,
            expiration=date(2026, 3, 1), dte=0, atm_iv=0.22,
        )
        for leg in legs:
            if "long" in leg.role:
                assert leg.action == LegAction.BUY_TO_OPEN
            else:
                assert leg.action == LegAction.SELL_TO_OPEN

    def test_orb_aware_uses_range_for_long_strikes(self) -> None:
        """When ORB range provided, long strikes placed at ORB edges."""
        from market_analyzer.opportunity.option_plays._trade_spec_helpers import (
            build_inverse_iron_condor_legs,
        )
        legs, _ = build_inverse_iron_condor_legs(
            price=100.0, atr=5.0, regime_id=1,
            expiration=date(2026, 3, 1), dte=0, atm_iv=0.22,
            orb_range_high=102.0, orb_range_low=98.0,
        )
        long_put = next(l for l in legs if l.role == "long_put")
        long_call = next(l for l in legs if l.role == "long_call")

        # Long strikes should be at/near ORB range edges (snapped to $1 ticks at $100)
        assert long_put.strike == 98.0
        assert long_call.strike == 102.0

    def test_without_orb_uses_atr_for_long_strikes(self) -> None:
        from market_analyzer.opportunity.option_plays._trade_spec_helpers import (
            build_inverse_iron_condor_legs,
        )
        legs, _ = build_inverse_iron_condor_legs(
            price=100.0, atr=5.0, regime_id=1,
            expiration=date(2026, 3, 1), dte=0, atm_iv=0.22,
        )
        long_put = next(l for l in legs if l.role == "long_put")
        long_call = next(l for l in legs if l.role == "long_call")
        # Without ORB, long strikes at 0.5 ATR from price (R1)
        assert long_put.strike < 100.0
        assert long_call.strike > 100.0

    def test_labels_mention_orb(self) -> None:
        from market_analyzer.opportunity.option_plays._trade_spec_helpers import (
            build_inverse_iron_condor_legs,
        )
        legs, _ = build_inverse_iron_condor_legs(
            price=100.0, atr=5.0, regime_id=1,
            expiration=date(2026, 3, 1), dte=0, atm_iv=0.22,
            orb_range_high=102.0, orb_range_low=98.0,
        )
        long_put = next(l for l in legs if l.role == "long_put")
        assert "ORB" in long_put.strike_label


# --- Integration: NO_GO has no trade_spec ---

class TestStructureTypeAndExitFields:
    """Tests for structure_type, order_side, and exit guidance fields."""

    def test_ic_has_structure_type_and_side(self) -> None:
        vs = _vol_surface()
        spec = build_single_expiry_trade_spec(
            ticker="SPY", price=580.0, atr=5.8, regime_id=1, vol_surface=vs,
            structure_type="iron_condor",
        )
        assert spec is not None
        assert spec.structure_type == "iron_condor"
        assert spec.order_side == "credit"

    def test_ifly_has_structure_type_and_side(self) -> None:
        vs = _vol_surface()
        spec = build_single_expiry_trade_spec(
            ticker="SPY", price=580.0, atr=5.8, regime_id=2, vol_surface=vs,
            structure_type="iron_butterfly",
        )
        assert spec is not None
        assert spec.structure_type == "iron_butterfly"
        assert spec.order_side == "credit"

    def test_ratio_has_structure_type(self) -> None:
        vs = _vol_surface()
        spec = build_single_expiry_trade_spec(
            ticker="SPY", price=580.0, atr=5.8, regime_id=1, vol_surface=vs,
            structure_type="ratio_spread", direction="bullish",
        )
        assert spec is not None
        assert spec.structure_type == "ratio_spread"
        assert spec.order_side == "credit"

    def test_calendar_has_structure_type(self) -> None:
        vs = _vol_surface()
        spec = build_dual_expiry_trade_spec(
            ticker="SPY", price=580.0, atr=5.8, vol_surface=vs,
            structure_type="calendar", strategy_type="atm_calendar",
        )
        assert spec is not None
        assert spec.structure_type == "calendar"
        assert spec.order_side == "debit"

    def test_diagonal_has_structure_type(self) -> None:
        vs = _vol_surface()
        spec = build_dual_expiry_trade_spec(
            ticker="SPY", price=580.0, atr=5.8, vol_surface=vs,
            structure_type="diagonal", strategy_type="bull_call_diagonal",
            trend_direction="bullish",
        )
        assert spec is not None
        assert spec.structure_type == "diagonal"
        assert spec.order_side == "debit"

    def test_ic_exit_fields_populated(self) -> None:
        vs = _vol_surface()
        spec = build_single_expiry_trade_spec(
            ticker="SPY", price=580.0, atr=5.8, regime_id=1, vol_surface=vs,
            structure_type="iron_condor",
        )
        assert spec is not None
        assert spec.profit_target_pct == 0.50
        assert spec.stop_loss_pct == 2.0
        assert spec.exit_dte == 21
        assert spec.max_profit_desc is not None
        assert spec.max_loss_desc is not None
        assert len(spec.exit_notes) > 0

    def test_ifly_exit_fields_populated(self) -> None:
        vs = _vol_surface()
        spec = build_single_expiry_trade_spec(
            ticker="SPY", price=580.0, atr=5.8, regime_id=2, vol_surface=vs,
            structure_type="iron_butterfly",
        )
        assert spec is not None
        assert spec.profit_target_pct == 0.25
        assert spec.stop_loss_pct == 2.0
        assert spec.exit_dte == 14

    def test_calendar_exit_dte_before_front_expiry(self) -> None:
        vs = _vol_surface()
        spec = build_dual_expiry_trade_spec(
            ticker="SPY", price=580.0, atr=5.8, vol_surface=vs,
            structure_type="calendar", strategy_type="atm_calendar",
        )
        assert spec is not None
        # exit_dte should be front_dte - 7 (or 0 if front is very short)
        assert spec.exit_dte is not None
        assert spec.exit_dte <= spec.front_dte

    def test_ratio_exit_notes_warn_naked_leg(self) -> None:
        vs = _vol_surface()
        spec = build_single_expiry_trade_spec(
            ticker="SPY", price=580.0, atr=5.8, regime_id=1, vol_surface=vs,
            structure_type="ratio_spread", direction="bullish",
        )
        assert spec is not None
        assert any("NAKED" in note for note in spec.exit_notes)
        assert "UNLIMITED" in spec.max_loss_desc

    def test_exit_summary_property(self) -> None:
        vs = _vol_surface()
        spec = build_single_expiry_trade_spec(
            ticker="SPY", price=580.0, atr=5.8, regime_id=1, vol_surface=vs,
            structure_type="iron_condor",
        )
        assert spec is not None
        summary = spec.exit_summary
        assert "TP 50%" in summary
        assert "credit" in summary.lower()

    def test_order_data_has_instrument_type(self) -> None:
        vs = _vol_surface()
        spec = build_single_expiry_trade_spec(
            ticker="SPY", price=580.0, atr=5.8, regime_id=1, vol_surface=vs,
            structure_type="iron_condor",
        )
        assert spec is not None
        for od in spec.order_data:
            assert od["instrument_type"] == "EQUITY_OPTION"

    def test_default_trade_spec_fields_are_none(self) -> None:
        """TradeSpec without new fields should default to None/empty."""
        spec = TradeSpec(
            ticker="SPY",
            legs=[
                LegSpec(role="short_put", action=LegAction.SELL_TO_OPEN,
                        option_type="put", strike=570.0,
                        strike_label="", expiration=date(2026, 3, 27),
                        days_to_expiry=31, atm_iv_at_expiry=0.22),
            ],
            underlying_price=580.0,
            target_dte=31,
            target_expiration=date(2026, 3, 27),
            spec_rationale="test",
        )
        assert spec.structure_type is None
        assert spec.order_side is None
        assert spec.profit_target_pct is None
        assert spec.stop_loss_pct is None
        assert spec.exit_dte is None
        assert spec.exit_notes == []
        assert spec.exit_summary == ""


class TestNoGoNoTradeSpec:
    def test_iron_condor_no_go_no_spec(self) -> None:
        from market_analyzer.models.regime import RegimeID, RegimeResult
        from market_analyzer.opportunity.option_plays.iron_condor import assess_iron_condor

        regime = RegimeResult(
            ticker="SPY", regime=RegimeID.R4_HIGH_VOL_TREND, confidence=0.85,
            regime_probabilities={1: 0.05, 2: 0.05, 3: 0.05, 4: 0.85},
            as_of_date=date(2026, 3, 1), model_version="test", trend_direction=None,
        )
        from tests.test_iron_condor import _technicals, _vol_surface as _ic_vol
        result = assess_iron_condor("SPY", regime, _technicals(), _ic_vol())
        assert result.verdict.value == "no_go"
        assert result.trade_spec is None
