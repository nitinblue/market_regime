"""Tests for TradeSpec helpers and integration with assessors."""

from datetime import date, timedelta

import pytest

from market_analyzer.models.opportunity import LegSpec, TradeSpec
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

    def test_ratio_spread_produces_3_legs(self) -> None:
        vs = _vol_surface()
        spec = build_single_expiry_trade_spec(
            ticker="SPY", price=580.0, atr=5.8, regime_id=1, vol_surface=vs,
            structure_type="ratio_spread", direction="bullish",
        )
        assert spec is not None
        assert len(spec.legs) == 3

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
            role="short_put", option_type="put", strike=570.0,
            strike_label="test", expiration=date(2026, 3, 27),
            days_to_expiry=31, atm_iv_at_expiry=0.22,
        )
        assert leg.short_code == "570P 3/27"

    def test_short_code_half_dollar_strike(self) -> None:
        leg = LegSpec(
            role="short_put", option_type="call", strike=32.5,
            strike_label="test", expiration=date(2026, 4, 17),
            days_to_expiry=47, atm_iv_at_expiry=0.30,
        )
        assert leg.short_code == "32.5C 4/17"

    def test_osi_symbol(self) -> None:
        leg = LegSpec(
            role="short_put", option_type="put", strike=570.0,
            strike_label="test", expiration=date(2026, 3, 27),
            days_to_expiry=31, atm_iv_at_expiry=0.22,
        )
        assert leg.osi_symbol == "260327P00570000"

    def test_osi_symbol_call(self) -> None:
        leg = LegSpec(
            role="long_call", option_type="call", strike=590.0,
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
                LegSpec(role="short_put", option_type="put", strike=570.0,
                        strike_label="", expiration=date(2026, 3, 27),
                        days_to_expiry=31, atm_iv_at_expiry=0.22),
                LegSpec(role="short_call", option_type="call", strike=590.0,
                        strike_label="", expiration=date(2026, 3, 27),
                        days_to_expiry=31, atm_iv_at_expiry=0.22),
            ],
            underlying_price=580.0,
            target_dte=31,
            target_expiration=date(2026, 3, 27),
            spec_rationale="test",
        )
        codes = spec.leg_codes
        assert codes == ["SPY 570P 3/27", "SPY 590C 3/27"]

    def test_streamer_symbols(self) -> None:
        spec = TradeSpec(
            ticker="SPY",
            legs=[
                LegSpec(role="short_put", option_type="put", strike=570.0,
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


# --- Integration: NO_GO has no trade_spec ---

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
