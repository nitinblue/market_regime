"""Shared helper functions for computing TradeSpec across all option play assessors.

Pure functions — no data fetching, no side effects.
"""

from __future__ import annotations

import math
from datetime import date

from market_analyzer.models.opportunity import LegAction, LegSpec, OrderSide, StructureType, TradeSpec
from market_analyzer.models.vol_surface import TermStructurePoint, VolatilitySurface


def action_from_role(role: str) -> LegAction:
    """Derive BTO/STO from a leg role string.

    Roles starting with 'short' or 'sell' → STO, everything else → BTO.
    """
    lower = role.lower()
    if lower.startswith("short") or lower.startswith("sell"):
        return LegAction.SELL_TO_OPEN
    return LegAction.BUY_TO_OPEN


def snap_strike(raw_strike: float, underlying_price: float) -> float:
    """Snap a raw strike to the nearest standard tick size.

    Rules: <$50 -> $0.50 ticks, <$200 -> $1.00 ticks, >= $200 -> $5.00 ticks.
    """
    if underlying_price < 50:
        tick = 0.50
    elif underlying_price < 200:
        tick = 1.00
    else:
        tick = 5.00
    return round(round(raw_strike / tick) * tick, 2)


def find_best_expiration(
    term_structure: list[TermStructurePoint],
    target_dte_min: int,
    target_dte_max: int,
) -> TermStructurePoint | None:
    """Find the expiration closest to target DTE range from vol surface term structure."""
    if not term_structure:
        return None

    # Prefer expirations within range
    in_range = [pt for pt in term_structure if target_dte_min <= pt.days_to_expiry <= target_dte_max]
    if in_range:
        mid = (target_dte_min + target_dte_max) / 2
        return min(in_range, key=lambda pt: abs(pt.days_to_expiry - mid))

    # Fallback: closest to the range
    mid = (target_dte_min + target_dte_max) / 2
    return min(term_structure, key=lambda pt: abs(pt.days_to_expiry - mid))


def compute_otm_strike(
    price: float,
    atr: float,
    multiplier: float,
    direction: str,
    underlying_price: float,
) -> float:
    """Compute OTM strike = price +/- (multiplier * ATR), snapped to tick.

    direction: "put" means below price, "call" means above price.
    """
    if direction == "put":
        raw = price - (multiplier * atr)
    else:
        raw = price + (multiplier * atr)
    return snap_strike(raw, underlying_price)


def compute_atm_strike(price: float) -> float:
    """Nearest ATM strike, snapped to tick."""
    return snap_strike(price, price)


def build_iron_condor_legs(
    price: float,
    atr: float,
    regime_id: int,
    expiration: date,
    dte: int,
    atm_iv: float,
) -> tuple[list[LegSpec], float]:
    """Build iron condor legs. Returns (legs, wing_width_points)."""
    # Short strike distance from price (in ATR multiples)
    short_mult = 1.0 if regime_id == 1 else 1.5
    # Wing width beyond short strike
    wing_mult = 0.5 if regime_id == 1 else 0.7

    wing_width = atr * wing_mult

    short_put = compute_otm_strike(price, atr, short_mult, "put", price)
    short_call = compute_otm_strike(price, atr, short_mult, "call", price)
    long_put = snap_strike(short_put - wing_width, price)
    long_call = snap_strike(short_call + wing_width, price)
    wing_width_points = short_put - long_put

    legs = [
        LegSpec(
            role="short_put", action=LegAction.SELL_TO_OPEN, option_type="put", strike=short_put,
            strike_label=f"{short_mult:.1f} ATR OTM put",
            expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
        ),
        LegSpec(
            role="long_put", action=LegAction.BUY_TO_OPEN, option_type="put", strike=long_put,
            strike_label=f"wing {wing_mult:.1f} ATR below short put",
            expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
        ),
        LegSpec(
            role="short_call", action=LegAction.SELL_TO_OPEN, option_type="call", strike=short_call,
            strike_label=f"{short_mult:.1f} ATR OTM call",
            expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
        ),
        LegSpec(
            role="long_call", action=LegAction.BUY_TO_OPEN, option_type="call", strike=long_call,
            strike_label=f"wing {wing_mult:.1f} ATR above short call",
            expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
        ),
    ]
    return legs, wing_width_points


def build_inverse_iron_condor_legs(
    price: float,
    atr: float,
    regime_id: int,
    expiration: date,
    dte: int,
    atm_iv: float,
    orb_range_high: float | None = None,
    orb_range_low: float | None = None,
) -> tuple[list[LegSpec], float]:
    """Build inverse iron condor (Iron Man) legs. Returns (legs, wing_width_points).

    Net debit structure — profits from big moves in either direction.
    BTO closer-to-ATM strikes, STO further OTM strikes (wings).

    If ORB range is provided, inner (long) strikes are placed near ORB boundaries
    so that the trade profits when price breaks out of the opening range.
    """
    # Inner (long) strike distance — closer to ATM
    if orb_range_high is not None and orb_range_low is not None:
        # ORB-aware: place long strikes at ORB range edges
        long_put = snap_strike(orb_range_low, price)
        long_call = snap_strike(orb_range_high, price)
    else:
        # Default: 0.5 ATR from price (tighter than standard IC)
        inner_mult = 0.5 if regime_id in (1, 2) else 0.3
        long_put = compute_otm_strike(price, atr, inner_mult, "put", price)
        long_call = compute_otm_strike(price, atr, inner_mult, "call", price)

    # Outer (short) strikes — wings, further OTM
    wing_mult = 0.5 if regime_id in (1, 2) else 0.4
    wing_width = atr * wing_mult
    short_put = snap_strike(long_put - wing_width, price)
    short_call = snap_strike(long_call + wing_width, price)
    wing_width_points = long_put - short_put

    legs = [
        LegSpec(
            role="long_put", action=LegAction.BUY_TO_OPEN, option_type="put", strike=long_put,
            strike_label="inner put (near ORB low)" if orb_range_low else "inner put",
            expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
        ),
        LegSpec(
            role="short_put", action=LegAction.SELL_TO_OPEN, option_type="put", strike=short_put,
            strike_label="wing put (further OTM)",
            expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
        ),
        LegSpec(
            role="long_call", action=LegAction.BUY_TO_OPEN, option_type="call", strike=long_call,
            strike_label="inner call (near ORB high)" if orb_range_high else "inner call",
            expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
        ),
        LegSpec(
            role="short_call", action=LegAction.SELL_TO_OPEN, option_type="call", strike=short_call,
            strike_label="wing call (further OTM)",
            expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
        ),
    ]
    return legs, wing_width_points


def build_iron_butterfly_legs(
    price: float,
    atr: float,
    regime_id: int,
    expiration: date,
    dte: int,
    atm_iv: float,
) -> tuple[list[LegSpec], float]:
    """Build iron butterfly legs. Returns (legs, wing_width_points)."""
    atm = compute_atm_strike(price)
    wing_mult = 1.0 if regime_id == 2 else 1.2

    wing_width = atr * wing_mult
    long_put = snap_strike(atm - wing_width, price)
    long_call = snap_strike(atm + wing_width, price)
    wing_width_points = atm - long_put

    legs = [
        LegSpec(
            role="short_put", action=LegAction.SELL_TO_OPEN, option_type="put", strike=atm,
            strike_label="ATM put (short straddle)",
            expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
        ),
        LegSpec(
            role="short_call", action=LegAction.SELL_TO_OPEN, option_type="call", strike=atm,
            strike_label="ATM call (short straddle)",
            expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
        ),
        LegSpec(
            role="long_put", action=LegAction.BUY_TO_OPEN, option_type="put", strike=long_put,
            strike_label=f"wing {wing_mult:.1f} ATR below ATM",
            expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
        ),
        LegSpec(
            role="long_call", action=LegAction.BUY_TO_OPEN, option_type="call", strike=long_call,
            strike_label=f"wing {wing_mult:.1f} ATR above ATM",
            expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
        ),
    ]
    return legs, wing_width_points


def build_calendar_legs(
    price: float,
    front_exp: TermStructurePoint,
    back_exp: TermStructurePoint,
    strategy_type: str,
    atr: float | None = None,
) -> list[LegSpec]:
    """Build calendar spread legs (same strike, different expirations)."""
    # Determine strike based on strategy variant
    if strategy_type in ("otm_call_calendar", "otm_call"):
        if atr:
            strike = snap_strike(price + 0.5 * atr, price)
            label = "0.5 ATR OTM call"
        else:
            strike = snap_strike(price * 1.02, price)
            label = "~2% OTM call"
        opt_type = "call"
    elif strategy_type in ("otm_put_calendar", "otm_put"):
        if atr:
            strike = snap_strike(price - 0.5 * atr, price)
            label = "0.5 ATR OTM put"
        else:
            strike = snap_strike(price * 0.98, price)
            label = "~2% OTM put"
        opt_type = "put"
    else:
        # ATM calendar (default)
        strike = compute_atm_strike(price)
        label = "ATM"
        opt_type = "call"  # Convention: ATM calendars use calls

    legs = [
        LegSpec(
            role="short_front", action=LegAction.SELL_TO_OPEN, option_type=opt_type, strike=strike,
            strike_label=f"sell front {label}",
            expiration=front_exp.expiration, days_to_expiry=front_exp.days_to_expiry,
            atm_iv_at_expiry=front_exp.atm_iv,
        ),
        LegSpec(
            role="long_back", action=LegAction.BUY_TO_OPEN, option_type=opt_type, strike=strike,
            strike_label=f"buy back {label}",
            expiration=back_exp.expiration, days_to_expiry=back_exp.days_to_expiry,
            atm_iv_at_expiry=back_exp.atm_iv,
        ),
    ]
    return legs


def build_double_calendar_legs(
    price: float,
    front_exp: TermStructurePoint,
    back_exp: TermStructurePoint,
    atr: float | None = None,
) -> list[LegSpec]:
    """Build double calendar: put calendar below + call calendar above = 4 legs.

    Put calendar at put_strike (below price), call calendar at call_strike (above price).
    Each calendar: sell front, buy back at the same strike.
    """
    offset = 0.5 * atr if atr else price * 0.02
    call_strike = snap_strike(price + offset, price)
    put_strike = snap_strike(price - offset, price)

    return [
        # Put calendar (below price)
        LegSpec(
            role="short_front_put", action=LegAction.SELL_TO_OPEN, option_type="put", strike=put_strike,
            strike_label="sell front put (below)",
            expiration=front_exp.expiration, days_to_expiry=front_exp.days_to_expiry,
            atm_iv_at_expiry=front_exp.atm_iv,
        ),
        LegSpec(
            role="long_back_put", action=LegAction.BUY_TO_OPEN, option_type="put", strike=put_strike,
            strike_label="buy back put (below)",
            expiration=back_exp.expiration, days_to_expiry=back_exp.days_to_expiry,
            atm_iv_at_expiry=back_exp.atm_iv,
        ),
        # Call calendar (above price)
        LegSpec(
            role="short_front_call", action=LegAction.SELL_TO_OPEN, option_type="call", strike=call_strike,
            strike_label="sell front call (above)",
            expiration=front_exp.expiration, days_to_expiry=front_exp.days_to_expiry,
            atm_iv_at_expiry=front_exp.atm_iv,
        ),
        LegSpec(
            role="long_back_call", action=LegAction.BUY_TO_OPEN, option_type="call", strike=call_strike,
            strike_label="buy back call (above)",
            expiration=back_exp.expiration, days_to_expiry=back_exp.days_to_expiry,
            atm_iv_at_expiry=back_exp.atm_iv,
        ),
    ]


def build_diagonal_legs(
    price: float,
    front_exp: TermStructurePoint,
    back_exp: TermStructurePoint,
    trend_direction: str,
    strategy_type: str,
    atr: float | None = None,
) -> list[LegSpec]:
    """Build diagonal spread legs (different strike, different expiration)."""
    if strategy_type == "pmcc_diagonal" or (trend_direction == "bullish" and strategy_type != "bear_put_diagonal"):
        # Bull diagonal / PMCC: sell OTM front call, buy ATM/ITM back call
        if atr:
            front_strike = snap_strike(price + 0.5 * atr, price)
        else:
            front_strike = snap_strike(price * 1.02, price)
        back_strike = compute_atm_strike(price)

        if strategy_type == "pmcc_diagonal":
            # PMCC: buy deep ITM back call
            if atr:
                back_strike = snap_strike(price - 1.0 * atr, price)
            else:
                back_strike = snap_strike(price * 0.95, price)
            back_label = "deep ITM back call (PMCC)"
        else:
            back_label = "ATM back call"

        legs = [
            LegSpec(
                role="short_front", action=LegAction.SELL_TO_OPEN, option_type="call", strike=front_strike,
                strike_label="OTM front call",
                expiration=front_exp.expiration, days_to_expiry=front_exp.days_to_expiry,
                atm_iv_at_expiry=front_exp.atm_iv,
            ),
            LegSpec(
                role="long_back", action=LegAction.BUY_TO_OPEN, option_type="call", strike=back_strike,
                strike_label=back_label,
                expiration=back_exp.expiration, days_to_expiry=back_exp.days_to_expiry,
                atm_iv_at_expiry=back_exp.atm_iv,
            ),
        ]
    else:
        # Bear diagonal: sell OTM front put, buy ATM/ITM back put
        if atr:
            front_strike = snap_strike(price - 0.5 * atr, price)
        else:
            front_strike = snap_strike(price * 0.98, price)
        back_strike = compute_atm_strike(price)

        legs = [
            LegSpec(
                role="short_front", action=LegAction.SELL_TO_OPEN, option_type="put", strike=front_strike,
                strike_label="OTM front put",
                expiration=front_exp.expiration, days_to_expiry=front_exp.days_to_expiry,
                atm_iv_at_expiry=front_exp.atm_iv,
            ),
            LegSpec(
                role="long_back", action=LegAction.BUY_TO_OPEN, option_type="put", strike=back_strike,
                strike_label="ATM back put",
                expiration=back_exp.expiration, days_to_expiry=back_exp.days_to_expiry,
                atm_iv_at_expiry=back_exp.atm_iv,
            ),
        ]
    return legs


def build_ratio_spread_legs(
    price: float,
    atr: float,
    direction: str,
    expiration: date,
    dte: int,
    atm_iv: float,
) -> list[LegSpec]:
    """Build ratio spread legs (buy 1 ATM, sell 2 OTM)."""
    atm = compute_atm_strike(price)

    if direction == "bullish":
        otm_strike = compute_otm_strike(price, atr, 1.0, "call", price)
        legs = [
            LegSpec(
                role="long_call", action=LegAction.BUY_TO_OPEN, option_type="call", strike=atm,
                strike_label="buy 1 ATM call",
                expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
            ),
            LegSpec(
                role="short_call", action=LegAction.SELL_TO_OPEN, quantity=2,
                option_type="call", strike=otm_strike,
                strike_label="sell 2x 1.0 ATR OTM call",
                expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
            ),
        ]
    else:
        otm_strike = compute_otm_strike(price, atr, 1.0, "put", price)
        legs = [
            LegSpec(
                role="long_put", action=LegAction.BUY_TO_OPEN, option_type="put", strike=atm,
                strike_label="buy 1 ATM put",
                expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
            ),
            LegSpec(
                role="short_put", action=LegAction.SELL_TO_OPEN, quantity=2,
                option_type="put", strike=otm_strike,
                strike_label="sell 2x 1.0 ATR OTM put",
                expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
            ),
        ]
    return legs


def build_single_expiry_trade_spec(
    ticker: str,
    price: float,
    atr: float,
    regime_id: int,
    vol_surface: VolatilitySurface,
    structure_type: str,
    target_dte_min: int = 30,
    target_dte_max: int = 45,
    direction: str | None = None,
) -> TradeSpec | None:
    """Build a TradeSpec for single-expiry structures (IC, IFly, ratio)."""
    exp_pt = find_best_expiration(vol_surface.term_structure, target_dte_min, target_dte_max)
    if exp_pt is None:
        return None

    if structure_type == "iron_condor":
        legs, wing_width = build_iron_condor_legs(
            price, atr, regime_id, exp_pt.expiration, exp_pt.days_to_expiry, exp_pt.atm_iv,
        )
        rationale = f"Target {target_dte_min}-{target_dte_max} DTE, matched {exp_pt.expiration} ({exp_pt.days_to_expiry}d). " \
                     f"Short strikes at {'1.0' if regime_id == 1 else '1.5'} ATR OTM, " \
                     f"wings {'0.5' if regime_id == 1 else '0.7'} ATR wide."
        return TradeSpec(
            ticker=ticker, legs=legs, underlying_price=price,
            target_dte=exp_pt.days_to_expiry, target_expiration=exp_pt.expiration,
            wing_width_points=wing_width,
            max_risk_per_spread=f"${wing_width * 100:.0f} - credit received",
            spec_rationale=rationale,
            structure_type=StructureType.IRON_CONDOR,
            order_side=OrderSide.CREDIT,
            profit_target_pct=0.50,
            stop_loss_pct=2.0,
            exit_dte=21,
            max_profit_desc="Credit received",
            max_loss_desc=f"Wing width (${wing_width:.0f}) minus credit",
            exit_notes=["Close at 50% of credit received",
                        "Close if short strike tested on either side",
                        "Close at 21 DTE to avoid gamma risk"],
        )

    if structure_type == "iron_butterfly":
        legs, wing_width = build_iron_butterfly_legs(
            price, atr, regime_id, exp_pt.expiration, exp_pt.days_to_expiry, exp_pt.atm_iv,
        )
        rationale = f"Target {target_dte_min}-{target_dte_max} DTE, matched {exp_pt.expiration} ({exp_pt.days_to_expiry}d). " \
                     f"Short straddle at ATM, wings {'1.0' if regime_id == 2 else '1.2'} ATR."
        return TradeSpec(
            ticker=ticker, legs=legs, underlying_price=price,
            target_dte=exp_pt.days_to_expiry, target_expiration=exp_pt.expiration,
            wing_width_points=wing_width,
            max_risk_per_spread=f"${wing_width * 100:.0f} - credit received",
            spec_rationale=rationale,
            structure_type=StructureType.IRON_BUTTERFLY,
            order_side=OrderSide.CREDIT,
            profit_target_pct=0.25,
            stop_loss_pct=2.0,
            exit_dte=14,
            max_profit_desc="Credit received (larger than IC due to ATM straddle)",
            max_loss_desc=f"Wing width (${wing_width:.0f}) minus credit",
            exit_notes=["Close at 25% of credit received",
                        "Close if underlying moves beyond ATM strike significantly",
                        "Close at 14 DTE to avoid pin risk"],
        )

    if structure_type == "ratio_spread":
        dir_ = direction or "bullish"
        legs = build_ratio_spread_legs(
            price, atr, dir_, exp_pt.expiration, exp_pt.days_to_expiry, exp_pt.atm_iv,
        )
        rationale = f"Target {target_dte_min}-{target_dte_max} DTE, matched {exp_pt.expiration} ({exp_pt.days_to_expiry}d). " \
                     f"Buy 1 ATM, sell 2 OTM at 1.0 ATR. {dir_.title()} direction."
        return TradeSpec(
            ticker=ticker, legs=legs, underlying_price=price,
            target_dte=exp_pt.days_to_expiry, target_expiration=exp_pt.expiration,
            spec_rationale=rationale,
            structure_type=StructureType.RATIO_SPREAD,
            order_side=OrderSide.CREDIT,
            profit_target_pct=0.50,
            stop_loss_pct=2.0,
            exit_dte=21,
            max_profit_desc="Net credit + OTM decay (max profit at short strike at expiry)",
            max_loss_desc="UNLIMITED beyond naked short strike",
            exit_notes=["NAKED LEG RISK: unlimited loss beyond short strikes",
                        "Close at 50% of credit or if short strike tested",
                        "Close at 21 DTE — gamma risk on naked leg"],
        )

    return None


def build_dual_expiry_trade_spec(
    ticker: str,
    price: float,
    atr: float,
    vol_surface: VolatilitySurface,
    structure_type: str,
    strategy_type: str,
    front_dte_min: int = 20,
    front_dte_max: int = 30,
    back_dte_min: int = 50,
    back_dte_max: int = 70,
    trend_direction: str = "neutral",
) -> TradeSpec | None:
    """Build a TradeSpec for dual-expiry structures (calendar, diagonal)."""
    # Use best_calendar_expiries from vol surface if available
    front_pt = find_best_expiration(vol_surface.term_structure, front_dte_min, front_dte_max)
    back_pt = find_best_expiration(vol_surface.term_structure, back_dte_min, back_dte_max)

    if front_pt is None or back_pt is None:
        # Try with broader ranges
        if len(vol_surface.term_structure) >= 2:
            sorted_ts = sorted(vol_surface.term_structure, key=lambda p: p.days_to_expiry)
            front_pt = sorted_ts[0]
            back_pt = sorted_ts[-1]
        else:
            return None

    # Ensure front < back
    if front_pt.days_to_expiry >= back_pt.days_to_expiry:
        return None

    iv_diff = (front_pt.atm_iv - back_pt.atm_iv) / back_pt.atm_iv * 100 if back_pt.atm_iv > 0 else 0.0

    if structure_type == "calendar" and strategy_type == "double_calendar":
        legs = build_double_calendar_legs(price, front_pt, back_pt, atr)
        rationale = (
            f"Double calendar: put cal + call cal bracketing price. "
            f"Front {front_pt.expiration} ({front_pt.days_to_expiry}d, IV {front_pt.atm_iv:.1%}) / "
            f"Back {back_pt.expiration} ({back_pt.days_to_expiry}d, IV {back_pt.atm_iv:.1%}). "
            f"IV diff: {iv_diff:+.1f}%."
        )
        st = StructureType.DOUBLE_CALENDAR
        exit_dte = max(front_pt.days_to_expiry - 7, 0)
        exit_notes = ["Close before front leg expiry to avoid assignment risk",
                      "Roll front legs on 25% profit",
                      "Close if underlying moves beyond either strike"]
    elif structure_type == "calendar":
        legs = build_calendar_legs(price, front_pt, back_pt, strategy_type, atr)
        rationale = (
            f"Front {front_pt.expiration} ({front_pt.days_to_expiry}d, IV {front_pt.atm_iv:.1%}) / "
            f"Back {back_pt.expiration} ({back_pt.days_to_expiry}d, IV {back_pt.atm_iv:.1%}). "
            f"IV diff: {iv_diff:+.1f}%."
        )
        st = StructureType.CALENDAR
        exit_dte = max(front_pt.days_to_expiry - 7, 0)
        exit_notes = ["Close before front leg expiry to avoid assignment risk",
                      "Roll front leg on 25% profit",
                      "Close if underlying moves >1 ATR from strike"]
    elif structure_type == "diagonal":
        legs = build_diagonal_legs(
            price, front_pt, back_pt, trend_direction, strategy_type, atr,
        )
        rationale = (
            f"Front {front_pt.expiration} ({front_pt.days_to_expiry}d) sell OTM / "
            f"Back {back_pt.expiration} ({back_pt.days_to_expiry}d) buy ATM. "
            f"{trend_direction.title()} diagonal. IV diff: {iv_diff:+.1f}%."
        )
        st = StructureType.DIAGONAL
        exit_dte = max(front_pt.days_to_expiry - 7, 0)
        exit_notes = ["Roll front leg on profit for recurring income",
                      "Close if underlying moves against back leg significantly",
                      "Monitor back leg delta — adjust if trend reverses"]
    else:
        return None

    return TradeSpec(
        ticker=ticker,
        legs=legs,
        underlying_price=price,
        target_dte=front_pt.days_to_expiry,
        target_expiration=front_pt.expiration,
        front_expiration=front_pt.expiration,
        front_dte=front_pt.days_to_expiry,
        back_expiration=back_pt.expiration,
        back_dte=back_pt.days_to_expiry,
        iv_at_front=front_pt.atm_iv,
        iv_at_back=back_pt.atm_iv,
        iv_differential_pct=iv_diff,
        spec_rationale=rationale,
        structure_type=st,
        order_side=OrderSide.DEBIT,
        profit_target_pct=0.25,
        stop_loss_pct=0.50,
        exit_dte=exit_dte,
        max_profit_desc="Front leg decay minus back leg decay",
        max_loss_desc="Net debit paid",
        exit_notes=exit_notes,
    )


# --- Simple structure builders (for 0DTE, LEAP, earnings, setups) ---


def build_long_option_legs(
    price: float,
    option_type: str,
    expiration: date,
    dte: int,
    atm_iv: float,
    otm_multiplier: float | None = None,
    atr: float | None = None,
) -> list[LegSpec]:
    """Build a single long option leg.

    If otm_multiplier and atr are provided, strike is OTM. Otherwise ATM.
    """
    if otm_multiplier is not None and atr is not None:
        strike = compute_otm_strike(price, atr, otm_multiplier, option_type, price)
        label = f"{otm_multiplier:.1f} ATR OTM {option_type}"
    else:
        strike = compute_atm_strike(price)
        label = f"ATM {option_type}"

    return [
        LegSpec(
            role=f"long_{option_type}", action=LegAction.BUY_TO_OPEN,
            option_type=option_type, strike=strike, strike_label=label,
            expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
        ),
    ]


def build_debit_spread_legs(
    price: float,
    atr: float,
    direction: str,
    expiration: date,
    dte: int,
    atm_iv: float,
    width_multiplier: float = 0.5,
) -> list[LegSpec]:
    """Build a debit spread (bullish call spread or bearish put spread).

    Long leg near ATM, short leg OTM by width_multiplier * ATR.
    """
    if direction == "bullish":
        long_strike = compute_atm_strike(price)
        short_strike = compute_otm_strike(price, atr, width_multiplier, "call", price)
        return [
            LegSpec(
                role="long_call", action=LegAction.BUY_TO_OPEN,
                option_type="call", strike=long_strike, strike_label="ATM call",
                expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
            ),
            LegSpec(
                role="short_call", action=LegAction.SELL_TO_OPEN,
                option_type="call", strike=short_strike,
                strike_label=f"{width_multiplier:.1f} ATR OTM call",
                expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
            ),
        ]
    else:
        long_strike = compute_atm_strike(price)
        short_strike = compute_otm_strike(price, atr, width_multiplier, "put", price)
        return [
            LegSpec(
                role="long_put", action=LegAction.BUY_TO_OPEN,
                option_type="put", strike=long_strike, strike_label="ATM put",
                expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
            ),
            LegSpec(
                role="short_put", action=LegAction.SELL_TO_OPEN,
                option_type="put", strike=short_strike,
                strike_label=f"{width_multiplier:.1f} ATR OTM put",
                expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
            ),
        ]


def build_credit_spread_legs(
    price: float,
    atr: float,
    direction: str,
    expiration: date,
    dte: int,
    atm_iv: float,
    short_multiplier: float = 1.0,
    wing_multiplier: float = 0.5,
) -> tuple[list[LegSpec], float]:
    """Build a credit spread. Returns (legs, wing_width_points).

    direction='bullish' -> bull put credit spread (sell put, buy lower put).
    direction='bearish' -> bear call credit spread (sell call, buy higher call).
    """
    if direction == "bullish":
        short_strike = compute_otm_strike(price, atr, short_multiplier, "put", price)
        wing_width = atr * wing_multiplier
        long_strike = snap_strike(short_strike - wing_width, price)
        wing_pts = short_strike - long_strike
        return [
            LegSpec(
                role="short_put", action=LegAction.SELL_TO_OPEN,
                option_type="put", strike=short_strike,
                strike_label=f"{short_multiplier:.1f} ATR OTM put",
                expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
            ),
            LegSpec(
                role="long_put", action=LegAction.BUY_TO_OPEN,
                option_type="put", strike=long_strike,
                strike_label=f"wing {wing_multiplier:.1f} ATR below short",
                expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
            ),
        ], wing_pts
    else:
        short_strike = compute_otm_strike(price, atr, short_multiplier, "call", price)
        wing_width = atr * wing_multiplier
        long_strike = snap_strike(short_strike + wing_width, price)
        wing_pts = long_strike - short_strike
        return [
            LegSpec(
                role="short_call", action=LegAction.SELL_TO_OPEN,
                option_type="call", strike=short_strike,
                strike_label=f"{short_multiplier:.1f} ATR OTM call",
                expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
            ),
            LegSpec(
                role="long_call", action=LegAction.BUY_TO_OPEN,
                option_type="call", strike=long_strike,
                strike_label=f"wing {wing_multiplier:.1f} ATR above short",
                expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
            ),
        ], wing_pts


def build_straddle_legs(
    price: float,
    action: str,
    expiration: date,
    dte: int,
    atm_iv: float,
    otm_offset_multiplier: float | None = None,
    atr: float | None = None,
) -> list[LegSpec]:
    """Build straddle/strangle legs.

    action: 'buy' or 'sell'.
    If otm_offset_multiplier + atr provided, builds OTM strangle instead of ATM straddle.
    """
    leg_action = LegAction.BUY_TO_OPEN if action == "buy" else LegAction.SELL_TO_OPEN
    role_prefix = "long" if action == "buy" else "short"

    if otm_offset_multiplier is not None and atr is not None:
        put_strike = compute_otm_strike(price, atr, otm_offset_multiplier, "put", price)
        call_strike = compute_otm_strike(price, atr, otm_offset_multiplier, "call", price)
        label = f"{otm_offset_multiplier:.1f} ATR OTM"
    else:
        atm = compute_atm_strike(price)
        put_strike = atm
        call_strike = atm
        label = "ATM"

    return [
        LegSpec(
            role=f"{role_prefix}_put", action=leg_action,
            option_type="put", strike=put_strike,
            strike_label=f"{label} put",
            expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
        ),
        LegSpec(
            role=f"{role_prefix}_call", action=leg_action,
            option_type="call", strike=call_strike,
            strike_label=f"{label} call",
            expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
        ),
    ]


def build_pmcc_legs(
    price: float,
    atr: float,
    front_exp: TermStructurePoint,
    back_exp: TermStructurePoint,
) -> list[LegSpec]:
    """Build PMCC legs: deep ITM back LEAP call + OTM front short call."""
    # Back: deep ITM call (~1.0 ATR ITM)
    back_strike = snap_strike(price - 1.0 * atr, price)
    # Front: OTM call (~0.5 ATR OTM)
    front_strike = snap_strike(price + 0.5 * atr, price)

    return [
        LegSpec(
            role="long_back", action=LegAction.BUY_TO_OPEN,
            option_type="call", strike=back_strike,
            strike_label="deep ITM LEAP call",
            expiration=back_exp.expiration, days_to_expiry=back_exp.days_to_expiry,
            atm_iv_at_expiry=back_exp.atm_iv,
        ),
        LegSpec(
            role="short_front", action=LegAction.SELL_TO_OPEN,
            option_type="call", strike=front_strike,
            strike_label="OTM front call",
            expiration=front_exp.expiration, days_to_expiry=front_exp.days_to_expiry,
            atm_iv_at_expiry=front_exp.atm_iv,
        ),
    ]


def build_setup_trade_spec(
    ticker: str,
    price: float,
    atr: float,
    direction: str,
    regime_id: int,
    vol_surface: VolatilitySurface | None,
    target_dte_min: int = 30,
    target_dte_max: int = 45,
) -> TradeSpec | None:
    """Build a suggested default TradeSpec for a setup (breakout, momentum, MR, ORB).

    Income-first bias: credit spreads in R1/R2, debit spreads in R3, None in R4.
    """
    if vol_surface is None or not vol_surface.term_structure:
        return None
    if regime_id == 4:
        return None

    exp_pt = find_best_expiration(vol_surface.term_structure, target_dte_min, target_dte_max)
    if exp_pt is None:
        return None

    if direction == "neutral" and regime_id in (1, 2):
        # Iron condor for neutral setups
        legs, wing_width = build_iron_condor_legs(
            price, atr, regime_id, exp_pt.expiration, exp_pt.days_to_expiry, exp_pt.atm_iv,
        )
        rationale = f"Suggested default: iron condor (neutral + R{regime_id}). {exp_pt.days_to_expiry} DTE."
        return TradeSpec(
            ticker=ticker, legs=legs, underlying_price=price,
            target_dte=exp_pt.days_to_expiry, target_expiration=exp_pt.expiration,
            wing_width_points=wing_width,
            max_risk_per_spread=f"${wing_width * 100:.0f} - credit received",
            spec_rationale=rationale,
            structure_type=StructureType.IRON_CONDOR,
            order_side=OrderSide.CREDIT,
            profit_target_pct=0.50,
            stop_loss_pct=2.0,
            exit_dte=21,
            max_profit_desc="Credit received",
            max_loss_desc=f"Wing width (${wing_width:.0f}) minus credit",
            exit_notes=["Suggested default structure for neutral setup",
                        "Close at 50% of credit received",
                        "Close if short strike tested on either side"],
        )

    if regime_id in (1, 2):
        # Credit spread (income-first)
        cr_dir = direction if direction in ("bullish", "bearish") else "bullish"
        legs, wing_pts = build_credit_spread_legs(
            price, atr, cr_dir, exp_pt.expiration, exp_pt.days_to_expiry, exp_pt.atm_iv,
        )
        rationale = (
            f"Suggested default: {cr_dir} credit spread (income-first, R{regime_id}). "
            f"{exp_pt.days_to_expiry} DTE."
        )
        return TradeSpec(
            ticker=ticker, legs=legs, underlying_price=price,
            target_dte=exp_pt.days_to_expiry, target_expiration=exp_pt.expiration,
            wing_width_points=wing_pts,
            max_risk_per_spread=f"${wing_pts * 100:.0f} - credit received",
            spec_rationale=rationale,
            structure_type=StructureType.CREDIT_SPREAD,
            order_side=OrderSide.CREDIT,
            profit_target_pct=0.50,
            stop_loss_pct=2.0,
            exit_dte=21,
            max_profit_desc="Credit received",
            max_loss_desc=f"Wing width (${wing_pts:.0f}) minus credit",
            exit_notes=["Suggested default structure for setup",
                        "Close at 50% of credit received",
                        "Close if short strike tested"],
        )

    # R3: debit spread (directional)
    db_dir = direction if direction in ("bullish", "bearish") else "bullish"
    legs = build_debit_spread_legs(
        price, atr, db_dir, exp_pt.expiration, exp_pt.days_to_expiry, exp_pt.atm_iv,
    )
    rationale = (
        f"Suggested default: {db_dir} debit spread (directional, R{regime_id}). "
        f"{exp_pt.days_to_expiry} DTE."
    )
    return TradeSpec(
        ticker=ticker, legs=legs, underlying_price=price,
        target_dte=exp_pt.days_to_expiry, target_expiration=exp_pt.expiration,
        spec_rationale=rationale,
        structure_type=StructureType.DEBIT_SPREAD,
        order_side=OrderSide.DEBIT,
        profit_target_pct=0.50,
        stop_loss_pct=0.50,
        exit_dte=14,
        max_profit_desc="Spread width minus debit paid",
        max_loss_desc="Net debit paid",
        exit_notes=["Suggested default structure for setup",
                    "Target 50% of max profit",
                    "Close at 50% loss of debit paid"],
    )


# --- Fill price estimation (rough Black-Scholes approximation) ---


def _bs_price(
    underlying: float,
    strike: float,
    dte_years: float,
    iv: float,
    option_type: str,
) -> float:
    """Rough Black-Scholes option price (no dividends, risk-free ~0).

    Good enough for fill-price cutoff estimation — cotrader has real bid/ask.
    """
    if dte_years <= 0 or iv <= 0:
        # Intrinsic only
        if option_type == "call":
            return max(underlying - strike, 0.0)
        return max(strike - underlying, 0.0)

    sqrt_t = math.sqrt(dte_years)
    d1 = (math.log(underlying / strike) + 0.5 * iv * iv * dte_years) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t

    nd1 = _norm_cdf(d1)
    nd2 = _norm_cdf(d2)

    if option_type == "call":
        return underlying * nd1 - strike * nd2
    else:
        return strike * _norm_cdf(-d2) - underlying * _norm_cdf(-d1)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def estimate_trade_price(trade_spec: TradeSpec) -> float | None:
    """Rough BS-estimated net price for the trade.

    Returns positive for net credit, negative for net debit.
    Returns None if estimation is not possible.
    """
    if not trade_spec.legs:
        return None

    total = 0.0
    for leg in trade_spec.legs:
        dte_years = max(leg.days_to_expiry, 1) / 365.0
        price = _bs_price(
            trade_spec.underlying_price,
            leg.strike,
            dte_years,
            leg.atm_iv_at_expiry,
            leg.option_type,
        )
        # STO adds credit (positive), BTO costs money (negative)
        if leg.action == LegAction.SELL_TO_OPEN:
            total += price * leg.quantity
        else:
            total -= price * leg.quantity

    return round(total, 2)


def compute_max_entry_price(trade_spec: TradeSpec, slippage_pct: float = 0.20) -> float | None:
    """Max price cotrader should pay/accept. Returns absolute value.

    For credits: max_entry = estimated_credit * (1 - slippage_pct)
        → don't accept less than (1 - slippage) of theoretical credit
    For debits: max_entry = |estimated_debit| * (1 + slippage_pct)
        → don't pay more than (1 + slippage) of theoretical debit
    For long options: tighter tolerance (slippage_pct * 0.75).
    """
    estimated = estimate_trade_price(trade_spec)
    if estimated is None:
        return None

    st = trade_spec.structure_type
    if estimated > 0:
        # Net credit structure — don't accept less than X% of theoretical
        return round(estimated * (1.0 - slippage_pct), 2)
    elif estimated < 0:
        debit = abs(estimated)
        if st and st in (StructureType.LONG_OPTION,):
            # Tighter tolerance for single options
            return round(debit * (1.0 + slippage_pct * 0.75), 2)
        return round(debit * (1.0 + slippage_pct), 2)
    return None
