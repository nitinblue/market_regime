"""Shared helper functions for computing TradeSpec across all option play assessors.

Pure functions — no data fetching, no side effects.
"""

from __future__ import annotations

from datetime import date

from market_analyzer.models.opportunity import LegSpec, TradeSpec
from market_analyzer.models.vol_surface import TermStructurePoint, VolatilitySurface


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
            role="short_put", option_type="put", strike=short_put,
            strike_label=f"{short_mult:.1f} ATR OTM put",
            expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
        ),
        LegSpec(
            role="long_put", option_type="put", strike=long_put,
            strike_label=f"wing {wing_mult:.1f} ATR below short put",
            expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
        ),
        LegSpec(
            role="short_call", option_type="call", strike=short_call,
            strike_label=f"{short_mult:.1f} ATR OTM call",
            expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
        ),
        LegSpec(
            role="long_call", option_type="call", strike=long_call,
            strike_label=f"wing {wing_mult:.1f} ATR above short call",
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
            role="short_put", option_type="put", strike=atm,
            strike_label="ATM put (short straddle)",
            expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
        ),
        LegSpec(
            role="short_call", option_type="call", strike=atm,
            strike_label="ATM call (short straddle)",
            expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
        ),
        LegSpec(
            role="long_put", option_type="put", strike=long_put,
            strike_label=f"wing {wing_mult:.1f} ATR below ATM",
            expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
        ),
        LegSpec(
            role="long_call", option_type="call", strike=long_call,
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
            role="short_front", option_type=opt_type, strike=strike,
            strike_label=f"sell front {label}",
            expiration=front_exp.expiration, days_to_expiry=front_exp.days_to_expiry,
            atm_iv_at_expiry=front_exp.atm_iv,
        ),
        LegSpec(
            role="long_back", option_type=opt_type, strike=strike,
            strike_label=f"buy back {label}",
            expiration=back_exp.expiration, days_to_expiry=back_exp.days_to_expiry,
            atm_iv_at_expiry=back_exp.atm_iv,
        ),
    ]
    return legs


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
                role="short_front", option_type="call", strike=front_strike,
                strike_label="OTM front call",
                expiration=front_exp.expiration, days_to_expiry=front_exp.days_to_expiry,
                atm_iv_at_expiry=front_exp.atm_iv,
            ),
            LegSpec(
                role="long_back", option_type="call", strike=back_strike,
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
                role="short_front", option_type="put", strike=front_strike,
                strike_label="OTM front put",
                expiration=front_exp.expiration, days_to_expiry=front_exp.days_to_expiry,
                atm_iv_at_expiry=front_exp.atm_iv,
            ),
            LegSpec(
                role="long_back", option_type="put", strike=back_strike,
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
                role="long_call", option_type="call", strike=atm,
                strike_label="buy 1 ATM call",
                expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
            ),
            LegSpec(
                role="short_call_1", option_type="call", strike=otm_strike,
                strike_label="sell 1.0 ATR OTM call (leg 1 of 2)",
                expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
            ),
            LegSpec(
                role="short_call_2", option_type="call", strike=otm_strike,
                strike_label="sell 1.0 ATR OTM call (leg 2 of 2)",
                expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
            ),
        ]
    else:
        otm_strike = compute_otm_strike(price, atr, 1.0, "put", price)
        legs = [
            LegSpec(
                role="long_put", option_type="put", strike=atm,
                strike_label="buy 1 ATM put",
                expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
            ),
            LegSpec(
                role="short_put_1", option_type="put", strike=otm_strike,
                strike_label="sell 1.0 ATR OTM put (leg 1 of 2)",
                expiration=expiration, days_to_expiry=dte, atm_iv_at_expiry=atm_iv,
            ),
            LegSpec(
                role="short_put_2", option_type="put", strike=otm_strike,
                strike_label="sell 1.0 ATR OTM put (leg 2 of 2)",
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

    if structure_type == "calendar":
        legs = build_calendar_legs(price, front_pt, back_pt, strategy_type, atr)
        rationale = (
            f"Front {front_pt.expiration} ({front_pt.days_to_expiry}d, IV {front_pt.atm_iv:.1%}) / "
            f"Back {back_pt.expiration} ({back_pt.days_to_expiry}d, IV {back_pt.atm_iv:.1%}). "
            f"IV diff: {iv_diff:+.1f}%."
        )
    elif structure_type == "diagonal":
        legs = build_diagonal_legs(
            price, front_pt, back_pt, trend_direction, strategy_type, atr,
        )
        rationale = (
            f"Front {front_pt.expiration} ({front_pt.days_to_expiry}d) sell OTM / "
            f"Back {back_pt.expiration} ({back_pt.days_to_expiry}d) buy ATM. "
            f"{trend_direction.title()} diagonal. IV diff: {iv_diff:+.1f}%."
        )
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
    )
