"""Volatility surface computation from options chain data.

Pure functions — no data fetching. Accepts a DataFrame (options chain) and
returns a VolatilitySurface model.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from market_analyzer.models.vol_surface import (
    SkewSlice,
    TermStructurePoint,
    VolatilitySurface,
)


def compute_vol_surface(
    chain_df: pd.DataFrame,
    underlying_price: float,
    ticker: str,
    as_of: date | None = None,
) -> VolatilitySurface:
    """Build a VolatilitySurface from an options chain DataFrame.

    Args:
        chain_df: DataFrame with columns expiration, strike, option_type,
                  bid, ask, implied_volatility, open_interest, volume.
        underlying_price: Current price of the underlying.
        ticker: Ticker symbol.
        as_of: Reference date (defaults to today).

    Returns:
        VolatilitySurface with term structure, skew, and calendar metrics.
    """
    today = as_of or date.today()
    df = chain_df.copy()

    # Ensure expiration is date type
    if not pd.api.types.is_object_dtype(df["expiration"]):
        df["expiration"] = pd.to_datetime(df["expiration"]).dt.date

    expirations = sorted(df["expiration"].unique())

    # Compute DTE for each expiration
    dte_map = {exp: (exp - today).days for exp in expirations}

    # Filter to expirations with DTE > 0
    valid_exps = [exp for exp in expirations if dte_map[exp] > 0]
    if not valid_exps:
        valid_exps = expirations  # fallback: use all

    # --- Term structure ---
    term_structure: list[TermStructurePoint] = []
    for exp in valid_exps:
        exp_df = df[df["expiration"] == exp]
        atm_strike = _find_atm_strike(exp_df["strike"].unique(), underlying_price)
        atm_iv = _get_atm_iv(exp_df, atm_strike)
        if atm_iv is not None and atm_iv > 0:
            term_structure.append(TermStructurePoint(
                expiration=exp,
                days_to_expiry=max(dte_map[exp], 1),
                atm_iv=atm_iv,
                atm_strike=atm_strike,
            ))

    # Sort by DTE
    term_structure.sort(key=lambda t: t.days_to_expiry)

    # Front / back IV
    front_iv = term_structure[0].atm_iv if term_structure else 0.0
    back_iv = _find_back_iv(term_structure)
    term_slope = (back_iv - front_iv) / front_iv if front_iv > 0 else 0.0

    # --- Skew per expiration ---
    skew_by_expiry: list[SkewSlice] = []
    for exp in valid_exps:
        exp_df = df[df["expiration"] == exp]
        atm_strike = _find_atm_strike(exp_df["strike"].unique(), underlying_price)
        skew = _compute_skew(exp_df, atm_strike, underlying_price, dte_map[exp])
        if skew is not None:
            skew_by_expiry.append(skew)

    # --- Calendar edge ---
    calendar_edge_score = _score_calendar_edge(term_structure)
    best_calendar = _find_best_calendar_expiries(term_structure)
    iv_diff_pct = ((front_iv - back_iv) / back_iv * 100) if back_iv > 0 else 0.0

    # --- Data quality ---
    total_contracts = len(df)
    avg_spread_pct = _avg_bid_ask_spread_pct(df, underlying_price)
    data_quality = _assess_data_quality(df, avg_spread_pct)

    # --- Summary ---
    parts = [f"{ticker} Vol Surface"]
    parts.append(f"Front IV: {front_iv:.1%}")
    parts.append(f"Back IV: {back_iv:.1%}")
    if term_slope > 0:
        parts.append(f"Contango ({term_slope:+.1%})")
    else:
        parts.append(f"Backwardation ({term_slope:+.1%})")
    parts.append(f"Calendar edge: {calendar_edge_score:.2f}")
    parts.append(f"Quality: {data_quality}")
    summary = " | ".join(parts)

    return VolatilitySurface(
        ticker=ticker,
        as_of_date=today,
        underlying_price=underlying_price,
        expirations=valid_exps,
        term_structure=term_structure,
        front_iv=front_iv,
        back_iv=back_iv,
        term_slope=term_slope,
        is_contango=back_iv > front_iv,
        is_backwardation=front_iv > back_iv,
        skew_by_expiry=skew_by_expiry,
        calendar_edge_score=calendar_edge_score,
        best_calendar_expiries=best_calendar,
        iv_differential_pct=iv_diff_pct,
        total_contracts=total_contracts,
        avg_bid_ask_spread_pct=avg_spread_pct,
        data_quality=data_quality,
        summary=summary,
    )


def _find_atm_strike(strikes: np.ndarray, underlying_price: float) -> float:
    """Find the strike closest to the underlying price."""
    strikes_arr = np.asarray(strikes, dtype=float)
    idx = np.argmin(np.abs(strikes_arr - underlying_price))
    return float(strikes_arr[idx])


def _get_atm_iv(exp_df: pd.DataFrame, atm_strike: float) -> float | None:
    """Get ATM implied volatility, preferring calls then averaging."""
    atm_rows = exp_df[exp_df["strike"] == atm_strike]
    if atm_rows.empty:
        return None

    ivs = atm_rows["implied_volatility"].dropna()
    if ivs.empty:
        return None
    return float(ivs.mean())


def _find_back_iv(term_structure: list[TermStructurePoint]) -> float:
    """Find the back-month ATM IV (~30-60 DTE, or furthest available)."""
    if not term_structure:
        return 0.0
    if len(term_structure) == 1:
        return term_structure[0].atm_iv

    # Prefer 30-60 DTE
    for ts in term_structure:
        if 30 <= ts.days_to_expiry <= 60:
            return ts.atm_iv

    # Fallback: use the second expiration or furthest if only 2
    if len(term_structure) >= 2:
        return term_structure[1].atm_iv
    return term_structure[-1].atm_iv


def _compute_skew(
    exp_df: pd.DataFrame,
    atm_strike: float,
    underlying_price: float,
    dte: int,
) -> SkewSlice | None:
    """Compute skew for one expiration.

    OTM put = ~5% below current price, OTM call = ~5% above.
    """
    atm_iv = _get_atm_iv(exp_df, atm_strike)
    if atm_iv is None or atm_iv <= 0:
        return None

    # Target OTM strikes: 5% away from underlying
    otm_put_target = underlying_price * 0.95
    otm_call_target = underlying_price * 1.05

    # Find nearest OTM put strike
    puts = exp_df[exp_df["option_type"] == "put"]
    otm_put_iv = _nearest_iv(puts, otm_put_target)

    # Find nearest OTM call strike
    calls = exp_df[exp_df["option_type"] == "call"]
    otm_call_iv = _nearest_iv(calls, otm_call_target)

    if otm_put_iv is None:
        otm_put_iv = atm_iv
    if otm_call_iv is None:
        otm_call_iv = atm_iv

    put_skew = otm_put_iv - atm_iv
    call_skew = otm_call_iv - atm_iv
    skew_ratio = put_skew / call_skew if call_skew != 0 else 0.0

    return SkewSlice(
        expiration=exp_df["expiration"].iloc[0],
        days_to_expiry=max(dte, 1),
        atm_iv=atm_iv,
        otm_put_iv=otm_put_iv,
        otm_call_iv=otm_call_iv,
        put_skew=put_skew,
        call_skew=call_skew,
        skew_ratio=skew_ratio,
    )


def _nearest_iv(df: pd.DataFrame, target_strike: float) -> float | None:
    """Find IV at the strike nearest to the target."""
    if df.empty:
        return None
    idx = (df["strike"] - target_strike).abs().idxmin()
    iv = df.loc[idx, "implied_volatility"]
    if pd.isna(iv) or iv <= 0:
        return None
    return float(iv)


def _score_calendar_edge(term_structure: list[TermStructurePoint]) -> float:
    """Score how favorable conditions are for calendar spreads (0-1).

    Higher score when:
    - Term structure is in contango (back > front)
    - Spread between front and back IV is meaningful
    - Multiple expirations available
    """
    if len(term_structure) < 2:
        return 0.0

    front = term_structure[0]
    back = _find_back_ts(term_structure)
    if back is None:
        return 0.0

    score = 0.0

    # Contango bonus (back IV > front IV is normal and good for calendars)
    iv_diff = back.atm_iv - front.atm_iv
    if iv_diff > 0:
        # Contango — moderate edge
        score += min(0.3, iv_diff * 3)  # Cap at 0.3 for 10% contango
    else:
        # Backwardation — front IV elevated, great for selling front
        score += min(0.5, abs(iv_diff) * 5)  # Stronger signal

    # Absolute IV level — higher IV = more premium
    avg_iv = (front.atm_iv + back.atm_iv) / 2
    if avg_iv >= 0.30:
        score += 0.25
    elif avg_iv >= 0.20:
        score += 0.15
    elif avg_iv >= 0.15:
        score += 0.10

    # Multiple expirations available
    if len(term_structure) >= 4:
        score += 0.15
    elif len(term_structure) >= 2:
        score += 0.10

    return min(1.0, score)


def _find_back_ts(term_structure: list[TermStructurePoint]) -> TermStructurePoint | None:
    """Find the back-month term structure point (~30-60 DTE)."""
    if len(term_structure) < 2:
        return None
    for ts in term_structure:
        if 30 <= ts.days_to_expiry <= 60:
            return ts
    return term_structure[1]


def _find_best_calendar_expiries(
    term_structure: list[TermStructurePoint],
) -> tuple[date, date] | None:
    """Find the best (front, back) pair for a calendar spread.

    Looks for largest IV differential between adjacent expirations.
    """
    if len(term_structure) < 2:
        return None

    best_diff = -float("inf")
    best_pair: tuple[date, date] | None = None

    for i in range(len(term_structure) - 1):
        front = term_structure[i]
        back = term_structure[i + 1]
        # Calendar sells front, buys back — profit from front theta decay
        # Best when front IV is elevated relative to back
        diff = front.atm_iv - back.atm_iv
        if diff > best_diff:
            best_diff = diff
            best_pair = (front.expiration, back.expiration)

    # Also check front vs 30-60 DTE (not necessarily adjacent)
    back_ts = _find_back_ts(term_structure)
    if back_ts is not None:
        front = term_structure[0]
        diff = front.atm_iv - back_ts.atm_iv
        if diff > best_diff:
            best_pair = (front.expiration, back_ts.expiration)

    return best_pair


def _avg_bid_ask_spread_pct(df: pd.DataFrame, underlying_price: float) -> float:
    """Average bid-ask spread as percent of underlying price."""
    if df.empty or underlying_price <= 0:
        return 0.0
    spreads = (df["ask"] - df["bid"]).clip(lower=0)
    # Filter out zero-bid contracts (likely illiquid/stale)
    valid = spreads[df["bid"] > 0]
    if valid.empty:
        return 0.0
    return float(valid.mean() / underlying_price * 100)


def _assess_data_quality(df: pd.DataFrame, avg_spread_pct: float) -> str:
    """Assess overall options chain data quality."""
    if df.empty:
        return "poor"

    # Check OI coverage
    avg_oi = df["open_interest"].mean()
    contracts_with_oi = (df["open_interest"] > 0).mean()

    if contracts_with_oi >= 0.5 and avg_oi >= 100 and avg_spread_pct < 1.0:
        return "good"
    elif contracts_with_oi >= 0.3 and avg_oi >= 50:
        return "fair"
    return "poor"
