"""Tests for vol surface computation."""

from datetime import date, timedelta

import pandas as pd
import pytest

from market_analyzer.features.vol_surface import (
    _assess_data_quality,
    _find_atm_strike,
    _score_calendar_edge,
    compute_vol_surface,
)
from market_analyzer.models.vol_surface import TermStructurePoint, VolatilitySurface


def _make_chain(
    expirations: list[date],
    underlying_price: float = 580.0,
    front_iv: float = 0.20,
    back_iv: float = 0.18,
    strikes_around: int = 5,
    oi: int = 1000,
    volume: int = 500,
) -> pd.DataFrame:
    """Build a synthetic options chain DataFrame for testing."""
    rows = []
    step = underlying_price * 0.01  # 1% strike spacing

    for i, exp in enumerate(expirations):
        # Linearly interpolate IV between front and back
        frac = i / max(len(expirations) - 1, 1)
        base_iv = front_iv + (back_iv - front_iv) * frac

        for j in range(-strikes_around, strikes_around + 1):
            strike = round(underlying_price + j * step, 2)
            moneyness = abs(j) / strikes_around
            # IV smile: higher for OTM puts, slightly higher for OTM calls
            put_iv = base_iv + moneyness * 0.03  # Put skew
            call_iv = base_iv + moneyness * 0.01  # Mild call skew

            for opt_type, iv in [("call", call_iv), ("put", put_iv)]:
                mid_price = max(0.5, 10 - abs(j) * 1.5)
                rows.append({
                    "expiration": exp,
                    "strike": strike,
                    "option_type": opt_type,
                    "bid": max(0.0, mid_price - 0.25),
                    "ask": mid_price + 0.25,
                    "last_price": mid_price,
                    "volume": volume,
                    "open_interest": oi,
                    "implied_volatility": iv,
                    "in_the_money": (opt_type == "call" and strike < underlying_price)
                    or (opt_type == "put" and strike > underlying_price),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def today() -> date:
    return date(2026, 3, 1)


@pytest.fixture
def expirations(today: date) -> list[date]:
    return [
        today + timedelta(days=7),   # ~1 week
        today + timedelta(days=30),  # ~1 month
        today + timedelta(days=60),  # ~2 months
    ]


@pytest.fixture
def chain_contango(expirations: list[date]) -> pd.DataFrame:
    """Chain with normal term structure (contango): back > front."""
    return _make_chain(expirations, front_iv=0.18, back_iv=0.22)


@pytest.fixture
def chain_backwardation(expirations: list[date]) -> pd.DataFrame:
    """Chain with inverted term structure: front > back."""
    return _make_chain(expirations, front_iv=0.28, back_iv=0.18)


class TestFindATMStrike:
    def test_exact_match(self) -> None:
        strikes = [575.0, 580.0, 585.0]
        assert _find_atm_strike(strikes, 580.0) == 580.0

    def test_nearest_match(self) -> None:
        strikes = [575.0, 580.0, 585.0]
        assert _find_atm_strike(strikes, 582.0) == 580.0

    def test_single_strike(self) -> None:
        assert _find_atm_strike([500.0], 580.0) == 500.0


class TestComputeVolSurface:
    def test_returns_volatility_surface(
        self, chain_contango: pd.DataFrame, today: date
    ) -> None:
        surf = compute_vol_surface(chain_contango, 580.0, "SPY", as_of=today)
        assert isinstance(surf, VolatilitySurface)
        assert surf.ticker == "SPY"
        assert surf.underlying_price == 580.0

    def test_term_structure_populated(
        self, chain_contango: pd.DataFrame, today: date, expirations: list[date]
    ) -> None:
        surf = compute_vol_surface(chain_contango, 580.0, "SPY", as_of=today)
        assert len(surf.term_structure) == len(expirations)
        # Sorted by DTE ascending
        dtes = [t.days_to_expiry for t in surf.term_structure]
        assert dtes == sorted(dtes)

    def test_contango_detected(
        self, chain_contango: pd.DataFrame, today: date
    ) -> None:
        surf = compute_vol_surface(chain_contango, 580.0, "SPY", as_of=today)
        assert surf.is_contango is True
        assert surf.is_backwardation is False
        assert surf.term_slope > 0

    def test_backwardation_detected(
        self, chain_backwardation: pd.DataFrame, today: date
    ) -> None:
        surf = compute_vol_surface(chain_backwardation, 580.0, "SPY", as_of=today)
        assert surf.is_backwardation is True
        assert surf.is_contango is False
        assert surf.term_slope < 0

    def test_front_and_back_iv(
        self, chain_contango: pd.DataFrame, today: date
    ) -> None:
        surf = compute_vol_surface(chain_contango, 580.0, "SPY", as_of=today)
        assert surf.front_iv > 0
        assert surf.back_iv > 0

    def test_skew_populated(
        self, chain_contango: pd.DataFrame, today: date, expirations: list[date]
    ) -> None:
        surf = compute_vol_surface(chain_contango, 580.0, "SPY", as_of=today)
        assert len(surf.skew_by_expiry) == len(expirations)
        # With our synthetic data, put skew should be positive (OTM puts have higher IV)
        for skew in surf.skew_by_expiry:
            assert skew.put_skew >= 0  # OTM put IV >= ATM IV

    def test_calendar_edge_score(
        self, chain_contango: pd.DataFrame, today: date
    ) -> None:
        surf = compute_vol_surface(chain_contango, 580.0, "SPY", as_of=today)
        assert 0.0 <= surf.calendar_edge_score <= 1.0

    def test_backwardation_higher_calendar_edge(
        self,
        chain_contango: pd.DataFrame,
        chain_backwardation: pd.DataFrame,
        today: date,
    ) -> None:
        """Backwardation (front IV elevated) should score higher for calendars."""
        surf_contango = compute_vol_surface(chain_contango, 580.0, "SPY", as_of=today)
        surf_backw = compute_vol_surface(chain_backwardation, 580.0, "SPY", as_of=today)
        assert surf_backw.calendar_edge_score > surf_contango.calendar_edge_score

    def test_best_calendar_expiries(
        self, chain_backwardation: pd.DataFrame, today: date
    ) -> None:
        surf = compute_vol_surface(chain_backwardation, 580.0, "SPY", as_of=today)
        assert surf.best_calendar_expiries is not None
        front_exp, back_exp = surf.best_calendar_expiries
        assert front_exp < back_exp

    def test_iv_differential_pct(
        self, chain_backwardation: pd.DataFrame, today: date
    ) -> None:
        surf = compute_vol_surface(chain_backwardation, 580.0, "SPY", as_of=today)
        # Backwardation: front > back, so iv_differential_pct > 0
        assert surf.iv_differential_pct > 0

    def test_data_quality(
        self, chain_contango: pd.DataFrame, today: date
    ) -> None:
        surf = compute_vol_surface(chain_contango, 580.0, "SPY", as_of=today)
        assert surf.data_quality in ("good", "fair", "poor")
        # With our synthetic data (good OI, tight spreads), should be good
        assert surf.data_quality == "good"

    def test_summary_string(
        self, chain_contango: pd.DataFrame, today: date
    ) -> None:
        surf = compute_vol_surface(chain_contango, 580.0, "SPY", as_of=today)
        assert "SPY" in surf.summary
        assert "Front IV" in surf.summary

    def test_single_expiration(self, today: date) -> None:
        """Surface with one expiration still works (limited metrics)."""
        exp = [today + timedelta(days=30)]
        chain = _make_chain(exp, front_iv=0.20, back_iv=0.20)
        surf = compute_vol_surface(chain, 580.0, "SPY", as_of=today)
        assert len(surf.term_structure) == 1
        assert surf.calendar_edge_score == 0.0  # Can't do calendars with 1 expiry
        assert surf.best_calendar_expiries is None


class TestScoreCalendarEdge:
    def test_empty_term_structure(self) -> None:
        assert _score_calendar_edge([]) == 0.0

    def test_single_point(self) -> None:
        ts = [TermStructurePoint(
            expiration=date(2026, 4, 1), days_to_expiry=30, atm_iv=0.20, atm_strike=580.0
        )]
        assert _score_calendar_edge(ts) == 0.0

    def test_high_iv_boosts_score(self) -> None:
        ts_low = [
            TermStructurePoint(expiration=date(2026, 3, 8), days_to_expiry=7, atm_iv=0.10, atm_strike=580.0),
            TermStructurePoint(expiration=date(2026, 4, 1), days_to_expiry=30, atm_iv=0.12, atm_strike=580.0),
        ]
        ts_high = [
            TermStructurePoint(expiration=date(2026, 3, 8), days_to_expiry=7, atm_iv=0.30, atm_strike=580.0),
            TermStructurePoint(expiration=date(2026, 4, 1), days_to_expiry=30, atm_iv=0.32, atm_strike=580.0),
        ]
        assert _score_calendar_edge(ts_high) > _score_calendar_edge(ts_low)


class TestDataQuality:
    def test_good_quality(self) -> None:
        df = pd.DataFrame({
            "open_interest": [500, 1000, 800],
            "bid": [5.0, 8.0, 3.0],
        })
        assert _assess_data_quality(df, avg_spread_pct=0.5) == "good"

    def test_poor_quality_low_oi(self) -> None:
        df = pd.DataFrame({
            "open_interest": [0, 0, 5],
            "bid": [0.0, 0.0, 0.5],
        })
        assert _assess_data_quality(df, avg_spread_pct=3.0) == "poor"

    def test_empty_is_poor(self) -> None:
        assert _assess_data_quality(pd.DataFrame(), avg_spread_pct=0.0) == "poor"
