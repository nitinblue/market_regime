"""Tests for BlackSwanService: tail-risk circuit breaker."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from market_analyzer.config import BlackSwanSettings
from market_analyzer.features.black_swan import (
    check_circuit_breakers,
    compute_black_swan_alert,
    score_credit_stress,
    score_em_contagion,
    score_put_call_ratio,
    score_rv_iv_gap,
    score_spy_drawdown,
    score_treasury_stress,
    score_vix_level,
    score_vix_term_structure,
    score_yield_curve,
)
from market_analyzer.models.black_swan import (
    AlertLevel,
    IndicatorStatus,
)


@pytest.fixture
def cfg() -> BlackSwanSettings:
    return BlackSwanSettings()


# ===========================================================================
# TestIndicatorScoring — 9 tests
# ===========================================================================


class TestIndicatorScoring:
    """Each indicator scores correctly at normal/warning/danger/critical."""

    def test_vix_normal(self, cfg: BlackSwanSettings) -> None:
        ind = score_vix_level(15.0, cfg)
        assert ind.status == IndicatorStatus.NORMAL
        assert ind.score < 0.2

    def test_vix_warning(self, cfg: BlackSwanSettings) -> None:
        ind = score_vix_level(25.0, cfg)
        assert ind.status == IndicatorStatus.WARNING
        assert 0.2 <= ind.score < 0.5

    def test_vix_danger(self, cfg: BlackSwanSettings) -> None:
        ind = score_vix_level(35.0, cfg)
        assert ind.status == IndicatorStatus.DANGER
        assert 0.5 <= ind.score < 0.85

    def test_vix_critical(self, cfg: BlackSwanSettings) -> None:
        ind = score_vix_level(45.0, cfg)
        assert ind.status == IndicatorStatus.CRITICAL
        assert ind.score == 1.0

    def test_vix_unavailable(self, cfg: BlackSwanSettings) -> None:
        ind = score_vix_level(None, cfg)
        assert ind.status == IndicatorStatus.UNAVAILABLE
        assert ind.score == 0.0

    def test_term_structure_contango(self, cfg: BlackSwanSettings) -> None:
        ind = score_vix_term_structure(0.85, cfg)
        # 0.85 is below warning threshold (0.95) but interpolation gives small score
        assert ind.score < 0.3

    def test_term_structure_backwardation(self, cfg: BlackSwanSettings) -> None:
        ind = score_vix_term_structure(1.25, cfg)
        assert ind.status == IndicatorStatus.CRITICAL
        assert ind.score == 1.0

    def test_credit_stress_normal(self, cfg: BlackSwanSettings) -> None:
        ind = score_credit_stress(0.1, cfg)
        assert ind.status == IndicatorStatus.NORMAL
        assert ind.score == 0.0

    def test_credit_stress_danger(self, cfg: BlackSwanSettings) -> None:
        ind = score_credit_stress(-2.0, cfg)
        assert ind.status == IndicatorStatus.DANGER
        assert 0.5 <= ind.score < 0.85

    def test_spy_drawdown_crash(self, cfg: BlackSwanSettings) -> None:
        ind = score_spy_drawdown(-5.0, cfg)
        assert ind.status == IndicatorStatus.CRITICAL
        assert ind.score == 1.0

    def test_spy_drawdown_normal(self, cfg: BlackSwanSettings) -> None:
        ind = score_spy_drawdown(0.5, cfg)
        assert ind.status == IndicatorStatus.NORMAL
        assert ind.score == 0.0

    def test_rv_iv_normal(self, cfg: BlackSwanSettings) -> None:
        ind = score_rv_iv_gap(-3.0, cfg)
        assert ind.status == IndicatorStatus.NORMAL
        assert ind.score == 0.0

    def test_rv_iv_danger(self, cfg: BlackSwanSettings) -> None:
        ind = score_rv_iv_gap(10.0, cfg)
        assert ind.status == IndicatorStatus.DANGER
        assert 0.5 <= ind.score < 0.85

    def test_treasury_normal(self, cfg: BlackSwanSettings) -> None:
        ind = score_treasury_stress(0.5, cfg)
        assert ind.status == IndicatorStatus.NORMAL
        assert ind.score == 0.0

    def test_treasury_critical(self, cfg: BlackSwanSettings) -> None:
        ind = score_treasury_stress(3.5, cfg)
        assert ind.status == IndicatorStatus.CRITICAL
        assert ind.score == 1.0

    def test_em_contagion_normal(self, cfg: BlackSwanSettings) -> None:
        ind = score_em_contagion(0.5, cfg)
        assert ind.status == IndicatorStatus.NORMAL
        assert ind.score == 0.0

    def test_em_contagion_danger(self, cfg: BlackSwanSettings) -> None:
        ind = score_em_contagion(-4.0, cfg)
        assert ind.status == IndicatorStatus.DANGER
        assert 0.5 <= ind.score < 0.85

    def test_yield_curve_normal(self, cfg: BlackSwanSettings) -> None:
        ind = score_yield_curve(100.0, cfg)
        assert ind.status == IndicatorStatus.NORMAL
        assert ind.score == 0.0

    def test_yield_curve_inverted(self, cfg: BlackSwanSettings) -> None:
        ind = score_yield_curve(-50.0, cfg)
        assert ind.status == IndicatorStatus.DANGER
        assert ind.score >= 0.5

    def test_put_call_normal(self, cfg: BlackSwanSettings) -> None:
        ind = score_put_call_ratio(0.65, cfg)
        assert ind.status == IndicatorStatus.NORMAL
        assert ind.score == 0.0

    def test_put_call_critical(self, cfg: BlackSwanSettings) -> None:
        ind = score_put_call_ratio(1.5, cfg)
        assert ind.status == IndicatorStatus.CRITICAL
        assert ind.score == 1.0


# ===========================================================================
# TestCircuitBreakers — 5 tests
# ===========================================================================


class TestCircuitBreakers:
    """Circuit breakers fire at their thresholds."""

    def test_vix_extreme_triggers(self, cfg: BlackSwanSettings) -> None:
        breakers = check_circuit_breakers(
            vix=45.0, vix_ratio=0.9, spy_return_pct=0.0, credit_daily_drop_pct=0.0, cfg=cfg
        )
        vix_b = next(b for b in breakers if b.name == "vix_extreme")
        assert vix_b.triggered is True

    def test_vix_backwardation_triggers(self, cfg: BlackSwanSettings) -> None:
        breakers = check_circuit_breakers(
            vix=25.0, vix_ratio=1.25, spy_return_pct=0.0, credit_daily_drop_pct=0.0, cfg=cfg
        )
        bt = next(b for b in breakers if b.name == "vix_backwardation")
        assert bt.triggered is True

    def test_spy_crash_triggers(self, cfg: BlackSwanSettings) -> None:
        breakers = check_circuit_breakers(
            vix=25.0, vix_ratio=0.9, spy_return_pct=-5.0, credit_daily_drop_pct=0.0, cfg=cfg
        )
        spy_b = next(b for b in breakers if b.name == "spy_crash")
        assert spy_b.triggered is True

    def test_credit_collapse_triggers(self, cfg: BlackSwanSettings) -> None:
        breakers = check_circuit_breakers(
            vix=25.0, vix_ratio=0.9, spy_return_pct=0.0, credit_daily_drop_pct=-4.0, cfg=cfg
        )
        cb = next(b for b in breakers if b.name == "credit_collapse")
        assert cb.triggered is True

    def test_multiple_breakers(self, cfg: BlackSwanSettings) -> None:
        breakers = check_circuit_breakers(
            vix=50.0, vix_ratio=1.30, spy_return_pct=-6.0, credit_daily_drop_pct=-5.0, cfg=cfg
        )
        triggered = [b for b in breakers if b.triggered]
        assert len(triggered) == 4


# ===========================================================================
# TestCompositeScore — 4 tests
# ===========================================================================


class TestCompositeScore:
    """Weighted composite and re-normalization."""

    def test_all_normal_low_score(self, cfg: BlackSwanSettings) -> None:
        alert = compute_black_swan_alert(
            vix=12.0, vix_ratio=0.85, credit_pct_change=0.1,
            credit_daily_drop_pct=0.0, spy_daily_return_pct=0.5,
            rv_iv_gap=-5.0, tlt_abs_return_pct=0.3,
            em_pct_change=0.5, yield_curve_bps=100.0,
            put_call_ratio=0.65, cfg=cfg,
        )
        assert alert.composite_score < 0.25

    def test_all_critical_high_score(self, cfg: BlackSwanSettings) -> None:
        alert = compute_black_swan_alert(
            vix=50.0, vix_ratio=1.30, credit_pct_change=-4.0,
            credit_daily_drop_pct=-4.0, spy_daily_return_pct=-6.0,
            rv_iv_gap=20.0, tlt_abs_return_pct=4.0,
            em_pct_change=-6.0, yield_curve_bps=-50.0,
            put_call_ratio=1.5, cfg=cfg,
        )
        assert alert.composite_score >= 0.75

    def test_renormalize_with_unavailable(self, cfg: BlackSwanSettings) -> None:
        # Only VIX provided, everything else None
        alert = compute_black_swan_alert(vix=35.0, cfg=cfg)
        # Should still compute meaningful score from VIX alone
        assert alert.composite_score > 0.0
        available = [i for i in alert.indicators if i.status != IndicatorStatus.UNAVAILABLE]
        assert len(available) == 1

    def test_all_unavailable_zero_score(self, cfg: BlackSwanSettings) -> None:
        alert = compute_black_swan_alert(cfg=cfg)
        assert alert.composite_score == 0.0


# ===========================================================================
# TestAlertLevels — 4 tests
# ===========================================================================


class TestAlertLevels:
    """Alert level derived from composite score and circuit breakers."""

    def test_normal_level(self, cfg: BlackSwanSettings) -> None:
        alert = compute_black_swan_alert(
            vix=12.0, vix_ratio=0.85, credit_pct_change=0.1,
            credit_daily_drop_pct=0.0, spy_daily_return_pct=0.5,
            rv_iv_gap=-5.0, tlt_abs_return_pct=0.3,
            em_pct_change=0.5, yield_curve_bps=100.0,
            put_call_ratio=0.65, cfg=cfg,
        )
        assert alert.alert_level == AlertLevel.NORMAL

    def test_elevated_level(self, cfg: BlackSwanSettings) -> None:
        # Multiple indicators at warning level
        alert = compute_black_swan_alert(
            vix=25.0, vix_ratio=1.0, credit_pct_change=-1.0,
            credit_daily_drop_pct=-0.5, spy_daily_return_pct=-1.5,
            rv_iv_gap=3.0, tlt_abs_return_pct=1.5,
            em_pct_change=-2.0, yield_curve_bps=30.0,
            put_call_ratio=0.9, cfg=cfg,
        )
        assert alert.alert_level in (AlertLevel.ELEVATED, AlertLevel.HIGH)

    def test_critical_from_composite(self, cfg: BlackSwanSettings) -> None:
        alert = compute_black_swan_alert(
            vix=38.0, vix_ratio=1.15, credit_pct_change=-2.5,
            credit_daily_drop_pct=-2.0, spy_daily_return_pct=-3.5,
            rv_iv_gap=12.0, tlt_abs_return_pct=2.5,
            em_pct_change=-4.5, yield_curve_bps=-20.0,
            put_call_ratio=1.2, cfg=cfg,
        )
        assert alert.alert_level in (AlertLevel.HIGH, AlertLevel.CRITICAL)

    def test_circuit_breaker_overrides_to_critical(self, cfg: BlackSwanSettings) -> None:
        # VIX at 50 triggers breaker even if other indicators are calm
        alert = compute_black_swan_alert(
            vix=50.0, vix_ratio=0.85, credit_pct_change=0.1,
            credit_daily_drop_pct=0.0, spy_daily_return_pct=0.5,
            rv_iv_gap=-5.0, tlt_abs_return_pct=0.3,
            em_pct_change=0.5, yield_curve_bps=100.0,
            put_call_ratio=0.65, cfg=cfg,
        )
        assert alert.alert_level == AlertLevel.CRITICAL
        assert alert.triggered_breakers >= 1


# ===========================================================================
# TestFREDFetcher — 3 tests
# ===========================================================================


class TestFREDFetcher:
    """FRED fetcher availability and graceful degradation."""

    def test_unavailable_without_api_key(self) -> None:
        from market_analyzer.data.providers.fred import FREDFetcher

        with patch.dict("os.environ", {}, clear=True):
            f = FREDFetcher()
            assert f.available is False

    def test_unavailable_without_fredapi(self) -> None:
        from market_analyzer.data.providers.fred import FREDFetcher

        with patch.dict("os.environ", {"FRED_API_KEY": "fake"}):
            with patch("builtins.__import__", side_effect=ImportError("no fredapi")):
                f = FREDFetcher()
                assert f.available is False

    def test_get_series_returns_none_when_unavailable(self) -> None:
        from market_analyzer.data.providers.fred import FREDFetcher

        with patch.dict("os.environ", {}, clear=True):
            f = FREDFetcher()
            assert f.get_series("T10Y2Y") is None


# ===========================================================================
# TestService — 4 tests
# ===========================================================================


def _make_ohlcv(
    n: int = 60,
    base_close: float = 100.0,
    daily_return: float = 0.001,
) -> pd.DataFrame:
    """Generate a simple OHLCV DataFrame for testing."""
    dates = pd.bdate_range(end=date.today(), periods=n)
    closes = base_close * np.cumprod(1 + np.random.default_rng(42).normal(daily_return, 0.01, n))
    return pd.DataFrame(
        {
            "Open": closes * 0.999,
            "High": closes * 1.005,
            "Low": closes * 0.995,
            "Close": closes,
            "Volume": np.random.default_rng(42).integers(1_000_000, 5_000_000, n),
        },
        index=dates,
    )


class TestService:
    """End-to-end with mocked DataService."""

    def test_alert_with_mocked_data(self) -> None:
        from market_analyzer.service.black_swan import BlackSwanService

        ds = MagicMock()
        ds.get_ohlcv.side_effect = lambda t: _make_ohlcv(
            base_close={"^VIX": 18.0, "^VIX3M": 20.0, "SPY": 450.0,
                        "HYG": 80.0, "LQD": 110.0, "TLT": 100.0, "EEM": 40.0}.get(t, 100.0)
        )
        svc = BlackSwanService(data_service=ds)
        alert = svc.alert()
        assert alert.alert_level in list(AlertLevel)
        assert 0.0 <= alert.composite_score <= 1.0

    def test_missing_data_graceful(self) -> None:
        from market_analyzer.service.black_swan import BlackSwanService

        ds = MagicMock()
        ds.get_ohlcv.side_effect = Exception("network error")
        svc = BlackSwanService(data_service=ds)
        alert = svc.alert()
        # All indicators unavailable → NORMAL
        assert alert.alert_level == AlertLevel.NORMAL
        assert alert.composite_score == 0.0

    def test_no_data_service_raises(self) -> None:
        from market_analyzer.service.black_swan import BlackSwanService

        svc = BlackSwanService()
        with pytest.raises(ValueError, match="requires a DataService"):
            svc.alert()

    def test_facade_wiring(self) -> None:
        from market_analyzer.service.analyzer import MarketAnalyzer
        from market_analyzer.service.black_swan import BlackSwanService as BSS

        ds = MagicMock()
        ma = MarketAnalyzer(data_service=ds)
        assert hasattr(ma, "black_swan")
        assert isinstance(ma.black_swan, BSS)


# ===========================================================================
# TestActionText — 3 tests
# ===========================================================================


class TestActionText:
    """Correct action text per alert level."""

    def test_normal_action(self, cfg: BlackSwanSettings) -> None:
        alert = compute_black_swan_alert(
            vix=12.0, vix_ratio=0.85, credit_pct_change=0.1,
            credit_daily_drop_pct=0.0, spy_daily_return_pct=0.5,
            rv_iv_gap=-5.0, tlt_abs_return_pct=0.3,
            em_pct_change=0.5, yield_curve_bps=100.0,
            put_call_ratio=0.65, cfg=cfg,
        )
        assert "Business as usual" in alert.action

    def test_critical_action(self, cfg: BlackSwanSettings) -> None:
        alert = compute_black_swan_alert(
            vix=50.0, vix_ratio=1.30, credit_pct_change=-4.0,
            credit_daily_drop_pct=-4.0, spy_daily_return_pct=-6.0,
            rv_iv_gap=20.0, tlt_abs_return_pct=4.0,
            em_pct_change=-6.0, yield_curve_bps=-50.0,
            put_call_ratio=1.5, cfg=cfg,
        )
        assert "HALT" in alert.action

    def test_elevated_action(self, cfg: BlackSwanSettings) -> None:
        alert = compute_black_swan_alert(
            vix=25.0, vix_ratio=1.0, credit_pct_change=-1.0,
            credit_daily_drop_pct=-0.5, spy_daily_return_pct=-1.5,
            rv_iv_gap=3.0, tlt_abs_return_pct=1.5,
            em_pct_change=-2.0, yield_curve_bps=30.0,
            put_call_ratio=0.9, cfg=cfg,
        )
        # Either "Reduce" (ELEVATED) or "Flatten" (HIGH)
        assert "Reduce" in alert.action or "Flatten" in alert.action
