"""BlackSwanService: tail-risk circuit breaker for pre-trade gating."""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from market_analyzer.config import get_settings
from market_analyzer.data.providers.fred import FREDFetcher
from market_analyzer.features.black_swan import compute_black_swan_alert
from market_analyzer.models.black_swan import BlackSwanAlert

if TYPE_CHECKING:
    from market_analyzer.data.service import DataService

logger = logging.getLogger(__name__)


class BlackSwanService:
    """Compute current tail-risk alert level.

    Fetches VIX, VIX3M, SPY, HYG, LQD, TLT, EEM via DataService.
    Fetches yield curve + put/call ratio from FRED (if available).
    """

    def __init__(
        self,
        data_service: DataService | None = None,
        fred: FREDFetcher | None = None,
    ) -> None:
        self.data_service = data_service
        self._fred = fred or FREDFetcher()

    def alert(self, as_of_date: date | None = None) -> BlackSwanAlert:
        """Compute current tail-risk alert.

        Raises ValueError if no data_service is configured.
        """
        if self.data_service is None:
            raise ValueError(
                "BlackSwanService requires a DataService for market data. "
                "Pass data_service to constructor."
            )

        cfg = get_settings().black_swan
        as_of_date = as_of_date or date.today()
        lookback = cfg.lookback_days

        # Fetch reference tickers
        tickers = ["^VIX", "^VIX3M", "SPY", "HYG", "LQD", "TLT", "EEM"]
        data: dict[str, pd.DataFrame | None] = {}
        for ticker in tickers:
            try:
                df = self.data_service.get_ohlcv(ticker)
                data[ticker] = df
            except Exception:
                logger.warning("Failed to fetch %s", ticker, exc_info=True)
                data[ticker] = None

        # Extract indicator values
        vix = self._latest_close(data.get("^VIX"))
        vix3m = self._latest_close(data.get("^VIX3M"))
        vix_ratio = (vix / vix3m) if (vix is not None and vix3m is not None and vix3m > 0) else None

        # Credit stress: HYG/LQD ratio % change from 20d avg
        credit_pct_change, credit_daily_drop = self._credit_stress(
            data.get("HYG"), data.get("LQD"), lookback
        )

        # SPY 1-day return
        spy_return = self._daily_return_pct(data.get("SPY"))

        # RV vs IV gap
        rv_iv_gap = self._rv_iv_gap(data.get("SPY"), vix, lookback)

        # Treasury stress: TLT absolute 1-day return
        tlt_abs = self._abs_daily_return_pct(data.get("TLT"))

        # EM contagion: EEM/SPY ratio % change from 20d avg
        em_pct_change = self._ratio_pct_change(
            data.get("EEM"), data.get("SPY"), lookback
        )

        # FRED indicators
        yield_curve_bps = self._fetch_yield_curve()
        put_call_ratio = self._fetch_put_call()

        return compute_black_swan_alert(
            vix=vix,
            vix_ratio=vix_ratio,
            credit_pct_change=credit_pct_change,
            credit_daily_drop_pct=credit_daily_drop,
            spy_daily_return_pct=spy_return,
            rv_iv_gap=rv_iv_gap,
            tlt_abs_return_pct=tlt_abs,
            em_pct_change=em_pct_change,
            yield_curve_bps=yield_curve_bps,
            put_call_ratio=put_call_ratio,
            as_of_date=as_of_date,
            cfg=cfg,
        )

    # --- Data extraction helpers ---

    @staticmethod
    def _latest_close(df: pd.DataFrame | None) -> float | None:
        if df is None or df.empty:
            return None
        return float(df["Close"].iloc[-1])

    @staticmethod
    def _daily_return_pct(df: pd.DataFrame | None) -> float | None:
        if df is None or len(df) < 2:
            return None
        close = df["Close"]
        return float((close.iloc[-1] / close.iloc[-2] - 1) * 100)

    @staticmethod
    def _abs_daily_return_pct(df: pd.DataFrame | None) -> float | None:
        if df is None or len(df) < 2:
            return None
        close = df["Close"]
        return float(abs((close.iloc[-1] / close.iloc[-2] - 1) * 100))

    @staticmethod
    def _credit_stress(
        hyg: pd.DataFrame | None,
        lqd: pd.DataFrame | None,
        lookback: int,
    ) -> tuple[float | None, float | None]:
        """Return (pct change from 20d avg, 1-day drop pct) of HYG/LQD ratio."""
        if hyg is None or lqd is None or hyg.empty or lqd.empty:
            return None, None
        # Align indexes
        hyg_c = hyg["Close"]
        lqd_c = lqd["Close"]
        ratio = hyg_c / lqd_c
        ratio = ratio.dropna()
        if len(ratio) < lookback + 1:
            return None, None
        avg_20d = ratio.iloc[-(lookback + 1):-1].mean()
        current = ratio.iloc[-1]
        pct_from_avg = (current / avg_20d - 1) * 100 if avg_20d != 0 else None

        prev = ratio.iloc[-2]
        daily_drop = (current / prev - 1) * 100 if prev != 0 else None

        return pct_from_avg, daily_drop

    @staticmethod
    def _rv_iv_gap(
        spy: pd.DataFrame | None, vix: float | None, lookback: int
    ) -> float | None:
        if spy is None or vix is None or len(spy) < lookback:
            return None
        log_returns = np.log(spy["Close"] / spy["Close"].shift(1)).dropna()
        if len(log_returns) < lookback:
            return None
        rv_20d = float(log_returns.iloc[-lookback:].std() * np.sqrt(252) * 100)
        return rv_20d - vix

    @staticmethod
    def _ratio_pct_change(
        numer: pd.DataFrame | None,
        denom: pd.DataFrame | None,
        lookback: int,
    ) -> float | None:
        if numer is None or denom is None or numer.empty or denom.empty:
            return None
        ratio = numer["Close"] / denom["Close"]
        ratio = ratio.dropna()
        if len(ratio) < lookback + 1:
            return None
        avg = ratio.iloc[-(lookback + 1):-1].mean()
        current = ratio.iloc[-1]
        return float((current / avg - 1) * 100) if avg != 0 else None

    def _fetch_yield_curve(self) -> float | None:
        series = self._fred.get_series("T10Y2Y", lookback_days=30)
        if series is None or series.empty:
            return None
        # T10Y2Y is in percentage points; convert to bps
        return float(series.iloc[-1] * 100)

    def _fetch_put_call(self) -> float | None:
        series = self._fred.get_series("EQUITPC", lookback_days=30)
        if series is None or series.empty:
            return None
        return float(series.iloc[-1])
