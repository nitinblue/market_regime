"""TechnicalService: technical indicators and ORB computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from market_analyzer.models.technicals import ORBData, TechnicalSnapshot

if TYPE_CHECKING:
    from market_analyzer.data.service import DataService


class TechnicalService:
    """Compute technical indicators and Opening Range Breakout levels."""

    def __init__(self, data_service: DataService | None = None) -> None:
        self.data_service = data_service

    def _get_ohlcv(self, ticker: str, ohlcv: pd.DataFrame | None) -> pd.DataFrame:
        if ohlcv is not None:
            return ohlcv
        if self.data_service is None:
            raise ValueError(
                "Either provide ohlcv DataFrame or initialize TechnicalService with a DataService"
            )
        return self.data_service.get_ohlcv(ticker)

    def snapshot(
        self, ticker: str, ohlcv: pd.DataFrame | None = None
    ) -> TechnicalSnapshot:
        """Compute technical indicators for a single instrument."""
        df = self._get_ohlcv(ticker, ohlcv)
        from market_analyzer.features.technicals import compute_technicals

        return compute_technicals(df, ticker)

    def orb(
        self,
        ticker: str,
        intraday: pd.DataFrame | None = None,
        daily_atr: float | None = None,
    ) -> ORBData:
        """Compute Opening Range Breakout from intraday data.

        Args:
            ticker: Instrument ticker.
            intraday: Intraday OHLCV DataFrame (1m/5m bars).
                      If None, fetches via yfinance.
            daily_atr: Optional daily ATR for context. Auto-computed if
                       data_service available and daily_atr is None.
        """
        from market_analyzer.features.orb import compute_orb

        if intraday is None:
            import yfinance as yf

            intraday = yf.download(ticker, period="1d", interval="5m", progress=False)
            if intraday.empty:
                raise ValueError(f"No intraday data available for {ticker}")
            if isinstance(intraday.columns, pd.MultiIndex):
                intraday.columns = intraday.columns.get_level_values(0)

        if daily_atr is None and self.data_service is not None:
            try:
                from market_analyzer.features.technicals import compute_atr

                ohlcv = self.data_service.get_ohlcv(ticker)
                atr_series = compute_atr(
                    ohlcv["High"], ohlcv["Low"], ohlcv["Close"], 14
                )
                if not atr_series.empty and not pd.isna(atr_series.iloc[-1]):
                    daily_atr = float(atr_series.iloc[-1])
            except Exception:
                pass  # ATR is optional context; don't fail ORB over it

        return compute_orb(intraday, ticker, daily_atr=daily_atr)
