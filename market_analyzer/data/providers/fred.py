"""Lightweight FRED API helper for tail-risk indicators.

Not a full DataProvider (FRED doesn't return OHLCV).
Used only by BlackSwanService for yield curve and put/call ratio.
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


class FREDFetcher:
    """Fetch time series from FRED (Federal Reserve Economic Data).

    Requires ``fredapi`` package and ``FRED_API_KEY`` environment variable.
    If either is missing, ``available`` returns False and ``get_series``
    returns None — callers degrade gracefully.
    """

    def __init__(self) -> None:
        self._fred: object | None = None
        self._checked = False
        self._available = False

    @property
    def available(self) -> bool:
        """True if fredapi is installed and FRED_API_KEY is set."""
        if not self._checked:
            self._init()
        return self._available

    def _init(self) -> None:
        self._checked = True
        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            logger.debug("FRED_API_KEY not set — FRED indicators disabled")
            return
        try:
            from fredapi import Fred  # type: ignore[import-untyped]

            self._fred = Fred(api_key=api_key)
            self._available = True
        except ImportError:
            logger.debug("fredapi not installed — FRED indicators disabled")
        except Exception:
            logger.warning("Failed to initialize FRED client", exc_info=True)

    def get_series(
        self, series_id: str, lookback_days: int = 30
    ) -> pd.Series | None:
        """Fetch a FRED time series. Returns None if unavailable."""
        if not self.available:
            return None
        try:
            end = date.today()
            start = end - timedelta(days=lookback_days)
            series = self._fred.get_series(  # type: ignore[union-attr]
                series_id,
                observation_start=start,
                observation_end=end,
            )
            if series is None or series.empty:
                return None
            return series.dropna()
        except Exception:
            logger.warning("FRED fetch failed for %s", series_id, exc_info=True)
            return None
