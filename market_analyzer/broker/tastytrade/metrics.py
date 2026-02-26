"""TastyTrade market metrics — IV rank, IV percentile, beta, liquidity.

Adapted from eTrading tastytrade_adapter.py get_market_metrics().
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from market_analyzer.broker.base import MarketMetricsProvider
from market_analyzer.models.quotes import MarketMetrics

if TYPE_CHECKING:
    from market_analyzer.broker.tastytrade.session import TastyTradeBrokerSession

logger = logging.getLogger(__name__)


class TastyTradeMetrics(MarketMetricsProvider):
    """IV rank, IV percentile, beta, liquidity from TastyTrade API."""

    def __init__(self, session: TastyTradeBrokerSession) -> None:
        self._session = session

    def get_metrics(self, tickers: list[str]) -> dict[str, MarketMetrics]:
        """Fetch market metrics for tickers via TastyTrade API."""
        from tastytrade.metrics import get_market_metrics

        raw = get_market_metrics(self._session.sdk_session, tickers)

        result: dict[str, MarketMetrics] = {}
        for m in raw:
            earnings_date = None
            if m.earnings and hasattr(m.earnings, "expected_report_date"):
                earnings_date = m.earnings.expected_report_date

            result[m.symbol] = MarketMetrics(
                ticker=m.symbol,
                iv_rank=float(m.implied_volatility_index_rank) if m.implied_volatility_index_rank else None,
                iv_percentile=float(m.implied_volatility_percentile) if m.implied_volatility_percentile else None,
                iv_index=float(m.implied_volatility_index) if m.implied_volatility_index else None,
                iv_30_day=float(m.implied_volatility_30_day) if m.implied_volatility_30_day else None,
                hv_30_day=float(m.historical_volatility_30_day) if m.historical_volatility_30_day else None,
                hv_60_day=float(m.historical_volatility_60_day) if m.historical_volatility_60_day else None,
                beta=float(m.beta) if m.beta else None,
                corr_spy=float(m.corr_spy_3month) if m.corr_spy_3month else None,
                liquidity_rating=float(m.liquidity_rating) if m.liquidity_rating else None,
                earnings_date=earnings_date,
            )

        return result
