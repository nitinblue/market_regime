"""TastyTrade broker integration — optional sub-package.

Requires ``tastytrade`` SDK: ``pip install tastytrade-sdk``

Usage::

    from market_analyzer.broker.tastytrade import connect_tastytrade

    market_data, metrics = connect_tastytrade()
    ma = MarketAnalyzer(data_service=DataService(),
                        market_data=market_data, market_metrics=metrics)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from market_analyzer.broker.tastytrade.market_data import TastyTradeMarketData
    from market_analyzer.broker.tastytrade.metrics import TastyTradeMetrics

logger = logging.getLogger(__name__)


def connect_tastytrade(
    config_path: str = "tastytrade_broker.yaml",
    is_paper: bool = False,
) -> tuple[TastyTradeMarketData, TastyTradeMetrics]:
    """Convenience: authenticate and return MarketDataProvider + MetricsProvider.

    Raises on auth failure.
    """
    from market_analyzer.broker.tastytrade.market_data import TastyTradeMarketData
    from market_analyzer.broker.tastytrade.metrics import TastyTradeMetrics
    from market_analyzer.broker.tastytrade.session import TastyTradeBrokerSession

    session = TastyTradeBrokerSession(config_path=config_path, is_paper=is_paper)
    if not session.connect():
        raise ConnectionError("Failed to authenticate with TastyTrade")

    return TastyTradeMarketData(session), TastyTradeMetrics(session)
