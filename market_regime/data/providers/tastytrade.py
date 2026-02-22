"""TastyTradeProvider: broker history via tastytrade-sdk."""

import pandas as pd

from market_regime.data.providers.base import DataProvider
from market_regime.models.data import DataRequest, DataType, ProviderType


class TastyTradeProvider(DataProvider):
    """Fetches broker trade history from TastyTrade.

    Requires TASTYTRADE_USERNAME and TASTYTRADE_PASSWORD env vars.
    """

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.TASTYTRADE

    @property
    def supported_data_types(self) -> list[DataType]:
        return [DataType.BROKER_HISTORY]

    def fetch(self, request: DataRequest) -> pd.DataFrame:
        """Fetch trade history from TastyTrade."""
        raise NotImplementedError

    def validate_ticker(self, ticker: str) -> bool:
        """Check if ticker has history on TastyTrade."""
        raise NotImplementedError
