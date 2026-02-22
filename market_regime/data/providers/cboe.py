"""CBOEProvider: options/IV data via CBOE API."""

import pandas as pd

from market_regime.data.providers.base import DataProvider
from market_regime.models.data import DataRequest, DataType, ProviderType


class CBOEProvider(DataProvider):
    """Fetches options/IV data from CBOE. Requires CBOE_API_KEY env var."""

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.CBOE

    @property
    def supported_data_types(self) -> list[DataType]:
        return [DataType.OPTIONS_IV]

    def fetch(self, request: DataRequest) -> pd.DataFrame:
        """Fetch options/IV data from CBOE."""
        raise NotImplementedError

    def validate_ticker(self, ticker: str) -> bool:
        """Check if ticker is available on CBOE."""
        raise NotImplementedError
