"""DataProvider abstract base class."""

from abc import ABC, abstractmethod

import pandas as pd

from market_regime.models.data import DataRequest, DataType, ProviderType


class DataProvider(ABC):
    """Base class for all data providers."""

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType: ...

    @property
    @abstractmethod
    def supported_data_types(self) -> list[DataType]: ...

    @abstractmethod
    def fetch(self, request: DataRequest) -> pd.DataFrame:
        """Fetch data from remote source. Raises on failure."""
        ...

    @abstractmethod
    def validate_ticker(self, ticker: str) -> bool:
        """Check if ticker is valid for this provider."""
        ...
