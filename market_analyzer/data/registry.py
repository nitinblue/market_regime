"""Maps (ticker, data_type) to the correct provider."""

from __future__ import annotations

from market_analyzer.data.exceptions import NoProviderError
from market_analyzer.data.providers.base import DataProvider
from market_analyzer.models.data import DataType


class ProviderRegistry:
    """Registry that resolves which provider handles a given (ticker, data_type)."""

    def __init__(self) -> None:
        self._providers: list[DataProvider] = []

    def register(self, provider: DataProvider) -> None:
        """Register a data provider."""
        self._providers.append(provider)

    def resolve(self, ticker: str, data_type: DataType) -> DataProvider:
        """Find the provider that supports the given data type."""
        for provider in self._providers:
            if data_type in provider.supported_data_types:
                return provider
        raise NoProviderError(data_type)
