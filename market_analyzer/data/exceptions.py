"""Typed exceptions for the data layer."""


class DataFetchError(Exception):
    """Provider network/API failure."""

    def __init__(self, provider: str, ticker: str, message: str) -> None:
        self.provider = provider
        self.ticker = ticker
        super().__init__(f"[{provider}] Failed to fetch {ticker}: {message}")


class InvalidTickerError(Exception):
    """Ticker not recognized by provider."""

    def __init__(self, provider: str, ticker: str) -> None:
        self.provider = provider
        self.ticker = ticker
        super().__init__(f"[{provider}] Invalid ticker: {ticker}")


class CacheError(Exception):
    """Cache read/write failure."""


class NoProviderError(Exception):
    """No provider registered for a given data type."""

    def __init__(self, data_type: str) -> None:
        self.data_type = data_type
        super().__init__(f"No provider registered for data_type={data_type}")
