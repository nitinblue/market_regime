"""Data service models."""

from datetime import date, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel


class DataType(StrEnum):
    OHLCV = "ohlcv"
    OPTIONS_IV = "options_iv"
    BROKER_HISTORY = "broker_history"


class ProviderType(StrEnum):
    YFINANCE = "yfinance"
    CBOE = "cboe"
    TASTYTRADE = "tastytrade"


class DataRequest(BaseModel):
    ticker: str
    data_type: DataType
    start_date: date | None = None
    end_date: date | None = None


class CacheMeta(BaseModel):
    ticker: str
    data_type: DataType
    provider: ProviderType
    first_date: date
    last_date: date
    last_fetched: datetime
    row_count: int
    file_path: Path


class DataResult(BaseModel):
    ticker: str
    data_type: DataType
    provider: ProviderType
    from_cache: bool
    date_range: tuple[date, date]
    row_count: int

    model_config = {"arbitrary_types_allowed": True}
