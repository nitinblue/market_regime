"""ParquetCache: read/write/freshness checks for cached market data."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from market_analyzer.data.exceptions import CacheError
from market_analyzer.models.data import CacheMeta, DataType


class ParquetCache:
    """Manages parquet files in ~/.market_analyzer/cache/."""

    def __init__(self, cache_dir: Path | None = None, staleness_hours: float | None = None) -> None:
        from market_analyzer.config import get_settings

        cache_cfg = get_settings().cache
        if cache_dir is None:
            cache_dir = Path(cache_cfg.cache_dir) if cache_cfg.cache_dir else None
        default_dir = Path.home() / ".market_analyzer" / "cache"
        legacy_dir = Path.home() / ".market_regime" / "cache"
        if not default_dir.exists() and legacy_dir.exists():
            self.cache_dir = cache_dir or legacy_dir
        else:
            self.cache_dir = cache_dir or default_dir
        self.staleness_hours = staleness_hours if staleness_hours is not None else cache_cfg.staleness_hours

    @property
    def _meta_path(self) -> Path:
        return self.cache_dir / "_meta.json"

    def _parquet_path(self, ticker: str, data_type: DataType) -> Path:
        return self.cache_dir / data_type.value / f"{ticker.upper()}.parquet"

    def _load_meta(self) -> dict[str, CacheMeta]:
        """Load _meta.json. Returns dict keyed by 'TICKER:data_type'."""
        if not self._meta_path.exists():
            return {}
        try:
            raw = json.loads(self._meta_path.read_text())
            return {
                key: CacheMeta(**entry) for key, entry in raw.items()
            }
        except (json.JSONDecodeError, Exception) as e:
            raise CacheError(f"Failed to read {self._meta_path}: {e}") from e

    def _save_meta(self, meta: dict[str, CacheMeta]) -> None:
        """Write _meta.json atomically."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        serialized = {
            key: entry.model_dump(mode="json") for key, entry in meta.items()
        }
        data = json.dumps(serialized, indent=2, default=str)
        # Atomic write: temp file + rename
        fd, tmp = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            os.write(fd, data.encode())
            os.close(fd)
            os.replace(tmp, self._meta_path)
        except Exception as e:
            os.close(fd) if not os.get_inheritable(fd) else None
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise CacheError(f"Failed to write meta: {e}") from e

    @staticmethod
    def _meta_key(ticker: str, data_type: DataType) -> str:
        return f"{ticker.upper()}:{data_type.value}"

    def read(self, ticker: str, data_type: DataType) -> pd.DataFrame | None:
        """Read cached data. Returns None on cache miss."""
        path = self._parquet_path(ticker, data_type)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            return df
        except Exception as e:
            raise CacheError(f"Failed to read {path}: {e}") from e

    def write(self, ticker: str, data_type: DataType, df: pd.DataFrame, meta: CacheMeta) -> None:
        """Write data to cache (atomic: temp file + rename)."""
        path = self._parquet_path(ticker, data_type)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: temp file + os.replace
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        os.close(fd)
        try:
            df.to_parquet(tmp, engine="pyarrow")
            os.replace(tmp, path)
        except Exception as e:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise CacheError(f"Failed to write {path}: {e}") from e

        # Update meta
        all_meta = self._load_meta()
        all_meta[self._meta_key(ticker, data_type)] = meta
        self._save_meta(all_meta)

    def is_stale(self, ticker: str, data_type: DataType) -> bool:
        """Check if cached data is stale (older than staleness threshold).

        Weekend awareness: if today is Sat/Sun and last cached date is
        the most recent Friday, data is considered fresh.
        """
        meta = self.get_meta(ticker, data_type)
        if meta is None:
            return True  # No cache = stale

        now = datetime.now()
        age = now - meta.last_fetched

        # If within staleness window, it's fresh
        if age < timedelta(hours=self.staleness_hours):
            return False

        # Weekend awareness: if today is Sat(5) or Sun(6),
        # and last_date >= last Friday, data is fresh
        today = date.today()
        weekday = today.weekday()
        if weekday in (5, 6):  # Saturday or Sunday
            # Find last Friday
            days_since_friday = weekday - 4  # Sat=1, Sun=2
            last_friday = today - timedelta(days=days_since_friday)
            if meta.last_date >= last_friday:
                return False

        return True

    def get_meta(self, ticker: str, data_type: DataType) -> CacheMeta | None:
        """Get cache metadata for a ticker/data_type. None if not cached."""
        key = self._meta_key(ticker, data_type)
        all_meta = self._load_meta()
        return all_meta.get(key)

    def delta_dates(self, ticker: str, data_type: DataType, end_date: date) -> tuple[date, date] | None:
        """Compute (start, end) dates needed for delta-fetch. None if cache is fresh."""
        if not self.is_stale(ticker, data_type):
            return None

        meta = self.get_meta(ticker, data_type)
        if meta is None:
            return None  # No cache at all — caller should do full fetch

        start = meta.last_date + timedelta(days=1)
        if start > end_date:
            return None
        return (start, end_date)

    def invalidate(self, ticker: str, data_type: DataType | None = None) -> None:
        """Remove cached data for a ticker."""
        types_to_clear = [data_type] if data_type else list(DataType)
        all_meta = self._load_meta()

        for dt in types_to_clear:
            path = self._parquet_path(ticker, dt)
            if path.exists():
                path.unlink()
            key = self._meta_key(ticker, dt)
            all_meta.pop(key, None)

        self._save_meta(all_meta)
