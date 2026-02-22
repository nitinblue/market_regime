"""Shared test fixtures for market_regime tests."""

import numpy as np
import pandas as pd
import pytest

from market_regime.features.pipeline import compute_features
from market_regime.hmm.trainer import HMMTrainer
from market_regime.models.features import FeatureConfig
from market_regime.models.regime import RegimeConfig


def _make_ohlcv(
    start: str,
    periods: int,
    base_price: float = 100.0,
    trend: float = 0.0,
    volatility: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data.

    Args:
        start: Start date string.
        periods: Number of trading days.
        base_price: Starting price.
        trend: Daily drift (e.g., 0.001 for uptrend).
        volatility: Daily return std.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=periods)
    returns = rng.normal(trend, volatility, periods)
    prices = base_price * np.exp(np.cumsum(returns))

    # Synthetic OHLCV
    daily_range = prices * volatility * rng.uniform(0.5, 2.0, periods)
    high = prices + daily_range / 2
    low = prices - daily_range / 2
    open_prices = prices + rng.normal(0, volatility * prices * 0.3, periods)
    volume = rng.integers(1_000_000, 10_000_000, periods).astype(float)

    return pd.DataFrame(
        {
            "Open": open_prices,
            "High": high,
            "Low": low,
            "Close": prices,
            "Volume": volume,
        },
        index=dates,
    )


@pytest.fixture
def sample_ohlcv_trending() -> pd.DataFrame:
    """250 rows uptrend, low volatility."""
    return _make_ohlcv("2024-01-01", 250, trend=0.001, volatility=0.008, seed=42)


@pytest.fixture
def sample_ohlcv_choppy() -> pd.DataFrame:
    """250 rows range-bound, high volatility."""
    return _make_ohlcv("2024-01-01", 250, trend=0.0, volatility=0.025, seed=99)


@pytest.fixture
def sample_ohlcv_mixed() -> pd.DataFrame:
    """500 rows: 250 trending then 250 choppy."""
    trending = _make_ohlcv("2023-01-01", 250, trend=0.001, volatility=0.008, seed=42)
    last_price = trending["Close"].iloc[-1]
    choppy = _make_ohlcv(
        "2024-01-01", 250, base_price=last_price, trend=0.0, volatility=0.025, seed=99
    )
    return pd.concat([trending, choppy])


@pytest.fixture
def sample_feature_matrix(sample_ohlcv_mixed: pd.DataFrame) -> pd.DataFrame:
    """Pre-computed features from mixed data."""
    return compute_features(sample_ohlcv_mixed)


@pytest.fixture
def fitted_trainer(sample_feature_matrix: pd.DataFrame) -> HMMTrainer:
    """HMMTrainer fitted on sample feature data."""
    trainer = HMMTrainer(RegimeConfig())
    trainer.fit(sample_feature_matrix)
    return trainer
