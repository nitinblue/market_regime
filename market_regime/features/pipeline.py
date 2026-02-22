"""Feature pipeline: computes feature matrix from OHLCV DataFrame."""

import numpy as np
import pandas as pd

from market_regime.models.features import FeatureConfig, FeatureInspection

FEATURE_NAMES = [
    "log_return_1d",
    "log_return_5d",
    "realized_vol",
    "atr_normalized",
    "trend_strength",
    "volume_anomaly",
]


def _validate_ohlcv(df: pd.DataFrame) -> None:
    """Validate OHLCV DataFrame has required columns and DatetimeIndex."""
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV DataFrame missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("OHLCV DataFrame must have a DatetimeIndex")
    if df.empty:
        raise ValueError("OHLCV DataFrame is empty")


def _compute_raw_features(ohlcv: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Compute raw (unnormalized) features from OHLCV data."""
    close = ohlcv["Close"]
    high = ohlcv["High"]
    low = ohlcv["Low"]
    volume = ohlcv["Volume"]

    features = pd.DataFrame(index=ohlcv.index)

    # Log returns
    features["log_return_1d"] = np.log(close / close.shift(1))
    features["log_return_5d"] = np.log(close / close.shift(5))

    # Realized volatility (annualized)
    features["realized_vol"] = (
        features["log_return_1d"].rolling(config.realized_vol_window).std()
        * np.sqrt(252)
    )

    # ATR normalized by close price
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(config.atr_window).mean()
    # Guard: avoid division by zero for penny stocks or zero-price rows
    safe_close = close.replace(0, np.nan)
    features["atr_normalized"] = atr / safe_close

    # Trend strength: slope proxy of SMA, normalized
    sma = close.rolling(config.trend_window).mean()
    safe_sma = sma.replace(0, np.nan)
    features["trend_strength"] = (sma - sma.shift(config.trend_window)) / (
        config.trend_window * safe_sma
    )

    # Volume anomaly
    vol_avg = volume.rolling(config.volume_window).mean()
    safe_vol_avg = vol_avg.replace(0, np.nan)
    features["volume_anomaly"] = volume / safe_vol_avg

    return features


def _normalize_features(
    raw: pd.DataFrame, lookback: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Z-score normalize features using rolling window.

    Returns:
        (normalized_df, rolling_means, rolling_stds) — all same shape as raw after dropna.
    """
    rolling_means = raw.rolling(lookback).mean()
    rolling_stds = raw.rolling(lookback).std()

    # Clip std to avoid division by zero
    rolling_stds = rolling_stds.clip(lower=1e-8)

    normalized = (raw - rolling_means) / rolling_stds

    # Drop rows where any feature is NaN (warmup period)
    valid_mask = normalized.notna().all(axis=1)
    normalized = normalized.loc[valid_mask]
    rolling_means = rolling_means.loc[valid_mask]
    rolling_stds = rolling_stds.loc[valid_mask]

    return normalized, rolling_means, rolling_stds


def compute_features(
    ohlcv: pd.DataFrame, config: FeatureConfig = FeatureConfig()
) -> pd.DataFrame:
    """Compute normalized feature matrix from OHLCV DataFrame.

    Args:
        ohlcv: DataFrame with Open, High, Low, Close, Volume columns and DatetimeIndex.
        config: Feature computation parameters.

    Returns:
        DataFrame with computed features, indexed same as input (rows with NaN dropped).
    """
    _validate_ohlcv(ohlcv)
    raw = _compute_raw_features(ohlcv, config)

    lookback = max(
        config.realized_vol_window,
        config.atr_window,
        config.trend_window,
        config.volume_window,
    )
    normalized, _, _ = _normalize_features(raw, lookback)
    if normalized.empty:
        raise ValueError(
            f"Feature normalization produced no valid rows. "
            f"Need at least ~{lookback * 2} trading days of OHLCV data "
            f"(have {len(ohlcv)} rows, lookback={lookback})."
        )
    return normalized


def compute_features_with_inspection(
    ohlcv: pd.DataFrame,
    ticker: str,
    config: FeatureConfig = FeatureConfig(),
    lookback: int | None = None,
) -> tuple[pd.DataFrame, FeatureInspection]:
    """Compute features and return full inspection data.

    Args:
        ohlcv: OHLCV DataFrame.
        ticker: Instrument ticker.
        config: Feature computation parameters.
        lookback: Rolling normalization window. Defaults to max of config windows.

    Returns:
        (normalized_df, FeatureInspection) tuple.
    """
    _validate_ohlcv(ohlcv)
    raw = _compute_raw_features(ohlcv, config)

    if lookback is None:
        lookback = max(
            config.realized_vol_window,
            config.atr_window,
            config.trend_window,
            config.volume_window,
        )

    normalized, rolling_means, rolling_stds = _normalize_features(raw, lookback)

    if normalized.empty:
        raise ValueError(
            f"Feature normalization for {ticker} produced no valid rows. "
            f"Need at least ~{lookback * 2} trading days of OHLCV data "
            f"(have {len(ohlcv)} rows, lookback={lookback})."
        )

    # Drop NaN rows from raw to match normalized index for inspection
    raw_valid = raw.dropna()

    inspection = FeatureInspection(
        ticker=ticker,
        date_range=(
            normalized.index[0].date(),
            normalized.index[-1].date(),
        ),
        raw_row_count=len(raw_valid),
        normalized_row_count=len(normalized),
        feature_names=list(normalized.columns),
        raw_features=raw_valid.reset_index().assign(
            date=lambda df: df.iloc[:, 0].dt.strftime("%Y-%m-%d")
        ).drop(columns=[raw_valid.index.name or raw_valid.reset_index().columns[0]])
        .to_dict(orient="records")
        if not raw_valid.empty
        else [],
        normalized_features=normalized.reset_index().assign(
            date=lambda df: df.iloc[:, 0].dt.strftime("%Y-%m-%d")
        ).drop(columns=[normalized.index.name or normalized.reset_index().columns[0]])
        .to_dict(orient="records")
        if not normalized.empty
        else [],
        normalization_means=rolling_means.tail(1).to_dict(orient="records"),
        normalization_stds=rolling_stds.tail(1).to_dict(orient="records"),
    )

    return normalized, inspection
