"""Technical indicator computation from OHLCV DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_regime.config import TechnicalsSettings, get_settings
from market_regime.models.technicals import (
    BollingerBands,
    MACDData,
    MovingAverages,
    RSIData,
    SignalDirection,
    SignalStrength,
    StochasticData,
    SupportResistance,
    TechnicalSignal,
    TechnicalSnapshot,
)


def compute_sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return close.rolling(window).mean()


def compute_ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return close.ewm(span=span, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """RSI using Wilder's smoothing method."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss is 0 (all gains), RSI = 100
    rsi = rsi.fillna(100.0)
    return rsi


def compute_bollinger(
    close: pd.Series, window: int, num_std: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: (upper, middle, lower)."""
    middle = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def compute_macd(
    close: pd.Series, fast: int, slow: int, signal: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD: (macd_line, signal_line, histogram)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_vwma(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Volume-weighted moving average."""
    return (close * volume).rolling(window).sum() / volume.rolling(window).sum()


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> pd.Series:
    """Average True Range."""
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def compute_stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int, d_period: int
) -> tuple[pd.Series, pd.Series]:
    """Stochastic oscillator: (%K, %D)."""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    denom = highest_high - lowest_low
    k = 100.0 * (close - lowest_low) / denom.replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def _detect_macd_crossover(
    macd_line: pd.Series, signal_line: pd.Series
) -> tuple[bool, bool]:
    """Detect MACD crossover on the most recent bar.

    Returns (is_bullish_crossover, is_bearish_crossover).
    """
    if len(macd_line) < 2:
        return False, False
    prev_diff = macd_line.iloc[-2] - signal_line.iloc[-2]
    curr_diff = macd_line.iloc[-1] - signal_line.iloc[-1]
    bullish = bool(prev_diff <= 0 and curr_diff > 0)
    bearish = bool(prev_diff >= 0 and curr_diff < 0)
    return bullish, bearish


def _detect_golden_death_cross(
    sma50: pd.Series, sma200: pd.Series
) -> tuple[bool, bool]:
    """Detect golden cross / death cross on most recent bar.

    Returns (is_golden_cross, is_death_cross).
    """
    if len(sma50) < 2:
        return False, False
    prev_diff = sma50.iloc[-2] - sma200.iloc[-2]
    curr_diff = sma50.iloc[-1] - sma200.iloc[-1]
    golden = bool(prev_diff <= 0 and curr_diff > 0)
    death = bool(prev_diff >= 0 and curr_diff < 0)
    return golden, death


def _generate_signals(
    price: float,
    rsi_val: float,
    macd_bullish: bool,
    macd_bearish: bool,
    golden_cross: bool,
    death_cross: bool,
    bb_upper: float,
    bb_lower: float,
    stoch_k: float,
    sma_200: float,
    vwma_val: float,
    settings: TechnicalsSettings,
) -> list[TechnicalSignal]:
    """Generate technical signals from current indicator values."""
    signals: list[TechnicalSignal] = []

    # RSI signals
    if rsi_val < settings.rsi_oversold:
        strength = SignalStrength.STRONG if rsi_val < 20 else SignalStrength.MODERATE
        signals.append(TechnicalSignal(
            name="RSI Oversold",
            direction=SignalDirection.BULLISH,
            strength=strength,
            description=f"RSI at {rsi_val:.1f} — below {settings.rsi_oversold:.0f} oversold threshold",
        ))
    elif rsi_val > settings.rsi_overbought:
        strength = SignalStrength.STRONG if rsi_val > 80 else SignalStrength.MODERATE
        signals.append(TechnicalSignal(
            name="RSI Overbought",
            direction=SignalDirection.BEARISH,
            strength=strength,
            description=f"RSI at {rsi_val:.1f} — above {settings.rsi_overbought:.0f} overbought threshold",
        ))

    # MACD crossover
    if macd_bullish:
        signals.append(TechnicalSignal(
            name="MACD Bullish Crossover",
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.MODERATE,
            description="MACD line crossed above signal line",
        ))
    if macd_bearish:
        signals.append(TechnicalSignal(
            name="MACD Bearish Crossover",
            direction=SignalDirection.BEARISH,
            strength=SignalStrength.MODERATE,
            description="MACD line crossed below signal line",
        ))

    # Golden / Death cross
    if golden_cross:
        signals.append(TechnicalSignal(
            name="Golden Cross",
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.STRONG,
            description="SMA(50) crossed above SMA(200)",
        ))
    if death_cross:
        signals.append(TechnicalSignal(
            name="Death Cross",
            direction=SignalDirection.BEARISH,
            strength=SignalStrength.STRONG,
            description="SMA(50) crossed below SMA(200)",
        ))

    # Bollinger Band signals
    if price < bb_lower:
        signals.append(TechnicalSignal(
            name="Below Lower Bollinger",
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.MODERATE,
            description=f"Price ({price:.2f}) below lower Bollinger Band ({bb_lower:.2f})",
        ))
    elif price > bb_upper:
        signals.append(TechnicalSignal(
            name="Above Upper Bollinger",
            direction=SignalDirection.BEARISH,
            strength=SignalStrength.MODERATE,
            description=f"Price ({price:.2f}) above upper Bollinger Band ({bb_upper:.2f})",
        ))

    # Stochastic signals
    if stoch_k < settings.stochastic_oversold:
        signals.append(TechnicalSignal(
            name="Stochastic Oversold",
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.MODERATE,
            description=f"Stochastic %K at {stoch_k:.1f} — below {settings.stochastic_oversold:.0f}",
        ))
    elif stoch_k > settings.stochastic_overbought:
        signals.append(TechnicalSignal(
            name="Stochastic Overbought",
            direction=SignalDirection.BEARISH,
            strength=SignalStrength.MODERATE,
            description=f"Stochastic %K at {stoch_k:.1f} — above {settings.stochastic_overbought:.0f}",
        ))

    # Trend context: price vs SMA(200)
    if not np.isnan(sma_200):
        if price > sma_200:
            signals.append(TechnicalSignal(
                name="Above 200 SMA",
                direction=SignalDirection.BULLISH,
                strength=SignalStrength.WEAK,
                description=f"Price above 200-day SMA ({sma_200:.2f})",
            ))
        else:
            signals.append(TechnicalSignal(
                name="Below 200 SMA",
                direction=SignalDirection.BEARISH,
                strength=SignalStrength.WEAK,
                description=f"Price below 200-day SMA ({sma_200:.2f})",
            ))

    # Price vs VWMA
    if not np.isnan(vwma_val):
        if price > vwma_val:
            signals.append(TechnicalSignal(
                name="Above VWMA",
                direction=SignalDirection.BULLISH,
                strength=SignalStrength.WEAK,
                description=f"Price above VWMA(20) ({vwma_val:.2f})",
            ))
        else:
            signals.append(TechnicalSignal(
                name="Below VWMA",
                direction=SignalDirection.BEARISH,
                strength=SignalStrength.WEAK,
                description=f"Price below VWMA(20) ({vwma_val:.2f})",
            ))

    return signals


def compute_technicals(
    ohlcv: pd.DataFrame,
    ticker: str,
    settings: TechnicalsSettings | None = None,
) -> TechnicalSnapshot:
    """Compute full technical snapshot from OHLCV DataFrame.

    Args:
        ohlcv: DataFrame with Open, High, Low, Close, Volume and DatetimeIndex.
        ticker: Instrument ticker symbol.
        settings: Optional settings override. Uses global config if None.

    Returns:
        TechnicalSnapshot with all indicators and signals.

    Raises:
        ValueError: If required columns are missing or DataFrame is empty.
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"OHLCV DataFrame missing columns: {missing}")
    if ohlcv.empty:
        raise ValueError("OHLCV DataFrame is empty")

    if settings is None:
        settings = get_settings().technicals

    close = ohlcv["Close"]
    high = ohlcv["High"]
    low = ohlcv["Low"]
    volume = ohlcv["Volume"]
    price = float(close.iloc[-1])

    # Moving averages
    sma_20 = compute_sma(close, 20)
    sma_50 = compute_sma(close, 50)
    sma_200 = compute_sma(close, 200)
    ema_9 = compute_ema(close, 9)
    ema_21 = compute_ema(close, 21)

    def _pct_vs(ma: pd.Series) -> float:
        val = ma.iloc[-1]
        if pd.isna(val) or val == 0:
            return 0.0
        return float((price - val) / val * 100)

    ma = MovingAverages(
        sma_20=float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else 0.0,
        sma_50=float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else 0.0,
        sma_200=float(sma_200.iloc[-1]) if not pd.isna(sma_200.iloc[-1]) else 0.0,
        ema_9=float(ema_9.iloc[-1]) if not pd.isna(ema_9.iloc[-1]) else 0.0,
        ema_21=float(ema_21.iloc[-1]) if not pd.isna(ema_21.iloc[-1]) else 0.0,
        price_vs_sma_20_pct=_pct_vs(sma_20),
        price_vs_sma_50_pct=_pct_vs(sma_50),
        price_vs_sma_200_pct=_pct_vs(sma_200),
    )

    # RSI
    rsi_series = compute_rsi(close, settings.rsi_period)
    rsi_val = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
    rsi = RSIData(
        value=rsi_val,
        is_overbought=rsi_val > settings.rsi_overbought,
        is_oversold=rsi_val < settings.rsi_oversold,
    )

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = compute_bollinger(
        close, settings.bollinger_window, settings.bollinger_std
    )
    bb_upper_val = float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else price
    bb_middle_val = float(bb_middle.iloc[-1]) if not pd.isna(bb_middle.iloc[-1]) else price
    bb_lower_val = float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else price
    bb_bw = bb_upper_val - bb_lower_val
    bb_range = bb_upper_val - bb_lower_val
    pct_b = (price - bb_lower_val) / bb_range if bb_range != 0 else 0.5
    bollinger = BollingerBands(
        upper=bb_upper_val,
        middle=bb_middle_val,
        lower=bb_lower_val,
        bandwidth=bb_bw / bb_middle_val if bb_middle_val != 0 else 0.0,
        percent_b=pct_b,
    )

    # MACD
    macd_line, signal_line, histogram = compute_macd(
        close, settings.macd_fast, settings.macd_slow, settings.macd_signal
    )
    macd_bullish, macd_bearish = _detect_macd_crossover(macd_line, signal_line)
    macd_data = MACDData(
        macd_line=float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
        signal_line=float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0,
        histogram=float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0,
        is_bullish_crossover=macd_bullish,
        is_bearish_crossover=macd_bearish,
    )

    # Stochastic
    stoch_k, stoch_d = compute_stochastic(
        high, low, close, settings.stochastic_k, settings.stochastic_d
    )
    stoch_k_val = float(stoch_k.iloc[-1]) if not pd.isna(stoch_k.iloc[-1]) else 50.0
    stoch_d_val = float(stoch_d.iloc[-1]) if not pd.isna(stoch_d.iloc[-1]) else 50.0
    stochastic = StochasticData(
        k=stoch_k_val,
        d=stoch_d_val,
        is_overbought=stoch_k_val > settings.stochastic_overbought,
        is_oversold=stoch_k_val < settings.stochastic_oversold,
    )

    # ATR
    atr_series = compute_atr(high, low, close, settings.atr_period)
    atr_val = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0
    atr_pct = atr_val / price * 100 if price != 0 else 0.0

    # VWMA
    vwma_series = compute_vwma(close, volume, settings.vwma_window)
    vwma_val = float(vwma_series.iloc[-1]) if not pd.isna(vwma_series.iloc[-1]) else price

    # Support / Resistance (reuse swing detection from phases)
    from market_regime.phases.price_structure import detect_swing_highs, detect_swing_lows
    from market_regime.config import PhaseSettings

    phase_settings = get_settings().phases
    swing_highs = detect_swing_highs(high, phase_settings.swing_lookback, phase_settings.swing_threshold_pct)
    swing_lows = detect_swing_lows(low, phase_settings.swing_lookback, phase_settings.swing_threshold_pct)

    support_price = swing_lows[-1].price if swing_lows else None
    resistance_price = swing_highs[-1].price if swing_highs else None

    sr = SupportResistance(
        support=support_price,
        resistance=resistance_price,
        price_vs_support_pct=(
            (price - support_price) / support_price * 100
            if support_price is not None and support_price != 0
            else None
        ),
        price_vs_resistance_pct=(
            (price - resistance_price) / resistance_price * 100
            if resistance_price is not None and resistance_price != 0
            else None
        ),
    )

    # Golden/Death cross
    golden_cross, death_cross = _detect_golden_death_cross(sma_50, sma_200)

    # Generate signals
    sma_200_val = float(sma_200.iloc[-1]) if not pd.isna(sma_200.iloc[-1]) else float("nan")
    signals = _generate_signals(
        price=price,
        rsi_val=rsi_val,
        macd_bullish=macd_bullish,
        macd_bearish=macd_bearish,
        golden_cross=golden_cross,
        death_cross=death_cross,
        bb_upper=bb_upper_val,
        bb_lower=bb_lower_val,
        stoch_k=stoch_k_val,
        sma_200=sma_200_val,
        vwma_val=vwma_val,
        settings=settings,
    )

    as_of = ohlcv.index[-1]
    as_of_date = as_of.date() if hasattr(as_of, "date") else as_of

    return TechnicalSnapshot(
        ticker=ticker,
        as_of_date=as_of_date,
        current_price=price,
        atr=atr_val,
        atr_pct=atr_pct,
        vwma_20=vwma_val,
        moving_averages=ma,
        rsi=rsi,
        bollinger=bollinger,
        macd=macd_data,
        stochastic=stochastic,
        support_resistance=sr,
        signals=signals,
    )
