"""Technical indicator computation from OHLCV DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_regime.config import TechnicalsSettings, get_settings
from market_regime.models.technicals import (
    BollingerBands,
    MACDData,
    MarketPhase,
    MovingAverages,
    PhaseIndicator,
    RSIData,
    SignalDirection,
    SignalStrength,
    StochasticData,
    SupportResistance,
    TechnicalSignal,
    TechnicalSnapshot,
    VCPData,
    VCPStage,
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


def compute_vcp(
    ohlcv: pd.DataFrame,
    price: float,
    sma_50_val: float,
    sma_200_val: float,
    volume: pd.Series,
    settings: TechnicalsSettings,
) -> VCPData | None:
    """Detect Volatility Contraction Pattern (Minervini VCP).

    Scans swing highs/lows for successively tightening contractions,
    declining volume, and proximity to a pivot breakout level.

    Returns None if insufficient data for VCP analysis.
    """
    from market_regime.phases.price_structure import (
        compute_range_compression,
        compute_volume_trend,
        detect_swing_highs,
        detect_swing_lows,
    )

    lookback = settings.vcp_lookback_days
    if len(ohlcv) < lookback:
        return None

    window = ohlcv.iloc[-lookback:]
    phase_settings = get_settings().phases

    swing_highs = detect_swing_highs(
        window["High"], phase_settings.swing_lookback, phase_settings.swing_threshold_pct,
    )
    swing_lows = detect_swing_lows(
        window["Low"], phase_settings.swing_lookback, phase_settings.swing_threshold_pct,
    )

    # Shared computations used by both the no-pattern and full-pattern paths
    above_50 = price > sma_50_val if not np.isnan(sma_50_val) else False
    above_200 = price > sma_200_val if not np.isnan(sma_200_val) else False
    range_compression = compute_range_compression(
        window, phase_settings.range_analysis_window,
    )
    vol_trend = compute_volume_trend(
        volume.iloc[-lookback:],
        phase_settings.volume_trend_window,
        phase_settings.volume_decline_threshold,
    )

    def _no_pattern() -> VCPData:
        """Return a NONE-stage VCPData when no tightening sequence is found."""
        return VCPData(
            stage=VCPStage.NONE,
            contraction_count=0,
            contraction_pcts=[],
            current_range_pct=0.0,
            range_compression=range_compression,
            volume_trend=vol_trend,
            pivot_price=None,
            pivot_distance_pct=None,
            days_in_base=0,
            above_sma_50=above_50,
            above_sma_200=above_200,
            score=0.0,
            description="No VCP pattern detected.",
        )

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return _no_pattern()

    # Build contraction pairs: match each swing high with the nearest
    # subsequent swing low to form (high, low) contraction segments.
    contractions: list[tuple[float, float]] = []  # (high_price, low_price)
    contraction_dates: list = []  # date of swing high (start of contraction)
    low_idx = 0
    for sh in swing_highs:
        # Find the first swing low that comes after this swing high
        while low_idx < len(swing_lows) and swing_lows[low_idx].date <= sh.date:
            low_idx += 1
        if low_idx < len(swing_lows):
            sl = swing_lows[low_idx]
            contractions.append((sh.price, sl.price))
            contraction_dates.append(sh.date)

    if len(contractions) < 2:
        return _no_pattern()

    # Compute range % for each contraction
    contraction_pcts: list[float] = []
    for h, l in contractions:
        mid = (h + l) / 2
        if mid == 0:
            contraction_pcts.append(0.0)
        else:
            contraction_pcts.append((h - l) / mid * 100)

    # Find the longest tightening sequence ending at the most recent contraction.
    # Walk backward: each prior contraction must be wider.
    tightening_end = len(contraction_pcts) - 1
    tightening_start = tightening_end
    ratio = settings.vcp_tightening_ratio
    for i in range(tightening_end - 1, -1, -1):
        if contraction_pcts[i] > contraction_pcts[i + 1] * (1.0 / ratio):
            # Prior contraction is meaningfully wider — extend the sequence
            tightening_start = i
        else:
            break

    t_count = tightening_end - tightening_start + 1
    tightening_pcts = contraction_pcts[tightening_start:tightening_end + 1]
    current_range_pct = tightening_pcts[-1] if tightening_pcts else 0.0
    first_range_pct = tightening_pcts[0] if tightening_pcts else 0.0

    # Reject if first contraction is too small (no real base)
    if first_range_pct < settings.vcp_min_contraction_pct:
        t_count = 0
        tightening_pcts = []

    # Pivot = highest swing high in the tightening sequence
    if t_count >= settings.vcp_min_contractions:
        pivot_candidates = contractions[tightening_start:tightening_end + 1]
        pivot_price = max(h for h, _ in pivot_candidates)
    else:
        pivot_price = None

    # Pivot distance
    pivot_distance_pct: float | None = None
    if pivot_price is not None and pivot_price != 0:
        pivot_distance_pct = (pivot_price - price) / pivot_price * 100

    # Days in base
    if t_count >= settings.vcp_min_contractions and contraction_dates:
        first_date = contraction_dates[tightening_start]
        last_date = window.index[-1].date() if hasattr(window.index[-1], "date") else window.index[-1]
        days_in_base = (last_date - first_date).days
    else:
        days_in_base = 0

    # Volume surge check for breakout
    vol_avg_20 = float(volume.iloc[-20:].mean()) if len(volume) >= 20 else 0.0
    latest_vol = float(volume.iloc[-1]) if len(volume) > 0 else 0.0
    vol_surge = (vol_avg_20 > 0) and (latest_vol > 1.5 * vol_avg_20)

    # Classify stage
    if t_count < settings.vcp_min_contractions:
        stage = VCPStage.NONE
    elif (
        pivot_price is not None
        and price > pivot_price
        and vol_surge
    ):
        stage = VCPStage.BREAKOUT
    elif (
        current_range_pct <= settings.vcp_ready_range_pct
        and pivot_distance_pct is not None
        and pivot_distance_pct <= settings.vcp_pivot_proximity_pct
        and vol_trend == "declining"
    ):
        stage = VCPStage.READY
    elif t_count > settings.vcp_min_contractions or range_compression > 0.3:
        stage = VCPStage.MATURING
    else:
        stage = VCPStage.FORMING

    # Composite score (0–1)
    if stage == VCPStage.NONE:
        score = 0.0
    else:
        contraction_score = min(t_count / 4.0, 1.0) * 0.30
        tightness_score = (
            (1.0 - current_range_pct / first_range_pct) * 0.25
            if first_range_pct > 0
            else 0.0
        )
        tightness_score = max(tightness_score, 0.0)
        volume_score = (
            1.0 if vol_trend == "declining"
            else 0.5 if vol_trend == "stable"
            else 0.0
        ) * 0.20
        ma_score = (float(above_50) + float(above_200)) / 2.0 * 0.15
        proximity_score = (
            max(0.0, 1.0 - abs(pivot_distance_pct) / 10.0) * 0.10
            if pivot_distance_pct is not None
            else 0.0
        )
        score = contraction_score + tightness_score + volume_score + ma_score + proximity_score

    # Description
    if stage == VCPStage.NONE:
        description = "No VCP pattern detected."
    elif stage == VCPStage.BREAKOUT:
        description = (
            f"VCP breakout: price cleared pivot {pivot_price:.2f} on volume surge. "
            f"{t_count} contractions ({', '.join(f'{p:.1f}%' for p in tightening_pcts)})."
        )
    elif stage == VCPStage.READY:
        description = (
            f"VCP ready: {t_count} contractions tightening to {current_range_pct:.1f}% range, "
            f"declining volume, pivot at {pivot_price:.2f} ({abs(pivot_distance_pct or 0):.1f}% away)."
        )
    elif stage == VCPStage.MATURING:
        description = (
            f"VCP maturing: {t_count} contractions "
            f"({', '.join(f'{p:.1f}%' for p in tightening_pcts)}). "
            f"Volume {vol_trend}. Pivot at {pivot_price:.2f}."
            if pivot_price
            else f"VCP maturing: {t_count} contractions. Volume {vol_trend}."
        )
    else:
        description = (
            f"VCP forming: {t_count} contractions detected. "
            f"Range: {current_range_pct:.1f}%. Volume {vol_trend}."
        )

    return VCPData(
        stage=stage,
        contraction_count=t_count,
        contraction_pcts=tightening_pcts,
        current_range_pct=current_range_pct,
        range_compression=range_compression,
        volume_trend=vol_trend,
        pivot_price=pivot_price,
        pivot_distance_pct=pivot_distance_pct,
        days_in_base=days_in_base,
        above_sma_50=above_50,
        above_sma_200=above_200,
        score=score,
        description=description,
    )


def _generate_vcp_signals(vcp: VCPData | None) -> list[TechnicalSignal]:
    """Generate TechnicalSignal entries for VCP pattern."""
    if vcp is None or vcp.stage == VCPStage.NONE:
        return []

    signals: list[TechnicalSignal] = []

    if vcp.stage == VCPStage.BREAKOUT:
        signals.append(TechnicalSignal(
            name="VCP Breakout",
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.STRONG,
            description=vcp.description,
        ))
    elif vcp.stage == VCPStage.READY:
        signals.append(TechnicalSignal(
            name="VCP Ready",
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.STRONG,
            description=vcp.description,
        ))
    elif vcp.stage == VCPStage.MATURING:
        signals.append(TechnicalSignal(
            name="VCP Maturing",
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.MODERATE,
            description=vcp.description,
        ))
    elif vcp.stage == VCPStage.FORMING:
        signals.append(TechnicalSignal(
            name="VCP Forming",
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.WEAK,
            description=vcp.description,
        ))

    return signals


def _generate_phase_signals(phase: PhaseIndicator) -> list[TechnicalSignal]:
    """Generate TechnicalSignal entries for the detected market phase."""
    signals: list[TechnicalSignal] = []

    if phase.phase == MarketPhase.MARKUP:
        signals.append(TechnicalSignal(
            name="Phase: Markup",
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.STRONG if phase.confidence >= 0.65 else SignalStrength.MODERATE,
            description=phase.description,
        ))
    elif phase.phase == MarketPhase.MARKDOWN:
        signals.append(TechnicalSignal(
            name="Phase: Markdown",
            direction=SignalDirection.BEARISH,
            strength=SignalStrength.STRONG if phase.confidence >= 0.65 else SignalStrength.MODERATE,
            description=phase.description,
        ))
    elif phase.phase == MarketPhase.ACCUMULATION:
        signals.append(TechnicalSignal(
            name="Phase: Accumulation",
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.WEAK,
            description=phase.description,
        ))
    elif phase.phase == MarketPhase.DISTRIBUTION:
        signals.append(TechnicalSignal(
            name="Phase: Distribution",
            direction=SignalDirection.BEARISH,
            strength=SignalStrength.WEAK,
            description=phase.description,
        ))

    return signals


def compute_phase_indicator(
    ohlcv: pd.DataFrame,
    price_vs_sma_50: float,
) -> PhaseIndicator:
    """Classify market phase from price structure alone (no HMM required).

    Uses swing patterns, volume trend, range compression, and price vs SMA
    to determine Accumulation / Markup / Distribution / Markdown.

    This is a lightweight indicator. For regime-aware Wyckoff phase detection,
    use PhaseDetector from phases/detector.py via RegimeService.detect_phase().
    """
    from market_regime.phases.price_structure import compute_price_structure
    from market_regime.config import get_settings

    settings = get_settings().phases
    ps = compute_price_structure(ohlcv, settings)

    # Classification logic (mirrors phases/detector.py fallback paths)
    # Trending: HH+HL = Markup, LH+LL = Markdown
    # Range-bound: HL+compression+declining vol = Accumulation, LH+expansion = Distribution
    # Fallback: price vs SMA

    phase: MarketPhase
    description: str
    confidence = 0.50

    if ps.higher_highs and ps.higher_lows:
        phase = MarketPhase.MARKUP
        description = "Higher highs and higher lows — uptrend in progress."
        confidence = 0.70
        if ps.volume_trend == "rising":
            confidence += 0.10
            description += " Volume confirming."
    elif ps.lower_highs and ps.lower_lows:
        phase = MarketPhase.MARKDOWN
        description = "Lower highs and lower lows — downtrend in progress."
        confidence = 0.70
        if ps.volume_trend == "rising":
            confidence += 0.10
            description += " Panic volume."
    elif ps.higher_lows and not ps.higher_highs:
        # Higher lows without higher highs = base building
        phase = MarketPhase.ACCUMULATION
        description = "Higher lows forming — potential accumulation base."
        confidence = 0.55
        if ps.range_compression > 0.2:
            confidence += 0.10
            description += " Range compressing."
        if ps.volume_trend == "declining":
            confidence += 0.10
            description += " Volume drying up (supply exhaustion)."
    elif ps.lower_highs and not ps.lower_lows:
        # Lower highs without lower lows = topping
        phase = MarketPhase.DISTRIBUTION
        description = "Lower highs forming — potential distribution top."
        confidence = 0.55
        if ps.range_compression < -0.2:
            confidence += 0.05
            description += " Range expanding."
        if ps.volume_trend == "rising":
            confidence += 0.10
            description += " Rising volume on declines."
    else:
        # No clear swing pattern — use price vs SMA + range compression
        if price_vs_sma_50 < -2.0:
            phase = MarketPhase.ACCUMULATION
            description = f"Price {price_vs_sma_50:.1f}% below SMA(50) — possible accumulation."
            confidence = 0.40
        elif price_vs_sma_50 > 2.0:
            if ps.range_compression > 0.2:
                phase = MarketPhase.MARKUP
                description = f"Price {price_vs_sma_50:.1f}% above SMA(50), range tightening."
                confidence = 0.45
            else:
                phase = MarketPhase.DISTRIBUTION
                description = f"Price {price_vs_sma_50:.1f}% above SMA(50) — possible distribution."
                confidence = 0.40
        else:
            # Near SMA — use range compression as tiebreaker
            if ps.range_compression > 0.2:
                phase = MarketPhase.ACCUMULATION
                description = "Range compressing near SMA — possible base building."
                confidence = 0.35
            else:
                phase = MarketPhase.DISTRIBUTION
                description = "No clear directional pattern — consolidation/distribution."
                confidence = 0.30

    confidence = max(0.10, min(0.95, confidence))

    return PhaseIndicator(
        phase=phase,
        confidence=round(confidence, 2),
        description=description,
        higher_highs=ps.higher_highs,
        higher_lows=ps.higher_lows,
        lower_highs=ps.lower_highs,
        lower_lows=ps.lower_lows,
        range_compression=round(ps.range_compression, 3),
        volume_trend=ps.volume_trend,
        price_vs_sma_50_pct=round(price_vs_sma_50, 2),
    )


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

    # Support = nearest swing low BELOW current price (search backward)
    support_price = next(
        (s.price for s in reversed(swing_lows) if s.price < price), None
    )
    # Resistance = nearest swing high ABOVE current price (search backward)
    resistance_price = next(
        (s.price for s in reversed(swing_highs) if s.price > price), None
    )

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

    # VCP detection
    sma_50_val = float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else float("nan")
    vcp = compute_vcp(ohlcv, price, sma_50_val, sma_200_val, volume, settings)
    signals.extend(_generate_vcp_signals(vcp))

    # Phase indicator (price-structure-only, no HMM required)
    phase_indicator = compute_phase_indicator(ohlcv, _pct_vs(sma_50))

    # Phase signal
    signals.extend(_generate_phase_signals(phase_indicator))

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
        phase=phase_indicator,
        vcp=vcp,
        signals=signals,
    )
