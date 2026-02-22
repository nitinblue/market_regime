"""Regime chart: price with colored regime bands, technical overlays,
volume, RSI, and confidence panels.

Usage (after pip install):
    regime-plot
    regime-plot --tickers AAPL MSFT
    regime-plot --tickers GLD --save
"""

import argparse
from datetime import date

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from market_regime.config import get_settings
from market_regime.data.service import DataService
from market_regime.features.technicals import (
    compute_bollinger,
    compute_rsi,
    compute_sma,
)
from market_regime.models.phase import PhaseResult
from market_regime.models.regime import RegimeExplanation, RegimeID
from market_regime.service.regime_service import RegimeService


def _regime_colors() -> dict[RegimeID, str]:
    """Build RegimeID -> color mapping from settings."""
    colors = get_settings().regimes.colors
    return {RegimeID(k): v for k, v in colors.items()}


def _regime_labels() -> dict[RegimeID, str]:
    """Build RegimeID -> label mapping from settings."""
    labels = get_settings().regimes.labels
    return {RegimeID(k): v for k, v in labels.items()}


def plot_ticker(
    ticker: str,
    explanation: RegimeExplanation,
    ohlcv_df: pd.DataFrame,
    save: bool = False,
    phase_result: PhaseResult | None = None,
) -> None:
    """Four-panel chart: price+technicals, volume, RSI, confidence."""
    settings = get_settings()
    plot_cfg = settings.display.plot
    tech_cfg = settings.technicals
    regime_colors = _regime_colors()
    regime_labels = _regime_labels()

    entries = explanation.regime_series.entries
    if not entries:
        print(f"  No regime series for {ticker}, skipping.")
        return

    # Build aligned arrays from regime series
    dates = [e.date for e in entries]
    regimes = [e.regime for e in entries]
    confidences = [e.confidence for e in entries]

    # Align OHLCV to regime dates
    ohlcv_df = ohlcv_df.copy()
    ohlcv_df.index = ohlcv_df.index.date  # normalize to date objects
    prices = [ohlcv_df.loc[d, "Close"] if d in ohlcv_df.index else np.nan for d in dates]

    # Build full aligned series for technicals (using ohlcv_df rows matching regime dates)
    close_series = pd.Series(prices, index=dates, dtype=float)
    open_vals = [ohlcv_df.loc[d, "Open"] if d in ohlcv_df.index else np.nan for d in dates]
    high_vals = [ohlcv_df.loc[d, "High"] if d in ohlcv_df.index else np.nan for d in dates]
    low_vals = [ohlcv_df.loc[d, "Low"] if d in ohlcv_df.index else np.nan for d in dates]
    vol_vals = [ohlcv_df.loc[d, "Volume"] if d in ohlcv_df.index else np.nan for d in dates]

    # Compute technical indicators from the full ohlcv (better lookback), then align
    full_close = ohlcv_df["Close"]
    sma50_full = compute_sma(full_close, 50)
    sma200_full = compute_sma(full_close, 200)
    bb_upper_full, _bb_mid, bb_lower_full = compute_bollinger(
        full_close, tech_cfg.bollinger_window, tech_cfg.bollinger_std,
    )
    rsi_full = compute_rsi(full_close, tech_cfg.rsi_period)

    # Align technicals to regime dates
    sma50 = [float(sma50_full.loc[d]) if d in sma50_full.index and not pd.isna(sma50_full.loc[d]) else np.nan for d in dates]
    sma200 = [float(sma200_full.loc[d]) if d in sma200_full.index and not pd.isna(sma200_full.loc[d]) else np.nan for d in dates]
    bb_upper = [float(bb_upper_full.loc[d]) if d in bb_upper_full.index and not pd.isna(bb_upper_full.loc[d]) else np.nan for d in dates]
    bb_lower = [float(bb_lower_full.loc[d]) if d in bb_lower_full.index and not pd.isna(bb_lower_full.loc[d]) else np.nan for d in dates]
    rsi_vals = [float(rsi_full.loc[d]) if d in rsi_full.index and not pd.isna(rsi_full.loc[d]) else np.nan for d in dates]

    # Volume MA (20-day)
    vol_series = pd.Series(vol_vals, index=dates, dtype=float)
    vol_ma = vol_series.rolling(20).mean()

    # Support / Resistance via swing detection
    from market_regime.phases.price_structure import detect_swing_highs, detect_swing_lows

    phase_cfg = settings.phases
    full_high = ohlcv_df["High"]
    full_low = ohlcv_df["Low"]
    swing_highs = detect_swing_highs(full_high, phase_cfg.swing_lookback, phase_cfg.swing_threshold_pct)
    swing_lows = detect_swing_lows(full_low, phase_cfg.swing_lookback, phase_cfg.swing_threshold_pct)

    current_price = prices[-1] if prices else np.nan
    # Support = nearest swing low BELOW current price
    support_price = next((s.price for s in reversed(swing_lows) if s.price < current_price), None)
    # Resistance = nearest swing high ABOVE current price
    resistance_price = next((s.price for s in reversed(swing_highs) if s.price > current_price), None)

    # Find regime transitions
    transitions = []
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i - 1]:
            transitions.append(i)

    # --- Create 4-panel figure ---
    fig, (ax_price, ax_vol, ax_rsi, ax_conf) = plt.subplots(
        4, 1,
        figsize=tuple(plot_cfg.figure_size),
        height_ratios=plot_cfg.height_ratios,
        sharex=True,
        gridspec_kw={"hspace": 0.05},
    )

    # ========== PANEL 1: Price + Technical Overlays ==========
    ax_price.plot(dates, prices, color="black", linewidth=0.8, zorder=3)

    # Regime-colored background bands
    i = 0
    while i < len(dates):
        j = i + 1
        while j < len(dates) and regimes[j] == regimes[i]:
            j += 1
        color = regime_colors[regimes[i]]
        ax_price.axvspan(dates[i], dates[min(j, len(dates) - 1)], alpha=0.15, color=color, zorder=1)
        i = j

    # Bollinger Bands
    ax_price.plot(dates, bb_upper, color="gray", linewidth=0.5, linestyle="--", zorder=2)
    ax_price.plot(dates, bb_lower, color="gray", linewidth=0.5, linestyle="--", zorder=2)
    ax_price.fill_between(dates, bb_upper, bb_lower, alpha=0.06, color="gray", zorder=1, label="Bollinger Band")

    # SMA 50 & 200
    ax_price.plot(dates, sma50, color="#2196F3", linewidth=0.7, zorder=2, label="SMA 50")
    ax_price.plot(dates, sma200, color="#F44336", linewidth=0.7, zorder=2, label="SMA 200")

    # Support / Resistance horizontal lines
    if support_price is not None:
        ax_price.axhline(support_price, color="#4CAF50", linestyle="--", linewidth=0.8, alpha=0.7, zorder=2)
        ax_price.text(dates[-1], support_price, f"  S: {support_price:.2f}",
                      fontsize=7, color="#4CAF50", va="top", ha="left", zorder=5)
    if resistance_price is not None:
        ax_price.axhline(resistance_price, color="#F44336", linestyle="--", linewidth=0.8, alpha=0.7, zorder=2)
        ax_price.text(dates[-1], resistance_price, f"  R: {resistance_price:.2f}",
                      fontsize=7, color="#F44336", va="bottom", ha="left", zorder=5)

    # Transition markers (extend through all panels)
    for idx in transitions:
        d = dates[idx]
        for ax in (ax_price, ax_vol, ax_rsi, ax_conf):
            ax.axvline(d, color="gray", linestyle="--", linewidth=0.7, alpha=0.7, zorder=2)
        if not np.isnan(prices[idx]):
            ax_price.plot(d, prices[idx], marker="v", color=regime_colors[regimes[idx]],
                          markersize=6, zorder=4)

    ax_price.set_ylabel("Close Price")
    ax_price.set_title(f"{ticker} — Regime Detection", fontsize=13, fontweight="bold")
    ax_price.tick_params(axis="x", labelbottom=False)

    # Legend: regime bands + technical overlays
    legend_handles = [
        plt.Line2D([0], [0], color=c, linewidth=6, alpha=0.4, label=regime_labels[rid])
        for rid, c in regime_colors.items()
    ]
    legend_handles.extend([
        plt.Line2D([0], [0], color="#2196F3", linewidth=0.7, label="SMA 50"),
        plt.Line2D([0], [0], color="#F44336", linewidth=0.7, label="SMA 200"),
        plt.Line2D([0], [0], color="gray", linewidth=0.5, linestyle="--", label="Bollinger Band"),
    ])
    if support_price is not None:
        legend_handles.append(plt.Line2D([0], [0], color="#4CAF50", linewidth=0.8, linestyle="--", label="Support"))
    if resistance_price is not None:
        legend_handles.append(plt.Line2D([0], [0], color="#F44336", linewidth=0.8, linestyle="--", label="Resistance"))

    ax_price.legend(
        handles=legend_handles, loc="upper left",
        fontsize=plot_cfg.font_size, framealpha=plot_cfg.legend_alpha, ncol=2,
    )

    # Phase annotation (upper right)
    if phase_result is not None:
        phase_text = f"Phase: {phase_result.phase_name} ({phase_result.confidence:.0%})"
        ax_price.text(
            0.98, 0.95, phase_text,
            transform=ax_price.transAxes,
            fontsize=plot_cfg.font_size + 1,
            fontweight="bold",
            ha="right", va="top",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=settings.phases.colors.get(int(phase_result.phase), "#888888"),
                alpha=0.3,
            ),
        )

    # ========== PANEL 2: Volume ==========
    vol_colors = [
        "#4CAF50" if (not np.isnan(prices[k]) and not np.isnan(open_vals[k]) and prices[k] >= open_vals[k])
        else "#F44336"
        for k in range(len(dates))
    ]
    ax_vol.bar(dates, vol_vals, color=vol_colors, width=1.0, alpha=0.7, zorder=2)
    ax_vol.plot(dates, vol_ma.values, color="#FF9800", linewidth=0.8, zorder=3, label="Vol MA(20)")
    ax_vol.set_ylabel("Volume")
    ax_vol.tick_params(axis="x", labelbottom=False)
    ax_vol.legend(loc="upper left", fontsize=plot_cfg.font_size, framealpha=plot_cfg.legend_alpha)

    # ========== PANEL 3: RSI ==========
    ax_rsi.plot(dates, rsi_vals, color="#7B1FA2", linewidth=0.8, zorder=3, label="RSI(14)")
    # Overbought / Oversold shading
    ax_rsi.fill_between(dates, 70, 100, alpha=0.08, color="red", zorder=1)
    ax_rsi.fill_between(dates, 0, 30, alpha=0.08, color="green", zorder=1)
    # Reference lines
    ax_rsi.axhline(70, color="red", linestyle="--", linewidth=0.5, alpha=0.6)
    ax_rsi.axhline(50, color="gray", linestyle="-", linewidth=0.4, alpha=0.5)
    ax_rsi.axhline(30, color="green", linestyle="--", linewidth=0.5, alpha=0.6)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.tick_params(axis="x", labelbottom=False)
    ax_rsi.legend(loc="upper left", fontsize=plot_cfg.font_size, framealpha=plot_cfg.legend_alpha)

    # ========== PANEL 4: Confidence (existing) ==========
    bar_colors = [regime_colors[r] for r in regimes]
    ax_conf.bar(dates, confidences, color=bar_colors, width=1.0, alpha=0.7)
    ax_conf.set_ylabel("Confidence")
    ax_conf.set_ylim(0, 1)
    ax_conf.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Format x-axis (only bottom panel shows labels)
    ax_conf.xaxis.set_major_locator(mdates.MonthLocator(interval=plot_cfg.month_interval))
    ax_conf.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=plot_cfg.xaxis_rotation)

    plt.tight_layout()

    if save:
        path = f"{ticker}_regime.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Plot regime detection chart")
    parser.add_argument(
        "--tickers", nargs="+", default=settings.display.default_tickers,
        help=f"Tickers to plot (default: {' '.join(settings.display.default_tickers)})",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save to PNG instead of showing interactive window",
    )
    args = parser.parse_args()

    print("Initializing services...")
    data_svc = DataService()
    regime_svc = RegimeService(data_service=data_svc)

    from market_regime.phases.detector import PhaseDetector

    phase_detector = PhaseDetector()
    for ticker in args.tickers:
        print(f"Processing {ticker}...")
        try:
            explanation = regime_svc.explain(ticker)
            ohlcv = data_svc.get_ohlcv(ticker)
            phase = phase_detector.detect(ticker, ohlcv, explanation.regime_series)
            plot_ticker(ticker, explanation, ohlcv, save=args.save, phase_result=phase)
        except Exception as e:
            print(f"  ERROR on {ticker}: {e}")


if __name__ == "__main__":
    main()
