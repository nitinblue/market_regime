"""Regime chart: price with colored regime bands and transition markers.

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

from market_regime.config import get_settings
from market_regime.data.service import DataService
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
    ohlcv_df,
    save: bool = False,
    phase_result: PhaseResult | None = None,
) -> None:
    """Two-panel chart: price with regime bands (top), confidence bars (bottom)."""
    settings = get_settings()
    plot_cfg = settings.display.plot
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

    # Align price to regime dates
    ohlcv_df = ohlcv_df.copy()
    ohlcv_df.index = ohlcv_df.index.date  # normalize to date objects
    prices = [ohlcv_df.loc[d, "Close"] if d in ohlcv_df.index else np.nan for d in dates]

    # Find regime transitions
    transitions = []
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i - 1]:
            transitions.append(i)

    # Create figure
    fig, (ax_price, ax_conf) = plt.subplots(
        2, 1,
        figsize=tuple(plot_cfg.figure_size),
        height_ratios=plot_cfg.height_ratios,
        sharex=True,
        gridspec_kw={"hspace": 0.05},
    )

    # --- Top panel: price + regime background bands ---
    ax_price.plot(dates, prices, color="black", linewidth=0.8, zorder=3)

    # Colored background bands per regime segment
    i = 0
    while i < len(dates):
        j = i + 1
        while j < len(dates) and regimes[j] == regimes[i]:
            j += 1
        color = regime_colors[regimes[i]]
        ax_price.axvspan(dates[i], dates[min(j, len(dates) - 1)], alpha=0.15, color=color, zorder=1)
        i = j

    # Transition markers
    for idx in transitions:
        d = dates[idx]
        ax_price.axvline(d, color="gray", linestyle="--", linewidth=0.7, alpha=0.7, zorder=2)
        ax_conf.axvline(d, color="gray", linestyle="--", linewidth=0.7, alpha=0.7, zorder=2)
        if not np.isnan(prices[idx]):
            ax_price.plot(d, prices[idx], marker="v", color=regime_colors[regimes[idx]],
                          markersize=6, zorder=4)

    ax_price.set_ylabel("Close Price")
    ax_price.set_title(f"{ticker} — Regime Detection", fontsize=13, fontweight="bold")
    ax_price.tick_params(axis="x", labelbottom=False)

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], color=c, linewidth=6, alpha=0.4, label=regime_labels[rid])
        for rid, c in regime_colors.items()
    ]
    ax_price.legend(
        handles=legend_handles, loc="upper left",
        fontsize=plot_cfg.font_size, framealpha=plot_cfg.legend_alpha,
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
                facecolor=get_settings().phases.colors.get(int(phase_result.phase), "#888888"),
                alpha=0.3,
            ),
        )

    # --- Bottom panel: confidence bars ---
    bar_colors = [regime_colors[r] for r in regimes]
    ax_conf.bar(dates, confidences, color=bar_colors, width=1.0, alpha=0.7)
    ax_conf.set_ylabel("Confidence")
    ax_conf.set_ylim(0, 1)
    ax_conf.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Format x-axis
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
