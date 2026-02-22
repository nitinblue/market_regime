"""Interactive exploration harness for the regime detection pipeline.

Usage (after pip install):
    regime-explore
    regime-explore --tickers SPY GLD QQQ TLT
"""

import argparse

from tabulate import tabulate

from market_regime.config import get_settings
from market_regime.data.service import DataService
from market_regime.models.regime import (
    CrossTickerEntry,
    ResearchReport,
    TickerResearch,
)
from market_regime.service.regime_service import RegimeService


def print_section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def print_ticker_research(r: TickerResearch) -> None:
    print_section(f"{r.ticker} — Full Regime Explanation")

    print(f"\n{r.explanation_text}")

    # --- Transition Matrix ---
    print("\n--- Transition Matrix ---")
    rows = []
    for tr in r.transition_matrix:
        probs = {f"-> R{k}": f"{v:.3f}" for k, v in sorted(tr.to_probabilities.items())}
        comment = tr.stability
        if tr.likely_transition_target is not None:
            comment += f", often -> R{tr.likely_transition_target}"
        rows.append({"From": f"R{tr.from_regime}", **probs, "Comment": comment})
    print(tabulate(rows, headers="keys", tablefmt="simple", stralign="right"))

    # --- State Means ---
    print("\n--- State Means (feature means per regime) ---")
    rows = []
    for sm in r.state_means:
        row: dict[str, str] = {"Regime": f"R{sm.regime}"}
        for name, val in sm.feature_means.items():
            row[name] = f"{val:+.4f}"
        row["Comment"] = f"{sm.vol_character}, {sm.trend_character}"
        rows.append(row)
    print(tabulate(rows, headers="keys", tablefmt="simple", stralign="right"))

    # --- Label Alignment ---
    print("\n--- Label Alignment ---")
    rows = []
    for la in r.label_alignment:
        rows.append({
            "Regime": f"R{la.regime}",
            "Vol Mean": f"{la.vol_mean:+.4f}",
            "Trend Mean": f"{la.trend_mean:+.4f}",
            "Comment": f"{la.vol_side}-vol, {la.trend_side} (threshold: vol={la.vol_threshold:+.4f}, trend={la.trend_threshold:+.4f})",
        })
    print(tabulate(rows, headers="keys", tablefmt="simple", stralign="right"))

    # --- Current Feature Z-Scores ---
    print("\n--- Current Feature Z-Scores ---")
    rows = []
    for fz in r.current_features:
        rows.append({
            "Feature": fz.feature,
            "Z-Score": f"{fz.z_score:+.2f}",
            "Comment": fz.comment,
        })
    print(tabulate(rows, headers="keys", tablefmt="simple", stralign="right"))

    # --- Recent Regime History (last 20 days) ---
    print("\n--- Recent Regime History (last 20 days) ---")
    rows = []
    for day in r.recent_history:
        bar = "#" * int(day.confidence * 20)
        direction = ""
        if day.trend_direction:
            direction = "^" if day.trend_direction == "bullish" else "v"
        comment = ""
        if day.changed_from is not None:
            comment = f"<< changed from R{day.changed_from}"
        rows.append({
            "Date": str(day.date),
            "Regime": f"R{day.regime}",
            "Dir": direction,
            "Confidence": f"{min(day.confidence * 100, get_settings().display.confidence_cap):.1f}%",
            "Bar": bar,
            "Comment": comment,
        })
    print(tabulate(rows, headers="keys", tablefmt="simple", stralign="right"))

    # --- Regime Distribution ---
    print("\n--- Regime Distribution (full history) ---")
    rows = []
    for rd in r.regime_distribution:
        bar = "#" * int(rd.percentage / 2)
        comment = ""
        if rd.is_dominant:
            comment = "dominant regime"
        elif rd.is_rare:
            comment = "rare"
        rows.append({
            "Regime": f"R{rd.regime}",
            "Name": rd.name,
            "Days": rd.days,
            "Pct": f"{rd.percentage:.1f}%",
            "Bar": bar,
            "Comment": comment,
        })
    print(tabulate(rows, headers="keys", tablefmt="simple", stralign="right"))


def print_comparison(comparison: list[CrossTickerEntry]) -> None:
    print_section("Cross-Ticker Comparison")

    rows = []
    for c in comparison:
        probs = " ".join(
            f"R{k}:{v:.0%}" for k, v in sorted(c.regime_probabilities.items())
        )
        rows.append({
            "Ticker": c.ticker,
            "Regime": f"R{c.regime}",
            "Direction": c.trend_direction or "",
            "Confidence": f"{min(c.confidence * 100, get_settings().display.confidence_cap):.1f}%",
            "Probabilities": probs,
            "Comment": c.strategy_comment,
        })

    print(tabulate(rows, headers="keys", tablefmt="simple", stralign="right"))


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Explore regime detection pipeline")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=settings.display.default_tickers,
        help=f"Tickers to analyze (default: {' '.join(settings.display.default_tickers)})",
    )
    args = parser.parse_args()

    print_section("Regime Definitions")
    defs = []
    for rid in sorted(settings.regimes.names.keys()):
        defs.append({
            "Regime": f"R{rid}",
            "Name": settings.regimes.names[rid],
            "Strategy": settings.regimes.strategies.get(rid, ""),
        })
    print(tabulate(defs, headers="keys", tablefmt="simple", stralign="left"))

    print("\nInitializing services...")
    data_svc = DataService()
    regime_svc = RegimeService(data_service=data_svc)

    # Use the research API
    try:
        report = regime_svc.research_batch(tickers=args.tickers)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return

    for ticker_research in report.tickers:
        try:
            print_ticker_research(ticker_research)
        except Exception as e:
            print(f"\n  ERROR on {ticker_research.ticker}: {e}")

    if report.comparison:
        try:
            print_comparison(report.comparison)
        except Exception as e:
            print(f"\n  ERROR in comparison: {e}")

    print_section("Done")


if __name__ == "__main__":
    main()
