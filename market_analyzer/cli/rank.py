"""Rank trade ideas across tickers and strategies.

Usage (after pip install):
    analyzer-rank
    analyzer-rank --tickers AAPL MSFT AMZN
    analyzer-rank --strategies zero_dte breakout
"""

import argparse

from tabulate import tabulate

from market_analyzer.config import get_settings
from market_analyzer.data.service import DataService
from market_analyzer.models.ranking import StrategyType
from market_analyzer.service.analyzer import MarketAnalyzer


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Rank trade ideas across tickers")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=settings.display.default_tickers,
        help=f"Tickers to rank (default: {' '.join(settings.display.default_tickers)})",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=[s.value for s in StrategyType],
        default=None,
        help="Strategy types to evaluate (default: all)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top trades to show (default: 10)",
    )
    args = parser.parse_args()

    strategies = (
        [StrategyType(s) for s in args.strategies] if args.strategies else None
    )

    print("Initializing services...")
    ma = MarketAnalyzer(data_service=DataService())

    print(f"Ranking {len(args.tickers)} tickers: {', '.join(args.tickers)}")
    if strategies:
        print(f"Strategies: {', '.join(s.value for s in strategies)}")
    print()

    try:
        result = ma.ranking.rank(args.tickers, strategies=strategies)
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # Black swan gate
    if result.black_swan_gate:
        print(f"BLACK SWAN GATE: {result.black_swan_level}")
        print(result.summary)
        return

    print(f"Alert level: {result.black_swan_level}")
    print()

    # Top trades table
    if result.top_trades:
        rows = []
        for entry in result.top_trades[: args.top]:
            rows.append({
                "Rank": f"#{entry.rank}",
                "Ticker": entry.ticker,
                "Strategy": entry.strategy_type.value,
                "Verdict": entry.verdict.value.upper(),
                "Score": f"{entry.composite_score:.3f}",
                "Specific": entry.strategy_name,
                "Direction": entry.direction,
                "Rationale": entry.rationale[:60],
            })
        print("--- Top Trades ---")
        print(tabulate(rows, headers="keys", tablefmt="simple", stralign="right"))

    # By ticker summary
    print("\n--- Best Per Ticker ---")
    ticker_rows = []
    for ticker in args.tickers:
        entries = result.by_ticker.get(ticker, [])
        if entries:
            best = entries[0]
            ticker_rows.append({
                "Ticker": ticker,
                "Best Strategy": best.strategy_type.value,
                "Score": f"{best.composite_score:.3f}",
                "Verdict": best.verdict.value.upper(),
                "Specific": best.strategy_name,
            })
        else:
            ticker_rows.append({
                "Ticker": ticker,
                "Best Strategy": "-",
                "Score": "-",
                "Verdict": "-",
                "Specific": "-",
            })
    print(tabulate(ticker_rows, headers="keys", tablefmt="simple", stralign="right"))

    # By strategy summary
    active_strategies = strategies or list(StrategyType)
    print("\n--- Best Per Strategy ---")
    strat_rows = []
    for strat in active_strategies:
        entries = result.by_strategy.get(strat, [])
        if entries:
            best = entries[0]
            strat_rows.append({
                "Strategy": strat.value,
                "Best Ticker": best.ticker,
                "Score": f"{best.composite_score:.3f}",
                "Verdict": best.verdict.value.upper(),
                "Specific": best.strategy_name,
            })
        else:
            strat_rows.append({
                "Strategy": strat.value,
                "Best Ticker": "-",
                "Score": "-",
                "Verdict": "-",
                "Specific": "-",
            })
    print(tabulate(strat_rows, headers="keys", tablefmt="simple", stralign="right"))

    # Score breakdown for #1
    if result.top_trades:
        best = result.top_trades[0]
        bd = best.breakdown
        print(f"\n--- Score Breakdown: #{best.rank} {best.ticker} {best.strategy_type.value} ---")
        print(f"  Verdict:           {bd.verdict_score:.2f}")
        print(f"  Confidence:        {bd.confidence_score:.2f}")
        print(f"  Regime alignment:  {bd.regime_alignment:.2f}")
        print(f"  Risk/Reward:       {bd.risk_reward:.2f}")
        print(f"  Technical quality: {bd.technical_quality:.2f}")
        print(f"  Phase alignment:   {bd.phase_alignment:.2f}")
        print(f"  Income bias:      +{bd.income_bias_boost:.2f}")
        print(f"  Macro penalty:    -{bd.macro_penalty:.2f}")
        print(f"  Earnings penalty: -{bd.earnings_penalty:.2f}")
        print(f"  Black swan:       x{1 - bd.black_swan_penalty:.2f}")
        if best.risk_notes:
            print(f"  Risk notes: {'; '.join(best.risk_notes)}")

    print(f"\n{result.summary}")


if __name__ == "__main__":
    main()
