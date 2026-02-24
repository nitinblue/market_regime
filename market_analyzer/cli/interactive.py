"""Interactive REPL for market_analyzer — Claude-like interface.

Usage:
    analyzer-cli
    analyzer-cli --market india
"""

from __future__ import annotations

import argparse
import cmd
import sys
import traceback
from datetime import date

from tabulate import tabulate


def _styled(text: str, style: str = "") -> str:
    """Basic ANSI styling. Falls back to plain text if terminal doesn't support it."""
    codes = {
        "bold": "\033[1m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "red": "\033[31m",
        "cyan": "\033[36m",
        "dim": "\033[2m",
        "reset": "\033[0m",
    }
    if not sys.stdout.isatty():
        return text
    prefix = codes.get(style, "")
    return f"{prefix}{text}{codes['reset']}" if prefix else text


def _print_header(title: str) -> None:
    print(f"\n{_styled('=' * 60, 'dim')}")
    print(f"  {_styled(title, 'bold')}")
    print(f"{_styled('=' * 60, 'dim')}")


class AnalyzerCLI(cmd.Cmd):
    """Interactive REPL for market analysis."""

    intro = (
        "\n"
        + _styled("market_analyzer", "bold")
        + " — interactive analysis REPL\n"
        + _styled("Type 'help' for commands, 'quit' to exit.", "dim")
        + "\n"
    )
    prompt = _styled("market_analyzer> ", "cyan") if sys.stdout.isatty() else "market_analyzer> "

    def __init__(self, market: str = "US") -> None:
        super().__init__()
        self._market = market
        self._ma = None  # Lazy-init

    def _get_ma(self):
        """Lazy-initialize MarketAnalyzer (avoids slow import on startup)."""
        if self._ma is None:
            print("Initializing services...")
            from market_analyzer import DataService, MarketAnalyzer
            self._ma = MarketAnalyzer(data_service=DataService(), market=self._market)
            print(_styled("Ready.", "green"))
        return self._ma

    def _parse_tickers(self, arg: str) -> list[str]:
        """Parse space-separated tickers from command argument."""
        return arg.upper().split() if arg.strip() else []

    # --- Commands ---

    def do_context(self, arg: str) -> None:
        """Show market environment assessment.\nUsage: context"""
        try:
            ma = self._get_ma()
            ctx = ma.context.assess()

            _print_header(f"Market Context — {ctx.market} ({ctx.as_of_date})")
            print(f"\n  Environment:  {_styled(ctx.environment_label, 'bold')}")
            print(f"  Trading:      {'ALLOWED' if ctx.trading_allowed else _styled('HALTED', 'red')}")
            print(f"  Size Factor:  {ctx.position_size_factor:.0%}")
            print(f"  Black Swan:   {ctx.black_swan.alert_level} (score: {ctx.black_swan.composite_score:.2f})")

            # Macro events
            events_7 = ctx.macro.events_next_7_days
            if events_7:
                print(f"\n  Macro events next 7 days:")
                for e in events_7:
                    print(f"    {e.date} | {e.name} ({e.impact})")

            # Intermarket
            if ctx.intermarket.entries:
                print(f"\n  Intermarket dashboard:")
                rows = []
                for entry in ctx.intermarket.entries:
                    rows.append({
                        "Ticker": entry.ticker,
                        "Regime": f"R{entry.regime}",
                        "Confidence": f"{entry.confidence:.0%}",
                        "Direction": entry.trend_direction or "",
                    })
                print(tabulate(rows, headers="keys", tablefmt="simple", stralign="right"))

            print(f"\n  {_styled(ctx.summary, 'dim')}")

        except Exception as exc:
            print(f"{_styled('ERROR:', 'red')} {exc}")

    def do_analyze(self, arg: str) -> None:
        """Show full instrument analysis.\nUsage: analyze SPY"""
        tickers = self._parse_tickers(arg)
        if not tickers:
            print("Usage: analyze TICKER [TICKER ...]")
            return

        ma = self._get_ma()
        for ticker in tickers:
            try:
                a = ma.instrument.analyze(ticker, include_opportunities=True)
                _print_header(f"{ticker} — Instrument Analysis ({a.as_of_date})")

                print(f"\n  Regime:       R{a.regime_id} ({a.regime.confidence:.0%})")
                print(f"  Phase:        {a.phase.phase_name} ({a.phase.confidence:.0%})")
                print(f"  Trend Bias:   {a.trend_bias}")
                print(f"  Volatility:   {a.volatility_label}")
                print(f"  Price:        ${a.technicals.current_price:.2f}")
                print(f"  RSI:          {a.technicals.rsi.value:.1f}")
                print(f"  ATR%:         {a.technicals.atr_pct:.2f}%")

                # Levels
                if a.levels:
                    print(f"\n  Levels:")
                    if a.levels.stop_loss:
                        print(f"    Stop:     ${a.levels.stop_loss.price:.2f} ({a.levels.stop_loss.distance_pct:+.1f}%)")
                    if a.levels.best_target:
                        print(f"    Target:   ${a.levels.best_target.price:.2f} (R:R {a.levels.best_target.risk_reward_ratio:.1f})")

                # Opportunities
                if a.actionable_setups:
                    print(f"\n  Actionable:   {', '.join(a.actionable_setups)}")

                print(f"\n  {_styled(a.summary, 'dim')}")

            except Exception as exc:
                print(f"{_styled('ERROR:', 'red')} {ticker}: {exc}")

    def do_screen(self, arg: str) -> None:
        """Screen tickers for setups.\nUsage: screen SPY GLD QQQ TLT"""
        tickers = self._parse_tickers(arg)
        if not tickers:
            from market_analyzer.config import get_settings
            tickers = get_settings().display.default_tickers
            print(f"Using default tickers: {' '.join(tickers)}")

        try:
            ma = self._get_ma()
            result = ma.screening.scan(tickers)
            _print_header(f"Screening Results ({result.as_of_date})")

            if not result.candidates:
                print("\n  No candidates found.")
            else:
                rows = []
                for c in result.candidates:
                    rows.append({
                        "Ticker": c.ticker,
                        "Screen": c.screen,
                        "Score": f"{c.score:.2f}",
                        "Regime": f"R{c.regime_id}",
                        "RSI": f"{c.rsi:.0f}",
                        "ATR%": f"{c.atr_pct:.2f}",
                        "Reason": c.reason[:60],
                    })
                print()
                print(tabulate(rows, headers="keys", tablefmt="simple", stralign="right"))

            print(f"\n  {_styled(result.summary, 'dim')}")

        except Exception as exc:
            print(f"{_styled('ERROR:', 'red')} {exc}")

    def do_entry(self, arg: str) -> None:
        """Confirm entry signal.\nUsage: entry SPY breakout"""
        parts = arg.strip().split()
        if len(parts) < 2:
            print("Usage: entry TICKER TRIGGER_TYPE")
            print("  Trigger types: breakout, pullback, momentum, mean_reversion, orb")
            return

        ticker = parts[0].upper()
        trigger_map = {
            "breakout": "breakout_confirmed",
            "pullback": "pullback_to_support",
            "momentum": "momentum_continuation",
            "mean_reversion": "mean_reversion_extreme",
            "orb": "orb_breakout",
        }
        trigger_name = trigger_map.get(parts[1].lower(), parts[1].lower())

        try:
            from market_analyzer.models.entry import EntryTriggerType
            trigger = EntryTriggerType(trigger_name)

            ma = self._get_ma()
            result = ma.entry.confirm(ticker, trigger)

            _print_header(f"{ticker} — Entry Confirmation ({result.trigger_type.value})")
            status = _styled("CONFIRMED", "green") if result.confirmed else _styled("NOT CONFIRMED", "red")
            print(f"\n  Status:      {status}")
            print(f"  Confidence:  {result.confidence:.0%}")
            print(f"  Conditions:  {result.conditions_met}/{result.conditions_total}")

            if result.suggested_entry_price:
                print(f"  Entry Price: ${result.suggested_entry_price:.2f}")
            if result.suggested_stop_price:
                print(f"  Stop Price:  ${result.suggested_stop_price:.2f}")
            if result.risk_per_share:
                print(f"  Risk/Share:  ${result.risk_per_share:.2f}")

            print("\n  Conditions:")
            for c in result.conditions:
                icon = _styled("+", "green") if c.met else _styled("-", "red")
                print(f"    {icon} {c.name}: {c.description}")

        except ValueError as exc:
            print(f"{_styled('ERROR:', 'red')} {exc}")
        except Exception as exc:
            print(f"{_styled('ERROR:', 'red')} {exc}")

    def do_strategy(self, arg: str) -> None:
        """Show strategy recommendation.\nUsage: strategy SPY"""
        tickers = self._parse_tickers(arg)
        if not tickers:
            print("Usage: strategy TICKER")
            return

        try:
            ma = self._get_ma()
            ticker = tickers[0]
            ohlcv = ma.data.get_ohlcv(ticker) if ma.data else None
            regime = ma.regime.detect(ticker, ohlcv)
            technicals = ma.technicals.snapshot(ticker, ohlcv)

            result = ma.strategy.select(ticker, regime=regime, technicals=technicals)

            _print_header(f"{ticker} — Strategy Recommendation")
            p = result.primary_structure
            print(f"\n  Structure:   {p.structure_type.value}")
            print(f"  Direction:   {p.direction}")
            print(f"  Max Loss:    {p.max_loss}")
            print(f"  Theta:       {p.theta_exposure}")
            print(f"  Vega:        {p.vega_exposure}")
            print(f"  DTE Range:   {result.suggested_dte_range[0]}-{result.suggested_dte_range[1]}")
            print(f"  Delta Range: {result.suggested_delta_range[0]:.0%}-{result.suggested_delta_range[1]:.0%}")
            if result.wing_width_suggestion:
                print(f"  Wing Width:  {result.wing_width_suggestion}")

            # Position size
            size = ma.strategy.size(result, current_price=technicals.current_price)
            print(f"\n  Position Sizing:")
            print(f"    Account:     ${size.account_size:,.0f}")
            print(f"    Max Risk:    ${size.max_risk_dollars:,.0f} ({size.max_risk_pct:.0f}%)")
            print(f"    Contracts:   {size.suggested_contracts} (max {size.max_contracts})")
            if size.margin_estimate:
                print(f"    Margin Est:  ${size.margin_estimate:,.0f}")

            print(f"\n  {_styled(result.regime_rationale, 'dim')}")

            if result.alternative_structures:
                print(f"\n  Alternatives:")
                for alt in result.alternative_structures:
                    print(f"    - {alt.structure_type.value} ({alt.direction}): {alt.rationale}")

        except Exception as exc:
            print(f"{_styled('ERROR:', 'red')} {exc}")

    def do_exit_plan(self, arg: str) -> None:
        """Show exit plan for a position.\nUsage: exit_plan SPY 580"""
        parts = arg.strip().split()
        if len(parts) < 2:
            print("Usage: exit_plan TICKER ENTRY_PRICE")
            return

        ticker = parts[0].upper()
        try:
            entry_price = float(parts[1])
        except ValueError:
            print("Entry price must be a number.")
            return

        try:
            ma = self._get_ma()
            ohlcv = ma.data.get_ohlcv(ticker) if ma.data else None
            regime = ma.regime.detect(ticker, ohlcv)
            technicals = ma.technicals.snapshot(ticker, ohlcv)
            levels = ma.levels.analyze(ticker, ohlcv=ohlcv)
            strategy = ma.strategy.select(ticker, regime=regime, technicals=technicals)

            plan = ma.exit.plan(
                ticker, strategy, entry_price=entry_price,
                regime=regime, technicals=technicals, levels=levels,
            )

            _print_header(f"{ticker} — Exit Plan ({plan.strategy_type} @ ${entry_price:.2f})")

            if plan.profit_targets:
                print(f"\n  Profit Targets:")
                for t in plan.profit_targets:
                    print(f"    ${t.price:.2f} ({t.pct_from_entry:+.1f}%) — {t.action}: {t.description}")

            if plan.stop_loss:
                print(f"\n  Stop Loss:")
                print(f"    ${plan.stop_loss.price:.2f} ({plan.stop_loss.pct_from_entry:+.1f}%) — {plan.stop_loss.description}")

            if plan.trailing_stop:
                print(f"\n  Trailing Stop:")
                print(f"    ${plan.trailing_stop.price:.2f} — {plan.trailing_stop.description}")

            if plan.risk_reward_ratio:
                print(f"\n  R:R Ratio:   {plan.risk_reward_ratio:.1f}")

            if plan.dte_exit_threshold:
                print(f"  Time Exit:   Close at {plan.dte_exit_threshold} DTE")
            if plan.theta_decay_exit_pct:
                print(f"  Theta Exit:  Close at {plan.theta_decay_exit_pct:.0f}% max profit")

            if plan.adjustments:
                print(f"\n  Adjustments:")
                for adj in plan.adjustments:
                    print(f"    [{adj.urgency}] {adj.condition}")
                    print(f"      → {adj.action}")

            print(f"\n  Regime Change: {plan.regime_change_action}")

        except Exception as exc:
            print(f"{_styled('ERROR:', 'red')} {exc}")

    def do_rank(self, arg: str) -> None:
        """Rank trades across tickers.\nUsage: rank SPY GLD QQQ TLT"""
        tickers = self._parse_tickers(arg)
        if not tickers:
            from market_analyzer.config import get_settings
            tickers = get_settings().display.default_tickers

        try:
            ma = self._get_ma()
            result = ma.ranking.rank(tickers)
            _print_header(f"Trade Ranking ({result.as_of_date})")

            if result.black_swan_gate:
                print(f"\n  {_styled('TRADING HALTED — Black Swan CRITICAL', 'red')}")

            if result.top_trades:
                rows = []
                for e in result.top_trades[:10]:
                    rows.append({
                        "#": e.rank,
                        "Ticker": e.ticker,
                        "Strategy": e.strategy_type,
                        "Verdict": e.verdict,
                        "Score": f"{e.composite_score:.2f}",
                        "Direction": e.direction,
                        "Rationale": e.rationale[:50],
                    })
                print()
                print(tabulate(rows, headers="keys", tablefmt="simple", stralign="right"))

            print(f"\n  {_styled(result.summary, 'dim')}")

        except Exception as exc:
            print(f"{_styled('ERROR:', 'red')} {exc}")

    def do_regime(self, arg: str) -> None:
        """Detect regime for ticker(s).\nUsage: regime SPY GLD"""
        tickers = self._parse_tickers(arg)
        if not tickers:
            from market_analyzer.config import get_settings
            tickers = get_settings().display.default_tickers

        try:
            ma = self._get_ma()
            results = ma.regime.detect_batch(tickers=tickers)
            _print_header("Regime Detection")
            rows = []
            for t, r in results.items():
                rows.append({
                    "Ticker": t,
                    "Regime": f"R{r.regime}",
                    "Confidence": f"{r.confidence:.0%}",
                    "Direction": r.trend_direction or "",
                    "Date": str(r.as_of_date),
                })
            print()
            print(tabulate(rows, headers="keys", tablefmt="simple", stralign="right"))

        except Exception as exc:
            print(f"{_styled('ERROR:', 'red')} {exc}")

    def do_technicals(self, arg: str) -> None:
        """Show technical snapshot.\nUsage: technicals SPY"""
        tickers = self._parse_tickers(arg)
        if not tickers:
            print("Usage: technicals TICKER")
            return

        try:
            ma = self._get_ma()
            ticker = tickers[0]
            t = ma.technicals.snapshot(ticker)

            _print_header(f"{ticker} — Technical Snapshot ({t.as_of_date})")
            print(f"\n  Price:       ${t.current_price:.2f}")
            print(f"  RSI:         {t.rsi.value:.1f} {'(OB)' if t.rsi.is_overbought else '(OS)' if t.rsi.is_oversold else ''}")
            print(f"  ATR:         ${t.atr:.2f} ({t.atr_pct:.2f}%)")
            print(f"  MACD:        {t.macd.histogram:+.4f} {'↑' if t.macd.is_bullish_crossover else '↓' if t.macd.is_bearish_crossover else ''}")

            ma_data = t.moving_averages
            print(f"\n  Moving Averages:")
            print(f"    SMA 20:    ${ma_data.sma_20:.2f} ({ma_data.price_vs_sma_20_pct:+.1f}%)")
            print(f"    SMA 50:    ${ma_data.sma_50:.2f} ({ma_data.price_vs_sma_50_pct:+.1f}%)")
            print(f"    SMA 200:   ${ma_data.sma_200:.2f} ({ma_data.price_vs_sma_200_pct:+.1f}%)")

            print(f"\n  Bollinger:   BW={t.bollinger.bandwidth:.4f}, %B={t.bollinger.percent_b:.2f}")
            print(f"  Stochastic:  K={t.stochastic.k:.0f}, D={t.stochastic.d:.0f}")
            print(f"  Phase:       {t.phase.phase.value} ({t.phase.confidence:.0%})")

            if t.signals:
                print(f"\n  Signals:")
                for s in t.signals[:5]:
                    print(f"    [{s.direction}] {s.name}: {s.description}")

        except Exception as exc:
            print(f"{_styled('ERROR:', 'red')} {exc}")

    def do_levels(self, arg: str) -> None:
        """Show support/resistance levels.\nUsage: levels SPY"""
        tickers = self._parse_tickers(arg)
        if not tickers:
            print("Usage: levels TICKER")
            return

        try:
            ma = self._get_ma()
            ticker = tickers[0]
            result = ma.levels.analyze(ticker)
            _print_header(f"{ticker} — Levels Analysis ({result.as_of_date})")
            print(f"\n{result.summary}")

        except Exception as exc:
            print(f"{_styled('ERROR:', 'red')} {exc}")

    def do_macro(self, arg: str) -> None:
        """Show macro economic calendar.\nUsage: macro"""
        try:
            ma = self._get_ma()
            cal = ma.macro.calendar()
            _print_header("Macro Calendar")

            if cal.next_event:
                print(f"\n  Next event:  {cal.next_event.name} ({cal.next_event.date}) — {cal.days_to_next}d")
            if cal.next_fomc:
                print(f"  Next FOMC:   {cal.next_fomc.date} — {cal.days_to_next_fomc}d")

            if cal.events_next_30_days:
                print(f"\n  Next 30 days:")
                rows = []
                for e in cal.events_next_30_days:
                    rows.append({
                        "Date": str(e.date),
                        "Event": e.name,
                        "Impact": e.impact,
                    })
                print(tabulate(rows, headers="keys", tablefmt="simple", stralign="right"))

        except Exception as exc:
            print(f"{_styled('ERROR:', 'red')} {exc}")

    def do_stress(self, arg: str) -> None:
        """Show black swan / tail-risk alert.\nUsage: stress"""
        try:
            ma = self._get_ma()
            alert = ma.black_swan.alert()
            _print_header(f"Tail-Risk Alert ({alert.as_of_date})")

            level_color = {
                "normal": "green",
                "elevated": "yellow",
                "high": "yellow",
                "critical": "red",
            }
            color = level_color.get(alert.alert_level, "")
            print(f"\n  Alert:   {_styled(alert.alert_level.upper(), color)}")
            print(f"  Score:   {alert.composite_score:.2f}")
            print(f"  Action:  {alert.action}")

            if alert.indicators:
                print(f"\n  Indicators:")
                rows = []
                for ind in alert.indicators:
                    rows.append({
                        "Name": ind.name,
                        "Status": ind.status,
                        "Score": f"{ind.score:.2f}",
                        "Value": f"{ind.value:.2f}" if ind.value is not None else "N/A",
                    })
                print(tabulate(rows, headers="keys", tablefmt="simple", stralign="right"))

            if alert.triggered_breakers > 0:
                print(f"\n  {_styled(f'{alert.triggered_breakers} circuit breaker(s) triggered!', 'red')}")

            print(f"\n  {_styled(alert.summary, 'dim')}")

        except Exception as exc:
            print(f"{_styled('ERROR:', 'red')} {exc}")

    def do_quit(self, arg: str) -> bool:
        """Exit the REPL."""
        print("Goodbye.")
        return True

    def do_exit(self, arg: str) -> bool:
        """Exit the REPL."""
        return self.do_quit(arg)

    do_EOF = do_quit

    def default(self, line: str) -> None:
        """Handle unknown commands."""
        print(f"Unknown command: '{line}'. Type 'help' for available commands.")

    def emptyline(self) -> None:
        """Do nothing on empty line (don't repeat last command)."""
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive market analyzer REPL")
    parser.add_argument(
        "--market",
        default="US",
        choices=["US", "India"],
        help="Default market (default: US)",
    )
    args = parser.parse_args()

    try:
        cli = AnalyzerCLI(market=args.market)
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
