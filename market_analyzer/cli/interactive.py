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


def _profile_tag(structure_type: str | None, order_side: str | None = None,
                 direction: str | None = None) -> str:
    """Format a compact profile tag: '/‾‾\\ neutral · defined'."""
    if not structure_type:
        return ""
    from market_analyzer.models.opportunity import get_structure_profile, RiskProfile
    p = get_structure_profile(structure_type, order_side, direction)
    risk_str = (
        _styled("UNDEFINED", "red") if p.risk_profile == RiskProfile.UNDEFINED
        else _styled("defined", "green")
    )
    return f"{p.payoff_graph} {p.bias} · {risk_str}"


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
                from market_analyzer.models.opportunity import get_structure_profile, RiskProfile
                rows = []
                for e in result.top_trades[:10]:
                    legs_str = ""
                    exit_str = ""
                    graph = ""
                    risk = ""
                    if e.trade_spec is not None:
                        legs_str = " | ".join(e.trade_spec.leg_codes[:2])
                        if len(e.trade_spec.leg_codes) > 2:
                            legs_str += " ..."
                        exit_str = e.trade_spec.exit_summary
                        if e.trade_spec.structure_type:
                            p = get_structure_profile(
                                e.trade_spec.structure_type,
                                e.trade_spec.order_side,
                                e.direction,
                            )
                            graph = p.payoff_graph
                            risk = p.risk_profile.value.upper() if p.risk_profile == RiskProfile.UNDEFINED else p.risk_profile.value
                    rows.append({
                        "#": e.rank,
                        "Ticker": e.ticker,
                        "Strategy": e.strategy_type,
                        "Payoff": graph or "—",
                        "Bias": e.direction,
                        "Risk": risk or "—",
                        "Verdict": e.verdict,
                        "Score": f"{e.composite_score:.2f}",
                        "Legs": legs_str or "—",
                        "Exit": exit_str or "—",
                    })
                print()
                print(tabulate(rows, headers="keys", tablefmt="simple", stralign="right"))

            print(f"\n  {_styled(result.summary, 'dim')}")

        except Exception as exc:
            print(f"{_styled('ERROR:', 'red')} {exc}")

    def do_plan(self, arg: str) -> None:
        """Generate daily trading plan.\nUsage: plan [TICKER ...] [--date YYYY-MM-DD]"""
        parts = arg.strip().split()
        tickers: list[str] = []
        plan_date = None

        i = 0
        while i < len(parts):
            if parts[i] == "--date" and i + 1 < len(parts):
                try:
                    plan_date = date.fromisoformat(parts[i + 1])
                except ValueError:
                    print(f"Invalid date: {parts[i + 1]}")
                    return
                i += 2
            else:
                tickers.append(parts[i].upper())
                i += 1

        try:
            ma = self._get_ma()
            plan = ma.plan.generate(
                tickers=tickers or None,
                plan_date=plan_date,
            )

            # Header
            day_str = plan.plan_for_date.strftime("%a %b %d, %Y")
            _print_header(f"Daily Trading Plan — {day_str}")

            # Day verdict
            verdict_color = {
                "trade": "green", "trade_light": "yellow",
                "avoid": "red", "no_trade": "red",
            }
            vc = verdict_color.get(plan.day_verdict, "")
            print(f"\n  Day: {_styled(plan.day_verdict.value.upper().replace('_', ' '), vc)}")
            if plan.day_verdict_reasons:
                for r in plan.day_verdict_reasons:
                    print(f"       {_styled(r, 'dim')}")

            # Risk budget
            b = plan.risk_budget
            print(f"  Risk: max {b.max_new_positions} new positions | "
                  f"${b.max_daily_risk_dollars:,.0f} daily risk budget | "
                  f"sizing {b.position_size_factor:.0%}")

            # Expiry events
            if plan.expiry_events:
                labels = [f"{e.label} ({e.date})" for e in plan.expiry_events]
                print(f"  Expiry: {', '.join(labels)}")
            if plan.upcoming_expiries:
                future = [e for e in plan.upcoming_expiries if e.date > plan.plan_for_date]
                if future:
                    nxt = future[0]
                    print(f"  Next: {nxt.label} ({nxt.date})")

            if not plan.all_trades:
                print(f"\n  {_styled('No actionable trades.', 'dim')}")
            else:
                # Group by horizon
                from market_analyzer.models.trading_plan import PlanHorizon
                horizon_labels = {
                    PlanHorizon.ZERO_DTE: "0DTE",
                    PlanHorizon.WEEKLY: "Weekly",
                    PlanHorizon.MONTHLY: "Monthly",
                    PlanHorizon.LEAP: "LEAP",
                }
                for h in PlanHorizon:
                    trades = plan.trades_by_horizon.get(h, [])
                    if not trades:
                        continue
                    print(f"\n  {_styled(f'--- {horizon_labels[h]} ({len(trades)} trades) ---', 'bold')}")
                    for t in trades:
                        v_color = {"go": "green", "caution": "yellow"}.get(t.verdict, "")
                        v_text = _styled(t.verdict.value.upper(), v_color)

                        legs_str = ""
                        exit_str = ""
                        st_type = None
                        side = None
                        if t.trade_spec is not None:
                            legs_str = " | ".join(t.trade_spec.leg_codes)
                            exit_str = t.trade_spec.exit_summary
                            st_type = t.trade_spec.structure_type
                            side = t.trade_spec.order_side

                        tag = _profile_tag(st_type, side, t.direction)
                        print(f"  #{t.rank} {_styled(t.ticker, 'bold')}  {t.strategy_type}  "
                              f"{v_text}  {t.composite_score:.2f}")
                        if tag:
                            print(f"     {tag}")
                        if legs_str:
                            print(f"     {legs_str}")
                        # Max profit / max loss
                        if t.trade_spec:
                            mp = t.trade_spec.max_profit_desc or ""
                            ml = t.trade_spec.max_loss_desc or ""
                            if mp or ml:
                                print(f"     Max profit: {mp} | Max loss: {ml}")
                            if exit_str:
                                print(f"     {exit_str}")
                        # Chase limit
                        if t.max_entry_price is not None:
                            print(f"     Chase limit: ${t.max_entry_price:.2f}")
                        # Expiry note
                        if t.expiry_note:
                            print(f"     {_styled(f'NOTE: {t.expiry_note}', 'yellow')}")

            print(f"\n  {_styled(plan.summary, 'dim')}")

        except Exception as exc:
            print(f"{_styled('ERROR:', 'red')} {exc}")
            if "--debug" in arg:
                traceback.print_exc()

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

    def do_vol(self, arg: str) -> None:
        """Show volatility surface.\nUsage: vol SPY"""
        tickers = self._parse_tickers(arg)
        if not tickers:
            print("Usage: vol TICKER")
            return

        try:
            ma = self._get_ma()
            ticker = tickers[0]
            surf = ma.vol_surface.surface(ticker)

            _print_header(f"{ticker} — Volatility Surface ({surf.as_of_date})")
            print(f"\n  Underlying:  ${surf.underlying_price:.2f}")
            print(f"  Front IV:    {surf.front_iv:.1%}")
            print(f"  Back IV:     {surf.back_iv:.1%}")
            print(f"  Term Slope:  {surf.term_slope:+.1%} ({'contango' if surf.is_contango else 'backwardation'})")
            print(f"  Calendar Edge: {surf.calendar_edge_score:.2f}")
            print(f"  Data Quality:  {surf.data_quality}")

            if surf.term_structure:
                print(f"\n  Term Structure:")
                rows = []
                for pt in surf.term_structure:
                    rows.append({
                        "Expiry": str(pt.expiration),
                        "DTE": pt.days_to_expiry,
                        "ATM IV": f"{pt.atm_iv:.1%}",
                        "Strike": f"${pt.atm_strike:.0f}",
                    })
                print(tabulate(rows, headers="keys", tablefmt="simple", stralign="right"))

            if surf.skew_by_expiry:
                print(f"\n  Skew (front expiry):")
                sk = surf.skew_by_expiry[0]
                print(f"    ATM IV:     {sk.atm_iv:.1%}")
                print(f"    OTM Put IV: {sk.otm_put_iv:.1%} (skew: +{sk.put_skew:.1%})")
                print(f"    OTM Call IV:{sk.otm_call_iv:.1%} (skew: +{sk.call_skew:.1%})")
                print(f"    Skew Ratio: {sk.skew_ratio:.1f}")

            if surf.best_calendar_expiries:
                f, b = surf.best_calendar_expiries
                print(f"\n  Best Calendar: sell {f} / buy {b} (diff: {surf.iv_differential_pct:+.1f}%)")

            print(f"\n  {_styled(surf.summary, 'dim')}")

        except Exception as exc:
            print(f"{_styled('ERROR:', 'red')} {exc}")

    def do_setup(self, arg: str) -> None:
        """Assess price-based setups (breakout, momentum, mean_reversion, orb).\nUsage: setup SPY [type]\n  Types: breakout, momentum, mr (mean_reversion), orb, all (default)\n  Note: ORB requires intraday data; shows NO_GO without it."""
        parts = arg.strip().split()
        if not parts:
            print("Usage: setup TICKER [type]")
            print("  Types: breakout, momentum, mr (mean_reversion), orb, all (default)")
            return

        ticker = parts[0].upper()
        setup_type = parts[1].lower() if len(parts) > 1 else "all"

        type_map = {
            "breakout": ["breakout"],
            "momentum": ["momentum"],
            "mr": ["mean_reversion"],
            "mean_reversion": ["mean_reversion"],
            "orb": ["orb"],
            "all": ["breakout", "momentum", "mean_reversion", "orb"],
        }
        setups = type_map.get(setup_type)
        if setups is None:
            print(f"Unknown setup: '{setup_type}'. Use: breakout, momentum, mr, orb, all")
            return

        try:
            ma = self._get_ma()
            _print_header(f"{ticker} — Setup Assessment")

            for s in setups:
                try:
                    if s == "breakout":
                        result = ma.opportunity.assess_breakout(ticker)
                    elif s == "momentum":
                        result = ma.opportunity.assess_momentum(ticker)
                    elif s == "mean_reversion":
                        result = ma.opportunity.assess_mean_reversion(ticker)
                    elif s == "orb":
                        # ORB without intraday data → NO_GO (expected)
                        from market_analyzer.opportunity.setups.orb import assess_orb as _orb_assess
                        regime = ma.regime.detect(ticker)
                        technicals = ma.technicals.snapshot(ticker)
                        result = _orb_assess(ticker, regime, technicals, orb=None)
                    else:
                        continue

                    verdict_color = {"go": "green", "caution": "yellow", "no_go": "red"}
                    v = result.verdict if isinstance(result.verdict, str) else result.verdict.value
                    v_color = verdict_color.get(v, "")
                    v_text = _styled(v.upper(), v_color)

                    name = s.replace("_", " ").title()
                    conf = result.confidence if hasattr(result, "confidence") else 0
                    print(f"\n  {_styled(name, 'bold')}: {v_text} ({conf:.0%})")

                    if hasattr(result, "hard_stops") and result.hard_stops:
                        for hs in result.hard_stops[:2]:
                            print(f"    {_styled('STOP:', 'red')} {hs.description}")

                    if hasattr(result, "direction") and result.direction != "neutral":
                        print(f"    Direction: {result.direction.title()}")
                    if hasattr(result, "strategy") and isinstance(result.strategy, str):
                        print(f"    Strategy:  {result.strategy.replace('_', ' ').title()}")

                    # ORB-specific fields
                    if hasattr(result, "orb_status") and result.orb_status != "none":
                        print(f"    ORB Status: {result.orb_status}")
                        print(f"    Range:     {result.range_pct:.2f}%")
                        if result.range_vs_daily_atr_pct is not None:
                            print(f"    Range/ATR: {result.range_vs_daily_atr_pct:.0f}%")

                    if hasattr(result, "signals"):
                        for sig in result.signals[:3]:
                            icon = _styled("+", "green") if sig.favorable else _styled("-", "red")
                            desc = sig.description if isinstance(sig.description, str) else str(sig.description)
                            print(f"    {icon} {desc[:70]}")

                    # Trade spec (actionable parameters)
                    if hasattr(result, "trade_spec") and result.trade_spec is not None:
                        ts = result.trade_spec
                        direction = getattr(result, "direction", None)
                        if direction is None and hasattr(result, "strategy") and hasattr(result.strategy, "direction"):
                            direction = result.strategy.direction
                        if ts.structure_type:
                            tag = _profile_tag(ts.structure_type, ts.order_side, direction)
                            print(f"    {tag}")
                        print(f"    Legs:")
                        for code in ts.leg_codes:
                            print(f"      {code}")
                        if ts.exit_summary:
                            print(f"    Exit:      {ts.exit_summary}")

                    print(f"    {_styled(result.summary, 'dim')}")

                except Exception as exc:
                    print(f"\n  {s.replace('_', ' ').title()}: {_styled(str(exc), 'red')}")

        except Exception as exc:
            print(f"{_styled('ERROR:', 'red')} {exc}")

    def do_opportunity(self, arg: str) -> None:
        """Assess option play opportunities.\nUsage: opportunity SPY [play]\n  Plays: ic, ifly, calendar, diagonal, ratio, zero_dte, leap, earnings, all\n  Default: all"""
        parts = arg.strip().split()
        if not parts:
            print("Usage: opportunity TICKER [play]")
            print("  Plays: ic (iron condor), ifly (iron butterfly), calendar, diagonal,")
            print("         ratio (ratio spread), zero_dte, leap, earnings, all (default)")
            return

        ticker = parts[0].upper()
        play = parts[1].lower() if len(parts) > 1 else "all"

        play_map = {
            "ic": ["iron_condor"],
            "iron_condor": ["iron_condor"],
            "ifly": ["iron_butterfly"],
            "iron_butterfly": ["iron_butterfly"],
            "calendar": ["calendar"],
            "cal": ["calendar"],
            "diagonal": ["diagonal"],
            "diag": ["diagonal"],
            "ratio": ["ratio_spread"],
            "ratio_spread": ["ratio_spread"],
            "zero_dte": ["zero_dte"],
            "0dte": ["zero_dte"],
            "leap": ["leap"],
            "earnings": ["earnings"],
            "all": ["iron_condor", "iron_butterfly", "calendar", "diagonal", "ratio_spread", "earnings"],
        }
        plays = play_map.get(play)
        if plays is None:
            print(f"Unknown play: '{play}'. Use: ic, ifly, calendar, diagonal, ratio, zero_dte, leap, earnings, all")
            return

        try:
            ma = self._get_ma()
            _print_header(f"{ticker} — Option Play Assessment")

            for p in plays:
                try:
                    method = getattr(ma.opportunity, f"assess_{p}")
                    result = method(ticker)

                    verdict_color = {"go": "green", "caution": "yellow", "no_go": "red"}
                    v_color = verdict_color.get(result.verdict.value, "")
                    v_text = _styled(result.verdict.value.upper(), v_color)

                    name = p.replace("_", " ").title()
                    print(f"\n  {_styled(name, 'bold')}: {v_text} ({result.confidence:.0%})")

                    if result.hard_stops:
                        for hs in result.hard_stops[:2]:
                            print(f"    {_styled('STOP:', 'red')} {hs.description}")

                    if hasattr(result, "strategy") and result.verdict != "no_go":
                        print(f"    Strategy:  {result.strategy.name}")
                        print(f"    Structure: {result.strategy.structure[:80]}")
                        if result.strategy.risk_notes:
                            print(f"    Risk:      {result.strategy.risk_notes[0]}")

                    # Iron condor specific
                    if hasattr(result, "wing_width_suggestion") and result.verdict != "no_go":
                        print(f"    Wings:     {result.wing_width_suggestion}")

                    # ORB decision (0DTE with ORB data)
                    if hasattr(result, "orb_decision") and result.orb_decision is not None:
                        od = result.orb_decision
                        print(f"    {_styled('ORB:', 'bold')}  {od.status} | {od.direction} | "
                              f"Range {od.range_low:.2f}–{od.range_high:.2f} ({od.range_pct:.1f}%)")
                        print(f"    ORB Decision: {od.decision[:100]}")
                        # Show key levels
                        level_strs = []
                        for k, v in od.key_levels.items():
                            if k not in ("range_high", "range_low"):
                                level_strs.append(f"{k}={v:.2f}")
                        if level_strs:
                            print(f"    ORB Levels: {', '.join(level_strs[:6])}")

                    # Trade spec (actionable parameters)
                    if hasattr(result, "trade_spec") and result.trade_spec is not None:
                        ts = result.trade_spec
                        direction = getattr(result, "direction", None)
                        if direction is None and hasattr(result, "strategy") and hasattr(result.strategy, "direction"):
                            direction = result.strategy.direction
                        if ts.structure_type:
                            tag = _profile_tag(ts.structure_type, ts.order_side, direction)
                            print(f"    {tag}")
                        print(f"    Expiry:    {ts.target_expiration} ({ts.target_dte}d)")
                        if ts.front_expiration and ts.back_expiration:
                            print(f"    Front:     {ts.front_expiration} ({ts.front_dte}d, IV {ts.iv_at_front:.1%})")
                            print(f"    Back:      {ts.back_expiration} ({ts.back_dte}d, IV {ts.iv_at_back:.1%})")
                        if ts.wing_width_points:
                            print(f"    Wing Width: ${ts.wing_width_points:.0f}")
                        print(f"    Legs:")
                        for code in ts.leg_codes:
                            print(f"      {code}")
                        # Exit guidance
                        if ts.exit_summary:
                            print(f"    Exit:      {ts.exit_summary}")
                        if ts.exit_notes:
                            for note in ts.exit_notes[:3]:
                                print(f"      - {note}")

                    # Ratio spread specific
                    if hasattr(result, "margin_warning") and result.margin_warning:
                        print(f"    {_styled('MARGIN:', 'yellow')} {result.margin_warning}")

                except Exception as exc:
                    print(f"\n  {p.replace('_', ' ').title()}: {_styled(str(exc), 'red')}")

        except Exception as exc:
            print(f"{_styled('ERROR:', 'red')} {exc}")

    def do_adjust(self, arg: str) -> None:
        """Analyze trade adjustments for a ticker.\nUsage: adjust TICKER"""
        tickers = self._parse_tickers(arg)
        if not tickers:
            print("Usage: adjust TICKER")
            return
        ticker = tickers[0]

        try:
            ma = self._get_ma()
            regime = ma.regime.detect(ticker)
            tech = ma.technicals.snapshot(ticker)
            price = tech.current_price
            atr = tech.atr

            # Build a representative IC trade for analysis
            from market_analyzer.models.opportunity import LegAction, LegSpec, OrderSide, StructureType, TradeSpec
            from market_analyzer.opportunity.option_plays._trade_spec_helpers import (
                build_iron_condor_legs,
                find_best_expiration,
            )

            vol_surface = None
            try:
                vol_surface = ma.vol_surface.get(ticker)
            except Exception:
                pass

            # Try to build from vol surface, fallback to synthetic
            exp_pt = None
            if vol_surface and vol_surface.term_structure:
                exp_pt = find_best_expiration(vol_surface.term_structure, 30, 45)

            if exp_pt:
                legs, wing_width = build_iron_condor_legs(
                    price, atr, regime.regime, exp_pt.expiration,
                    exp_pt.days_to_expiry, exp_pt.atm_iv,
                )
                trade = TradeSpec(
                    ticker=ticker, legs=legs, underlying_price=price,
                    target_dte=exp_pt.days_to_expiry, target_expiration=exp_pt.expiration,
                    wing_width_points=wing_width,
                    spec_rationale="Representative IC for adjustment analysis",
                    structure_type=StructureType.IRON_CONDOR,
                    order_side=OrderSide.CREDIT,
                    profit_target_pct=0.50, stop_loss_pct=2.0, exit_dte=21,
                )
            else:
                # Synthetic fallback
                from datetime import timedelta
                from market_analyzer.opportunity.option_plays._trade_spec_helpers import (
                    compute_otm_strike, snap_strike,
                )
                dte = 30
                exp = date.today() + timedelta(days=dte)
                short_put = compute_otm_strike(price, atr, 1.0, "put", price)
                short_call = compute_otm_strike(price, atr, 1.0, "call", price)
                long_put = snap_strike(short_put - atr * 0.5, price)
                long_call = snap_strike(short_call + atr * 0.5, price)
                ww = short_put - long_put

                def _leg(role, action, otype, strike):
                    return LegSpec(
                        role=role, action=action, option_type=otype, strike=strike,
                        strike_label=f"{strike:.0f} {otype}",
                        expiration=exp, days_to_expiry=dte, atm_iv_at_expiry=0.25,
                    )

                trade = TradeSpec(
                    ticker=ticker,
                    legs=[
                        _leg("short_put", LegAction.SELL_TO_OPEN, "put", short_put),
                        _leg("long_put", LegAction.BUY_TO_OPEN, "put", long_put),
                        _leg("short_call", LegAction.SELL_TO_OPEN, "call", short_call),
                        _leg("long_call", LegAction.BUY_TO_OPEN, "call", long_call),
                    ],
                    underlying_price=price, target_dte=dte, target_expiration=exp,
                    wing_width_points=ww,
                    spec_rationale="Synthetic IC for adjustment analysis",
                    structure_type=StructureType.IRON_CONDOR,
                    order_side=OrderSide.CREDIT,
                    profit_target_pct=0.50, stop_loss_pct=2.0, exit_dte=21,
                )

            result = ma.adjustment.analyze(trade, regime, tech, vol_surface)

            # Display
            _print_header(f"{ticker} — Trade Adjustment Analysis")

            # Position summary
            short_puts = [l for l in trade.legs
                          if l.option_type == "put" and l.action == LegAction.SELL_TO_OPEN]
            short_calls = [l for l in trade.legs
                           if l.option_type == "call" and l.action == LegAction.SELL_TO_OPEN]
            long_puts = [l for l in trade.legs
                         if l.option_type == "put" and l.action == LegAction.BUY_TO_OPEN]
            long_calls = [l for l in trade.legs
                          if l.option_type == "call" and l.action == LegAction.BUY_TO_OPEN]

            legs_desc = ""
            if short_puts and long_puts and short_calls and long_calls:
                sp = max(l.strike for l in short_puts)
                lp = min(l.strike for l in long_puts)
                sc = min(l.strike for l in short_calls)
                lc = max(l.strike for l in long_calls)
                legs_desc = f"{lp:.0f}P/{sp:.0f}P — {sc:.0f}C/{lc:.0f}C"

            profile = _profile_tag(trade.structure_type, trade.order_side)
            print(f"\n  Position: Iron Condor  {legs_desc}  {result.remaining_dte} DTE  {profile}")

            status_style = {
                "safe": "green", "tested": "yellow", "breached": "red", "max_loss": "red",
            }.get(result.position_status, "")
            tested_str = (
                f" ({result.tested_side} side)"
                if result.tested_side != "none" else ""
            )
            print(f"  Status: {_styled(result.position_status.upper(), status_style)}{tested_str}  |  "
                  f"Price: ${result.current_price:.0f}", end="")
            if result.distance_to_short_put_pct is not None:
                print(f"  |  Short put: {result.distance_to_short_put_pct:+.1f}%", end="")
            if result.distance_to_short_call_pct is not None:
                print(f"  |  Short call: {result.distance_to_short_call_pct:+.1f}%", end="")
            print()
            print(f"  P&L: ${result.pnl_estimate:+.2f}  |  Regime: R{result.regime_id}")

            # Adjustments
            print()
            for i, adj in enumerate(result.adjustments, 1):
                type_label = adj.adjustment_type.value.upper().replace("_", " ")
                print(f"  #{i}  {_styled(type_label, 'bold')} — {adj.rationale}")
                cost_str = f"${adj.estimated_cost:+.2f}" if adj.estimated_cost != 0 else "$0"
                risk_str = f"${adj.risk_change:+.0f}" if adj.risk_change != 0 else "unchanged"
                eff_str = f"{adj.efficiency:.2f}" if adj.efficiency is not None else ("∞" if adj.estimated_cost <= 0 and adj.risk_change < 0 else "—")
                urgency_style = {"immediate": "red", "soon": "yellow", "monitor": "dim"}.get(adj.urgency, "")
                print(f"      Cost: {cost_str}  |  Risk: {risk_str}  |  "
                      f"Efficiency: {eff_str}  |  "
                      f"Urgency: {_styled(adj.urgency, urgency_style)}")
                if adj.description and adj.description != adj.rationale:
                    print(f"      {_styled(adj.description, 'dim')}")
                # Warn on poor cost/risk ratio for paid adjustments
                if adj.estimated_cost > 0 and adj.risk_change < 0:
                    ratio = abs(adj.risk_change) / adj.estimated_cost if adj.estimated_cost > 0 else 0
                    if ratio < 1.0:
                        print(f"      {_styled(f'⚠ POOR — paying ${adj.estimated_cost:.2f} to reduce ${abs(adj.risk_change):.0f} risk', 'yellow')}")
                print()

            print(f"  {_styled(result.recommendation, 'bold')}")

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


def _ensure_utf8_stdout() -> None:
    """Reconfigure stdout to UTF-8 on Windows to handle Unicode payoff graphs."""
    import io
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    elif hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace",
        )


def main() -> None:
    _ensure_utf8_stdout()

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
