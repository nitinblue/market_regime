"""Harness runner: exercises every MarketAnalyzer API call in sequence.

Written as an external caller would use the library — pure API calls,
model attribute access, and tabulated output.

Usage:
    .venv/Scripts/python harness.py
    .venv/Scripts/python harness.py --tickers AAPL MSFT
    .venv/Scripts/python harness.py --tickers GLD --skip regime phase
    .venv/Scripts/python harness.py --only black_swan macro
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import date

from tabulate import tabulate

from market_analyzer import (
    DataService,
    MarketAnalyzer,
    get_settings,
)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_call_counter = 0


def next_call() -> int:
    global _call_counter
    _call_counter += 1
    return _call_counter


def section(num: int, title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  [{num}] {title}")
    print(f"{'=' * 72}")


def table(rows: list[list], headers: list[str] | None = None) -> None:
    """Print a 2-column key/value table or multi-column table."""
    if headers:
        print(tabulate(rows, headers=headers, tablefmt="simple", stralign="right"))
    else:
        print(tabulate(rows, tablefmt="plain", stralign="left"))


def kvtable(pairs: list[tuple[str, object]]) -> None:
    """Print key-value pairs as a clean 2-column table."""
    rows = [[k, v] for k, v in pairs]
    print(tabulate(rows, tablefmt="plain", stralign="left"))


def banner(api_call: str, elapsed: float) -> None:
    print(f"\n  API call:  {api_call}")
    print(f"  Status:    OK  ({elapsed:.2f}s)\n")


# ---------------------------------------------------------------------------
# API call wrappers
# ---------------------------------------------------------------------------

def call_data_service(ds: DataService, ticker: str) -> None:
    """DataService.get_ohlcv() + cache_status()"""
    num = next_call()
    section(num, f"DataService — {ticker}")

    t0 = time.perf_counter()
    df = ds.get_ohlcv(ticker)
    elapsed = time.perf_counter() - t0
    banner(f"ds.get_ohlcv('{ticker}')", elapsed)

    kvtable([
        ("Rows", len(df)),
        ("Columns", ", ".join(df.columns)),
        ("Date range", f"{df.index[0].date()} to {df.index[-1].date()}"),
        ("Last close", f"${df['Close'].iloc[-1]:.2f}"),
    ])

    t0 = time.perf_counter()
    metas = ds.cache_status(ticker)
    elapsed2 = time.perf_counter() - t0
    print(f"\n  ds.cache_status('{ticker}')  ({elapsed2:.2f}s)")

    if metas:
        rows = []
        for m in metas:
            rows.append([m.data_type.value, str(m.first_date), str(m.last_date),
                         m.row_count, str(m.last_fetched)])
        print()
        table(rows, headers=["Type", "From", "To", "Rows", "Fetched"])


def call_regime(ma: MarketAnalyzer, ticker: str) -> None:
    """ma.regime.detect(ticker)"""
    num = next_call()
    section(num, f"RegimeService.detect — {ticker}")

    t0 = time.perf_counter()
    r = ma.regime.detect(ticker)
    banner(f"ma.regime.detect('{ticker}')", time.perf_counter() - t0)

    settings = get_settings()
    name = settings.regimes.names.get(int(r.regime), "?")
    strategy = settings.regimes.strategies.get(int(r.regime), "")

    kvtable([
        ("Ticker", r.ticker),
        ("Regime", f"R{r.regime} — {name}"),
        ("Confidence", f"{r.confidence:.4f}"),
        ("Trend direction", r.trend_direction.value if r.trend_direction else "N/A"),
        ("As-of date", r.as_of_date),
        ("Model version", r.model_version),
        ("Strategy", strategy),
    ])

    # Probabilities table
    prob_rows = [[f"R{k}", f"{v:.4f}"] for k, v in sorted(r.regime_probabilities.items())]
    print()
    table(prob_rows, headers=["State", "Probability"])


def call_technicals(ma: MarketAnalyzer, ticker: str) -> None:
    """ma.technicals.snapshot(ticker)"""
    num = next_call()
    section(num, f"TechnicalService.snapshot — {ticker}")

    t0 = time.perf_counter()
    t = ma.technicals.snapshot(ticker)
    banner(f"ma.technicals.snapshot('{ticker}')", time.perf_counter() - t0)

    ma_data = t.moving_averages
    rsi_status = "overbought" if t.rsi.is_overbought else ("oversold" if t.rsi.is_oversold else "neutral")
    macd_cross = "bullish" if t.macd.is_bullish_crossover else ("bearish" if t.macd.is_bearish_crossover else "none")
    stoch_status = "overbought" if t.stochastic.is_overbought else ("oversold" if t.stochastic.is_oversold else "neutral")

    kvtable([
        ("Price", f"${t.current_price:.2f}"),
        ("ATR", f"${t.atr:.2f}  ({t.atr_pct:.2f}%)"),
        ("VWMA 20", f"${t.vwma_20:.2f}"),
    ])

    # Moving averages
    print()
    ma_rows = [
        ["SMA 20", f"${ma_data.sma_20:.2f}", f"{ma_data.price_vs_sma_20_pct:+.2f}%"],
        ["SMA 50", f"${ma_data.sma_50:.2f}", f"{ma_data.price_vs_sma_50_pct:+.2f}%"],
        ["SMA 200", f"${ma_data.sma_200:.2f}", f"{ma_data.price_vs_sma_200_pct:+.2f}%"],
        ["EMA 9", f"${ma_data.ema_9:.2f}", ""],
        ["EMA 21", f"${ma_data.ema_21:.2f}", ""],
    ]
    table(ma_rows, headers=["Moving Avg", "Value", "Price vs MA"])

    # Oscillators
    print()
    osc_rows = [
        ["RSI", f"{t.rsi.value:.2f}", rsi_status],
        ["Stochastic %K", f"{t.stochastic.k:.2f}", stoch_status],
        ["Stochastic %D", f"{t.stochastic.d:.2f}", ""],
        ["MACD Line", f"{t.macd.macd_line:.4f}", macd_cross],
        ["MACD Signal", f"{t.macd.signal_line:.4f}", ""],
        ["MACD Hist", f"{t.macd.histogram:.4f}", ""],
    ]
    table(osc_rows, headers=["Oscillator", "Value", "Status"])

    # Bollinger
    print()
    kvtable([
        ("Bollinger upper", f"${t.bollinger.upper:.2f}"),
        ("Bollinger mid", f"${t.bollinger.middle:.2f}"),
        ("Bollinger lower", f"${t.bollinger.lower:.2f}"),
        ("Bandwidth", f"{t.bollinger.bandwidth:.4f}"),
        ("%B", f"{t.bollinger.percent_b:.4f}"),
    ])

    # S/R
    sr = t.support_resistance
    print()
    sup_str = f"${sr.support:.2f} ({sr.price_vs_support_pct:+.2f}%)" if sr.support else "N/A"
    res_str = f"${sr.resistance:.2f} ({sr.price_vs_resistance_pct:+.2f}%)" if sr.resistance else "N/A"
    kvtable([
        ("Support", sup_str),
        ("Resistance", res_str),
    ])

    # VCP
    if t.vcp:
        print()
        kvtable([
            ("VCP stage", t.vcp.stage.value),
            ("VCP score", f"{t.vcp.score:.2f}"),
            ("Contractions", t.vcp.contraction_count),
            ("Days in base", t.vcp.days_in_base),
            ("Pivot price", f"${t.vcp.pivot_price:.2f}" if t.vcp.pivot_price else "N/A"),
        ])

    # Smart money
    if t.smart_money:
        sm = t.smart_money
        if sm.order_blocks:
            print()
            ob_rows = []
            for ob in sm.order_blocks[:3]:
                ob_rows.append([ob.type.value, f"${ob.low:.2f}", f"${ob.high:.2f}",
                                f"{ob.impulse_strength:.2f}x ATR",
                                "yes" if ob.is_tested else "no",
                                "yes" if ob.is_broken else "no"])
            table(ob_rows, headers=["OB Type", "Low", "High", "Impulse", "Tested", "Broken"])

        if sm.fair_value_gaps:
            print()
            fvg_rows = []
            for g in sm.fair_value_gaps[:3]:
                fvg_rows.append([g.type.value, f"${g.low:.2f}", f"${g.high:.2f}",
                                 f"{g.gap_size_pct:.2f}%",
                                 "yes" if g.is_filled else "no",
                                 f"{g.fill_pct:.0f}%"])
            table(fvg_rows, headers=["FVG Type", "Low", "High", "Gap%", "Filled", "Fill%"])

    # Signals
    if t.signals:
        print()
        sig_rows = [[s.name, s.direction.value, s.strength.value, s.description] for s in t.signals]
        table(sig_rows, headers=["Signal", "Direction", "Strength", "Description"])


def call_levels(ma: MarketAnalyzer, ticker: str) -> None:
    """ma.levels.analyze(ticker)"""
    num = next_call()
    section(num, f"LevelsService.analyze — {ticker}")

    t0 = time.perf_counter()
    lv = ma.levels.analyze(ticker)
    banner(f"ma.levels.analyze('{ticker}')", time.perf_counter() - t0)

    kvtable([
        ("Entry price", f"${lv.entry_price:.2f}"),
        ("Current price", f"${lv.current_price:.2f}"),
        ("Direction", lv.direction.value),
        ("Auto-detected", lv.direction_auto_detected),
        ("ATR", f"${lv.atr:.2f} ({lv.atr_pct:.2f}%)"),
    ])

    # Stop loss
    if lv.stop_loss:
        sl = lv.stop_loss
        print()
        kvtable([
            ("Stop price", f"${sl.price:.2f}"),
            ("Stop distance", f"{sl.distance_pct:.2f}%"),
            ("Dollar risk/share", f"${sl.dollar_risk_per_share:.2f}"),
            ("ATR buffer", f"${sl.atr_buffer:.2f}"),
            ("Stop reason", sl.description),
        ])
    else:
        print("\n  Stop loss: N/A (no suitable support level)")

    # Targets
    if lv.targets:
        print()
        tgt_rows = []
        for i, tgt in enumerate(lv.targets, 1):
            tgt_rows.append([f"T{i}", f"${tgt.price:.2f}", f"{tgt.distance_pct:+.2f}%",
                             f"${tgt.dollar_reward_per_share:.2f}",
                             f"{tgt.risk_reward_ratio:.2f}", tgt.description])
        table(tgt_rows, headers=["#", "Price", "Distance", "$/Share", "R:R", "Level"])

        if lv.best_target:
            bt = lv.best_target
            print(f"\n  Best target: ${bt.price:.2f}  R:R={bt.risk_reward_ratio:.2f}")

    # Support/Resistance levels
    if lv.support_levels:
        print()
        sr_rows = []
        for s in lv.support_levels[:5]:
            sources = ", ".join(src.value for src in s.sources)
            sr_rows.append(["support", f"${s.price:.2f}", f"{s.distance_pct:+.2f}%",
                            s.confluence_score, f"{s.strength:.2f}", sources])
        for r in lv.resistance_levels[:5]:
            sources = ", ".join(src.value for src in r.sources)
            sr_rows.append(["resist", f"${r.price:.2f}", f"{r.distance_pct:+.2f}%",
                            r.confluence_score, f"{r.strength:.2f}", sources])
        table(sr_rows, headers=["Role", "Price", "Distance", "Confluence", "Strength", "Sources"])

    print(f"\n  Summary: {lv.summary}")


def call_phase(ma: MarketAnalyzer, ticker: str) -> None:
    """ma.phase.detect(ticker)"""
    num = next_call()
    section(num, f"PhaseService.detect — {ticker}")

    t0 = time.perf_counter()
    p = ma.phase.detect(ticker)
    banner(f"ma.phase.detect('{ticker}')", time.perf_counter() - t0)

    kvtable([
        ("Phase", f"P{p.phase} — {p.phase_name}"),
        ("Confidence", f"{p.confidence:.4f}"),
        ("Age", f"{p.phase_age_days} days"),
        ("Prior phase", f"P{p.prior_phase}" if p.prior_phase else "N/A"),
        ("Cycle completion", f"{p.cycle_completion:.2%}"),
        ("Strategy", p.strategy_comment),
    ])

    # Evidence
    ev = p.evidence
    print()
    kvtable([
        ("Regime signal", ev.regime_signal),
        ("Price signal", ev.price_signal),
        ("Volume signal", ev.volume_signal),
    ])

    if ev.supporting:
        print()
        for s in ev.supporting:
            print(f"    + {s}")
    if ev.contradictions:
        for c in ev.contradictions:
            print(f"    - {c}")

    # Transitions
    if p.transitions:
        print()
        tr_rows = [[f"P{t.to_phase}", f"{t.probability:.2f}", ", ".join(t.triggers)]
                   for t in p.transitions]
        table(tr_rows, headers=["To Phase", "Prob", "Triggers"])


def call_fundamentals(ma: MarketAnalyzer, ticker: str) -> None:
    """ma.fundamentals.get(ticker)"""
    num = next_call()
    section(num, f"FundamentalService.get — {ticker}")

    t0 = time.perf_counter()
    f = ma.fundamentals.get(ticker)
    banner(f"ma.fundamentals.get('{ticker}')", time.perf_counter() - t0)

    if f is None:
        print("  Result: None (no fundamentals data)")
        return

    def fmt(val, suffix="", prefix="", mult=1):
        if val is None:
            return "N/A"
        return f"{prefix}{val * mult:.2f}{suffix}"

    def fmt_int(val, label=""):
        if val is None:
            return "N/A"
        if abs(val) >= 1e9:
            return f"${val / 1e9:.2f}B"
        if abs(val) >= 1e6:
            return f"${val / 1e6:.0f}M"
        return f"${val:,.0f}"

    kvtable([
        ("Ticker", f.ticker),
        ("Name", f.business.long_name),
        ("Sector", f.business.sector),
        ("Industry", f.business.industry),
        ("Beta", fmt(f.business.beta)),
    ])

    print()
    val_rows = [
        ["Trailing P/E", fmt(f.valuation.trailing_pe)],
        ["Forward P/E", fmt(f.valuation.forward_pe)],
        ["PEG Ratio", fmt(f.valuation.peg_ratio)],
        ["P/B", fmt(f.valuation.price_to_book)],
        ["P/S", fmt(f.valuation.price_to_sales)],
    ]
    table(val_rows, headers=["Valuation", "Value"])

    print()
    growth_rows = [
        ["Market Cap", fmt_int(f.revenue.market_cap)],
        ["Revenue", fmt_int(f.revenue.total_revenue)],
        ["Revenue Growth", fmt(f.revenue.revenue_growth, suffix="%", mult=100)],
        ["EPS (TTM)", fmt(f.earnings.trailing_eps, prefix="$")],
        ["EPS (Fwd)", fmt(f.earnings.forward_eps, prefix="$")],
        ["Earnings Growth", fmt(f.earnings.earnings_growth, suffix="%", mult=100)],
    ]
    table(growth_rows, headers=["Growth", "Value"])

    print()
    margin_rows = [
        ["Profit Margin", fmt(f.margins.profit_margins, suffix="%", mult=100)],
        ["Gross Margin", fmt(f.margins.gross_margins, suffix="%", mult=100)],
        ["Operating Margin", fmt(f.margins.operating_margins, suffix="%", mult=100)],
        ["Debt/Equity", fmt(f.debt.debt_to_equity)],
        ["Current Ratio", fmt(f.debt.current_ratio)],
    ]
    table(margin_rows, headers=["Financial Health", "Value"])

    print()
    w52 = f.fifty_two_week
    kvtable([
        ("52wk High", fmt(w52.high, prefix="$")),
        ("52wk Low", fmt(w52.low, prefix="$")),
        ("% from High", fmt(w52.pct_from_high, suffix="%")),
        ("% from Low", fmt(w52.pct_from_low, suffix="%")),
        ("Dividend Yield", fmt(f.dividends.dividend_yield, suffix="%", mult=100)),
    ])

    if f.upcoming_events.next_earnings_date:
        print()
        kvtable([
            ("Next earnings", f.upcoming_events.next_earnings_date),
            ("Days to earnings", f.upcoming_events.days_to_earnings),
        ])


def call_macro(ma: MarketAnalyzer) -> None:
    """ma.macro.calendar()"""
    num = next_call()
    section(num, f"MacroService.calendar")

    t0 = time.perf_counter()
    cal = ma.macro.calendar()
    banner("ma.macro.calendar()", time.perf_counter() - t0)

    kvtable([
        ("Total events", len(cal.events)),
        ("Days to next FOMC", cal.days_to_next_fomc if cal.days_to_next_fomc is not None else "N/A"),
        ("Events next 7d", len(cal.events_next_7_days)),
        ("Events next 30d", len(cal.events_next_30_days)),
    ])

    if cal.next_event:
        ne = cal.next_event
        print()
        kvtable([
            ("Next event", ne.event_type.value),
            ("Date", ne.date),
            ("Impact", ne.impact.value),
            ("Description", ne.description),
        ])

    if cal.events_next_7_days:
        print()
        rows = [[str(ev.date), ev.event_type.value, ev.impact.value, ev.description]
                for ev in cal.events_next_7_days[:5]]
        table(rows, headers=["Date", "Type", "Impact", "Description"])


def call_zero_dte(ma: MarketAnalyzer, ticker: str) -> None:
    """ma.opportunity.assess_zero_dte(ticker)"""
    num = next_call()
    section(num, f"OpportunityService.assess_zero_dte — {ticker}")

    t0 = time.perf_counter()
    z = ma.opportunity.assess_zero_dte(ticker)
    banner(f"ma.opportunity.assess_zero_dte('{ticker}')", time.perf_counter() - t0)

    kvtable([
        ("Verdict", z.verdict.value.upper()),
        ("Confidence", f"{z.confidence:.4f}"),
        ("0DTE strategy", z.zero_dte_strategy.value),
        ("Strategy", f"{z.strategy.name} ({z.strategy.direction})"),
        ("Structure", z.strategy.structure),
        ("Regime", f"R{z.regime_id} ({z.regime_confidence:.2%})"),
        ("ATR%", f"{z.atr_pct:.3f}%"),
        ("ORB status", z.orb_status or "N/A"),
        ("Macro event today", z.has_macro_event_today),
        ("Days to earnings", z.days_to_earnings if z.days_to_earnings is not None else "N/A"),
    ])

    if z.hard_stops:
        print()
        hs_rows = [[hs.name, hs.description] for hs in z.hard_stops]
        table(hs_rows, headers=["Hard Stop", "Reason"])

    if z.signals:
        print()
        sig_rows = [[s.name, "yes" if s.favorable else "no", f"{s.weight:.2f}", s.description]
                    for s in z.signals]
        table(sig_rows, headers=["Signal", "Favorable", "Weight", "Description"])

    print(f"\n  Summary: {z.summary}")


def call_leap(ma: MarketAnalyzer, ticker: str) -> None:
    """ma.opportunity.assess_leap(ticker)"""
    num = next_call()
    section(num, f"OpportunityService.assess_leap — {ticker}")

    t0 = time.perf_counter()
    lp = ma.opportunity.assess_leap(ticker)
    banner(f"ma.opportunity.assess_leap('{ticker}')", time.perf_counter() - t0)

    kvtable([
        ("Verdict", lp.verdict.value.upper()),
        ("Confidence", f"{lp.confidence:.4f}"),
        ("LEAP strategy", lp.leap_strategy.value),
        ("Strategy", f"{lp.strategy.name} ({lp.strategy.direction})"),
        ("Structure", lp.strategy.structure),
        ("Regime", f"R{lp.regime_id} ({lp.regime_confidence:.2%})"),
        ("Phase", f"P{lp.phase_id} {lp.phase_name} ({lp.phase_confidence:.2%})"),
        ("IV environment", lp.iv_environment),
        ("Fundamental score", f"{lp.fundamental_score.score:.4f}"),
        ("Fund description", lp.fundamental_score.description),
        ("Days to earnings", lp.days_to_earnings if lp.days_to_earnings is not None else "N/A"),
        ("Macro events 30d", lp.macro_events_next_30_days),
    ])

    if lp.hard_stops:
        print()
        hs_rows = [[hs.name, hs.description] for hs in lp.hard_stops]
        table(hs_rows, headers=["Hard Stop", "Reason"])

    if lp.signals:
        print()
        sig_rows = [[s.name, "yes" if s.favorable else "no", f"{s.weight:.2f}", s.description]
                    for s in lp.signals]
        table(sig_rows, headers=["Signal", "Favorable", "Weight", "Description"])

    print(f"\n  Summary: {lp.summary}")


def call_breakout(ma: MarketAnalyzer, ticker: str) -> None:
    """ma.opportunity.assess_breakout(ticker)"""
    num = next_call()
    section(num, f"OpportunityService.assess_breakout — {ticker}")

    t0 = time.perf_counter()
    bo = ma.opportunity.assess_breakout(ticker)
    banner(f"ma.opportunity.assess_breakout('{ticker}')", time.perf_counter() - t0)

    kvtable([
        ("Verdict", bo.verdict.value.upper()),
        ("Confidence", f"{bo.confidence:.4f}"),
        ("Breakout strategy", bo.breakout_strategy.value),
        ("Breakout type", bo.breakout_type.value),
        ("Strategy", f"{bo.strategy.name} ({bo.strategy.direction})"),
        ("Structure", bo.strategy.structure),
        ("Regime", f"R{bo.regime_id} ({bo.regime_confidence:.2%})"),
        ("Phase", f"P{bo.phase_id} {bo.phase_name} ({bo.phase_confidence:.2%})"),
        ("Pivot price", f"${bo.pivot_price:.2f}" if bo.pivot_price else "N/A"),
        ("Days to earnings", bo.days_to_earnings if bo.days_to_earnings is not None else "N/A"),
    ])

    # Setup details
    s = bo.setup
    print()
    kvtable([
        ("VCP stage", s.vcp_stage),
        ("VCP score", f"{s.vcp_score:.4f}"),
        ("Bollinger squeeze", s.bollinger_squeeze),
        ("Bollinger bandwidth", f"{s.bollinger_bandwidth:.4f}"),
        ("Range compression", f"{s.range_compression:.4f}"),
        ("Volume pattern", s.volume_pattern),
        ("Smart money align", s.smart_money_alignment),
        ("Days in base", s.days_in_base if s.days_in_base is not None else "N/A"),
    ])

    if bo.hard_stops:
        print()
        hs_rows = [[hs.name, hs.description] for hs in bo.hard_stops]
        table(hs_rows, headers=["Hard Stop", "Reason"])

    if bo.signals:
        print()
        sig_rows = [[s.name, "yes" if s.favorable else "no", f"{s.weight:.2f}", s.description]
                    for s in bo.signals]
        table(sig_rows, headers=["Signal", "Favorable", "Weight", "Description"])

    print(f"\n  Summary: {bo.summary}")


def call_momentum(ma: MarketAnalyzer, ticker: str) -> None:
    """ma.opportunity.assess_momentum(ticker)"""
    num = next_call()
    section(num, f"OpportunityService.assess_momentum — {ticker}")

    t0 = time.perf_counter()
    mo = ma.opportunity.assess_momentum(ticker)
    banner(f"ma.opportunity.assess_momentum('{ticker}')", time.perf_counter() - t0)

    kvtable([
        ("Verdict", mo.verdict.value.upper()),
        ("Confidence", f"{mo.confidence:.4f}"),
        ("Momentum strategy", mo.momentum_strategy.value),
        ("Momentum direction", mo.momentum_direction.value),
        ("Strategy", f"{mo.strategy.name} ({mo.strategy.direction})"),
        ("Structure", mo.strategy.structure),
        ("Regime", f"R{mo.regime_id} ({mo.regime_confidence:.2%})"),
        ("Phase", f"P{mo.phase_id} {mo.phase_name} ({mo.phase_confidence:.2%})"),
        ("Days to earnings", mo.days_to_earnings if mo.days_to_earnings is not None else "N/A"),
    ])

    # Score details
    sc = mo.score
    print()
    score_rows = [
        ["MACD histogram", sc.macd_histogram_trend],
        ["MACD crossover", sc.macd_crossover],
        ["RSI zone", sc.rsi_zone],
        ["MA alignment", sc.price_vs_ma_alignment],
        ["Golden/Death cross", sc.golden_death_cross or "none"],
        ["Structure", sc.structural_pattern],
        ["Volume confirm", "yes" if sc.volume_confirmation else "no"],
        ["Stochastic confirm", "yes" if sc.stochastic_confirmation else "no"],
        ["ATR trend", sc.atr_trend],
    ]
    table(score_rows, headers=["Momentum Score", "Value"])

    if mo.hard_stops:
        print()
        hs_rows = [[hs.name, hs.description] for hs in mo.hard_stops]
        table(hs_rows, headers=["Hard Stop", "Reason"])

    if mo.signals:
        print()
        sig_rows = [[s.name, "yes" if s.favorable else "no", f"{s.weight:.2f}", s.description]
                    for s in mo.signals]
        table(sig_rows, headers=["Signal", "Favorable", "Weight", "Description"])

    print(f"\n  Summary: {mo.summary}")


def call_black_swan(ma: MarketAnalyzer) -> None:
    """ma.black_swan.alert()"""
    num = next_call()
    section(num, "BlackSwanService.alert")

    t0 = time.perf_counter()
    alert = ma.black_swan.alert()
    banner("ma.black_swan.alert()", time.perf_counter() - t0)

    kvtable([
        ("As-of date", alert.as_of_date),
        ("Alert level", alert.alert_level.value.upper()),
        ("Composite score", f"{alert.composite_score:.4f}"),
        ("Triggered breakers", alert.triggered_breakers),
        ("Action", alert.action),
    ])

    # Circuit breakers
    print()
    cb_rows = []
    for cb in alert.circuit_breakers:
        val = f"{cb.value:.4f}" if cb.value is not None else "N/A"
        status = "TRIGGERED" if cb.triggered else "ok"
        cb_rows.append([cb.name, val, cb.threshold, status, cb.description])
    table(cb_rows, headers=["Breaker", "Value", "Threshold", "Status", "Description"])

    # Indicators
    print()
    ind_rows = []
    for ind in alert.indicators:
        val = f"{ind.value:.4f}" if ind.value is not None else "N/A"
        ind_rows.append([ind.name, val, f"{ind.score:.4f}", ind.status.value, f"{ind.weight:.2f}"])
    table(ind_rows, headers=["Indicator", "Value", "Score", "Status", "Weight"])

    print(f"\n  Summary: {alert.summary}")


# ---------------------------------------------------------------------------
# Registry and main
# ---------------------------------------------------------------------------

ALL_SERVICES = [
    "data", "regime", "technicals", "levels", "phase",
    "fundamentals", "macro", "zero_dte", "leap", "breakout",
    "momentum", "black_swan",
]

GLOBAL_SERVICES = {"macro", "black_swan"}

PER_TICKER_CALLS = {
    "data": call_data_service,
    "regime": call_regime,
    "technicals": call_technicals,
    "levels": call_levels,
    "phase": call_phase,
    "fundamentals": call_fundamentals,
    "zero_dte": call_zero_dte,
    "leap": call_leap,
    "breakout": call_breakout,
    "momentum": call_momentum,
}


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Harness: exercises every MarketAnalyzer API call"
    )
    parser.add_argument(
        "--tickers", nargs="+",
        default=settings.display.default_tickers,
        help=f"Tickers (default: {' '.join(settings.display.default_tickers)})",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        choices=ALL_SERVICES,
        help="Services to skip",
    )
    parser.add_argument(
        "--only", nargs="*", default=None,
        choices=ALL_SERVICES,
        help="Run only these services",
    )
    args = parser.parse_args()

    active = set(args.only) if args.only else set(ALL_SERVICES)
    active -= set(args.skip)

    n_per_ticker = len([s for s in ALL_SERVICES if s not in GLOBAL_SERVICES and s in active])
    n_global = len([s for s in GLOBAL_SERVICES if s in active])
    total_calls = n_global + n_per_ticker * len(args.tickers)

    section(0, "MarketAnalyzer API Harness")
    kvtable([
        ("Date", date.today().isoformat()),
        ("Tickers", ", ".join(args.tickers)),
        ("Services", ", ".join(s for s in ALL_SERVICES if s in active)),
        ("Expected calls", total_calls),
    ])

    print("\n  Constructing services...")
    t0 = time.perf_counter()
    ds = DataService()
    ma = MarketAnalyzer(data_service=ds)
    print(f"  Init: {time.perf_counter() - t0:.2f}s")

    errors: list[tuple[str, str]] = []

    # --- Global services ---

    if "black_swan" in active:
        try:
            call_black_swan(ma)
        except Exception as e:
            errors.append(("black_swan.alert()", str(e)))
            traceback.print_exc()

    if "macro" in active:
        try:
            call_macro(ma)
        except Exception as e:
            errors.append(("macro.calendar()", str(e)))
            traceback.print_exc()

    # --- Per-ticker services ---

    per_ticker = [s for s in ALL_SERVICES if s not in GLOBAL_SERVICES and s in active]

    for ticker in args.tickers:
        for svc_name in per_ticker:
            fn = PER_TICKER_CALLS.get(svc_name)
            if fn is None:
                continue
            try:
                if svc_name == "data":
                    fn(ds, ticker)
                else:
                    fn(ma, ticker)
            except Exception as e:
                api_name = f"{svc_name}('{ticker}')"
                errors.append((api_name, str(e)))
                print(f"\n  ERROR in {api_name}: {e}")
                traceback.print_exc()

    # --- Results ---

    section(_call_counter + 1, "Harness Complete")
    total = time.perf_counter() - t0

    summary_rows = [
        ["Total time", f"{total:.1f}s"],
        ["API calls made", _call_counter],
        ["Tickers", len(args.tickers)],
        ["Errors", len(errors)],
    ]
    table(summary_rows, headers=["Metric", "Value"])

    if errors:
        print()
        err_rows = [[name, msg] for name, msg in errors]
        table(err_rows, headers=["Failed Call", "Error"])
        sys.exit(1)
    else:
        print("\n  All API calls completed successfully.")


if __name__ == "__main__":
    main()
