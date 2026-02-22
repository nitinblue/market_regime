# market_regime — API Reference

**Historical market data service, HMM-based regime detection, and options opportunity assessment.**

---

## Quick Start

```python
from market_regime import RegimeService, DataService

svc = RegimeService(data_service=DataService())

# What regime is SPY in?
regime = svc.detect("SPY")
print(f"{regime.ticker}: R{regime.regime} ({regime.confidence:.0%})")

# Should I trade 0DTE today?
z = svc.assess_zero_dte("SPY")
print(f"0DTE: {z.verdict} — {z.strategy.name}")

# Is AAPL a good LEAP candidate?
lp = svc.assess_leap("AAPL")
print(f"LEAP: {lp.verdict} — {lp.strategy.name}")
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  RegimeService  (single entry point for everything)     │
│                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │
│  │  Regime   │ │Technical │ │  Phase   │ │Opportunity│  │
│  │Detection  │ │Indicators│ │Detection │ │Assessment │  │
│  │ R1–R4    │ │RSI,MACD..│ │ P1–P4    │ │0DTE, LEAP │  │
│  └──────────┘ └──────────┘ └──────────┘ └───────────┘  │
│                                                         │
│  ┌──────────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ Fundamentals  │  │  Macro   │  │  Opening Range    │  │
│  │ (yfinance)   │  │ Calendar │  │  Breakout (ORB)   │  │
│  └──────────────┘  └──────────┘  └───────────────────┘  │
└───────────────────────┬─────────────────────────────────┘
                        │
                ┌───────┴────────┐
                │  DataService   │
                │ (cache-first)  │
                └────────────────┘
```

Three conceptual layers:

| Layer | What it does | Examples |
|-------|-------------|---------|
| **Analysis** | Produces signals & indicators | Regime R1–R4, RSI, Phase P1–P4, ORB |
| **Opportunity** | Per-horizon go/no-go + recommended strategy | "0DTE: GO, sell iron condor" |
| **Strategy** | Specific trade structure (data only) | "Iron Condor", "Bull Call LEAP 18mo" |

---

## Use Cases

### 1. Regime Detection — "What regime is this ticker in?"

Detect the current volatility/trend regime for any instrument. Every downstream decision depends on this.

```python
from market_regime import RegimeService, DataService

svc = RegimeService(data_service=DataService())

# Single ticker
result = svc.detect("GLD")
print(result.regime)           # RegimeID.R1_LOW_VOL_MR
print(result.confidence)       # 0.82
print(result.trend_direction)  # TrendDirection.BULLISH

# Batch detection
results = svc.detect_batch(tickers=["SPY", "GLD", "QQQ", "TLT"])
for ticker, r in results.items():
    print(f"{ticker}: R{r.regime} ({r.confidence:.0%})")
```

**Regime definitions:**

| ID | Name | Description | Options bias |
|----|------|-------------|-------------|
| R1 | Low-Vol Mean Reverting | Range-bound, IV compression | Theta harvesting (iron condors) |
| R2 | High-Vol Mean Reverting | Wide swings, no sustained trend | Selective theta, wider wings |
| R3 | Low-Vol Trending | Slow persistent directional move | Directional spreads |
| R4 | High-Vol Trending | Explosive moves, IV expansion | Risk-defined only, long vega |

**Returns:** `RegimeResult`

| Field | Type | Description |
|-------|------|-------------|
| `ticker` | `str` | Instrument symbol |
| `regime` | `RegimeID` | R1–R4 enum (also an int 1–4) |
| `confidence` | `float` | 0.0–1.0 posterior probability |
| `regime_probabilities` | `dict[int, float]` | All 4 state probabilities |
| `trend_direction` | `TrendDirection \| None` | BULLISH or BEARISH |
| `as_of_date` | `date` | Date of detection |
| `model_version` | `str` | Model identifier |

`RegimeID` has convenience properties: `.is_mean_reverting`, `.is_trending`, `.is_low_vol`, `.is_high_vol`.

---

### 2. Regime Research — "Explain everything about this ticker's regime"

Deep dive with transition probabilities, feature z-scores, recent history, and strategy guidance.

```python
# Single ticker deep research
research = svc.research("GLD")
print(research.strategy_comment)
# "R1: Low-Vol MR — ideal for theta harvesting. Iron condors, strangles."

for feat in research.current_features:
    print(f"  {feat.feature}: z={feat.z_score:+.2f} ({feat.comment})")

for row in research.transition_matrix:
    print(f"  R{row.from_regime} → stay {row.stay_probability:.0%} ({row.stability})")

# Multi-ticker comparison
report = svc.research_batch(tickers=["SPY", "GLD", "QQQ"])
for entry in report.comparison:
    print(f"{entry.ticker}: R{entry.regime} P{entry.phase} — {entry.strategy_comment}")
```

**Returns:** `TickerResearch`

| Field | Type | Description |
|-------|------|-------------|
| `regime_result` | `RegimeResult` | Current regime |
| `explanation_text` | `str` | Human-readable regime explanation |
| `transition_matrix` | `list[TransitionRow]` | Regime transition probabilities |
| `state_means` | `list[StateMeansRow]` | Feature means per regime state |
| `current_features` | `list[FeatureZScore]` | Current features with z-scores and interpretation |
| `recent_history` | `list[RegimeHistoryDay]` | Last N days of regime labels |
| `regime_distribution` | `list[RegimeDistributionEntry]` | % time in each regime |
| `strategy_comment` | `str` | Actionable strategy guidance |
| `phase_result` | `PhaseResult \| None` | Wyckoff phase if available |
| `model_info` | `HMMModelInfo` | HMM internals (means, covariances, transition matrix) |

---

### 3. Technical Analysis — "What do the indicators say?"

Full technical snapshot: moving averages, RSI, Bollinger, MACD, Stochastic, ATR, support/resistance, VCP, smart money concepts.

```python
technicals = svc.get_technicals("AAPL")

print(f"Price: ${technicals.current_price:.2f}")
print(f"RSI: {technicals.rsi.value:.1f} (overbought={technicals.rsi.is_overbought})")
print(f"ATR: {technicals.atr_pct:.2f}%")
print(f"Bollinger %B: {technicals.bollinger.percent_b:.2f}")
print(f"MACD: {'bullish' if technicals.macd.is_bullish_crossover else 'bearish'} crossover")
print(f"Support: ${technicals.support_resistance.support}")
print(f"Resistance: ${technicals.support_resistance.resistance}")

# VCP (Minervini Volatility Contraction Pattern)
if technicals.vcp:
    print(f"VCP: {technicals.vcp.stage} (score={technicals.vcp.score:.2f})")
    print(f"  Contractions: {technicals.vcp.contraction_count}")
    print(f"  Pivot: ${technicals.vcp.pivot_price}")

# Smart Money Concepts (order blocks + fair value gaps)
if technicals.smart_money:
    for ob in technicals.smart_money.order_blocks[:3]:
        print(f"  {ob.type} OB: ${ob.low:.2f}–${ob.high:.2f} ({ob.distance_pct:+.1f}%)")

# Signals summary
for sig in technicals.signals:
    print(f"  [{sig.direction}] {sig.name}: {sig.description}")
```

**Returns:** `TechnicalSnapshot`

| Field | Type | Description |
|-------|------|-------------|
| `current_price` | `float` | Latest close |
| `atr` / `atr_pct` | `float` | ATR in dollars and as % of price |
| `moving_averages` | `MovingAverages` | SMA 20/50/200, EMA 9/21, price-vs-MA % |
| `rsi` | `RSIData` | RSI value + overbought/oversold flags |
| `bollinger` | `BollingerBands` | Upper/lower/middle, bandwidth, %B |
| `macd` | `MACDData` | MACD/signal/histogram + crossover flags |
| `stochastic` | `StochasticData` | %K/%D + overbought/oversold flags |
| `support_resistance` | `SupportResistance` | Swing-based S/R with distance % |
| `phase` | `PhaseIndicator` | Price-structure-derived phase (HH/HL/LH/LL) |
| `vcp` | `VCPData \| None` | Minervini VCP detection |
| `smart_money` | `SmartMoneyData \| None` | Order blocks + fair value gaps |
| `signals` | `list[TechnicalSignal]` | All active signals with direction/strength |

---

### 4. Phase Detection — "Where is this stock in its Wyckoff cycle?"

Detect accumulation/markup/distribution/markdown phases using regime history + price structure.

```python
phase = svc.detect_phase("AAPL")

print(f"Phase: {phase.phase_name}")        # "Markup"
print(f"Confidence: {phase.confidence:.0%}")
print(f"Age: {phase.phase_age_days} days")
print(f"Prior phase: P{phase.prior_phase}") # "Accumulation"
print(f"Cycle: {phase.cycle_completion:.0%}")
print(phase.strategy_comment)
# "P2 Markup: ride the trend. Directional LEAP entry or PMCC."

# What might come next?
for t in phase.transitions:
    print(f"  → {t.to_phase.name}: {t.probability:.0%} if {', '.join(t.triggers)}")

# Price structure evidence
ps = phase.price_structure
print(f"  Higher highs: {ps.higher_highs}, Higher lows: {ps.higher_lows}")
print(f"  Volume trend: {ps.volume_trend}")
```

**Phase definitions:**

| Phase | Name | Description | LEAP guidance |
|-------|------|-------------|--------------|
| P1 | Accumulation | Base building, range-bound after decline | LEAP entry zone |
| P2 | Markup | Sustained advance, HH/HL | Ride the trend (PMCC) |
| P3 | Distribution | Topping, range after advance | LEAP exit zone |
| P4 | Markdown | Sustained decline, LH/LL | Protective puts or bear LEAPs |

**Returns:** `PhaseResult`

| Field | Type | Description |
|-------|------|-------------|
| `phase` | `PhaseID` | P1–P4 (IntEnum) |
| `phase_name` | `str` | Human-readable name |
| `confidence` | `float` | 0.0–1.0 |
| `phase_age_days` | `int` | How long in this phase |
| `prior_phase` | `PhaseID \| None` | Previous phase |
| `cycle_completion` | `float` | 0.0–1.0 progress through full Wyckoff cycle |
| `price_structure` | `PriceStructure` | Swings, HH/HL/LH/LL, range compression |
| `evidence` | `PhaseEvidence` | Regime/price/volume signals + contradictions |
| `transitions` | `list[PhaseTransition]` | Likely next phases with triggers |
| `strategy_comment` | `str` | Actionable guidance |

---

### 5. 0DTE Opportunity Assessment — "Should I trade 0DTE today?"

Combines regime, technicals, ORB, macro calendar, and fundamentals into a single go/no-go verdict with strategy recommendation.

```python
z = svc.assess_zero_dte("SPY")

print(f"Verdict: {z.verdict}")       # "go" | "caution" | "no_go"
print(f"Confidence: {z.confidence:.0%}")
print(f"Strategy: {z.strategy.name} — {z.strategy.structure}")
print(f"Rationale: {z.strategy.rationale}")

# Hard stops (any one → NO_GO)
for stop in z.hard_stops:
    print(f"  STOP: {stop.name} — {stop.description}")

# Scoring signals
for sig in z.signals:
    mark = "+" if sig.favorable else "-"
    print(f"  [{mark}] {sig.name} (w={sig.weight:.2f}): {sig.description}")

# Context
print(f"Regime: R{z.regime_id} ({z.regime_confidence:.0%})")
print(f"ATR: {z.atr_pct:.2f}%")
print(f"ORB: {z.orb_status}")
print(f"Macro today: {z.has_macro_event_today}")
print(f"Earnings in: {z.days_to_earnings} days")
print(z.summary)
```

**How it works:**

1. **Hard stops** — any one triggers NO_GO:

| Hard Stop | Condition |
|-----------|-----------|
| `earnings_blackout` | Earnings within 1 day |
| `macro_event_today` | HIGH-impact macro event today |
| `atr_too_low` | ATR < 0.3% (no movement to trade) |
| `atr_too_high` | ATR > 3.0% (too volatile for 0DTE) |
| `r4_high_confidence` | R4 regime with > 70% confidence |

2. **Scoring signals** — weighted 0DTE favorability:

| Signal | Weight | Favorable when |
|--------|--------|---------------|
| `regime_favorable` | 0.25 | R1 or R2 |
| `atr_sweet_spot` | 0.15 | 0.5% ≤ ATR ≤ 1.5% |
| `orb_alignment` | 0.15 | ORB breakout matches bias |
| `rsi_not_extreme` | 0.10 | 30 ≤ RSI ≤ 70 |
| `bollinger_favorable` | 0.10 | Price inside bands |
| `no_macro_tomorrow` | 0.10 | No HIGH events tomorrow |
| `sr_levels_defined` | 0.10 | Support & resistance present |
| `volume_normal` | 0.05 | No extreme volume anomaly |

3. **Verdict**: `confidence = sum(favorable_weights) × regime_multiplier`
   - GO if ≥ 0.55, CAUTION if ≥ 0.35, NO_GO otherwise (hard stops override)

4. **Strategy selection** (Regime × ORB):

| Regime | WITHIN range | Breakout Long | Breakout Short |
|--------|-------------|--------------|----------------|
| R1 | Iron Condor | Credit Put Spread | Credit Call Spread |
| R2 | Straddle/Strangle | Directional Spread | Directional Spread |
| R3 | Directional w/ trend | Directional Long | Directional Short |
| R4 | No Trade | No Trade | No Trade |

**Returns:** `ZeroDTEOpportunity`

| Field | Type | Description |
|-------|------|-------------|
| `verdict` | `Verdict` | GO / CAUTION / NO_GO |
| `confidence` | `float` | 0.0–1.0 composite score |
| `hard_stops` | `list[HardStop]` | Active blockers (empty if none) |
| `signals` | `list[OpportunitySignal]` | All scoring signals with weights |
| `strategy` | `StrategyRecommendation` | Recommended strategy with rationale |
| `zero_dte_strategy` | `ZeroDTEStrategy` | Strategy enum for programmatic use |
| `regime_id` | `int` | Current regime (1–4) |
| `atr_pct` | `float` | Current ATR as % of price |
| `orb_status` | `str \| None` | ORB status if intraday data available |
| `has_macro_event_today` | `bool` | Macro event flag |
| `days_to_earnings` | `int \| None` | Days until next earnings |
| `summary` | `str` | Human-readable assessment |

---

### 6. LEAP Opportunity Assessment — "Is this a good LEAP candidate?"

Combines regime, phase, technicals, fundamentals, and macro into a long-horizon (1–2 year) opportunity verdict.

```python
lp = svc.assess_leap("AAPL")

print(f"Verdict: {lp.verdict}")
print(f"Confidence: {lp.confidence:.0%}")
print(f"Phase: {lp.phase_name}, IV: {lp.iv_environment}")
print(f"Strategy: {lp.strategy.name} — {lp.strategy.structure}")

# Fundamental quality
fs = lp.fundamental_score
print(f"Fundamentals: {fs.score:.0%} — {fs.description}")
print(f"  Earnings growth: {fs.earnings_growth_signal}")
print(f"  Revenue growth: {fs.revenue_growth_signal}")
print(f"  Margins: {fs.margin_signal}")
print(f"  Debt: {fs.debt_signal}")
print(f"  Valuation: {fs.valuation_signal}")

# Hard stops
for stop in lp.hard_stops:
    print(f"  STOP: {stop.name} — {stop.description}")

# Signals
for sig in lp.signals:
    mark = "+" if sig.favorable else "-"
    print(f"  [{mark}] {sig.name} (w={sig.weight:.2f}): {sig.description}")

print(lp.summary)
```

**How it works:**

1. **Hard stops** — any one triggers NO_GO:

| Hard Stop | Condition |
|-----------|-----------|
| `iv_expensive` | R4 regime with > 70% confidence |
| `distribution_top` | P3 (Distribution) with > 65% confidence |
| `markdown_phase` | P4 (Markdown) with > 65% confidence |
| `earnings_imminent` | Earnings within 5 days |
| `weak_fundamentals` | Fundamental score < 20% |

2. **Fundamental score** (0–1 composite):

| Component | Weight | Strong | Moderate | Weak |
|-----------|--------|--------|----------|------|
| Earnings growth | 25% | > 15% | > 5% | > 0% |
| Revenue growth | 25% | > 15% | > 5% | > 0% |
| Profit margins | 20% | > 20% | > 10% | > 0% |
| Debt-to-equity | 15% | < 50 | < 100 | < 200 |
| Forward P/E | 15% | < 15 | < 25 | < 40 |

Missing data defaults to 0.5 (neutral). ETFs with no earnings data handled gracefully.

3. **Scoring signals:**

| Signal | Weight | Favorable when |
|--------|--------|---------------|
| `phase_entry_zone` | 0.25 | P1 (Accumulation) |
| `iv_cheap` | 0.20 | R1 or R3 |
| `fundamental_quality` | 0.20 | Score ≥ 0.6 |
| `trend_alignment` | 0.10 | Phase direction matches LEAP direction |
| `52wk_position` | 0.10 | Price in lower 40% of 52-week range |
| `macro_clear` | 0.05 | No HIGH events in next 7 days |
| `rsi_not_overbought` | 0.05 | RSI < 65 |

4. **Strategy selection** (Phase × Regime):

| Phase | R1 (cheap IV) | R2 (moderate) | R3 (moderate) | R4 (expensive) |
|-------|--------------|---------------|---------------|----------------|
| P1 Accumulation | Bull Call LEAP | Bull Call Spread | Bull Call LEAP | No Trade |
| P2 Markup | PMCC | PMCC / Call Spread | Bull Call LEAP | No Trade |
| P3 Distribution | Protective Put | Protective Put | No Trade | Bear Put LEAP |
| P4 Markdown | No Trade | Bear Put LEAP | No Trade | Bear Put LEAP |

**Returns:** `LEAPOpportunity`

| Field | Type | Description |
|-------|------|-------------|
| `verdict` | `Verdict` | GO / CAUTION / NO_GO |
| `confidence` | `float` | 0.0–1.0 composite score |
| `hard_stops` | `list[HardStop]` | Active blockers |
| `signals` | `list[OpportunitySignal]` | All scoring signals |
| `strategy` | `StrategyRecommendation` | Strategy with rationale & risk notes |
| `leap_strategy` | `LEAPStrategy` | Strategy enum for programmatic use |
| `phase_id` / `phase_name` | `int` / `str` | Current Wyckoff phase |
| `iv_environment` | `str` | "cheap" / "moderate" / "expensive" |
| `fundamental_score` | `FundamentalScore` | Composite with per-metric signals |
| `days_to_earnings` | `int \| None` | Days until next earnings |
| `macro_events_next_30_days` | `int` | Count of upcoming macro events |
| `summary` | `str` | Human-readable assessment |

---

### 7. Opening Range Breakout — "What's the ORB setup today?"

Compute first-30-minute opening range, detect breakouts, and provide extension levels.

```python
orb = svc.get_orb("SPY")

print(f"Range: ${orb.range_low:.2f} – ${orb.range_high:.2f} ({orb.range_pct:.2f}%)")
print(f"Status: {orb.status}")     # WITHIN | BREAKOUT_LONG | BREAKOUT_SHORT | FAILED_*
print(f"Current: ${orb.current_price:.2f}")
print(f"VWAP: ${orb.session_vwap:.2f}")

for level in orb.levels:
    print(f"  {level.label}: ${level.price:.2f} ({level.distance_pct:+.1f}%)")

for sig in orb.signals:
    print(f"  [{sig.direction}] {sig.name}: {sig.description}")
```

**Returns:** `ORBData`

| Field | Type | Description |
|-------|------|-------------|
| `range_high` / `range_low` | `float` | Opening range boundaries |
| `range_pct` | `float` | Range as % of range_low |
| `status` | `ORBStatus` | WITHIN, BREAKOUT_LONG/SHORT, FAILED_LONG/SHORT |
| `levels` | `list[ORBLevel]` | Extension levels (0.5×, 1×, 1.5×, 2× range) |
| `session_vwap` | `float \| None` | Session VWAP |
| `opening_volume_ratio` | `float` | Opening 30min volume vs prior day average |
| `range_vs_daily_atr_pct` | `float \| None` | ORB size relative to daily ATR |
| `retest_count` | `int` | Times price retested range boundary |
| `signals` | `list[TechnicalSignal]` | ORB-specific signals |

---

### 8. Fundamentals — "What are the fundamentals?"

Stock fundamentals via yfinance with in-memory TTL cache.

```python
fund = svc.get_fundamentals("AAPL")

print(f"{fund.business.long_name} ({fund.business.sector})")
print(f"Forward P/E: {fund.valuation.forward_pe}")
print(f"Earnings growth: {fund.earnings.earnings_growth:.1%}")
print(f"Revenue growth: {fund.revenue.revenue_growth:.1%}")
print(f"Profit margins: {fund.margins.profit_margins:.1%}")
print(f"Debt/Equity: {fund.debt.debt_to_equity}")
print(f"52-week: ${fund.fifty_two_week.low} – ${fund.fifty_two_week.high}")
print(f"  {fund.fifty_two_week.pct_from_high:+.1f}% from high")
print(f"Next earnings: {fund.upcoming_events.next_earnings_date}")
print(f"  in {fund.upcoming_events.days_to_earnings} days")
```

**Returns:** `FundamentalsSnapshot` — sections: `business`, `valuation`, `earnings`, `revenue`, `margins`, `cash`, `debt`, `returns`, `dividends`, `fifty_two_week`, `recent_earnings`, `upcoming_events`.

---

### 9. Macro Calendar — "What macro events are coming?"

Pre-built calendar of FOMC, CPI, NFP, PCE, GDP events (2025–2027).

```python
macro = svc.get_macro_calendar()

print(f"Next event: {macro.next_event.name} on {macro.next_event.date}")
print(f"  Impact: {macro.next_event.impact}")
print(f"  Options: {macro.next_event.options_impact}")
print(f"Next FOMC: {macro.next_fomc.date} ({macro.days_to_next_fomc} days)")

print(f"\nNext 7 days ({len(macro.events_next_7_days)} events):")
for e in macro.events_next_7_days:
    print(f"  {e.date} — {e.name} [{e.impact}]")
```

**Returns:** `MacroCalendar`

| Field | Type | Description |
|-------|------|-------------|
| `events` | `list[MacroEvent]` | All events in lookahead window |
| `next_event` | `MacroEvent \| None` | Closest upcoming event |
| `days_to_next` | `int \| None` | Days to closest event |
| `next_fomc` | `MacroEvent \| None` | Next FOMC specifically |
| `events_next_7_days` | `list[MacroEvent]` | Convenience: next week |
| `events_next_30_days` | `list[MacroEvent]` | Convenience: next month |

---

### 10. Historical Data — "Get OHLCV data"

Cache-first historical data service. Checks local parquet cache, fetches only missing dates.

```python
from market_regime import DataService

ds = DataService()

# Fetch OHLCV (cache-first, auto-fetches missing data)
df = ds.get_ohlcv("SPY")
# Returns: DataFrame with Open, High, Low, Close, Volume columns, DatetimeIndex

# With date range
from datetime import date
df = ds.get_ohlcv("GLD", start_date=date(2024, 1, 1))

# Check what's cached
for meta in ds.cache_status("SPY"):
    print(f"{meta.data_type}: {meta.first_date} → {meta.last_date} ({meta.row_count} rows)")

# Force re-fetch
ds.invalidate_cache("SPY")
```

---

## Standalone Functions

For callers who want to bring their own data instead of using `RegimeService`:

```python
from market_regime import (
    compute_features,      # OHLCV → feature matrix (log returns, vol, ATR, trend)
    compute_technicals,    # OHLCV → TechnicalSnapshot
    fetch_fundamentals,    # ticker → FundamentalsSnapshot
    get_macro_calendar,    # → MacroCalendar
    assess_zero_dte,       # pre-computed inputs → ZeroDTEOpportunity
    assess_leap,           # pre-computed inputs → LEAPOpportunity
    PhaseDetector,         # regime_series + OHLCV → PhaseResult
)

# Bring your own DataFrame
features_df = compute_features(my_ohlcv_df)
snapshot = compute_technicals(my_ohlcv_df, ticker="AAPL")

# Pure assessment functions (no data fetching)
from market_regime.opportunity.zero_dte import assess_zero_dte
result = assess_zero_dte(
    ticker="SPY",
    regime=my_regime_result,
    technicals=my_technicals,
    macro=my_macro_calendar,
    fundamentals=my_fundamentals,  # optional
    orb=my_orb_data,               # optional
)
```

---

## Configuration

All thresholds and parameters are configurable via YAML.

```python
from market_regime import get_settings

settings = get_settings()
print(settings.opportunity.zero_dte.go_threshold)   # 0.55
print(settings.opportunity.leap.go_threshold)        # 0.50
print(settings.regime.n_states)                      # 4
print(settings.cache.staleness_hours)                # 18.0
```

Override defaults by creating `~/.market_regime/config.yaml`:

```yaml
# Override any default setting
opportunity:
  zero_dte:
    go_threshold: 0.60          # Stricter 0DTE go threshold
    min_atr_pct: 0.4            # Require more movement
  leap:
    go_threshold: 0.45          # More lenient LEAP threshold
    earnings_blackout_days: 7   # Wider earnings buffer

cache:
  staleness_hours: 12           # Fresher data

technicals:
  rsi_period: 14
  rsi_overbought: 75            # Less sensitive overbought
```

### Key configuration sections

| Section | Controls |
|---------|---------|
| `regime` | HMM states, training lookback, refit frequency |
| `features` | Feature windows (vol, ATR, trend, volume) |
| `technicals` | MA periods, RSI/Bollinger/MACD params, VCP settings |
| `phases` | Swing detection, phase transition thresholds |
| `opportunity.zero_dte` | 0DTE hard stop thresholds, scoring weights, verdict thresholds |
| `opportunity.leap` | LEAP hard stop thresholds, fundamental scoring, verdict thresholds |
| `orb` | Opening range minutes, extension multipliers |
| `cache` | Staleness hours, cache directory |
| `fundamentals` | TTL cache minutes |
| `macro` | Lookahead days |

---

## Model Enums Reference

```python
from market_regime import RegimeID, PhaseID, TrendDirection
from market_regime.models.opportunity import Verdict, ZeroDTEStrategy, LEAPStrategy

# Regime (IntEnum — usable as int)
RegimeID.R1_LOW_VOL_MR      # 1
RegimeID.R2_HIGH_VOL_MR     # 2
RegimeID.R3_LOW_VOL_TREND   # 3
RegimeID.R4_HIGH_VOL_TREND  # 4

# Phase (IntEnum)
PhaseID.ACCUMULATION   # 1
PhaseID.MARKUP         # 2
PhaseID.DISTRIBUTION   # 3
PhaseID.MARKDOWN       # 4

# Trend
TrendDirection.BULLISH
TrendDirection.BEARISH

# Opportunity verdicts
Verdict.GO             # Conditions favor trading
Verdict.CAUTION        # Mixed signals — reduce size
Verdict.NO_GO          # Hard stop — do not trade

# 0DTE strategies
ZeroDTEStrategy.IRON_CONDOR
ZeroDTEStrategy.CREDIT_SPREAD
ZeroDTEStrategy.STRADDLE_STRANGLE
ZeroDTEStrategy.DIRECTIONAL_SPREAD
ZeroDTEStrategy.NO_TRADE

# LEAP strategies
LEAPStrategy.BULL_CALL_LEAP
LEAPStrategy.BULL_CALL_SPREAD
LEAPStrategy.BEAR_PUT_LEAP
LEAPStrategy.PROTECTIVE_PUT
LEAPStrategy.PMCC              # Poor Man's Covered Call
LEAPStrategy.NO_TRADE
```

---

## Integration Patterns

### Pattern 1: Full-service (recommended)

Let `RegimeService` handle everything — data fetching, caching, model fitting, analysis.

```python
svc = RegimeService(data_service=DataService())
result = svc.detect("SPY")           # auto-fetches, caches, fits, predicts
z = svc.assess_zero_dte("SPY")       # gathers all inputs automatically
```

### Pattern 2: Bring your own data

Provide pre-fetched DataFrames. Useful when you already have data from another source.

```python
svc = RegimeService()                 # no DataService
result = svc.detect("SPY", ohlcv=my_df)
technicals = svc.get_technicals("SPY", ohlcv=my_df)
```

### Pattern 3: Composable pure functions

Use standalone functions for maximum control. Each function takes pre-computed inputs and returns a model — no side effects.

```python
from market_regime.opportunity.zero_dte import assess_zero_dte
from market_regime.opportunity.leap import assess_leap

# You manage the data pipeline; we just score
z = assess_zero_dte("SPY", regime=r, technicals=t, macro=m)
lp = assess_leap("AAPL", regime=r, technicals=t, phase=p, macro=m, fundamentals=f)
```

### Pattern 4: Data layer only

Use `DataService` standalone for cache-first historical data. No regime detection needed.

```python
ds = DataService()
spy_daily = ds.get_ohlcv("SPY")                          # 2+ years cached
gld_recent = ds.get_ohlcv("GLD", start_date=date(2025, 1, 1))
```

---

## All Exports

```python
import market_regime

# Config
market_regime.Settings
market_regime.get_settings

# Services
market_regime.RegimeService
market_regime.DataService

# Regime
market_regime.RegimeID
market_regime.RegimeResult
market_regime.RegimeConfig
market_regime.RegimeExplanation
market_regime.HMMModelInfo
market_regime.RegimeTimeSeries
market_regime.RegimeTimeSeriesEntry
market_regime.TrendDirection

# Research
market_regime.TickerResearch
market_regime.CrossTickerEntry
market_regime.ResearchReport
market_regime.TransitionRow
market_regime.StateMeansRow
market_regime.LabelAlignmentDetail
market_regime.FeatureZScore
market_regime.RegimeHistoryDay
market_regime.RegimeDistributionEntry

# Phase
market_regime.PhaseID
market_regime.PhaseResult
market_regime.PhaseDetector

# Data
market_regime.DataType
market_regime.ProviderType
market_regime.DataRequest
market_regime.DataResult

# Features
market_regime.FeatureConfig
market_regime.FeatureInspection

# Technicals
market_regime.TechnicalSnapshot
market_regime.TechnicalSignal

# Fundamentals
market_regime.FundamentalsSnapshot
market_regime.fetch_fundamentals

# Macro
market_regime.MacroCalendar
market_regime.MacroEvent
market_regime.MacroEventType
market_regime.get_macro_calendar

# Opportunity
market_regime.Verdict
market_regime.ZeroDTEOpportunity
market_regime.LEAPOpportunity
market_regime.assess_zero_dte
market_regime.assess_leap

# Functions
market_regime.compute_features
market_regime.compute_technicals
```
