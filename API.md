# market_analyzer — API Reference

**Market analysis toolkit: HMM regime detection, technicals, unified price levels with R:R, phase detection, and opportunity assessment for options trading.**

---

## Quick Start

```python
from market_analyzer import MarketAnalyzer, DataService

ma = MarketAnalyzer(data_service=DataService())

# What regime is SPY in?
regime = ma.regime.detect("SPY")
print(f"{regime.ticker}: R{regime.regime} ({regime.confidence:.0%})")

# Full technical snapshot
tech = ma.technicals.snapshot("SPY")
print(f"RSI: {tech.rsi.value:.1f}, ATR: {tech.atr_pct:.2f}%")

# Unified price levels with stop loss, targets, R:R
levels = ma.levels.analyze("SPY")
print(f"Stop: ${levels.stop_loss.price:.2f}, Best R:R: {levels.best_target.risk_reward_ratio:.1f}")

# Should I trade 0DTE today?
z = ma.opportunity.assess_zero_dte("SPY")
print(f"0DTE: {z.verdict} — {z.strategy.name}")

# Is AAPL a good LEAP candidate?
lp = ma.opportunity.assess_leap("AAPL")
print(f"LEAP: {lp.verdict} — {lp.strategy.name}")

# Breakout setup?
bo = ma.opportunity.assess_breakout("AAPL")
print(f"Breakout: {bo.verdict} — {bo.breakout_strategy}")

# Momentum opportunity?
mo = ma.opportunity.assess_momentum("AAPL")
print(f"Momentum: {mo.verdict} — {mo.momentum_strategy}")
```

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  MarketAnalyzer  (facade — composes all services)                │
│                                                                  │
│  .regime        → RegimeService        HMM regime R1–R4          │
│  .technicals    → TechnicalService     RSI, MACD, VCP, ORB       │
│  .levels        → LevelsService        S/R, stop, targets, R:R   │
│  .phase         → PhaseService         Wyckoff P1–P4             │
│  .fundamentals  → FundamentalService   Valuation, earnings       │
│  .macro         → MacroService         FOMC, CPI, NFP, PCE       │
│  .opportunity   → OpportunityService   0DTE, LEAP, BO, MOM       │
│  .black_swan    → BlackSwanService     Tail-risk circuit breaker   │
│  .ranking       → TradeRankingService  Multi-ticker trade ranking  │
│  .data          → DataService          Cache-first OHLCV          │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                   ┌───────┴────────┐
                   │  DataService   │
                   │ (cache-first)  │
                   └────────────────┘
```

Each service is independently usable:

```python
from market_analyzer import TechnicalService, DataService

tech_svc = TechnicalService(data_service=DataService())
snap = tech_svc.snapshot("SPY")
```

Three conceptual layers:

| Layer | What it does | Examples |
|-------|-------------|---------|
| **Analysis** | Produces signals & indicators | Regime R1–R4, RSI, Phase P1–P4, ORB |
| **Levels** | Unified price levels, stop/target/R:R | Confluence S/R, stop loss, targets |
| **Opportunity** | Per-horizon go/no-go + recommended strategy | "0DTE: GO, sell iron condor" |

---

## Use Cases

### 1. Regime Detection — "What regime is this ticker in?"

Detect the current volatility/trend regime for any instrument. Every downstream decision depends on this.

```python
# Via facade
ma = MarketAnalyzer(data_service=DataService())
result = ma.regime.detect("GLD")

# Or standalone
from market_analyzer import RegimeService, DataService
svc = RegimeService(data_service=DataService())
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
research = ma.regime.research("GLD")
print(research.strategy_comment)
# "R1: Low-Vol MR — ideal for theta harvesting. Iron condors, strangles."

for feat in research.current_features:
    print(f"  {feat.feature}: z={feat.z_score:+.2f} ({feat.comment})")

for row in research.transition_matrix:
    print(f"  R{row.from_regime} → stay {row.stay_probability:.0%} ({row.stability})")

# Multi-ticker comparison
report = ma.regime.research_batch(tickers=["SPY", "GLD", "QQQ"])
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
technicals = ma.technicals.snapshot("AAPL")

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

### 4. Unified Price Levels — "Where are the key levels, and what's my R:R?"

Synthesizes all price level sources (swing S/R, MAs, Bollinger, VCP pivot, order blocks, FVGs, ORB, VWAP) into ranked support/resistance levels with confluence scoring, stop loss, take profit targets, and risk/reward ratios.

```python
# Basic usage — auto-detects direction from regime/phase
analysis = ma.levels.analyze("SPY")

# Ranked support levels (nearest first)
for lvl in analysis.support_levels:
    print(f"  S ${lvl.price:.2f} — strength={lvl.strength:.2f} sources={lvl.sources}")

# Ranked resistance levels (nearest first)
for lvl in analysis.resistance_levels:
    print(f"  R ${lvl.price:.2f} — strength={lvl.strength:.2f} sources={lvl.sources}")

# Trade planning
print(f"Direction: {analysis.direction}")
print(f"Entry: ${analysis.entry_price:.2f}")
if analysis.stop_loss:
    print(f"Stop: ${analysis.stop_loss.price:.2f} ({analysis.stop_loss.distance_pct:.1f}%)")
    print(f"  Risk/share: ${analysis.stop_loss.dollar_risk_per_share:.2f}")
for i, t in enumerate(analysis.targets, 1):
    print(f"  T{i}: ${t.price:.2f} R:R={t.risk_reward_ratio:.1f}")
if analysis.best_target:
    print(f"Best: ${analysis.best_target.price:.2f} R:R={analysis.best_target.risk_reward_ratio:.1f}")

print(analysis.summary)
# "SPY LONG | Entry $595.00 | Stop $590.50 (0.8%) | Best target $605.00 R:R=2.2 | 4S/3R levels"

# Override entry & direction
analysis = ma.levels.analyze("AAPL", entry_price=175.50, direction="long")

# Include ORB levels (intraday)
analysis = ma.levels.analyze("SPY", include_orb=True)

# Or standalone
from market_analyzer import LevelsService, TechnicalService, DataService
ds = DataService()
svc = LevelsService(
    technical_service=TechnicalService(data_service=ds),
)
analysis = svc.analyze("SPY")

# Pure function (bring your own TechnicalSnapshot)
from market_analyzer.features.levels import compute_levels
analysis = compute_levels(my_technicals, regime=my_regime, direction="long")
```

**How it works:**

1. **Extract** raw price levels from 8+ sources in `TechnicalSnapshot`:

| Source | Extraction | Weight |
|--------|-----------|--------|
| Swing S/R | `support_resistance.support/resistance` | 1.0 |
| Order Blocks | Non-broken OB high/low | 0.9 |
| VCP Pivot | `vcp.pivot_price` | 0.85 |
| SMA 200 | `moving_averages.sma_200` | 0.8 |
| SMA 50, FVGs | `sma_50`, unfilled FVG high/low | 0.7 |
| SMA 20, EMA 21, Bollinger, VWMA | Dynamic short-term levels | 0.5 |
| EMA 9 | Very short-term | 0.4 |
| ORB levels | Extension levels (if intraday) | 0.6 |

2. **Cluster** nearby levels within 0.5% proximity into confluence zones. Each cluster gets:
   - `confluence_score` = count of distinct sources merged
   - `strength` = sum of source weights / 3.0, capped at 1.0

3. **Classify** as support (below entry) or resistance (above entry), sorted nearest-first.

4. **Stop loss** from first support/resistance beyond min distance (0.3%), minus ATR buffer (0.5× ATR). Fallback: 2× ATR if no qualifying level.

5. **Targets** (up to 3) from opposite-side levels beyond min distance (0.5%). R:R = reward / risk for each.

6. **Direction auto-detection** priority: regime `trend_direction` → phase (markup/accumulation=long) → price vs SMA 50. Always overridable via `direction` parameter.

**Returns:** `LevelsAnalysis`

| Field | Type | Description |
|-------|------|-------------|
| `ticker` | `str` | Instrument symbol |
| `as_of_date` | `date` | Analysis date |
| `entry_price` | `float` | Entry price (current or overridden) |
| `direction` | `TradeDirection` | LONG or SHORT |
| `direction_auto_detected` | `bool` | True if direction was auto-detected |
| `current_price` | `float` | Latest price from snapshot |
| `atr` / `atr_pct` | `float` | ATR in dollars and as % of price |
| `support_levels` | `list[PriceLevel]` | Ranked support, nearest first (desc) |
| `resistance_levels` | `list[PriceLevel]` | Ranked resistance, nearest first (asc) |
| `stop_loss` | `StopLoss \| None` | Computed stop with ATR buffer |
| `targets` | `list[Target]` | Up to 3 targets, ordered by distance |
| `best_target` | `Target \| None` | Highest R:R above 1.5 threshold |
| `summary` | `str` | One-line human-readable summary |

`PriceLevel` fields:

| Field | Type | Description |
|-------|------|-------------|
| `price` | `float` | Level price |
| `role` | `LevelRole` | SUPPORT or RESISTANCE |
| `sources` | `list[LevelSource]` | All contributing sources |
| `confluence_score` | `int` | Count of distinct sources |
| `strength` | `float` | 0.0–1.0 weighted confluence |
| `distance_pct` | `float` | % distance from entry (negative = below) |
| `description` | `str` | Human-readable description |

`StopLoss` fields:

| Field | Type | Description |
|-------|------|-------------|
| `price` | `float` | Stop price |
| `distance_pct` | `float` | % distance from entry (always positive) |
| `dollar_risk_per_share` | `float` | Dollar risk per share |
| `level` | `PriceLevel` | Underlying support/resistance level |
| `atr_buffer` | `float` | ATR buffer applied |

`Target` fields:

| Field | Type | Description |
|-------|------|-------------|
| `price` | `float` | Target price |
| `distance_pct` | `float` | % distance from entry |
| `dollar_reward_per_share` | `float` | Dollar reward per share |
| `risk_reward_ratio` | `float` | R:R = reward / risk |
| `level` | `PriceLevel` | Underlying resistance/support level |

---

### 5. Phase Detection — "Where is this stock in its Wyckoff cycle?"

Detect accumulation/markup/distribution/markdown phases using regime history + price structure.

```python
phase = ma.phase.detect("AAPL")

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

### 6. 0DTE Opportunity Assessment — "Should I trade 0DTE today?"

Combines regime, technicals, ORB, macro calendar, and fundamentals into a single go/no-go verdict with strategy recommendation.

```python
z = ma.opportunity.assess_zero_dte("SPY")

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

### 7. LEAP Opportunity Assessment — "Is this a good LEAP candidate?"

Combines regime, phase, technicals, fundamentals, and macro into a long-horizon (1–2 year) opportunity verdict.

```python
lp = ma.opportunity.assess_leap("AAPL")

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

### 8. Breakout Opportunity Assessment — "Is a breakout setting up?"

Detects VCP, Bollinger squeeze, flag patterns, and evaluates breakout readiness using regime, technicals, phase, and smart money alignment.

```python
bo = ma.opportunity.assess_breakout("AAPL")

print(f"Verdict: {bo.verdict}")
print(f"Confidence: {bo.confidence:.0%}")
print(f"Type: {bo.breakout_type}")             # "bullish" | "bearish"
print(f"Strategy: {bo.breakout_strategy}")      # PIVOT_BREAKOUT, SQUEEZE_PLAY, etc.
print(f"Pivot: ${bo.pivot_price}")

# Setup quality
s = bo.setup
print(f"VCP: stage={s.vcp_stage}, score={s.vcp_score:.2f}")
print(f"BB Squeeze: {s.bollinger_squeeze}, BW={s.bollinger_bandwidth:.4f}")
print(f"Range compression: {s.range_compression:.2f}")
print(f"Volume: {s.volume_pattern}")
print(f"Days in base: {s.days_in_base}")
print(f"Smart money: {s.smart_money_alignment}")

# Hard stops (any one → NO_GO)
for stop in bo.hard_stops:
    print(f"  STOP: {stop.name} — {stop.description}")

# Scoring signals
for sig in bo.signals:
    mark = "+" if sig.favorable else "-"
    print(f"  [{mark}] {sig.name} (w={sig.weight:.2f}): {sig.description}")

print(bo.summary)
```

**How it works:**

1. **Hard stops** — any one triggers NO_GO:

| Hard Stop | Condition |
|-----------|-----------|
| `r4_high_confidence` | R4 regime with > 70% confidence |
| `earnings_imminent` | Earnings within 2 days |
| `no_base_established` | No VCP, no squeeze, no range compression |
| `already_extended` | Price > 5% above resistance |
| `r2_very_high_confidence` | R2 regime with > 80% confidence |

2. **Scoring signals:**

| Signal | Weight | Favorable when |
|--------|--------|---------------|
| `vcp_present` | 0.20 | VCP stage ≥ 2, score > 0.5 |
| `bollinger_squeeze` | 0.15 | Bandwidth in bottom 20% of range |
| `volume_contraction` | 0.10 | Declining volume during base |
| `support_resistance_defined` | 0.10 | Clear S/R levels |
| `regime_favorable` | 0.15 | R3 (trending) best |
| `phase_favorable` | 0.10 | P1 (Accumulation) or early P2 |
| `smart_money_alignment` | 0.05 | Order blocks / FVGs support direction |
| `macro_clear` | 0.05 | No HIGH events in next 2 days |
| `rsi_not_extreme` | 0.10 | RSI not overbought/oversold |

3. **Verdict**: `confidence = sum(favorable_weights) × regime_multiplier`
   - GO if ≥ 0.50, CAUTION if ≥ 0.30, NO_GO otherwise

4. **Strategy selection:**

| Pattern | Strategy |
|---------|---------|
| VCP present with pivot | `PIVOT_BREAKOUT` |
| Bollinger squeeze | `SQUEEZE_PLAY` |
| Higher lows, bullish structure | `BULL_FLAG` |
| Lower highs, bearish structure | `BEAR_FLAG` |
| Pullback to breakout level | `PULLBACK_TO_BREAKOUT` |

**Returns:** `BreakoutOpportunity`

| Field | Type | Description |
|-------|------|-------------|
| `verdict` | `Verdict` | GO / CAUTION / NO_GO |
| `confidence` | `float` | 0.0–1.0 composite score |
| `hard_stops` | `list[HardStop]` | Active blockers |
| `signals` | `list[OpportunitySignal]` | All scoring signals |
| `strategy` | `StrategyRecommendation` | Strategy with rationale |
| `breakout_strategy` | `BreakoutStrategy` | Strategy enum |
| `breakout_type` | `BreakoutType` | BULLISH or BEARISH |
| `setup` | `BreakoutSetup` | VCP, squeeze, volume, S/R, smart money details |
| `pivot_price` | `float \| None` | Key breakout price level |
| `regime_id` / `regime_confidence` | `int` / `float` | Current regime |
| `phase_id` / `phase_name` | `int` / `str` | Current phase |
| `days_to_earnings` | `int \| None` | Days until next earnings |
| `summary` | `str` | Human-readable assessment |

---

### 9. Momentum Opportunity Assessment — "Is momentum tradeable?"

Evaluates trend continuation, pullback, acceleration, and fade opportunities using MACD, RSI, MA alignment, stochastic, and volume confirmation.

```python
mo = ma.opportunity.assess_momentum("AAPL")

print(f"Verdict: {mo.verdict}")
print(f"Confidence: {mo.confidence:.0%}")
print(f"Direction: {mo.momentum_direction}")      # "bullish" | "bearish"
print(f"Strategy: {mo.momentum_strategy}")          # TREND_CONTINUATION, PULLBACK_ENTRY, etc.

# Momentum scoring
sc = mo.score
print(f"MACD histogram: {sc.macd_histogram_trend}")
print(f"MACD crossover: {sc.macd_crossover}")
print(f"RSI zone: {sc.rsi_zone}")
print(f"MA alignment: {sc.price_vs_ma_alignment}")
print(f"Golden/Death cross: {sc.golden_death_cross}")
print(f"Structure: {sc.structural_pattern}")
print(f"Volume: {sc.volume_confirmation}")
print(f"Stochastic: {sc.stochastic_confirmation}")
print(f"ATR trend: {sc.atr_trend}")

# Hard stops
for stop in mo.hard_stops:
    print(f"  STOP: {stop.name} — {stop.description}")

# Signals
for sig in mo.signals:
    mark = "+" if sig.favorable else "-"
    print(f"  [{mark}] {sig.name} (w={sig.weight:.2f}): {sig.description}")

print(mo.summary)
```

**How it works:**

1. **Hard stops** — any one triggers NO_GO:

| Hard Stop | Condition |
|-----------|-----------|
| `r1_high_confidence` | R1 (mean-reverting) with > 70% confidence |
| `earnings_imminent` | Earnings within 3 days |
| `rsi_extreme` | RSI > 85 or < 15 |
| `macd_crossover_against_trend` | MACD crossing against dominant trend |
| `volume_divergence_on_new_highs` | New price highs on declining volume |

2. **Scoring signals:**

| Signal | Weight | Favorable when |
|--------|--------|---------------|
| `macd_momentum` | 0.15 | Histogram expanding in trend direction |
| `rsi_favorable` | 0.10 | RSI 40–60 (pullback) or 55–75 (bullish continuation) |
| `ma_alignment` | 0.15 | Price > EMA 9 > SMA 20 > SMA 50 |
| `regime_favorable` | 0.15 | R3 (best) or R4 |
| `phase_favorable` | 0.10 | P2 (Markup) or P4 (Markdown) |
| `volume_confirmation` | 0.10 | Volume expanding with trend |
| `stochastic_confirmation` | 0.05 | %K > %D for bullish, %K < %D for bearish |
| `atr_expanding` | 0.05 | ATR trending up (movement increasing) |
| `structure_favorable` | 0.10 | HH/HL (bullish) or LH/LL (bearish) |
| `macro_clear` | 0.05 | No HIGH events in next 3 days |

3. **Verdict**: `confidence = sum(favorable_weights) × regime_multiplier`
   - GO if ≥ 0.50, CAUTION if ≥ 0.30, NO_GO otherwise

4. **Strategy selection:**

| Condition | Strategy |
|-----------|---------|
| Strong trend + expanding momentum | `TREND_CONTINUATION` |
| Trend intact + RSI pullback to neutral | `PULLBACK_ENTRY` |
| Momentum accelerating (histogram expanding) | `MOMENTUM_ACCELERATION` |
| Momentum exhaustion (RSI extreme + divergence) | `MOMENTUM_FADE` |

**Returns:** `MomentumOpportunity`

| Field | Type | Description |
|-------|------|-------------|
| `verdict` | `Verdict` | GO / CAUTION / NO_GO |
| `confidence` | `float` | 0.0–1.0 composite score |
| `hard_stops` | `list[HardStop]` | Active blockers |
| `signals` | `list[OpportunitySignal]` | All scoring signals |
| `strategy` | `StrategyRecommendation` | Strategy with rationale |
| `momentum_strategy` | `MomentumStrategy` | Strategy enum |
| `momentum_direction` | `MomentumDirection` | BULLISH or BEARISH |
| `score` | `MomentumScore` | Per-indicator scoring details |
| `regime_id` / `regime_confidence` | `int` / `float` | Current regime |
| `phase_id` / `phase_name` | `int` / `str` | Current phase |
| `days_to_earnings` | `int \| None` | Days until next earnings |
| `summary` | `str` | Human-readable assessment |

---

### 10. Opening Range Breakout — "What's the ORB setup today?"

Compute first-30-minute opening range, detect breakouts, and provide extension levels.

```python
orb = ma.technicals.orb("SPY")

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

### 11. Fundamentals — "What are the fundamentals?"

Stock fundamentals via yfinance with in-memory TTL cache.

```python
fund = ma.fundamentals.get("AAPL")

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

### 12. Macro Calendar — "What macro events are coming?"

Pre-built calendar of FOMC, CPI, NFP, PCE, GDP events (2025–2027).

```python
macro = ma.macro.calendar()

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

### 13. Historical Data — "Get OHLCV data"

Cache-first historical data service. Checks local parquet cache, fetches only missing dates.

```python
from market_analyzer import DataService

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

### 14. Black Swan / Tail-Risk Alert — "Is it safe to trade today?"

Daily circuit breaker that detects tail-risk conditions across 9 indicators. Pre-trade gate: if alert is CRITICAL, halt trading.

```python
ma = MarketAnalyzer(data_service=DataService())

alert = ma.black_swan.alert()

print(f"Alert: {alert.alert_level}")       # NORMAL | ELEVATED | HIGH | CRITICAL
print(f"Stress: {alert.composite_score:.0%}")

# Circuit breakers (any one → CRITICAL)
for cb in alert.circuit_breakers:
    if cb.triggered:
        print(f"  BREAKER: {cb.name} — {cb.description}")

# Individual indicators
for ind in alert.indicators:
    print(f"  {ind.name}: {ind.value} → {ind.status} (score={ind.score:.2f})")

print(f"Action: {alert.action}")
print(alert.summary)
```

**Indicators:** VIX level, VIX term structure, credit stress (HYG/LQD), SPY drawdown, realized vs implied vol, treasury stress (TLT), EM contagion (EEM/SPY), yield curve (FRED), put/call ratio (FRED).

**Circuit breakers:** VIX > 40, VIX/VIX3M > 1.20, SPY 1-day < -4%, HYG/LQD 1-day drop > 3%.

**Alert levels:**

| Level | Composite Score | Action |
|-------|----------------|--------|
| NORMAL | < 0.25 | Business as usual |
| ELEVATED | 0.25–0.50 | Reduce new positions, tighten stops |
| HIGH | 0.50–0.75 | Flatten directional, scale into hedges |
| CRITICAL | ≥ 0.75 or circuit breaker | Halt all new trades |

**Returns:** `BlackSwanAlert` — `alert_level`, `composite_score`, `circuit_breakers`, `indicators`, `triggered_breakers`, `action`, `summary`.

**Optional FRED data:** Install `pip install -e ".[fred]"` and set `FRED_API_KEY` env var for yield curve and put/call ratio indicators. Without it, those 2 indicators degrade gracefully and weights re-normalize.

---

### 15. Trade Ranking — "What's the best trade across all my tickers?"

Runs all opportunity assessments across multiple tickers and strategies, scores each combination, and returns ranked results. Answers both "best strategy for ticker X" and "best ticker for strategy Y."

```python
ma = MarketAnalyzer(data_service=DataService())

result = ma.ranking.rank(["SPY", "AAPL", "GLD", "MSFT"])

# Overall top trades
for entry in result.top_trades[:5]:
    print(f"#{entry.rank} {entry.ticker} {entry.strategy_type} "
          f"score={entry.composite_score:.2f} verdict={entry.verdict}")

# Best strategy per ticker
for ticker, entries in result.by_ticker.items():
    best = entries[0]
    print(f"{ticker}: {best.strategy_type} (score={best.composite_score:.2f})")

# Best ticker per strategy
for strat, entries in result.by_strategy.items():
    best = entries[0]
    print(f"{strat}: {best.ticker} (score={best.composite_score:.2f})")

# Black swan gate
print(f"Alert: {result.black_swan_level}, gate={result.black_swan_gate}")
print(result.summary)
```

**How it works:**

1. **Black swan gate** — if CRITICAL, halt all trading (empty ranking).

2. **For each ticker x strategy**, run opportunity assessment (0DTE, LEAP, breakout, momentum).

3. **Score each** with a weighted composite:

| Component | Weight | Source |
|-----------|--------|--------|
| `verdict_score` | 0.25 | GO=1.0, CAUTION=0.5, NO_GO=0.0 |
| `confidence_score` | 0.25 | Opportunity `.confidence` |
| `regime_alignment` | 0.15 | Regime x Strategy matrix |
| `risk_reward` | 0.15 | LevelsAnalysis R:R ratio |
| `technical_quality` | 0.10 | RSI + MACD + MA alignment + Stochastic |
| `phase_alignment` | 0.10 | Phase x Strategy matrix |

4. **Adjustments:**
   - Income-first bias: +0.05 boost for 0DTE in R1/R2 (theta harvesting)
   - Macro penalty: -0.02 per event in next 7 days (max -0.10)
   - Earnings penalty: -0.10 if earnings within 3 days
   - Black swan: multiplicative penalty `score x (1 - black_swan_score)`

5. **Sort** by composite score descending, assign ranks, group by ticker and strategy.

**Regime x Strategy alignment:**

| | R1 (Low-Vol MR) | R2 (High-Vol MR) | R3 (Low-Vol Trend) | R4 (High-Vol Trend) |
|---|---|---|---|---|
| zero_dte | 1.0 | 0.6 | 0.5 | 0.3 |
| leap | 0.3 | 0.2 | 1.0 | 0.5 |
| breakout | 0.4 | 0.3 | 0.8 | 1.0 |
| momentum | 0.2 | 0.3 | 1.0 | 0.8 |

**Phase x Strategy alignment:**

| | P1 (Accum) | P2 (Markup) | P3 (Distrib) | P4 (Markdown) |
|---|---|---|---|---|
| zero_dte | 0.7 | 0.8 | 0.7 | 0.4 |
| leap | 0.9 | 0.7 | 0.2 | 0.1 |
| breakout | 1.0 | 0.5 | 0.3 | 0.2 |
| momentum | 0.3 | 1.0 | 0.4 | 0.6 |

**Returns:** `TradeRankingResult`

| Field | Type | Description |
|-------|------|-------------|
| `as_of_date` | `date` | Ranking date |
| `tickers` | `list[str]` | Tickers assessed |
| `top_trades` | `list[RankedEntry]` | All entries sorted by score desc |
| `by_ticker` | `dict[str, list[RankedEntry]]` | Per-ticker, best first |
| `by_strategy` | `dict[StrategyType, list[RankedEntry]]` | Per-strategy, best first |
| `black_swan_level` | `str` | Current alert level |
| `black_swan_gate` | `bool` | True if CRITICAL (all halted) |
| `total_assessed` | `int` | Total ticker x strategy pairs |
| `total_actionable` | `int` | Entries with verdict != NO_GO |
| `summary` | `str` | Human-readable summary |

`RankedEntry` fields:

| Field | Type | Description |
|-------|------|-------------|
| `rank` | `int` | 1-based rank |
| `ticker` | `str` | Instrument |
| `strategy_type` | `StrategyType` | ZERO_DTE / LEAP / BREAKOUT / MOMENTUM |
| `verdict` | `Verdict` | GO / CAUTION / NO_GO |
| `composite_score` | `float` | 0.0-1.0 final score |
| `breakdown` | `ScoreBreakdown` | All component scores |
| `strategy_name` | `str` | Specific strategy (e.g. "iron_condor") |
| `direction` | `str` | "neutral" / "bullish" / "bearish" |
| `rationale` | `str` | Why this strategy |
| `risk_notes` | `list[str]` | Risk warnings |

**ML hook:** `WeightProvider` ABC allows swapping config-based weights for learned weights in the future.

```python
# Record feedback for future RL training
from market_analyzer import RankingFeedback, StrategyType, Verdict

feedback = RankingFeedback(
    as_of_date=date.today(),
    ticker="SPY",
    strategy_type=StrategyType.ZERO_DTE,
    composite_score=0.85,
    verdict=Verdict.GO,
    outcome_5d_return=0.012,
    outcome_20d_return=0.03,
)
ma.ranking.record_feedback(feedback)
```

**Configuration** (`~/.market_analyzer/config.yaml`):

```yaml
ranking:
  weights:
    verdict: 0.25
    confidence: 0.25
    regime_alignment: 0.15
    risk_reward: 0.15
    technical_quality: 0.10
    phase_alignment: 0.10
  income_bias_boost: 0.05
  macro_penalty_per_event: 0.02
  macro_penalty_max: 0.10
  earnings_penalty: 0.10
  earnings_proximity_days: 3
  risk_reward_excellent: 3.0
  risk_reward_good: 2.0
  risk_reward_fair: 1.0
```

---

## Standalone Functions

For callers who want to bring their own data instead of using the facade:

```python
from market_analyzer import (
    compute_features,      # OHLCV → feature matrix (log returns, vol, ATR, trend)
    compute_technicals,    # OHLCV → TechnicalSnapshot
    fetch_fundamentals,    # ticker → FundamentalsSnapshot
    get_macro_calendar,    # → MacroCalendar
    assess_zero_dte,       # pre-computed inputs → ZeroDTEOpportunity
    assess_leap,           # pre-computed inputs → LEAPOpportunity
    assess_breakout,       # pre-computed inputs → BreakoutOpportunity
    assess_momentum,       # pre-computed inputs → MomentumOpportunity
    PhaseDetector,         # regime_series + OHLCV → PhaseResult
)

# Bring your own DataFrame
features_df = compute_features(my_ohlcv_df)
snapshot = compute_technicals(my_ohlcv_df, ticker="AAPL")

# Pure levels computation (bring your own TechnicalSnapshot)
from market_analyzer.features.levels import compute_levels
analysis = compute_levels(my_technicals, regime=my_regime, direction="long", entry_price=175.50)

# Pure assessment functions (no data fetching)
from market_analyzer.opportunity.breakout import assess_breakout
result = assess_breakout(
    ticker="AAPL",
    regime=my_regime_result,
    technicals=my_technicals,
    phase=my_phase_result,
    macro=my_macro_calendar,
    fundamentals=my_fundamentals,  # optional
)
```

---

## Configuration

All thresholds and parameters are configurable via YAML.

```python
from market_analyzer import get_settings

settings = get_settings()
print(settings.opportunity.zero_dte.go_threshold)      # 0.55
print(settings.opportunity.leap.go_threshold)           # 0.50
print(settings.opportunity.breakout.go_threshold)       # 0.50
print(settings.opportunity.momentum.go_threshold)       # 0.50
print(settings.levels.confluence_proximity_pct)          # 0.5
print(settings.levels.min_risk_reward)                   # 1.5
print(settings.regime.n_states)                          # 4
print(settings.cache.staleness_hours)                    # 18.0
```

Override defaults by creating `~/.market_analyzer/config.yaml`:

```yaml
# Override any default setting
levels:
  confluence_proximity_pct: 0.75    # Wider clustering
  min_risk_reward: 2.0              # Stricter R:R threshold
  max_targets: 4                    # More targets
  atr_stop_buffer_multiple: 0.75    # Larger ATR buffer on stops
  source_weights:
    swing_support: 1.2              # Boost swing S/R weight
    order_block_high: 1.0

opportunity:
  zero_dte:
    go_threshold: 0.60              # Stricter 0DTE go threshold
    min_atr_pct: 0.4                # Require more movement
  leap:
    go_threshold: 0.45              # More lenient LEAP threshold
    earnings_blackout_days: 7       # Wider earnings buffer
  breakout:
    go_threshold: 0.55              # Stricter breakout threshold
    earnings_blackout_days: 3
  momentum:
    go_threshold: 0.55              # Stricter momentum threshold
    rsi_extreme_high: 80

cache:
  staleness_hours: 12               # Fresher data

technicals:
  rsi_period: 14
  rsi_overbought: 75                # Less sensitive overbought
```

### Key configuration sections

| Section | Controls |
|---------|---------|
| `regime` | HMM states, training lookback, refit frequency |
| `features` | Feature windows (vol, ATR, trend, volume) |
| `technicals` | MA periods, RSI/Bollinger/MACD params, VCP settings |
| `phases` | Swing detection, phase transition thresholds |
| `levels` | Confluence proximity, stop/target distances, ATR buffers, R:R threshold, source weights |
| `opportunity.zero_dte` | 0DTE hard stop thresholds, scoring weights, verdict thresholds |
| `opportunity.leap` | LEAP hard stop thresholds, fundamental scoring, verdict thresholds |
| `opportunity.breakout` | Breakout hard stop thresholds, pattern detection, verdict thresholds |
| `opportunity.momentum` | Momentum hard stop thresholds, scoring weights, verdict thresholds |
| `ranking` | Scoring weights, bias/penalty parameters, R:R thresholds |
| `orb` | Opening range minutes, extension multipliers |
| `cache` | Staleness hours, cache directory |
| `fundamentals` | TTL cache minutes |
| `macro` | Lookahead days |

---

## Model Enums Reference

```python
from market_analyzer import RegimeID, PhaseID, TrendDirection
from market_analyzer.models.levels import (
    LevelSource,
    LevelRole,
    TradeDirection,
)
from market_analyzer.models.opportunity import (
    Verdict,
    ZeroDTEStrategy,
    LEAPStrategy,
    BreakoutStrategy,
    BreakoutType,
    MomentumStrategy,
    MomentumDirection,
)

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

# Trade direction (for levels)
TradeDirection.LONG
TradeDirection.SHORT

# Level role
LevelRole.SUPPORT
LevelRole.RESISTANCE

# Level sources (17 sources)
LevelSource.SWING_SUPPORT        # Historic S/R
LevelSource.SWING_RESISTANCE
LevelSource.SMA_20               # Moving averages
LevelSource.SMA_50
LevelSource.SMA_200
LevelSource.EMA_9
LevelSource.EMA_21
LevelSource.BOLLINGER_UPPER      # Bollinger Bands
LevelSource.BOLLINGER_MIDDLE
LevelSource.BOLLINGER_LOWER
LevelSource.VWMA_20              # Volume-weighted MA
LevelSource.VCP_PIVOT            # VCP pattern pivot
LevelSource.ORDER_BLOCK_HIGH     # Smart money OB zones
LevelSource.ORDER_BLOCK_LOW
LevelSource.FVG_HIGH             # Fair value gaps
LevelSource.FVG_LOW
LevelSource.ORB_LEVEL            # Opening range breakout

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

# Breakout strategies
BreakoutStrategy.PIVOT_BREAKOUT
BreakoutStrategy.SQUEEZE_PLAY
BreakoutStrategy.BULL_FLAG
BreakoutStrategy.BEAR_FLAG
BreakoutStrategy.PULLBACK_TO_BREAKOUT
BreakoutStrategy.NO_TRADE

# Breakout type
BreakoutType.BULLISH
BreakoutType.BEARISH

# Momentum strategies
MomentumStrategy.TREND_CONTINUATION
MomentumStrategy.PULLBACK_ENTRY
MomentumStrategy.MOMENTUM_ACCELERATION
MomentumStrategy.MOMENTUM_FADE
MomentumStrategy.NO_TRADE

# Momentum direction
MomentumDirection.BULLISH
MomentumDirection.BEARISH
```

---

## Integration Patterns

### Pattern 1: MarketAnalyzer facade (recommended)

Single entry point. All services wired together automatically.

```python
from market_analyzer import MarketAnalyzer, DataService

ma = MarketAnalyzer(data_service=DataService())
regime = ma.regime.detect("SPY")
levels = ma.levels.analyze("SPY")
bo = ma.opportunity.assess_breakout("SPY")
```

### Pattern 2: Individual services

Use specific services when you only need one capability.

```python
from market_analyzer import RegimeService, TechnicalService, LevelsService, DataService

ds = DataService()
regime_svc = RegimeService(data_service=ds)
tech_svc = TechnicalService(data_service=ds)
levels_svc = LevelsService(technical_service=tech_svc, regime_service=regime_svc)

result = regime_svc.detect("SPY")
snap = tech_svc.snapshot("SPY")
analysis = levels_svc.analyze("SPY")
```

### Pattern 3: Bring your own data

Provide pre-fetched DataFrames. Useful when you already have data from another source.

```python
from market_analyzer import RegimeService, TechnicalService, LevelsService

regime_svc = RegimeService()             # no DataService
result = regime_svc.detect("SPY", ohlcv=my_df)

tech_svc = TechnicalService()
snap = tech_svc.snapshot("SPY", ohlcv=my_df)

levels_svc = LevelsService(technical_service=tech_svc)
analysis = levels_svc.analyze("SPY", ohlcv=my_df, direction="long", entry_price=595.0)
```

### Pattern 4: Composable pure functions

Use standalone functions for maximum control. Each function takes pre-computed inputs and returns a model — no side effects.

```python
from market_analyzer.features.levels import compute_levels
from market_analyzer.opportunity.zero_dte import assess_zero_dte
from market_analyzer.opportunity.breakout import assess_breakout

# Levels from pre-computed snapshot
analysis = compute_levels(my_technicals, regime=my_regime, direction="long")

# Opportunity scoring from pre-computed inputs
z = assess_zero_dte("SPY", regime=r, technicals=t, macro=m)
bo = assess_breakout("SPY", regime=r, technicals=t, phase=p, macro=m)
```

### Pattern 5: Data layer only

Use `DataService` standalone for cache-first historical data. No analysis needed.

```python
ds = DataService()
spy_daily = ds.get_ohlcv("SPY")                          # 2+ years cached
gld_recent = ds.get_ohlcv("GLD", start_date=date(2025, 1, 1))
```

---

## All Exports

```python
import market_analyzer

# Facade
market_analyzer.MarketAnalyzer

# Services
market_analyzer.RegimeService
market_analyzer.TechnicalService
market_analyzer.LevelsService
market_analyzer.PhaseService
market_analyzer.FundamentalService
market_analyzer.MacroService
market_analyzer.OpportunityService
market_analyzer.TradeRankingService
market_analyzer.DataService

# Config
market_analyzer.Settings
market_analyzer.get_settings

# Regime
market_analyzer.RegimeID
market_analyzer.RegimeResult
market_analyzer.RegimeConfig
market_analyzer.RegimeExplanation
market_analyzer.HMMModelInfo
market_analyzer.RegimeTimeSeries
market_analyzer.RegimeTimeSeriesEntry
market_analyzer.TrendDirection

# Research
market_analyzer.TickerResearch
market_analyzer.CrossTickerEntry
market_analyzer.ResearchReport
market_analyzer.TransitionRow
market_analyzer.StateMeansRow
market_analyzer.LabelAlignmentDetail
market_analyzer.FeatureZScore
market_analyzer.RegimeHistoryDay
market_analyzer.RegimeDistributionEntry

# Phase
market_analyzer.PhaseID
market_analyzer.PhaseResult
market_analyzer.PhaseDetector

# Data
market_analyzer.DataType
market_analyzer.ProviderType
market_analyzer.DataRequest
market_analyzer.DataResult

# Features
market_analyzer.FeatureConfig
market_analyzer.FeatureInspection

# Technicals
market_analyzer.TechnicalSnapshot
market_analyzer.TechnicalSignal

# Fundamentals
market_analyzer.FundamentalsSnapshot
market_analyzer.fetch_fundamentals

# Macro
market_analyzer.MacroCalendar
market_analyzer.MacroEvent
market_analyzer.MacroEventType
market_analyzer.get_macro_calendar

# Levels
market_analyzer.LevelRole
market_analyzer.LevelSource
market_analyzer.LevelsAnalysis
market_analyzer.PriceLevel
market_analyzer.StopLoss
market_analyzer.Target
market_analyzer.TradeDirection

# Opportunity
market_analyzer.Verdict
market_analyzer.ZeroDTEOpportunity
market_analyzer.LEAPOpportunity
market_analyzer.BreakoutOpportunity
market_analyzer.MomentumOpportunity
market_analyzer.assess_zero_dte
market_analyzer.assess_leap
market_analyzer.assess_breakout
market_analyzer.assess_momentum

# Black Swan
market_analyzer.AlertLevel
market_analyzer.BlackSwanAlert
market_analyzer.CircuitBreaker
market_analyzer.IndicatorStatus
market_analyzer.StressIndicator
market_analyzer.BlackSwanService
market_analyzer.compute_black_swan_alert

# Ranking
market_analyzer.TradeRankingService
market_analyzer.StrategyType
market_analyzer.ScoreBreakdown
market_analyzer.RankedEntry
market_analyzer.TradeRankingResult
market_analyzer.RankingFeedback

# Functions
market_analyzer.compute_features
market_analyzer.compute_technicals
```
