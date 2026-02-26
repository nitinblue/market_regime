# market_analyzer API Reference

Complete API reference for cotrader and external consumers.

---

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from market_analyzer import MarketAnalyzer, DataService

ma = MarketAnalyzer(data_service=DataService())

# US market (default)
ma = MarketAnalyzer(data_service=DataService())

# India market
ma = MarketAnalyzer(data_service=DataService(), market="India")
```

All services are accessible as attributes of `MarketAnalyzer`.

---

## Package Structure

```
market_analyzer/
├── models/              # Pydantic data models (no logic)
│   ├── vol_surface.py       # VolatilitySurface, TermStructurePoint, SkewSlice
│   └── opportunity.py       # Verdict, HardStop, OpportunitySignal, all opportunity models
├── features/            # Feature computation (technicals, screening)
│   ├── patterns/            # Consolidated price patterns
│   │   ├── vcp.py               # Volatility Contraction Pattern (Minervini)
│   │   ├── smart_money.py       # Order Blocks, Fair Value Gaps, Smart Money
│   │   └── orb.py               # Opening Range Breakout analysis
│   └── vol_surface.py       # compute_vol_surface() — options chain → VolatilitySurface
├── hmm/                 # HMM training and inference
├── opportunity/         # Opportunity assessment
│   ├── setups/          # Price-based pattern detection
│   │   ├── breakout.py      # VCP, Bollinger squeeze, resistance proximity
│   │   ├── momentum.py      # MACD, RSI, MA alignment, stochastic
│   │   ├── mean_reversion.py # RSI extremes, Bollinger bands
│   │   └── orb.py           # Opening Range Breakout setup (intraday)
│   └── option_plays/    # Option structure recommendations by horizon
│       ├── zero_dte.py      # Same-day: iron condors, credit spreads, straddles
│       ├── leap.py          # Long-term: bull call LEAPs, PMCC
│       ├── earnings.py      # Event-driven: pre-earnings straddles, IV crush
│       ├── iron_condor.py   # #1 income play: sell OTM strangle + long wings
│       ├── iron_butterfly.py # Short ATM straddle + long OTM wings
│       ├── calendar.py      # Front/back month IV differential plays
│       ├── diagonal.py      # Calendar with different strikes (directional bias)
│       └── ratio_spread.py  # Buy 1 ATM, sell 2 OTM (naked leg, margin-intensive)
├── service/             # Service layer (facade + workflow services)
│   └── vol_surface.py       # VolSurfaceService
├── data/                # Data fetching, caching (parquet), providers
├── config/              # Settings, defaults.yaml, market definitions
├── macro/               # Macro calendar (FOMC, RBI, econ schedule)
└── cli/                 # CLI tools (explorer, plotter, interactive REPL)
```

---

## Trading Workflow APIs

These services follow the trading decision workflow: What to buy → When to buy → How much → When to sell.

### Q1a: Is the Environment Safe? — `ma.context`

```python
ctx = ma.context.assess()
# Returns: MarketContext

ctx.environment_label     # "risk-on" | "cautious" | "defensive" | "crisis"
ctx.trading_allowed       # bool
ctx.position_size_factor  # 0.0–1.0 (scale down in stress)
ctx.black_swan            # BlackSwanAlert
ctx.macro                 # MacroCalendar
ctx.intermarket           # IntermarketDashboard

# Intermarket dashboard separately:
dashboard = ma.context.intermarket()
# Returns: IntermarketDashboard with regime reads of reference tickers
```

### Q1b: What's This Ticker Doing? — `ma.instrument`

```python
analysis = ma.instrument.analyze("SPY")
analysis = ma.instrument.analyze("SPY", include_opportunities=True)
# Returns: InstrumentAnalysis

analysis.regime           # RegimeResult
analysis.phase            # PhaseResult
analysis.technicals       # TechnicalSnapshot
analysis.levels           # LevelsAnalysis | None
analysis.fundamentals     # FundamentalsSnapshot | None
analysis.regime_id        # RegimeID (1-4)
analysis.phase_id         # PhaseID (1-4)
analysis.trend_bias       # "bullish" | "bearish" | "neutral"
analysis.volatility_label # "low" | "high"
analysis.actionable_setups # ["breakout", "momentum", ...]
analysis.summary          # str

# With include_opportunities=True:
analysis.breakout         # BreakoutOpportunity | None
analysis.momentum         # MomentumOpportunity | None
analysis.leap             # LEAPOpportunity | None
analysis.zero_dte         # ZeroDTEOpportunity | None

# Batch:
results = ma.instrument.analyze_batch(["SPY", "GLD", "QQQ"])
# Returns: dict[str, InstrumentAnalysis]
```

### Q1c: Where Are the Setups? — `ma.screening`

```python
result = ma.screening.scan(["SPY", "GLD", "QQQ", "TLT"])
result = ma.screening.scan(tickers, screens=["breakout", "momentum"])
# Returns: ScreeningResult

result.candidates         # list[ScreenCandidate] sorted by score desc
result.by_screen          # dict[str, list[ScreenCandidate]]
result.tickers_scanned    # int
result.summary            # str

# Each candidate:
c.ticker                  # str
c.screen                  # "breakout" | "momentum" | "mean_reversion" | "income"
c.score                   # 0.0–1.0
c.reason                  # str
c.regime_id               # int
c.rsi                     # float
c.atr_pct                 # float
```

Available screens: `breakout`, `momentum`, `mean_reversion`, `income`

### Q2: Is the Entry Confirmed? — `ma.entry`

```python
from market_analyzer import EntryTriggerType

result = ma.entry.confirm("SPY", EntryTriggerType.BREAKOUT_CONFIRMED)
# Returns: EntryConfirmation

result.confirmed          # bool
result.confidence         # 0.0–1.0
result.conditions         # list[EntryCondition]
result.conditions_met     # int
result.conditions_total   # int
result.suggested_entry_price  # float | None
result.suggested_stop_price   # float | None
result.risk_per_share     # float | None
result.summary            # str
```

Trigger types:
- `BREAKOUT_CONFIRMED` — price near/above resistance, VCP setup, MACD positive
- `PULLBACK_TO_SUPPORT` — price near support, RSI oversold, uptrend intact
- `MOMENTUM_CONTINUATION` — MACD positive, RSI healthy, above key MAs
- `MEAN_REVERSION_EXTREME` — RSI extreme, Bollinger extreme, stochastic confirmation
- `ORB_BREAKOUT` — above SMA20, MACD positive, RSI healthy, ATR tradeable

### Q3: What Structure + How Much? — `ma.strategy`

```python
# Select structure
params = ma.strategy.select("SPY", regime=r, technicals=t)
params = ma.strategy.select("SPY", regime=r, technicals=t, setup_type="breakout")
# Returns: StrategyParameters

params.primary_structure          # OptionStructure
params.alternative_structures     # list[OptionStructure]
params.suggested_dte_range        # (30, 45)
params.suggested_delta_range      # (0.20, 0.35)
params.wing_width_suggestion      # "5-wide" | "10-wide" | "15-wide" | "20-wide"
params.regime_rationale           # str
params.summary                    # str

# OptionStructure fields:
params.primary_structure.structure_type   # OptionStructureType enum
params.primary_structure.direction        # "neutral" | "bullish" | "bearish"
params.primary_structure.max_loss         # "defined" | "undefined"
params.primary_structure.theta_exposure   # "positive" | "negative" | "neutral"
params.primary_structure.vega_exposure    # "short" | "long" | "neutral"

# Size position
size = ma.strategy.size(params, current_price=580.0)
size = ma.strategy.size(params, current_price=580.0, account_size=200_000)
# Returns: PositionSize

size.suggested_contracts    # int
size.max_contracts          # int
size.max_risk_dollars       # float
size.margin_estimate        # float | None
size.buying_power_usage_pct # float | None
```

### Q4: When to Sell? — `ma.exit`

```python
plan = ma.exit.plan(
    "SPY",
    strategy=params,
    entry_price=580.0,
    regime=r,
    technicals=t,
    levels=l,    # optional
)
# Returns: ExitPlan

plan.profit_targets       # list[ExitTarget] — ordered nearest first
plan.stop_loss            # ExitTarget | None
plan.trailing_stop        # ExitTarget | None
plan.dte_exit_threshold   # int | None (close at this DTE)
plan.theta_decay_exit_pct # float | None (close at X% max profit)
plan.adjustments          # list[AdjustmentTrigger]
plan.regime_change_action # str
plan.risk_reward_ratio    # float | None
plan.summary              # str

# Each ExitTarget:
target.price              # float
target.pct_from_entry     # float
target.reason             # ExitReason enum
target.action             # "close 50%", "close all", "trail stop"

# Each AdjustmentTrigger:
adj.trigger_type          # AdjustmentTriggerType enum
adj.condition             # str (human-readable)
adj.action                # str
adj.urgency               # "immediate" | "next_session" | "monitor"
```

---

## Building Block APIs

Lower-level services that the workflow APIs compose. Also usable directly.

### Regime Detection — `ma.regime`

```python
r = ma.regime.detect("SPY")
# Returns: RegimeResult

r.regime                # RegimeID (R1_LOW_VOL_MR, R2_HIGH_VOL_MR, R3_LOW_VOL_TREND, R4_HIGH_VOL_TREND)
r.confidence            # float (0.0–1.0)
r.regime_probabilities  # dict[int, float]
r.trend_direction       # TrendDirection | None
r.as_of_date            # date

# Batch:
results = ma.regime.detect_batch(tickers=["SPY", "GLD"])
# Returns: dict[str, RegimeResult]
```

### Technical Snapshot — `ma.technicals`

```python
t = ma.technicals.snapshot("SPY")
# Returns: TechnicalSnapshot

t.current_price, t.atr, t.atr_pct
t.rsi.value, t.rsi.is_overbought, t.rsi.is_oversold
t.bollinger.bandwidth, t.bollinger.percent_b
t.macd.histogram, t.macd.is_bullish_crossover
t.stochastic.k, t.stochastic.d
t.moving_averages.sma_20, t.moving_averages.sma_50, t.moving_averages.sma_200
t.phase.phase, t.phase.confidence
t.vcp                   # VCPData | None
t.smart_money           # SmartMoneyData | None
t.signals               # list[TechnicalSignal]
```

### Phase Detection — `ma.phase`

```python
p = ma.phase.detect("SPY")
# Returns: PhaseResult

p.phase             # PhaseID (ACCUMULATION, MARKUP, DISTRIBUTION, MARKDOWN)
p.phase_name        # str
p.confidence        # float
p.phase_age_days    # int
p.evidence          # PhaseEvidence
p.transitions       # list[PhaseTransition]
p.strategy_comment  # str (LEAP-specific)
```

### Levels Analysis — `ma.levels`

```python
l = ma.levels.analyze("SPY")
# Returns: LevelsAnalysis

l.support_levels    # list[PriceLevel]
l.resistance_levels # list[PriceLevel]
l.stop_loss         # StopLoss | None
l.targets           # list[Target]
l.best_target       # Target | None
l.summary           # str
```

### Volatility Surface — `ma.vol_surface`

Fetches options chain data from yfinance, computes term structure, skew, and calendar edge metrics.

```python
surf = ma.vol_surface.surface("SPY")
# Returns: VolatilitySurface

# Term structure
surf.front_iv              # float — nearest expiration ATM IV
surf.back_iv               # float — ~30-60 DTE ATM IV
surf.term_slope            # float — (back - front) / front; positive = contango
surf.is_contango           # bool — back_iv > front_iv (normal)
surf.is_backwardation      # bool — front_iv > back_iv (event-driven)
surf.term_structure        # list[TermStructurePoint] — ATM IV per expiration
surf.expirations           # list[date]

# Skew
surf.skew_by_expiry        # list[SkewSlice] — put/call skew per expiration

# SkewSlice fields:
sk = surf.skew_by_expiry[0]
sk.atm_iv                  # float
sk.otm_put_iv              # float — ~25-delta put IV
sk.otm_call_iv             # float — ~25-delta call IV
sk.put_skew                # float — otm_put_iv - atm_iv
sk.call_skew               # float — otm_call_iv - atm_iv
sk.skew_ratio              # float — put_skew / call_skew (>1 = put-heavy)

# Calendar spread metrics
surf.calendar_edge_score   # float 0-1 — how favorable for calendar spreads
surf.best_calendar_expiries # tuple[date, date] | None — (sell_front, buy_back)
surf.iv_differential_pct   # float — (front - back) / back as pct

# Data quality
surf.data_quality          # "good" | "fair" | "poor"
surf.total_contracts       # int
surf.avg_bid_ask_spread_pct # float
surf.underlying_price      # float
surf.summary               # str
```

Convenience methods:

```python
# Just term structure
ts = ma.vol_surface.term_structure("SPY")
# Returns: list[TermStructurePoint]

# Skew for nearest (or specified) expiration
sk = ma.vol_surface.skew("SPY")
sk = ma.vol_surface.skew("SPY", expiration=date(2026, 4, 17))
# Returns: SkewSlice

# Calendar edge score only
edge = ma.vol_surface.calendar_edge("SPY")
# Returns: float (0-1)
```

### Opportunity Assessment — `ma.opportunity`

All option play assessors return a consistent structure: `verdict` (GO/CAUTION/NO_GO), `confidence` (0-1), `hard_stops`, `signals`, `strategy` recommendation, and a play-specific strategy variant.

#### Setups — Price-Based Patterns

```python
bo = ma.opportunity.assess_breakout("SPY")       # BreakoutOpportunity
mo = ma.opportunity.assess_momentum("SPY")       # MomentumOpportunity
# Mean reversion: from market_analyzer import assess_mean_reversion
```

#### ORB Setup — `ma.opportunity.assess_orb()` / `assess_orb()`

Opening Range Breakout setup assessment. Evaluates ORB status, volume, range size,
regime alignment, RSI, and VWAP to determine if the ORB presents a tradeable setup.

**Requires intraday data** — returns NO_GO without it (ORB is an intraday pattern).

```python
# Via OpportunityService (provides intraday data):
orb = ma.opportunity.assess_orb("SPY", intraday=intraday_df)

# Via pure function (direct call with pre-computed data):
from market_analyzer import assess_orb
orb = assess_orb("SPY", regime, technicals, orb=orb_data, phase=phase)

# Returns: ORBSetupOpportunity
orb.verdict               # Verdict (GO, CAUTION, NO_GO)
orb.confidence            # float 0-1
orb.strategy              # ORBStrategy:
                          #   BREAKOUT_CONTINUATION     — clean breakout, no retest yet
                          #   BREAKOUT_WITH_RETEST      — breakout held after retesting range edge
                          #   FAILED_BREAKOUT_REVERSAL  — broke out then returned — reversal signal
                          #   NARROW_RANGE_ANTICIPATION — tight range, watching for directional move
                          #   NO_TRADE
orb.direction             # "bullish", "bearish", or "neutral"
orb.orb_status            # ORBStatus value (breakout_long, within, failed_long, etc.)
orb.range_pct             # float — opening range as % of midpoint
orb.opening_volume_ratio  # float — opening volume vs session average
orb.range_vs_daily_atr_pct # float | None — range as % of daily ATR
orb.hard_stops            # list[HardStop]
orb.signals               # list[OpportunitySignal]
orb.summary               # str
```

**Hard stops**: R4 high confidence, earnings today/tomorrow, opening range > 85% of daily ATR.

**Signals** (7): ORB status (0.25), opening volume (0.15), range vs ATR (0.15), retest confirmation (0.10), regime alignment (0.15), RSI alignment (0.10), VWAP alignment (0.10).

**Regime alignment**:
- R1/R2 (mean-reverting): Failed breakouts score higher, clean breakouts penalized
- R3 (trending): Clean breakouts score higher
- R4: Hard stop at high confidence

#### Features Patterns — `features/patterns/`

Consolidated price structure pattern detection (extracted from technicals.py):

```python
from market_analyzer.features.patterns import (
    compute_vcp,           # Volatility Contraction Pattern
    compute_order_blocks,  # Order Blocks (smart money demand/supply zones)
    compute_fair_value_gaps, # Fair Value Gaps (3-candle imbalances)
    compute_smart_money,   # OB + FVG orchestrator with confluence scoring
    compute_orb,           # Opening Range Breakout from intraday data
)

# All functions are pure — accept DataFrames, return model instances.
# Backward compat: these are still importable from features.technicals and features.orb.
```

#### Iron Condor — `ma.opportunity.assess_iron_condor()`

The **#1 income strategy**. Sell OTM put + OTM call, buy further OTM wings. Defined risk.
Best in R1 (prime theta harvesting), good in R2 (wider wings). Avoid R3/R4.

```python
ic = ma.opportunity.assess_iron_condor("SPY")
# Returns: IronCondorOpportunity

ic.verdict                # Verdict (GO, CAUTION, NO_GO)
ic.confidence             # float 0-1
ic.hard_stops             # list[HardStop]
ic.signals                # list[OpportunitySignal]
ic.strategy               # StrategyRecommendation (name, direction, structure, rationale, risk_notes)
ic.iron_condor_strategy   # IronCondorStrategy:
                          #   STANDARD_IRON_CONDOR — equal wings, R1 ideal
                          #   WIDE_IRON_CONDOR     — wider strikes for R2
                          #   UNBALANCED_IRON_CONDOR — directional tilt (RSI skewed)
                          #   NARROW_IRON_CONDOR   — tighter strikes for low IV
                          #   NO_TRADE
ic.front_iv               # float
ic.put_skew               # float
ic.call_skew              # float
ic.wing_width_suggestion  # str — e.g. "Short strikes ~1.0% OTM (6pt), wings +0.5% beyond"
ic.trade_spec             # TradeSpec | None — actionable legs, strikes, expirations (see TradeSpec section)
ic.regime_id              # int
ic.regime_confidence      # float
ic.days_to_earnings       # int | None
ic.summary                # str
```

**Hard stops**: R4 high confidence, R3 high confidence, no vol surface, IV < 10%, poor data quality, earnings imminent.

**Signals** (8, weighted to 1.0): IV level (0.20), regime R1/R2 (0.25), RSI centered (0.15), BB bandwidth (0.10), skew balance (0.10), no earnings (0.10), term structure (0.05), liquidity (0.05).

#### Iron Butterfly — `ma.opportunity.assess_iron_butterfly()`

Short ATM straddle + long OTM wings. Maximum premium at ATM.
Best in R2 (high IV + mean-reverting). Avoid R3/R4.

```python
ifly = ma.opportunity.assess_iron_butterfly("SPY")
# Returns: IronButterflyOpportunity

ifly.verdict              # Verdict
ifly.confidence           # float
ifly.iron_butterfly_strategy  # IronButterflyStrategy:
                              #   STANDARD_IRON_BUTTERFLY — R2 high confidence
                              #   BROKEN_WING_BUTTERFLY   — R1 + directional tilt
                              #   WIDE_IRON_BUTTERFLY     — lower confidence, more room
                              #   NO_TRADE
ifly.atm_iv               # float
ifly.front_iv             # float
ifly.trade_spec           # TradeSpec | None
ifly.days_to_earnings     # int | None
ifly.summary              # str
```

**Hard stops**: R3/R4 trending at high confidence, ATM IV < 15%, no vol surface, poor data, earnings imminent.

#### Calendar Spread — `ma.opportunity.assess_calendar()`

Sell front-month, buy back-month at same strike. Edge from IV differential + time decay.
Best in R1/R2 (range-bound, front IV elevated). Worst in R4.

```python
cal = ma.opportunity.assess_calendar("SPY")
# Returns: CalendarOpportunity

cal.verdict               # Verdict
cal.confidence            # float
cal.calendar_strategy     # CalendarStrategy:
                          #   ATM_CALENDAR      — same ATM strike, pure theta (R1)
                          #   OTM_CALL_CALENDAR — bullish bias (R3 mild uptrend)
                          #   OTM_PUT_CALENDAR  — bearish bias (R3 mild downtrend)
                          #   DOUBLE_CALENDAR   — two strikes bracketing price (R2)
                          #   NO_TRADE
cal.front_iv              # float
cal.back_iv               # float
cal.term_slope            # float
cal.calendar_edge_score   # float
cal.trade_spec            # TradeSpec | None — includes front/back expiration, IV diff
cal.days_to_earnings      # int | None
cal.summary               # str
```

**Hard stops**: R4 high confidence, no vol surface, poor data, wide bid-ask, earnings imminent.

#### Diagonal Spread — `ma.opportunity.assess_diagonal()`

Calendar with different strikes — sell OTM front, buy ATM/ITM back. Directional bias + theta.
Best in R3 (mild trend + time decay). Also works in R1 (PMCC variant).

```python
diag = ma.opportunity.assess_diagonal("SPY")
# Returns: DiagonalOpportunity

diag.verdict              # Verdict
diag.confidence           # float
diag.diagonal_strategy    # DiagonalStrategy:
                          #   BULL_CALL_DIAGONAL — sell OTM call front, buy ATM/ITM call back
                          #   BEAR_PUT_DIAGONAL  — sell OTM put front, buy ATM/ITM put back
                          #   PMCC_DIAGONAL      — poor man's covered call (deep ITM back)
                          #   NO_TRADE
diag.direction            # "bullish" | "bearish"
diag.front_iv             # float
diag.back_iv              # float
diag.phase_id             # int | None
diag.phase_name           # str | None
diag.trend_direction      # str | None
diag.trade_spec           # TradeSpec | None — includes front/back different strikes
diag.days_to_earnings     # int | None
diag.summary              # str
```

**Hard stops**: R4 high confidence, no vol surface, extreme skew, earnings between legs.

#### Ratio Spread — `ma.opportunity.assess_ratio_spread()`

Buy 1 ATM, sell 2 OTM. Profits from premium collection + limited directional move.
**Has naked leg — margin-intensive, undefined risk.** Best in R1 (range-bound) or R3 (mild trend).

```python
rs = ma.opportunity.assess_ratio_spread("SPY")
# Returns: RatioSpreadOpportunity

rs.verdict                # Verdict
rs.confidence             # float
rs.ratio_strategy         # RatioSpreadStrategy:
                          #   CALL_RATIO_SPREAD — buy 1 ATM call, sell 2 OTM calls (bullish)
                          #   PUT_RATIO_SPREAD  — buy 1 ATM put, sell 2 OTM puts (bearish)
                          #   CALL_BACK_RATIO   — sell 1 ATM call, buy 2 OTM calls (long vol)
                          #   PUT_BACK_RATIO    — sell 1 ATM put, buy 2 OTM puts (crash protection)
                          #   NO_TRADE
rs.direction              # "bullish" | "bearish"
rs.has_naked_leg          # bool — True for standard ratios
rs.margin_warning         # str | None — margin requirements notice
rs.front_iv               # float
rs.put_skew               # float
rs.call_skew              # float
rs.trade_spec             # TradeSpec | None — buy 1 ATM, sell 2 OTM legs
rs.days_to_earnings       # int | None
rs.summary                # str
```

**Hard stops**: R4 at ANY confidence, R2 high confidence, no vol surface, earnings imminent, skew too flat (< 2%).

#### Zero-DTE & LEAP

```python
zd = ma.opportunity.assess_zero_dte("SPY")       # ZeroDTEOpportunity
zd = ma.opportunity.assess_zero_dte("SPY", intraday=intraday_df)  # with ORB data

lo = ma.opportunity.assess_leap("SPY")            # LEAPOpportunity

# Earnings play (direct import):
from market_analyzer import assess_earnings_play
```

#### Common Output Pattern

All opportunity assessors share this structure:

```python
result.verdict            # Verdict.GO | Verdict.CAUTION | Verdict.NO_GO
result.confidence         # 0.0–1.0 (weighted signals * regime multiplier)
result.hard_stops         # list[HardStop] — if any, verdict = NO_GO
result.signals            # list[OpportunitySignal] — weighted scoring inputs
result.strategy           # StrategyRecommendation (name, direction, structure, rationale, risk_notes)
result.summary            # str — one-line summary

# HardStop:
hs.name                   # str — e.g. "R4 trending"
hs.description            # str — human-readable explanation

# OpportunitySignal:
sig.name                  # str
sig.favorable             # bool
sig.weight                # float (0-1, all weights sum to ~1.0)
sig.description           # str
```

#### TradeSpec — Actionable Trade Parameters

Every assessor (option plays, setups, 0DTE, LEAP, earnings) includes a `trade_spec` when verdict is GO or CAUTION. `None` for NO_GO.

```python
from market_analyzer import LegAction  # BTO, STO

ic = ma.opportunity.assess_iron_condor("SPY")
if ic.trade_spec:
    ts = ic.trade_spec
    ts.ticker                 # "SPY"
    ts.underlying_price       # 580.0
    ts.target_dte             # 35
    ts.target_expiration      # date(2026, 3, 27)

    # Legs — each with strike, expiration, role, action, quantity
    for leg in ts.legs:
        leg.role              # "short_put", "long_put", "short_call", "long_call"
        leg.action            # LegAction.SELL_TO_OPEN or LegAction.BUY_TO_OPEN
        leg.quantity          # 1 (default), 2 for ratio spread short leg
        leg.option_type       # "call" or "put"
        leg.strike            # 570.0 (snapped to standard tick)
        leg.strike_label      # "1.0 ATR OTM put"
        leg.expiration        # date(2026, 3, 27)
        leg.days_to_expiry    # 35
        leg.atm_iv_at_expiry  # 0.22
        leg.short_code        # "STO 1x 570P 3/27/26"
        leg.osi_symbol        # "260327P00570000"

    # Human-readable leg codes (with ticker, action, quantity)
    ts.leg_codes              # ["STO 1x SPY P570 3/27/26", "BTO 1x SPY P565 3/27/26", ...]

    # Full OCC streamer symbols
    ts.streamer_symbols       # ["SPY   260327P00570000", ...]

    # Machine-readable order data for cotrader
    ts.order_data             # [{"action": "STO", "quantity": 1, "symbol": "SPY",
                              #   "option_type": "put", "strike": 570.0,
                              #   "expiration": "2026-03-27",
                              #   "osi_symbol": "SPY   260327P00570000"}, ...]

    # Single-expiry structures (IC, IFly, ratio):
    ts.wing_width_points      # 5.0 (IC/IFly: distance between short and long strike)
    ts.max_risk_per_spread    # "$500 - credit received"

    # Calendar/diagonal structures:
    ts.front_expiration       # date — front month
    ts.front_dte              # int
    ts.back_expiration        # date — back month
    ts.back_dte               # int
    ts.iv_at_front            # float
    ts.iv_at_back             # float
    ts.iv_differential_pct    # float — (front - back) / back * 100

    ts.spec_rationale         # str — why these dates/strikes
```

**LegAction enum**: `LegAction.BUY_TO_OPEN` ("BTO") / `LegAction.SELL_TO_OPEN` ("STO").

**Quantity**: Default 1. Ratio spreads use `quantity=2` on the short leg (2 legs instead of 3).

**Strike snapping rules**: <$50 -> $0.50 ticks, <$200 -> $1.00 ticks, >= $200 -> $5.00 ticks.

**Expirations are always real**: sourced from the vol surface term structure (computed from actual options chain data).

#### Pure Function Access

All assessors are also available as pure functions (no service dependency):

```python
from market_analyzer import (
    assess_iron_condor,
    assess_iron_butterfly,
    assess_calendar,
    assess_diagonal,
    assess_ratio_spread,
    assess_zero_dte,
    assess_leap,
    assess_breakout,
    assess_momentum,
    assess_mean_reversion,
    assess_earnings_play,
)

# Pure function pattern: pass pre-computed analysis, no data fetching
result = assess_iron_condor(
    ticker="SPY",
    regime=regime_result,
    technicals=technical_snapshot,
    vol_surface=vol_surface,          # optional
    fundamentals=fundamentals,        # optional
    as_of=date.today(),               # optional
)
```

### Trade Ranking — `ma.ranking`

```python
result = ma.ranking.rank(["SPY", "GLD", "QQQ", "TLT"])
# Returns: TradeRankingResult

result.top_trades       # list[RankedEntry]
result.by_ticker        # dict[str, list[RankedEntry]]
result.by_strategy      # dict[StrategyType, list[RankedEntry]]
result.black_swan_gate  # bool (True = CRITICAL, halt trading)

# StrategyType enum (11 strategies):
#   ZERO_DTE, LEAP, BREAKOUT, MOMENTUM,
#   IRON_CONDOR, IRON_BUTTERFLY, CALENDAR, DIAGONAL, RATIO_SPREAD,
#   EARNINGS, MEAN_REVERSION

# Each RankedEntry now includes trade_spec:
entry = result.top_trades[0]
entry.trade_spec        # TradeSpec | None — concrete legs for the recommendation
entry.trade_spec.leg_codes    # ["STO 1x SPY P570 3/27/26", ...]
entry.trade_spec.order_data   # machine-readable dicts for cotrader
```

### Black Swan Alert — `ma.black_swan`

```python
alert = ma.black_swan.alert()
# Returns: BlackSwanAlert

alert.alert_level       # AlertLevel (NORMAL, ELEVATED, HIGH, CRITICAL)
alert.composite_score   # float (0.0–1.0)
alert.circuit_breakers  # list[CircuitBreaker]
alert.indicators        # list[StressIndicator]
```

### Macro Calendar — `ma.macro`

```python
cal = ma.macro.calendar()
# Returns: MacroCalendar

cal.next_event, cal.days_to_next
cal.next_fomc, cal.days_to_next_fomc
cal.events_next_7_days, cal.events_next_30_days
```

### Fundamentals — `ma.fundamentals`

```python
f = ma.fundamentals.get("SPY")
# Returns: FundamentalsSnapshot

f.valuation, f.earnings, f.revenue, f.margins
f.upcoming_events.days_to_earnings
```

### Data Service — `ma.data`

```python
from market_analyzer import DataService
ds = DataService()

df = ds.get_ohlcv("SPY")              # pd.DataFrame (OHLCV)
chain = ds.get_options_chain("SPY")    # pd.DataFrame (options chain)
status = ds.cache_status("SPY")        # list[CacheMeta]
ds.invalidate_cache("SPY")             # Force re-fetch
```

Options chain DataFrame columns:
```
expiration(date), strike(float), option_type("call"/"put"),
bid(float), ask(float), last_price(float), volume(int),
open_interest(int), implied_volatility(float), in_the_money(bool)
```

Cache behavior:
- OHLCV: time-series, 18hr staleness, delta-fetch (appends new dates)
- Options chain: snapshot, 4hr staleness, full refresh when stale

---

## Multi-Market Support

Yahoo Finance Indian stocks use `.NS` (NSE) or `.BO` (BSE) suffixes.

```python
# India market
ma = MarketAnalyzer(data_service=DataService(), market="India")
r = ma.regime.detect("RELIANCE.NS")
ctx = ma.context.assess()   # Uses India reference tickers (NIFTY, BANKNIFTY)

# Explicit tickers work regardless of market setting
r = ma.regime.detect("TCS.NS")
r = ma.regime.detect("INFY.NS")
```

Market-specific behavior:
- `MarketContextService` uses market-specific reference tickers
- `BlackSwanService` uses market-specific VIX (^INDIAVIX for India)
- All analysis logic (regime, phase, technicals, levels) is market-agnostic

---

## Interactive CLI

```bash
analyzer-cli                    # US market (default)
analyzer-cli --market india     # India market

# Or run without installing:
.venv_312/Scripts/python -m market_analyzer.cli.interactive
```

### Commands

| Command | Description | Example |
|---------|-------------|---------|
| `context` | Market environment assessment | `context` |
| `analyze <ticker>` | Full instrument analysis | `analyze SPY` |
| `screen <tickers...>` | Find setups across tickers | `screen SPY GLD QQQ TLT` |
| `entry <ticker> <type>` | Confirm entry signal | `entry SPY breakout` |
| `strategy <ticker>` | Strategy recommendation | `strategy SPY` |
| `exit_plan <ticker> <price>` | Exit plan | `exit_plan SPY 580` |
| `rank <tickers...>` | Trade ranking | `rank SPY GLD QQQ TLT` |
| `regime <tickers...>` | Regime detection | `regime SPY GLD` |
| `technicals <ticker>` | Technical snapshot | `technicals SPY` |
| `levels <ticker>` | Support/resistance levels | `levels SPY` |
| `vol <ticker>` | Volatility surface | `vol SPY` |
| `opportunity <ticker> [play]` | Option play assessment | `opportunity SPY ic` |
| `macro` | Macro calendar | `macro` |
| `stress` | Tail-risk alert | `stress` |
| `help` | List all commands | `help` |
| `quit` | Exit | `quit` |

### `opportunity` Command

Assesses vol-surface-dependent option plays for a ticker.

```
opportunity SPY              # all plays (default)
opportunity SPY ic           # iron condor only
opportunity SPY ifly         # iron butterfly only
opportunity SPY calendar     # calendar spread only
opportunity SPY diagonal     # diagonal spread only
opportunity SPY ratio        # ratio spread only
opportunity SPY zero_dte     # zero-DTE only
opportunity SPY leap         # LEAP only
opportunity SPY all          # all plays
```

Aliases: `ic` = iron_condor, `ifly` = iron_butterfly, `cal` = calendar, `diag` = diagonal, `0dte` = zero_dte

---

## Regime → Strategy Mapping

| Regime | Primary Strategy | Option Plays | Avoid | Vega |
|--------|-----------------|--------------|-------|------|
| R1: Low-Vol MR | Iron condors, strangles | IC (standard), Calendar (ATM), Diagonal (PMCC), Ratio spread | Directional | Short |
| R2: High-Vol MR | Wider wings, defined risk | IC (wide), Iron butterfly (standard), Calendar (double) | Directional | Neutral |
| R3: Low-Vol Trend | Directional spreads | Diagonal (bull/bear), Calendar (OTM), Ratio (back ratio) | Heavy theta | Neutral |
| R4: High-Vol Trend | Risk-defined only | Back ratios only (crash protection) | Naked/income | Long |

---

## Configuration

Override defaults via `~/.market_analyzer/config.yaml`:

```yaml
strategy:
  default_account_size: 100000
  max_position_pct: 0.03

screening:
  min_volume_20d_avg: 1000000

exit:
  profit_target_pcts: [40, 70, 90]
  time_exit_dte: 10

# Tune individual option play thresholds
opportunity:
  iron_condor:
    r4_confidence_threshold: 0.65
    min_iv: 0.12
    go_threshold: 0.50
  iron_butterfly:
    trending_confidence_threshold: 0.75
    min_atm_iv: 0.18
  calendar:
    r4_confidence_threshold: 0.65
    max_bid_ask_spread_pct: 1.5
  diagonal:
    r4_confidence_threshold: 0.65
    max_skew_ratio: 3.5
  ratio_spread:
    r2_confidence_threshold: 0.80
    min_skew_pct: 0.03

markets:
  default_market: "India"
```
