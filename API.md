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
├── features/            # Feature computation (technicals, screening)
├── hmm/                 # HMM training and inference
├── opportunity/         # Opportunity assessment
│   ├── setups/          # Price-based pattern detection
│   │   ├── breakout.py      # VCP, Bollinger squeeze, resistance proximity
│   │   ├── momentum.py      # MACD, RSI, MA alignment, stochastic
│   │   └── mean_reversion.py # RSI extremes, Bollinger bands
│   └── option_plays/    # Option structure recommendations by horizon
│       ├── zero_dte.py      # Same-day: iron condors, credit spreads, straddles
│       ├── leap.py          # Long-term: bull call LEAPs, PMCC
│       └── earnings.py      # Event-driven: pre-earnings straddles, IV crush
├── service/             # Service layer (facade + workflow services)
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

### Opportunity Assessment — `ma.opportunity`

Organized into two subpackages:

**Setups** — price-based directional pattern detection (`opportunity/setups/`):
```python
bo = ma.opportunity.assess_breakout("SPY")       # BreakoutOpportunity
mo = ma.opportunity.assess_momentum("SPY")       # MomentumOpportunity
# Mean reversion available via: from market_analyzer.opportunity.setups.mean_reversion import assess_mean_reversion
```

**Option Plays** — horizon-specific option structure recommendations (`opportunity/option_plays/`):
```python
lo = ma.opportunity.assess_leap("SPY")           # LEAPOpportunity
zd = ma.opportunity.assess_zero_dte("SPY")       # ZeroDTEOpportunity
# Earnings play available via: from market_analyzer.opportunity.option_plays.earnings import assess_earnings_play
```

### Trade Ranking — `ma.ranking`

```python
result = ma.ranking.rank(["SPY", "GLD", "QQQ", "TLT"])
# Returns: TradeRankingResult

result.top_trades       # list[RankedEntry]
result.by_ticker        # dict[str, list[RankedEntry]]
result.by_strategy      # dict[StrategyType, list[RankedEntry]]
result.black_swan_gate  # bool (True = CRITICAL, halt trading)
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

df = ds.get_ohlcv("SPY")           # pd.DataFrame
status = ds.cache_status("SPY")     # list[CacheMeta]
ds.invalidate_cache("SPY")          # Force re-fetch
```

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
| `macro` | Macro calendar | `macro` |
| `stress` | Tail-risk alert | `stress` |
| `help` | List all commands | `help` |
| `quit` | Exit | `quit` |

---

## Regime → Strategy Mapping

| Regime | Primary Strategy | Avoid | Vega Exposure |
|--------|-----------------|-------|---------------|
| R1: Low-Vol MR | Iron condors, strangles | Directional | Short |
| R2: High-Vol MR | Wider wings, defined risk | Directional | Neutral |
| R3: Low-Vol Trend | Directional spreads | Heavy theta | Neutral |
| R4: High-Vol Trend | Risk-defined only | Naked/income | Long |

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

markets:
  default_market: "India"
```
