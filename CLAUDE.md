# market_analyzer

**Historical market data service and HMM-based regime detection for options trading.**

Serves as the canonical historical data layer for the entire ecosystem (market_analyzer, cotrader, decision agent). Detects per-instrument regime state (R1–R4) using Hidden Markov Models, enabling regime-aware strategy selection for small options accounts. Real-time/streaming data remains with broker connections in cotrader.

---

## Ownership

| Section | Owner | Notes |
|---------|-------|-------|
| Domain rules, regime definitions, strategy mappings | Nitin | Trading philosophy, account constraints, regime semantics |
| Code architecture, module structure | Claude | Package layout, dependency graph, implementation |
| API contracts, data models | Claude | Pydantic models, input/output specs |
| Data provider contracts | Claude | ABC design, provider implementations, caching |
| Caching strategy | Claude | Parquet cache, delta-fetch, staleness logic |
| Data source selection | Nitin | Which providers to use, API key provisioning |
| Feature engineering | Claude | Domain guidance from Nitin; implementation by Claude |
| HMM model spec | Claude | Regime definitions from Nitin; hmmlearn config by Claude |
| Testing strategy | Claude | Unit, integration, regime validation |

---

## Standing Instructions for Claude

- **Read this file completely before any work.** All code decisions must align with the domain rules below.
- **No decision logic runs without a regime label.** This is the core invariant.
- **All decisions must be explainable.** No black-box outputs—every regime label must trace back to features and model state.
- **Per-instrument regime detection.** Never build a single global "market regime." Each ticker gets its own regime.
- **Income-first bias.** Default to theta-harvesting strategies; directional only when regime permits.
- **Hedging is same-ticker only.** No beta-weighted index hedging.
- **Small account constraints.** Design for 50K taxable, 200K IRA. Margin efficiency matters.
- **Keep it a library.** This package is imported by cotrader—no CLI, no server, no UI.
- **Prefer simplicity.** Minimal dependencies, no over-engineering, no speculative abstractions.
- **Type everything.** Use Pydantic models for all public interfaces. Type hints on all functions.
- **Historical data flows through this module.** All projects in the ecosystem (cotrader, decision agent) use `market_analyzer.data.DataService` for historical data. No other module fetches its own historical data.
- **Cache before fetch.** Always check parquet cache first. Only hit network for delta-fetch (missing date ranges). Never re-download data that's already cached.
- **Provider failures are not silent.** Raise typed exceptions on fetch failures, rate limits, bad tickers. Callers must be able to distinguish "no data exists" from "fetch failed."
- **No API keys in code.** All credentials come from environment variables or config files. Never hardcode keys, tokens, or passwords.
- **Data and regime modules are independently usable.** `data/` must work without `hmm/` or `features/`. `hmm/` must work with caller-provided DataFrames. No circular dependencies between the two halves.

---

## Domain Rules (Owner: Nitin)

### Trading Philosophy

- **All decisions are explainable**
- **Designed for small accounts (50K taxable, 200K IRA)**
- **Income-first (theta harvesting), directional only when regime allows**
- **Hedging is same-ticker only (no beta-weighted index hedging)**

### Why Regime First?

Options strategies **only make sense relative to regime**:
- Theta is fragile in trend acceleration
- Directional trades are expensive in chop
- Vega behaves differently in metals vs tech

> **No decision logic runs without a regime label**

### Regime States (4-State Model)

| Regime ID | Name | Description |
|-----------|------|-------------|
| R1 | Low-Vol Mean Reverting | Chop, range-bound, IV compression |
| R2 | High-Vol Mean Reverting | Wide swings, but no sustained trend |
| R3 | Low-Vol Trending | Slow, persistent directional move |
| R4 | High-Vol Trending | Explosive moves, IV expansion |

### Asset-Specific Regimes

- Regime is detected at **instrument level**
- Optional **sector-level HMM** for confirmation
- No single "market regime"

Examples:
- Gold can be trending while Tech is mean reverting
- Metals behave structurally differently from equities

### Regime -> Strategy Mapping

| Regime | Income (Theta) | Directional | Vega | Notes |
|--------|----------------|-------------|------|-------|
| R1: Low-Vol MR | Primary | Avoid | Short Vega | Ideal for iron condors, strangles |
| R2: High-Vol MR | Selective | Avoid | Neutral | Wider wings, defined risk |
| R3: Low-Vol Trend | Light | Primary | — | Directional spreads |
| R4: High-Vol Trend | Avoid | Selective | Long Vega | Risk-defined only |

### Income-First Bias

Default preference order:
1. Short theta
2. Neutral delta
3. Defined risk
4. Minimal margin usage

Directional trades only when:
- Regime = R3 or R4
- Portfolio delta budget allows

---

## Architecture & Module Structure (Owner: Claude)

```
market_analyzer/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── regime.py          # Pydantic models: RegimeState, RegimeResult, RegimeConfig
│   ├── features.py        # Pydantic models: FeatureConfig, FeatureVector
│   └── data.py            # Pydantic models: DataRequest, DataResult, CacheMeta
├── features/
│   ├── __init__.py
│   └── pipeline.py        # Feature computation: log returns, realized vol, ATR, trend strength
├── hmm/
│   ├── __init__.py
│   ├── trainer.py          # HMM training: fit, refit, persist
│   └── inference.py        # Regime inference: predict current regime from features
├── service/
│   ├── __init__.py
│   └── regime_service.py   # Top-level regime API: accepts DataFrame or auto-fetches
├── data/
│   ├── __init__.py
│   ├── service.py          # DataService: orchestrates cache + providers
│   ├── cache/
│   │   ├── __init__.py
│   │   └── parquet_cache.py  # ParquetCache: read/write/freshness checks
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py         # DataProvider ABC
│   │   ├── yfinance.py     # YFinanceProvider (OHLCV)
│   │   ├── cboe.py         # CBOEProvider (options/IV)
│   │   └── tastytrade.py   # TastyTradeProvider (broker history)
│   └── registry.py         # Maps (ticker, data_type) → provider
└── tests/
    ├── __init__.py
    ├── test_features.py
    ├── test_hmm.py
    ├── test_service.py
    ├── test_regime_validation.py
    ├── test_data_service.py
    ├── test_cache.py
    └── test_providers.py
```

### Dependency Graph

```
service/regime_service.py
    ├── features/pipeline.py
    │       └── models/features.py
    ├── hmm/trainer.py
    │       └── models/regime.py
    ├── hmm/inference.py
    │       └── models/regime.py
    └── data/service.py  (optional, for auto-fetch)
            ├── data/cache/parquet_cache.py
            ├── data/providers/yfinance.py
            ├── data/providers/cboe.py
            ├── data/providers/tastytrade.py
            ├── data/registry.py
            └── models/data.py

data/service.py  (independently usable by cotrader)
    ├── data/cache/parquet_cache.py
    ├── data/providers/*.py
    ├── data/registry.py
    └── models/data.py
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `models/` | Pydantic data models only. No logic. |
| `features/` | Compute feature vectors from OHLCV DataFrames. Normalization lives here. |
| `hmm/` | hmmlearn wrapper. Training, persistence, inference. |
| `service/` | Regime orchestration. Entry point for regime detection callers. |
| `data/service.py` | Data orchestration. Entry point for historical data callers. Cache-first fetch logic. |
| `data/cache/` | Parquet read/write, freshness checks, delta-date computation. |
| `data/providers/` | Network fetchers. Each provider implements the `DataProvider` ABC. |
| `data/registry.py` | Maps (ticker, data_type) to the correct provider. |

---

## API Contracts (Owner: Claude)

### Core Regime Models (Existing)

```python
from enum import IntEnum
from pydantic import BaseModel
import pandas as pd
from datetime import date

class RegimeID(IntEnum):
    R1_LOW_VOL_MR = 1    # Low-Vol Mean Reverting
    R2_HIGH_VOL_MR = 2   # High-Vol Mean Reverting
    R3_LOW_VOL_TREND = 3  # Low-Vol Trending
    R4_HIGH_VOL_TREND = 4 # High-Vol Trending

class RegimeResult(BaseModel):
    ticker: str
    regime: RegimeID
    confidence: float          # Posterior probability of assigned regime
    regime_probabilities: dict[RegimeID, float]  # All 4 state probabilities
    as_of_date: date
    model_version: str

class RegimeConfig(BaseModel):
    n_states: int = 4
    training_lookback_years: float = 2.0
    feature_lookback_days: int = 60
    refit_frequency_days: int = 30
```

### Data Models (New)

```python
from enum import StrEnum
from pydantic import BaseModel
from datetime import date, datetime
from pathlib import Path

class DataType(StrEnum):
    OHLCV = "ohlcv"
    OPTIONS_IV = "options_iv"
    BROKER_HISTORY = "broker_history"

class ProviderType(StrEnum):
    YFINANCE = "yfinance"
    CBOE = "cboe"
    TASTYTRADE = "tastytrade"

class DataRequest(BaseModel):
    ticker: str
    data_type: DataType
    start_date: date | None = None   # None = use default lookback
    end_date: date | None = None     # None = today

class CacheMeta(BaseModel):
    ticker: str
    data_type: DataType
    provider: ProviderType
    first_date: date                 # Earliest date in cached data
    last_date: date                  # Latest date in cached data
    last_fetched: datetime           # When we last hit the network
    row_count: int
    file_path: Path

class DataResult(BaseModel):
    ticker: str
    data_type: DataType
    provider: ProviderType
    from_cache: bool                 # True if served entirely from cache
    date_range: tuple[date, date]    # (first, last) date in returned data
    row_count: int
    # Actual DataFrame returned separately (not in Pydantic model)

    class Config:
        arbitrary_types_allowed = True
```

### Data Provider ABC

```python
from abc import ABC, abstractmethod

class DataProvider(ABC):
    @property
    @abstractmethod
    def provider_type(self) -> ProviderType: ...

    @property
    @abstractmethod
    def supported_data_types(self) -> list[DataType]: ...

    @abstractmethod
    def fetch(self, request: DataRequest) -> pd.DataFrame:
        """Fetch data from remote source. Raises on failure."""
        ...

    @abstractmethod
    def validate_ticker(self, ticker: str) -> bool:
        """Check if ticker is valid for this provider."""
        ...
```

### Data Service Interface

```python
class DataService:
    def get(self, request: DataRequest) -> tuple[pd.DataFrame, DataResult]:
        """Get data (cache-first, delta-fetch if stale)."""
        ...

    def get_ohlcv(self, ticker: str, start_date: date | None = None,
                  end_date: date | None = None) -> pd.DataFrame:
        """Convenience: fetch OHLCV data for a ticker."""
        ...

    def get_options_iv(self, ticker: str, start_date: date | None = None,
                       end_date: date | None = None) -> pd.DataFrame:
        """Convenience: fetch options/IV data for a ticker."""
        ...

    def cache_status(self, ticker: str,
                     data_type: DataType | None = None) -> list[CacheMeta]:
        """Check what's cached for a ticker."""
        ...

    def invalidate_cache(self, ticker: str,
                         data_type: DataType | None = None) -> None:
        """Force re-fetch on next request."""
        ...
```

### Regime Service Interface (Updated)

```python
class RegimeService:
    def __init__(self, config: RegimeConfig = RegimeConfig(),
                 data_service: DataService | None = None):
        """
        If data_service is provided, detect() can auto-fetch OHLCV data.
        If not, caller must always provide ohlcv DataFrame.
        """
        ...

    def detect(self, ticker: str,
               ohlcv: pd.DataFrame | None = None) -> RegimeResult:
        """
        Detect current regime for a single instrument.
        If ohlcv is None and data_service is available, auto-fetches.
        Raises ValueError if ohlcv is None and no data_service.
        """
        ...

    def detect_batch(self, tickers: list[str] | None = None,
                     data: dict[str, pd.DataFrame] | None = None
                     ) -> dict[str, RegimeResult]:
        """
        Detect regimes for multiple instruments.
        Can accept ticker list (auto-fetch) or dict of DataFrames.
        """
        ...

    def fit(self, ticker: str, ohlcv: pd.DataFrame | None = None) -> None:
        """Train/retrain HMM for a given instrument."""
        ...
```

---

## Feature Engineering (Owner: Claude, domain guidance from Nitin)

### Feature Pipeline

Computed per **instrument**, not globally. The feature pipeline is data-source-agnostic—it accepts a DataFrame and doesn't care whether it came from cache, yfinance, or the caller.

| Feature | Computation | Notes |
|---------|-------------|-------|
| Log returns (1d) | `log(close_t / close_{t-1})` | Primary return signal |
| Log returns (5d) | `log(close_t / close_{t-5})` | Weekly momentum |
| Realized volatility | Rolling std of log returns (20-day) | Volatility regime signal |
| ATR (normalized) | ATR / close price | Normalized for cross-asset comparison |
| Trend strength | Slope of 20-day SMA, normalized | Directional signal |
| Volume anomaly | Volume / 20-day avg volume | Optional; liquidity signal |

### Normalization

- All features are z-score normalized **per ticker** using a rolling window
- Window length matches `feature_lookback_days` in config
- IV Rank / IV Percentile deferred until options data integration with cotrader

### Historical Windows

| Component | Lookback |
|-----------|----------|
| Feature calculation | 30–90 days |
| HMM training | 1–3 years (rolling) |
| Regime inference | Daily or intraday |

---

## HMM Model Spec (Owner: Claude, regime definitions from Nitin)

### Model Configuration

- **Library:** hmmlearn `GaussianHMM`
- **n_components:** 4 (maps to R1–R4)
- **covariance_type:** "full" (captures feature correlations)
- **n_iter:** 100 (EM iterations)
- **random_state:** seeded for reproducibility

### Why HMM?

- Captures **latent market structure**
- Separates *observation noise* from *true regime*
- Well-understood, explainable, robust
- Online inference with periodic re-fitting

### Training Pipeline

1. Fetch or receive OHLCV data (1–3 years)
2. Compute feature matrix via `features/pipeline.py`
3. Fit `GaussianHMM` on feature matrix
4. **Post-fit label alignment:** HMM states are arbitrary integers. Map them to R1–R4 using feature means (e.g., lowest vol + lowest trend = R1, highest vol + highest trend = R4)
5. Persist fitted model (pickle or joblib)

### Inference Pipeline

1. Receive recent OHLCV data
2. Compute feature vector
3. Run `model.predict()` on recent window
4. Return mapped regime label + posterior probabilities

### Label Alignment Strategy

HMM hidden states have no inherent meaning. After fitting, align states to R1–R4 by sorting on:
- **Volatility axis:** mean realized vol per state (low vs high)
- **Trend axis:** mean absolute trend strength per state (mean-reverting vs trending)

This gives a 2x2 mapping that naturally produces R1–R4.

---

## Data Service (Owner: Claude, data source selection by Nitin)

### Scope

- **Historical data only.** This module serves all historical/daily data needs for the ecosystem.
- **Real-time data stays with cotrader.** Streaming quotes, live fills, and order book data come from broker connections in cotrader.
- **Boundary:** if it's bar data (daily or slower) and it's historical, it lives here.

### Data Providers

| Provider | Data Types | Auth Required | Notes |
|----------|-----------|---------------|-------|
| yfinance | OHLCV | No | Free, rate-limited. Primary source for price data. |
| CBOE | OPTIONS_IV | Yes (API key) | IV, skew, term structure. Details TBD. |
| TastyTrade | BROKER_HISTORY | Yes (username/password) | Trade history, P&L. Via tastytrade-sdk. |

### Cache Strategy

**Location:** `~/.market_analyzer/cache/`

**Directory layout:**
```
~/.market_analyzer/cache/
├── ohlcv/
│   ├── GLD.parquet
│   ├── SPY.parquet
│   └── AAPL.parquet
├── options_iv/
│   └── SPY.parquet
├── broker_history/
│   └── trades.parquet
└── _meta.json              # CacheMeta entries for all cached files
```

**Cache behavior:**
- **Staleness threshold:** 18 hours by default (configurable). Data older than this triggers a delta-fetch.
- **Delta-fetch:** Only request dates from `last_cached_date + 1` to today. Append to existing parquet.
- **Weekend/holiday awareness:** Don't mark cache as stale on Saturday/Sunday or market holidays. Last trading day's data is fresh until next trading day.
- **Atomic writes:** Write to temp file, then rename. No partial parquet files.
- **Cache miss:** Full fetch from provider, write complete parquet file.
- **`_meta.json`:** Single file tracking all cache entries (ticker, data_type, date range, last_fetched timestamp, row count, file path).

### DataFrame Contracts

**OHLCV DataFrame:**
- Columns: `Open`, `High`, `Low`, `Close`, `Volume`
- Index: `DatetimeIndex` (daily frequency)
- Sorted ascending by date
- No NaN values in required columns

**Options/IV DataFrame:** TBD (pending CBOE integration design)

**Broker History DataFrame:** TBD (pending TastyTrade integration design)

### Provider Configuration

All credentials via environment variables:

| Variable | Provider | Notes |
|----------|----------|-------|
| `CBOE_API_KEY` | CBOE | Required for options/IV data |
| `TASTYTRADE_USERNAME` | TastyTrade | Required for broker history |
| `TASTYTRADE_PASSWORD` | TastyTrade | Required for broker history |

### Usage Patterns

**Pattern 1: Auto-fetch via RegimeService**
```python
from market_analyzer.data.service import DataService
from market_analyzer.service.regime_service import RegimeService

data_svc = DataService()
regime_svc = RegimeService(data_service=data_svc)

# No DataFrame needed — auto-fetches and caches OHLCV
result = regime_svc.detect(ticker="GLD")
```

**Pattern 2: Direct DataService use by cotrader**
```python
from market_analyzer.data.service import DataService

data_svc = DataService()

# Any project can fetch historical data through this module
ohlcv = data_svc.get_ohlcv("SPY")
iv_data = data_svc.get_options_iv("SPY")
```

**Pattern 3: Caller provides data (backward compatible)**
```python
from market_analyzer.service.regime_service import RegimeService

regime_svc = RegimeService()  # No data_service
result = regime_svc.detect(ticker="GLD", ohlcv=my_dataframe)
```

---

## Tech Stack

| Dependency | Purpose | Required |
|------------|---------|----------|
| Python 3.11+ | Runtime | Yes |
| hmmlearn | HMM fitting and inference | Yes |
| pandas | DataFrame handling | Yes |
| numpy | Numerical computation | Yes |
| pydantic | Data models, validation | Yes |
| scikit-learn | hmmlearn dependency | Yes (transitive) |
| yfinance | OHLCV data fetching | Yes |
| pyarrow | Parquet read/write for cache | Yes |
| requests | HTTP client for CBOE provider | Yes |
| joblib | Model persistence | Yes |
| tastytrade-sdk | TastyTrade broker history | Optional |
| pytest | Testing | Dev |
| pytest-mock | Mock providers in tests | Dev |

---

## Testing Strategy (Owner: Claude)

### Test Layers

| Layer | What | How |
|-------|------|-----|
| Unit | Feature computation correctness | Known OHLCV -> expected features |
| Unit | Pydantic model validation | Invalid inputs rejected |
| Unit | ParquetCache read/write | Write parquet, read back, verify roundtrip |
| Unit | Cache freshness logic | Mock clock, verify staleness detection, weekend awareness |
| Unit | Delta date computation | Given cached range + today, compute correct fetch range |
| Integration | Full regime pipeline: OHLCV -> RegimeResult | Synthetic + real data |
| Integration | DataService full cycle: cache miss -> fetch -> cache hit | Mock provider, real parquet cache |
| Contract | DataProvider implementations | Each provider returns correct DataFrame schema |
| Regime validation | Label alignment makes sense | Verify R1 has lowest vol, R4 has highest vol+trend |
| Regression | Regime stability on known data | Fitted model produces consistent labels on historical data |
| Provider integration | Live provider tests | Real network calls, marked `@pytest.mark.integration` |

### Test Data

- **Synthetic data** for unit tests (deterministic, no network)
- **Real data via yfinance** for integration tests (marked with `@pytest.mark.integration`)
- **Fixture data** (saved CSVs) for regression tests
- **Mock providers** for DataService tests (deterministic, no network)

---

## Integration with cotrader

This library is used by `cotrader` (trading platform at `C:\Users\nitin\PythonProjects\eTrading`).

### Data Integration Contract

```
cotrader → market_analyzer.data.DataService.get_ohlcv() → cached OHLCV DataFrame
cotrader → market_analyzer.data.DataService.get_options_iv() → cached IV DataFrame
```

All historical data requests from cotrader flow through `market_analyzer.data.DataService`. Cotrader does not fetch its own historical data.

### Regime Integration Contract

```
cotrader → market_analyzer.RegimeService.detect() → RegimeResult
cotrader uses RegimeResult to gate strategy selection in Decision Agent
```

### Architecture Context

```
cotrader (execution, broker, real-time data)
    │
    ├── real-time quotes, fills, order book → broker connections (cotrader-owned)
    │
    ├── historical OHLCV, IV, trade history → market_analyzer.data.DataService
    │
    ▼
market_analyzer (this library) ← historical data + regime detection
    │
    ▼
Decision Agent (separate library) ← strategy selection
    │
    ▼
What-if Evaluator (part of cotrader) ← PnL, Greeks, Margin
```

**Boundary:** real-time data = cotrader, historical data = market_analyzer.

---

## Quick Reference — Command Lines

All commands assume you're in the project root (`C:\Users\nitin\PythonProjects\market_analyzer`).

### Setup

```bash
# Create venv (Python 3.12 — hmmlearn has no 3.14 wheels yet)
py -3.12 -m venv .venv

# Install package + dev deps
.venv/Scripts/pip install -e ".[dev]"
```

### Running Tests

```bash
# All tests
.venv/Scripts/python -m pytest tests/ -v

# Individual test files
.venv/Scripts/python -m pytest tests/test_features.py -v
.venv/Scripts/python -m pytest tests/test_hmm.py -v
.venv/Scripts/python -m pytest tests/test_cache.py -v
.venv/Scripts/python -m pytest tests/test_data_service.py -v
.venv/Scripts/python -m pytest tests/test_providers.py -v
.venv/Scripts/python -m pytest tests/test_service.py -v
.venv/Scripts/python -m pytest tests/test_regime_validation.py -v
.venv/Scripts/python -m pytest tests/test_technicals.py -v
.venv/Scripts/python -m pytest tests/test_phases.py -v
.venv/Scripts/python -m pytest tests/test_fundamentals.py -v
.venv/Scripts/python -m pytest tests/test_macro.py -v
.venv/Scripts/python -m pytest tests/test_opportunity.py -v
.venv/Scripts/python -m pytest tests/test_breakout.py -v
.venv/Scripts/python -m pytest tests/test_momentum.py -v
.venv/Scripts/python -m pytest tests/test_orb.py -v
.venv/Scripts/python -m pytest tests/test_price_structure.py -v
.venv/Scripts/python -m pytest tests/test_analyzer.py -v
.venv/Scripts/python -m pytest tests/test_ranking.py -v
.venv/Scripts/python -m pytest tests/test_context.py -v
.venv/Scripts/python -m pytest tests/test_instrument.py -v
.venv/Scripts/python -m pytest tests/test_screening.py -v
.venv/Scripts/python -m pytest tests/test_entry.py -v
.venv/Scripts/python -m pytest tests/test_strategy.py -v
.venv/Scripts/python -m pytest tests/test_exit.py -v
.venv/Scripts/python -m pytest tests/test_mean_reversion.py -v
.venv/Scripts/python -m pytest tests/test_earnings.py -v

# Run a single test by name
.venv/Scripts/python -m pytest tests/test_hmm.py::TestRegimeInference::test_predict_returns_regime_result -v

# Integration tests only (requires network)
.venv/Scripts/python -m pytest -m integration -v

# Skip integration tests
.venv/Scripts/python -m pytest -m "not integration" -v
```

### CLI Commands (after pip install)

```bash
# Interactive regime exploration (default tickers: SPY, GLD, QQQ, TLT)
analyzer-explore
analyzer-explore --tickers AAPL MSFT AMZN
analyzer-explore --tickers GLD

# Regime chart with price, volume, RSI, confidence panels (requires [plot])
analyzer-plot
analyzer-plot --tickers AAPL MSFT
analyzer-plot --tickers GLD --save

# Interactive REPL (Claude-like interface)
analyzer-cli
analyzer-cli --market india
```

### Script Wrappers (no install required)

```bash
.venv/Scripts/python explore.py
.venv/Scripts/python explore.py --tickers GLD
.venv/Scripts/python plot_regime.py
.venv/Scripts/python plot_regime.py --tickers GLD --save
```

### Quick Python Usage

```bash
# Detect regime (via facade)
.venv/Scripts/python -c "
from market_analyzer import MarketAnalyzer, DataService
ma = MarketAnalyzer(data_service=DataService())
r = ma.regime.detect('SPY')
print(f'{r.ticker}: R{r.regime} ({r.confidence:.0%})')
"

# Batch regime detection
.venv/Scripts/python -c "
from market_analyzer import MarketAnalyzer, DataService
ma = MarketAnalyzer(data_service=DataService())
for t, r in ma.regime.detect_batch(tickers=['SPY','GLD','QQQ','TLT']).items():
    print(f'{t}: R{r.regime} ({r.confidence:.0%})')
"

# Technical snapshot
.venv/Scripts/python -c "
from market_analyzer import MarketAnalyzer, DataService
ma = MarketAnalyzer(data_service=DataService())
t = ma.technicals.snapshot('SPY')
print(f'RSI: {t.rsi.value:.1f}, ATR: {t.atr_pct:.2f}%')
"

# Levels with R:R
.venv/Scripts/python -c "
from market_analyzer import MarketAnalyzer, DataService
ma = MarketAnalyzer(data_service=DataService())
print(ma.levels.analyze('SPY').summary)
"

# Rank trades across tickers
.venv/Scripts/python -c "
from market_analyzer import MarketAnalyzer, DataService
ma = MarketAnalyzer(data_service=DataService())
result = ma.ranking.rank(['SPY', 'GLD', 'QQQ', 'TLT'])
for e in result.top_trades[:5]:
    print(f'#{e.rank} {e.ticker} {e.strategy_type} score={e.composite_score:.2f} {e.verdict}')
"

# Black swan alert
.venv/Scripts/python -c "
from market_analyzer import MarketAnalyzer, DataService
ma = MarketAnalyzer(data_service=DataService())
alert = ma.black_swan.alert()
print(f'Alert: {alert.alert_level} (score={alert.composite_score:.2f})')
"

# Fetch OHLCV data only (cache-first)
.venv/Scripts/python -c "
from market_analyzer import DataService
df = DataService().get_ohlcv('GLD')
print(df.tail())
"

# Check what's cached
.venv/Scripts/python -c "
from market_analyzer import DataService
for m in DataService().cache_status('SPY'):
    print(f'{m.data_type}: {m.first_date} → {m.last_date} ({m.row_count} rows)')
"
```

### Workflow APIs (NEW)

```bash
# Market context (environment assessment)
.venv/Scripts/python -c "
from market_analyzer import MarketAnalyzer, DataService
ma = MarketAnalyzer(data_service=DataService())
ctx = ma.context.assess()
print(f'Environment: {ctx.environment_label}, Trading: {ctx.trading_allowed}')
"

# Full instrument analysis
.venv/Scripts/python -c "
from market_analyzer import MarketAnalyzer, DataService
ma = MarketAnalyzer(data_service=DataService())
a = ma.instrument.analyze('SPY')
print(f'{a.ticker}: R{a.regime_id} | {a.phase.phase_name} | RSI {a.technicals.rsi.value:.0f} | {a.trend_bias}')
"

# Screen for setups
.venv/Scripts/python -c "
from market_analyzer import MarketAnalyzer, DataService
ma = MarketAnalyzer(data_service=DataService())
result = ma.screening.scan(['SPY', 'GLD', 'QQQ', 'TLT'])
for c in result.candidates[:5]:
    print(f'{c.ticker} [{c.screen}] score={c.score:.2f}: {c.reason}')
"

# Entry confirmation
.venv/Scripts/python -c "
from market_analyzer import MarketAnalyzer, DataService, EntryTriggerType
ma = MarketAnalyzer(data_service=DataService())
e = ma.entry.confirm('SPY', EntryTriggerType.BREAKOUT_CONFIRMED)
print(f'Entry: {\"CONFIRMED\" if e.confirmed else \"NOT CONFIRMED\"} ({e.confidence:.0%})')
"

# Strategy selection + sizing
.venv/Scripts/python -c "
from market_analyzer import MarketAnalyzer, DataService
ma = MarketAnalyzer(data_service=DataService())
r = ma.regime.detect('SPY')
t = ma.technicals.snapshot('SPY')
params = ma.strategy.select('SPY', regime=r, technicals=t)
size = ma.strategy.size(params, current_price=t.current_price)
print(f'{params.primary_structure.structure_type} | {size.suggested_contracts} contracts | max risk \${size.max_risk_dollars:.0f}')
"
```

### Cache Management

```bash
# Cache location
ls ~/.market_analyzer/cache/ohlcv/

# Invalidate cache for a ticker (forces re-fetch on next request)
.venv/Scripts/python -c "
from market_analyzer import DataService
DataService().invalidate_cache('SPY')
print('Cache invalidated for SPY')
"

# Invalidate all cached data for a ticker
.venv/Scripts/python -c "
from market_analyzer import DataService
DataService().invalidate_cache('SPY', data_type=None)
"
```

---

## Change Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-21 | Initial design doc | Established 4-state HMM, per-instrument regime, income-first bias |
| 2026-02-21 | Package structure defined | models/, features/, hmm/, service/, data/ modules |
| 2026-02-21 | yfinance as optional dependency | Core library must work with caller-provided DataFrames |
| 2026-02-21 | Label alignment via vol+trend axes | 2x2 sorting maps arbitrary HMM states to R1–R4 semantically |
| 2026-02-21 | Expanded to canonical historical data service | market_analyzer owns all historical data for ecosystem; added DataService, parquet cache, three providers (yfinance, CBOE, TastyTrade); yfinance now required |
| 2026-02-23 | Trading workflow restructure | Added 6 workflow services (context, instrument, screening, entry, strategy, exit), 5 new model files, multi-market config (US + India), interactive CLI (analyzer-cli), API.md. Additive — no existing files moved, all 580 tests pass. |
