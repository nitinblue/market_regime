# market_regime — Integration Guide

API reference for external programs (cotrader, decision agent, etc.) consuming `market_regime` as a library.

---

## Installation

```bash
# From local checkout (editable)
pip install -e /path/to/market_regime

# Or with plot extras
pip install -e "/path/to/market_regime[plot]"

# Requires Python 3.11+. Use 3.12 for best compatibility (hmmlearn lacks 3.14 wheels).
```

All dependencies (pandas, hmmlearn, yfinance, pyarrow, pydantic, etc.) install automatically.

---

## Quick Start

```python
from market_regime import RegimeService, DataService

# DataService handles OHLCV fetching + parquet caching (~/.market_regime/cache/)
data_svc = DataService()

# RegimeService orchestrates HMM training, inference, and research
regime_svc = RegimeService(data_service=data_svc)
```

---

## API Tiers

### Tier 1: Just the Regime Label

Use when you only need the current regime to gate strategy selection.

```python
result = regime_svc.detect("SPY")

result.ticker              # "SPY"
result.regime              # RegimeID.R1_LOW_VOL_MR (IntEnum: 1-4)
result.confidence          # 0.92
result.trend_direction     # "bullish" | "bearish" | None
result.regime_probabilities  # {1: 0.92, 2: 0.03, 3: 0.04, 4: 0.01}
result.as_of_date          # date(2026, 2, 21)
result.model_version       # "hmm_v1_4s"
```

Batch variant:

```python
results = regime_svc.detect_batch(tickers=["SPY", "GLD", "QQQ", "TLT"])
# returns dict[str, RegimeResult]

for ticker, result in results.items():
    print(f"{ticker}: R{result.regime} ({result.confidence:.0%})")
```

### Tier 2: Full Research (Interpreted)

Use for research UIs, dashboards, or any context where you need the full interpreted analysis.

```python
r = regime_svc.research("SPY")
# returns TickerResearch — everything explore.py prints, as structured objects
```

**`TickerResearch` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `ticker` | `str` | Ticker symbol |
| `regime_result` | `RegimeResult` | Current regime label + confidence |
| `explanation_text` | `str` | Human-readable regime explanation |
| `transition_matrix` | `list[TransitionRow]` | 4 rows, one per regime, with stability comments |
| `state_means` | `list[StateMeansRow]` | Feature means per regime with vol/trend character |
| `label_alignment` | `list[LabelAlignmentDetail]` | How HMM states map to R1-R4 |
| `current_features` | `list[FeatureZScore]` | Current z-scores with semantic comments |
| `recent_history` | `list[RegimeHistoryDay]` | Last 20 trading days with change detection |
| `regime_distribution` | `list[RegimeDistributionEntry]` | Regime frequency over full training window |
| `strategy_comment` | `str` | Strategy recommendation for current regime |
| `model_info` | `HMMModelInfo` | Raw model internals for advanced consumers |

Batch variant with cross-ticker comparison:

```python
report = regime_svc.research_batch(tickers=["SPY", "GLD", "QQQ"])
# returns ResearchReport

for tr in report.tickers:
    print(f"{tr.ticker}: R{tr.regime_result.regime}")
    print(f"  Strategy: {tr.strategy_comment}")
    print(f"  Stability: {tr.transition_matrix[tr.regime_result.regime - 1].stability}")

# Cross-comparison (populated when 2+ tickers)
for c in report.comparison:
    print(f"{c.ticker}: R{c.regime} {c.trend_direction or ''} ({c.confidence:.0%})")
```

### Tier 3: Raw Data Only

Use when you just need OHLCV data (cache-first, auto delta-fetch).

```python
from market_regime import DataService

data_svc = DataService()

# Fetch OHLCV (cached to ~/.market_regime/cache/ohlcv/SPY.parquet)
df = data_svc.get_ohlcv("SPY")
# Returns DataFrame: index=DatetimeIndex, columns=[Open, High, Low, Close, Volume]

# With date range
from datetime import date
df = data_svc.get_ohlcv("GLD", start_date=date(2024, 1, 1))

# Check cache status
for meta in data_svc.cache_status("SPY"):
    print(f"{meta.data_type}: {meta.first_date} -> {meta.last_date} ({meta.row_count} rows)")

# Force re-fetch
data_svc.invalidate_cache("SPY")
```

### Tier 4: Caller-Provided Data (No Network)

Use when you already have OHLCV data and don't want any network calls.

```python
regime_svc = RegimeService()  # no data_service — no auto-fetch

# Caller provides DataFrame directly
result = regime_svc.detect("SPY", ohlcv=my_dataframe)
research = regime_svc.research("SPY", ohlcv=my_dataframe)

# Batch with caller data
report = regime_svc.research_batch(data={"SPY": spy_df, "GLD": gld_df})
```

DataFrame must have: `DatetimeIndex`, columns `[Open, High, Low, Close, Volume]`, sorted ascending, no NaNs.

---

## Model Reference

### RegimeID (IntEnum)

```
R1_LOW_VOL_MR  = 1   # Low-Vol Mean Reverting — ideal for iron condors, strangles
R2_HIGH_VOL_MR = 2   # High-Vol Mean Reverting — wider wings, defined risk
R3_LOW_VOL_TREND = 3  # Low-Vol Trending — directional spreads
R4_HIGH_VOL_TREND = 4 # High-Vol Trending — risk-defined directional, long vega
```

### Research Sub-Models

**TransitionRow** — one per regime in `TickerResearch.transition_matrix`:
```python
row.from_regime              # RegimeID
row.to_probabilities         # {1: 0.95, 2: 0.02, 3: 0.01, 4: 0.02}
row.stay_probability         # 0.95
row.stability                # "very sticky" | "sticky" | "moderately stable" | "unstable"
row.likely_transition_target # RegimeID | None (only if unstable)
```

**StateMeansRow** — one per regime in `TickerResearch.state_means`:
```python
row.regime          # RegimeID
row.feature_means   # {"realized_vol": 0.012, "trend_strength": -0.05, ...}
row.vol_character   # "high-vol" | "low-vol"
row.trend_character # "trending" | "mean-rev"
```

**FeatureZScore** — one per feature in `TickerResearch.current_features`:
```python
fz.feature   # "realized_vol"
fz.z_score   # 1.83
fz.comment   # "elevated vol (high)"
```

**RegimeHistoryDay** — last 20 days in `TickerResearch.recent_history`:
```python
day.date           # date(2026, 2, 21)
day.regime         # RegimeID
day.trend_direction # "bullish" | "bearish" | None
day.confidence     # 0.94
day.changed_from   # RegimeID | None (set on transition days)
```

**RegimeDistributionEntry** — one per observed regime in `TickerResearch.regime_distribution`:
```python
rd.regime      # RegimeID
rd.name        # "Low-Vol Mean Reverting"
rd.days        # 245
rd.percentage  # 48.3
rd.is_dominant # True
rd.is_rare     # False (True if < 10%)
```

**CrossTickerEntry** — one per ticker in `ResearchReport.comparison`:
```python
c.ticker               # "SPY"
c.regime               # RegimeID
c.trend_direction      # "bullish" | "bearish" | None
c.confidence           # 0.92
c.regime_probabilities # {1: 0.92, 2: 0.03, 3: 0.04, 4: 0.01}
c.strategy_comment     # "Primary: theta (IC, strangles). Avoid directional."
```

---

## Serialization

All models are Pydantic v2 `BaseModel`. Serialize to JSON/dict for APIs or storage:

```python
r = regime_svc.research("SPY")

# To dict
d = r.model_dump()

# To JSON string
j = r.model_json()

# From dict
r2 = TickerResearch.model_validate(d)
```

---

## Imports

Everything is exported from the top-level package:

```python
from market_regime import (
    # Services
    RegimeService,
    DataService,

    # Regime detection
    RegimeID,
    RegimeResult,
    RegimeConfig,

    # Research API
    TickerResearch,
    ResearchReport,
    CrossTickerEntry,
    TransitionRow,
    StateMeansRow,
    LabelAlignmentDetail,
    FeatureZScore,
    RegimeHistoryDay,
    RegimeDistributionEntry,

    # Lower-level (rarely needed)
    RegimeExplanation,
    HMMModelInfo,
    RegimeTimeSeries,
    RegimeTimeSeriesEntry,
    FeatureConfig,
    FeatureInspection,

    # Data layer
    DataType,
    ProviderType,
    DataRequest,
    DataResult,
)
```

---

## Typical cotrader Integration

```python
from market_regime import RegimeService, DataService, RegimeID

# Initialize once at startup
data_svc = DataService()
regime_svc = RegimeService(data_service=data_svc)

# In strategy selection loop:
result = regime_svc.detect("SPY")

if result.regime == RegimeID.R1_LOW_VOL_MR:
    # theta strategies: iron condors, strangles
    ...
elif result.regime == RegimeID.R3_LOW_VOL_TREND:
    # directional spreads
    ...

# For research UI section:
report = regime_svc.research_batch(tickers=watchlist)
# Render report.tickers[i].transition_matrix, .current_features, etc.
```

---

## Caching Behavior

- Cache location: `~/.market_regime/cache/`
- Staleness: 18 hours (weekday-aware — won't refetch on weekends)
- Delta-fetch: only downloads missing date range, appends to existing parquet
- First call for a ticker: full download + HMM training (~2 years of data)
- Subsequent calls: cache hit, instant return
- Models cached at: `~/.market_regime/models/{TICKER}.joblib`
