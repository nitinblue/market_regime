
- **All decisions are explainable**
- **Designed for small accounts (50K taxable, 200K IRA)**
- **Income-first (theta harvesting), directional only when regime allows**
- **Hedging is same-ticker only (no beta-weighted index hedging)**
---

## 2. High-Level Architecture

```

```

┌──────────────────────┐ ← Trading platform (cotrader) (C:\Users\nitin\PythonProjects\eTrading\CLAUDE.md)
│  (HMM-based)               │
│     cotrader         │
│ (execution, broker,  │
│  market data feeds)  │
└─────────┬────────────┘
│
│  In-memory containers
│  (Portfolio, Position, RiskFactor)
▼
┌────────────────────────────┐
│  Market Regime Service     │  ← Independent library (C:\Users\nitin\PythonProjects\eTrading\trading_cotrader\hmm_regime_claude.md)
│  (HMM-based)               │
└─────────┬──────────────────┘
│
▼
┌────────────────────────────┐
│  Decision Agent            │  ← Independent library (This md file is for decision agent)
│  (rule + objective based)  │
└─────────┬──────────────────┘
│
▼
┌────────────────────────────┐← Part of trading platform  (cotrader) (C:\Users\nitin\PythonProjects\eTrading\trading_cotrader\services\trade_booking_service.py)
│  What-if Evaluator         │
│  (PnL, Greeks, Margin)     │
└────────────────────────────┘
````

**Key Design Choice**  
- `cotrader` remains execution-focused  
- **Regime detection and decision intelligence live outside** and are *called* by cotrader

---

## 3. Market Regime Detection

### 3.1 Why Regime First?

Options strategies **only make sense relative to regime**:
- Theta is fragile in trend acceleration
- Directional trades are expensive in chop
- Vega behaves differently in metals vs tech

Therefore:
> **No decision logic runs without a regime label**

---

## 4. Regime Model Specification

### 4.1 Model Type

**Hidden Markov Model (HMM)**  
- Gaussian emissions
- Discrete hidden states
- Online inference with periodic re-fitting

#### Why HMM?
- Captures **latent market structure**
- Separates *observation noise* from *true regime*
- Well-understood, explainable, robust

---

### 4.2 Regime States (4-State Model)

| Regime ID | Name                     | Description |
|----------|--------------------------|------------|
| R1       | Low-Vol Mean Reverting   | Chop, range-bound, IV compression |
| R2       | High-Vol Mean Reverting  | Wide swings, but no sustained trend |
| R3       | Low-Vol Trending         | Slow, persistent directional move |
| R4       | High-Vol Trending        | Explosive moves, IV expansion |

---

### 4.3 Observations (Features)

Computed per **instrument**, not globally.

| Feature | Notes |
|------|------|
| Log returns | 1d, 5d |
| Realized volatility | Rolling (10–30 days) |
| ATR | Normalized |
| IV Rank / IV Percentile | Options-specific |
| Trend strength | e.g., slope of moving average |
| Volume anomaly | Optional |

All features are normalized **per ticker or per sector**.

---

### 4.4 Asset-Specific Regimes

**Important Design Choice**

- Regime is detected at **instrument level**
- Optional **sector-level HMM** for confirmation
- No single “market regime”

Examples:
- Gold can be trending while Tech is mean reverting
- Metals behave structurally differently from equities

---

### 4.5 Historical Window

| Component | Lookback |
|--------|----------|
| Feature calculation | 30–90 days |
| HMM training | 1–3 years (rolling) |
| Regime inference | Daily or intraday |

---

## 5. Regime → Action Mapping

### 5.1 Allowed Strategy Families per Regime

| Regime | Income (Theta) | Directional | Vega | Notes |
|------|----------------|-------------|------|------|
| R1: Low-Vol MR | ✅ Primary | ❌ Avoid | ❌ Short Vega | Ideal for iron condors, strangles |
| R2: High-Vol MR | ⚠️ Selective | ❌ Avoid | ⚠️ Neutral | Wider wings, defined risk |
| R3: Low-Vol Trend | ⚠️ Light | ✅ Primary | ❌ | Directional spreads |
| R4: High-Vol Trend | ❌ Avoid | ⚠️ Selective | ✅ Long Vega | Risk-defined only |

---

### 5.2 Income-First Bias

Default preference order:
1. Short theta
2. Neutral delta
3. Defined risk
4. Minimal margin usage

Directional trades only when:
- Regime = R3 or R4
- Portfolio delta budget allows


```
market_regime/
├── models/
│   └── hmm_spec.md
├── features/
│   └── feature_contracts.md
├── service/
│   └── regime_service.py
└── README.md
```

