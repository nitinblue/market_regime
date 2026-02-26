"""Pydantic models for options opportunity assessment."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel


class LegAction(StrEnum):
    """Buy or sell action for an option leg."""

    BUY_TO_OPEN = "BTO"
    SELL_TO_OPEN = "STO"


class StructureType(StrEnum):
    """Option structure types — tells cotrader what order structure to build."""

    IRON_CONDOR = "iron_condor"
    IRON_MAN = "iron_man"  # Inverse/long iron condor
    IRON_BUTTERFLY = "iron_butterfly"
    CREDIT_SPREAD = "credit_spread"
    DEBIT_SPREAD = "debit_spread"
    CALENDAR = "calendar"
    DIAGONAL = "diagonal"
    RATIO_SPREAD = "ratio_spread"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    DOUBLE_CALENDAR = "double_calendar"
    LONG_OPTION = "long_option"  # Single long call/put
    PMCC = "pmcc"


class OrderSide(StrEnum):
    """Net order side — credit (receive premium) or debit (pay premium)."""

    CREDIT = "credit"
    DEBIT = "debit"


class Verdict(StrEnum):
    """Go/no-go verdict for an opportunity assessment."""

    GO = "go"
    CAUTION = "caution"
    NO_GO = "no_go"


class RiskProfile(StrEnum):
    """Whether the strategy has a defined maximum loss."""

    DEFINED = "defined"
    UNDEFINED = "undefined"


class StructureProfile(BaseModel):
    """Quick-reference profile for an option structure.

    Shown in CLI next to every trade so the user instantly knows:
    what the payoff looks like, which direction it bets on, and
    whether they can lose more than they put in.
    """

    payoff_graph: str   # Compact ASCII payoff shape (5-7 chars)
    bias: str           # "neutral", "bullish", "bearish"
    risk_profile: RiskProfile
    label: str          # One-liner: "4-leg neutral credit · defined risk"


# Payoff graph + profile lookup.
# For direction-sensitive structures (spreads, long options), caller
# provides direction; default is "neutral".  For order-side-sensitive
# structures (straddle/strangle), caller provides order_side.

_PROFILES: dict[str, tuple[str, str, RiskProfile, str]] = {
    # (payoff_graph, default_bias, risk, label)
    "iron_condor":     ("/‾‾\\",  "neutral",  RiskProfile.DEFINED,   "4-leg neutral credit · defined risk"),
    "iron_man":        ("\\__/",  "neutral",  RiskProfile.DEFINED,   "4-leg neutral debit · defined risk"),
    "iron_butterfly":  ("/\\",    "neutral",  RiskProfile.DEFINED,   "4-leg ATM straddle + wings · defined risk"),
    "calendar":        ("/\\",    "neutral",  RiskProfile.DEFINED,   "2-leg time spread · defined risk"),
    "double_calendar": ("/‾‾\\",  "neutral",  RiskProfile.DEFINED,   "4-leg double time spread · defined risk"),
    "diagonal":        ("_/\\",   "neutral",  RiskProfile.DEFINED,   "2-leg time+strike spread · defined risk"),
    "pmcc":            ("_/\\",   "bullish",  RiskProfile.DEFINED,   "poor man's covered call · defined risk"),
}

# Direction-dependent structures
_SPREAD_PROFILES: dict[tuple[str, str], tuple[str, str, RiskProfile, str]] = {
    # (structure, direction): (graph, bias, risk, label)
    ("credit_spread", "bullish"):  ("__/‾",  "bullish",  RiskProfile.DEFINED,  "bull put credit spread · defined risk"),
    ("credit_spread", "bearish"):  ("‾\\__",  "bearish",  RiskProfile.DEFINED,  "bear call credit spread · defined risk"),
    ("credit_spread", "neutral"):  ("__/‾",  "neutral",  RiskProfile.DEFINED,  "credit spread · defined risk"),
    ("debit_spread", "bullish"):   ("__/‾",  "bullish",  RiskProfile.DEFINED,  "bull call debit spread · defined risk"),
    ("debit_spread", "bearish"):   ("‾\\__",  "bearish",  RiskProfile.DEFINED,  "bear put debit spread · defined risk"),
    ("debit_spread", "neutral"):   ("__/‾",  "neutral",  RiskProfile.DEFINED,  "debit spread · defined risk"),
    ("ratio_spread", "bullish"):   ("_/\\!",  "bullish",  RiskProfile.UNDEFINED, "call ratio spread · UNDEFINED risk"),
    ("ratio_spread", "bearish"):   ("!\\/\\_", "bearish",  RiskProfile.UNDEFINED, "put ratio spread · UNDEFINED risk"),
    ("ratio_spread", "neutral"):   ("_/\\!",  "neutral",  RiskProfile.UNDEFINED, "ratio spread · UNDEFINED risk"),
    ("long_option", "bullish"):    ("__/",   "bullish",  RiskProfile.DEFINED,  "long call · defined risk"),
    ("long_option", "bearish"):    ("\\__",   "bearish",  RiskProfile.DEFINED,  "long put · defined risk"),
    ("long_option", "neutral"):    ("__/",   "neutral",  RiskProfile.DEFINED,  "long option · defined risk"),
}

# Order-side-dependent structures (straddle, strangle)
_VOL_PROFILES: dict[tuple[str, str], tuple[str, str, RiskProfile, str]] = {
    # (structure, order_side): (graph, bias, risk, label)
    ("straddle", "credit"):   ("/\\",    "neutral", RiskProfile.UNDEFINED, "short straddle · UNDEFINED risk"),
    ("straddle", "debit"):    ("\\/",    "neutral", RiskProfile.DEFINED,   "long straddle · defined risk"),
    ("strangle", "credit"):   ("/‾\\",   "neutral", RiskProfile.UNDEFINED, "short strangle · UNDEFINED risk"),
    ("strangle", "debit"):    ("\\__/",  "neutral", RiskProfile.DEFINED,   "long strangle · defined risk"),
}


def get_structure_profile(
    structure_type: str | StructureType,
    order_side: str | OrderSide | None = None,
    direction: str | None = None,
) -> StructureProfile:
    """Look up the payoff profile for a structure.

    Args:
        structure_type: The option structure (from StructureType enum).
        order_side: "credit" or "debit" — needed for straddle/strangle.
        direction: "bullish", "bearish", or "neutral" — needed for spreads.

    Returns:
        StructureProfile with payoff_graph, bias, risk_profile, label.
    """
    st = str(structure_type)
    side = str(order_side) if order_side else None
    dir_ = direction or "neutral"

    # Check vol structures first (straddle/strangle depend on buy/sell)
    if st in ("straddle", "strangle") and side:
        key = (st, side)
        if key in _VOL_PROFILES:
            g, b, r, l = _VOL_PROFILES[key]
            return StructureProfile(payoff_graph=g, bias=b, risk_profile=r, label=l)

    # Check direction-dependent structures
    key_dir = (st, dir_)
    if key_dir in _SPREAD_PROFILES:
        g, b, r, l = _SPREAD_PROFILES[key_dir]
        return StructureProfile(payoff_graph=g, bias=b, risk_profile=r, label=l)

    # Check fixed-profile structures
    if st in _PROFILES:
        g, b, r, l = _PROFILES[st]
        return StructureProfile(payoff_graph=g, bias=b, risk_profile=r, label=l)

    # Fallback
    return StructureProfile(
        payoff_graph="???",
        bias=dir_,
        risk_profile=RiskProfile.DEFINED,
        label=f"{st} · check risk",
    )


class HardStop(BaseModel):
    """A condition that forces a NO_GO verdict."""

    name: str
    description: str


class OpportunitySignal(BaseModel):
    """A single contributing signal to the opportunity assessment."""

    name: str
    favorable: bool
    weight: float
    description: str


class StrategyRecommendation(BaseModel):
    """A specific trade structure recommendation."""

    name: str
    direction: str  # "neutral", "bullish", "bearish"
    structure: str
    rationale: str
    risk_notes: list[str]


# --- 0DTE ---


class ZeroDTEStrategy(StrEnum):
    IRON_CONDOR = "iron_condor"
    IRON_MAN = "iron_man"  # Inverse/long iron condor — debit, profits from big moves
    CREDIT_SPREAD = "credit_spread"
    STRADDLE_STRANGLE = "straddle_strangle"
    DIRECTIONAL_SPREAD = "directional_spread"
    NO_TRADE = "no_trade"


class ORBDecision(BaseModel):
    """ORB-based decision context for 0DTE strategies."""

    status: str  # ORBStatus value
    range_high: float
    range_low: float
    range_pct: float
    direction: str  # "bullish", "bearish", "neutral"
    decision: str  # Human-readable ORB-based decision rationale
    key_levels: dict[str, float]  # {"T1_long": 605.5, "T1_short": 595.2, "vwap": 600.1, ...}


class ZeroDTEOpportunity(BaseModel):
    """0DTE opportunity assessment for a single ticker."""

    ticker: str
    as_of_date: date
    verdict: Verdict
    confidence: float
    hard_stops: list[HardStop]
    signals: list[OpportunitySignal]
    strategy: StrategyRecommendation
    zero_dte_strategy: ZeroDTEStrategy
    regime_id: int
    regime_confidence: float
    atr_pct: float
    orb_status: str | None
    orb_decision: ORBDecision | None = None
    has_macro_event_today: bool
    days_to_earnings: int | None
    trade_spec: TradeSpec | None = None
    summary: str


# --- LEAP ---


class LEAPStrategy(StrEnum):
    BULL_CALL_LEAP = "bull_call_leap"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_LEAP = "bear_put_leap"
    PROTECTIVE_PUT = "protective_put"
    PMCC = "pmcc"
    NO_TRADE = "no_trade"


class FundamentalScore(BaseModel):
    """Composite score from fundamentals data."""

    score: float
    earnings_growth_signal: str
    revenue_growth_signal: str
    margin_signal: str
    debt_signal: str
    valuation_signal: str
    description: str


class LEAPOpportunity(BaseModel):
    """LEAP opportunity assessment for a single ticker."""

    ticker: str
    as_of_date: date
    verdict: Verdict
    confidence: float
    hard_stops: list[HardStop]
    signals: list[OpportunitySignal]
    strategy: StrategyRecommendation
    leap_strategy: LEAPStrategy
    regime_id: int
    regime_confidence: float
    phase_id: int
    phase_name: str
    phase_confidence: float
    iv_environment: str
    fundamental_score: FundamentalScore
    days_to_earnings: int | None
    macro_events_next_30_days: int
    trade_spec: TradeSpec | None = None
    summary: str


# --- Breakout ---


class BreakoutType(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"


class BreakoutStrategy(StrEnum):
    PIVOT_BREAKOUT = "pivot_breakout"
    SQUEEZE_PLAY = "squeeze_play"
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    PULLBACK_TO_BREAKOUT = "pullback_to_breakout"
    NO_TRADE = "no_trade"


class BreakoutSetup(BaseModel):
    """Describes the current breakout setup context."""

    vcp_stage: str
    vcp_score: float
    bollinger_squeeze: bool
    bollinger_bandwidth: float
    range_compression: float
    volume_pattern: str  # "declining_base" | "surge" | "normal" | "distribution"
    resistance_proximity_pct: float | None
    support_proximity_pct: float | None
    days_in_base: int | None
    smart_money_alignment: str  # "supportive" | "neutral" | "conflicting"
    description: str


class BreakoutOpportunity(BaseModel):
    """Breakout opportunity assessment for a single ticker."""

    ticker: str
    as_of_date: date
    verdict: Verdict
    confidence: float
    hard_stops: list[HardStop]
    signals: list[OpportunitySignal]
    strategy: StrategyRecommendation
    breakout_strategy: BreakoutStrategy
    breakout_type: BreakoutType
    regime_id: int
    regime_confidence: float
    phase_id: int
    phase_name: str
    phase_confidence: float
    setup: BreakoutSetup
    pivot_price: float | None
    days_to_earnings: int | None
    trade_spec: TradeSpec | None = None
    summary: str


# --- Momentum ---


class MomentumDirection(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"


class MomentumStrategy(StrEnum):
    TREND_CONTINUATION = "trend_continuation"
    PULLBACK_ENTRY = "pullback_entry"
    MOMENTUM_ACCELERATION = "momentum_acceleration"
    MOMENTUM_FADE = "momentum_fade"
    NO_TRADE = "no_trade"


class MomentumScore(BaseModel):
    """Composite momentum score details."""

    macd_histogram_trend: str  # "expanding" | "flat" | "contracting"
    macd_crossover: str  # "bullish" | "bearish" | "none"
    rsi_zone: str  # "oversold" | "healthy_bull" | "neutral" | "overbought" | "healthy_bear"
    price_vs_ma_alignment: str  # "strong_bull" | "bull" | "neutral" | "bear" | "strong_bear"
    golden_death_cross: str | None  # "golden_cross" | "death_cross" | None
    structural_pattern: str  # "HH_HL" | "LH_LL" | "mixed"
    volume_confirmation: bool
    stochastic_confirmation: bool
    atr_trend: str  # "rising" | "stable" | "falling"
    description: str


class MomentumOpportunity(BaseModel):
    """Momentum opportunity assessment for a single ticker."""

    ticker: str
    as_of_date: date
    verdict: Verdict
    confidence: float
    hard_stops: list[HardStop]
    signals: list[OpportunitySignal]
    strategy: StrategyRecommendation
    momentum_strategy: MomentumStrategy
    momentum_direction: MomentumDirection
    regime_id: int
    regime_confidence: float
    phase_id: int
    phase_name: str
    phase_confidence: float
    score: MomentumScore
    days_to_earnings: int | None
    trade_spec: TradeSpec | None = None
    summary: str


# --- TradeSpec (actionable trade parameters) ---


class LegSpec(BaseModel):
    """A single option leg in a trade spec."""

    role: str  # "short_put", "long_put", "short_call", "long_call", "short_straddle"
    action: LegAction  # BTO or STO
    quantity: int = 1  # default 1, ratio spreads use 2
    option_type: str  # "call" or "put"
    strike: float  # Suggested strike price (snapped to tick)
    strike_label: str  # Human-readable: "1.0 ATR OTM put" or "ATM call"
    expiration: date
    days_to_expiry: int
    atm_iv_at_expiry: float  # IV at this expiration (from term structure)

    @property
    def short_code(self) -> str:
        """Parseable short code: 'STO 1x 580P 3/27/26'."""
        p_or_c = "C" if self.option_type == "call" else "P"
        strike_str = f"{self.strike:.0f}" if self.strike == int(self.strike) else f"{self.strike:.1f}"
        yy = self.expiration.strftime("%y")
        return f"{self.action.value} {self.quantity}x {strike_str}{p_or_c} {self.expiration.month}/{self.expiration.day}/{yy}"

    @property
    def osi_symbol(self) -> str:
        """OCC option symbol format: 'SPY   260327P00580000'.

        Note: ticker must be set externally since LegSpec doesn't carry it.
        Returns the date+type+strike portion only.
        """
        p_or_c = "C" if self.option_type == "call" else "P"
        date_str = self.expiration.strftime("%y%m%d")
        # OCC: strike * 1000, zero-padded to 8 digits
        strike_int = int(self.strike * 1000)
        return f"{date_str}{p_or_c}{strike_int:08d}"


class TradeSpec(BaseModel):
    """Actionable trade parameters — the 'what to actually trade' output.

    Cotrader contract:
      1. Read ``structure_type`` + ``order_side`` to identify the trade.
      2. Read ``legs`` / ``order_data`` to build the multi-leg order.
      3. After fill, apply exit rules using ``profit_target_pct``,
         ``stop_loss_pct``, and ``exit_dte`` with the actual credit/debit.

    ``stop_loss_pct`` semantics depend on ``order_side``:
      - credit: multiple of credit received (2.0 → close when loss = 2× credit)
      - debit: fraction of debit paid to lose (0.50 → close at 50% loss of debit)
    """

    ticker: str  # Underlying symbol
    legs: list[LegSpec]
    underlying_price: float
    target_dte: int  # Primary DTE target (e.g., 35)
    target_expiration: date  # Best matching real expiration
    # Calendar/diagonal only:
    front_expiration: date | None = None
    front_dte: int | None = None
    back_expiration: date | None = None
    back_dte: int | None = None
    iv_at_front: float | None = None
    iv_at_back: float | None = None
    iv_differential_pct: float | None = None  # (front - back) / back * 100
    # Sizing context:
    wing_width_points: float | None = None  # IC/IFly: distance between short and long strike
    max_risk_per_spread: str | None = None  # "wing_width * 100 - credit"
    # Rationale:
    spec_rationale: str  # Why these specific dates/strikes were chosen
    # Structure identification (cotrader reads these for order routing):
    structure_type: str | None = None  # StructureType value
    order_side: str | None = None  # OrderSide value: "credit" or "debit"
    # Exit guidance (cotrader applies with actual fill data):
    profit_target_pct: float | None = None  # Close at X% of max profit (0.50 = 50%)
    stop_loss_pct: float | None = None  # Credit: X× credit; Debit: X fraction loss
    exit_dte: int | None = None  # Close when DTE drops to this
    max_profit_desc: str | None = None  # "Credit received" / "Spread width - debit"
    max_loss_desc: str | None = None  # "Wing width - credit" / "UNLIMITED"
    exit_notes: list[str] = []  # Structure-specific guidance
    max_entry_price: float | None = None  # Don't chase beyond this price

    @property
    def leg_codes(self) -> list[str]:
        """Parseable short codes: ['STO 1x SPY 580P 3/27/26', ...]."""
        p = self.ticker
        return [f"{leg.action.value} {leg.quantity}x {p} "
                f"{'C' if leg.option_type == 'call' else 'P'}"
                f"{f'{leg.strike:.0f}' if leg.strike == int(leg.strike) else f'{leg.strike:.1f}'} "
                f"{leg.expiration.month}/{leg.expiration.day}/{leg.expiration.strftime('%y')}"
                for leg in self.legs]

    @property
    def streamer_symbols(self) -> list[str]:
        """Full OCC option symbols: ['SPY   260327P00580000', ...]."""
        padded = f"{self.ticker:<6}"
        return [f"{padded}{leg.osi_symbol}" for leg in self.legs]

    @property
    def order_data(self) -> list[dict]:
        """Machine-readable dicts for cotrader order building.

        Each dict: {action, quantity, symbol, option_type, strike, expiration,
                     osi_symbol, instrument_type}
        """
        padded = f"{self.ticker:<6}"
        return [
            {
                "action": leg.action.value,
                "quantity": leg.quantity,
                "symbol": self.ticker,
                "option_type": leg.option_type,
                "strike": leg.strike,
                "expiration": leg.expiration.isoformat(),
                "osi_symbol": f"{padded}{leg.osi_symbol}",
                "instrument_type": "EQUITY_OPTION",
            }
            for leg in self.legs
        ]

    @property
    def exit_summary(self) -> str:
        """One-line exit guidance for display."""
        parts = []
        if self.profit_target_pct is not None:
            parts.append(f"TP {self.profit_target_pct:.0%}")
        if self.stop_loss_pct is not None:
            if self.order_side == "credit":
                parts.append(f"SL {self.stop_loss_pct:.0f}× credit")
            else:
                parts.append(f"SL {self.stop_loss_pct:.0%} debit")
        if self.exit_dte is not None:
            parts.append(f"close ≤{self.exit_dte} DTE")
        return " | ".join(parts) if parts else ""
