"""Top-level regime API: accepts DataFrame or auto-fetches via DataService."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from market_regime.features.pipeline import (
    compute_features,
    compute_features_with_inspection,
)
from market_regime.hmm.inference import RegimeInference
from market_regime.hmm.trainer import HMMTrainer
from market_regime.models.features import FeatureConfig
from market_regime.models.phase import PhaseResult
from market_regime.models.regime import (
    CrossTickerEntry,
    FeatureZScore,
    HMMModelInfo,
    LabelAlignmentDetail,
    RegimeConfig,
    RegimeDistributionEntry,
    RegimeExplanation,
    RegimeHistoryDay,
    RegimeID,
    RegimeResult,
    RegimeTimeSeries,
    ResearchReport,
    StateMeansRow,
    TickerResearch,
    TransitionRow,
)
from market_regime.phases.detector import PhaseDetector

if TYPE_CHECKING:
    from market_regime.data.service import DataService

def _default_model_dir() -> Path:
    from market_regime.config import get_settings

    cfg = get_settings().cache
    if cfg.model_dir:
        return Path(cfg.model_dir)
    return Path.home() / ".market_regime" / "models"


class RegimeService:
    """Orchestrates feature computation, HMM training, and regime inference."""

    def __init__(
        self,
        config: RegimeConfig = RegimeConfig(),
        data_service: DataService | None = None,
        model_dir: Path | None = None,
        feature_config: FeatureConfig = FeatureConfig(),
    ) -> None:
        self.config = config
        self.data_service = data_service
        self.model_dir = model_dir or _default_model_dir()
        self.feature_config = feature_config
        self._trainers: dict[str, HMMTrainer] = {}

    def _get_ohlcv(self, ticker: str, ohlcv: pd.DataFrame | None) -> pd.DataFrame:
        """Get OHLCV data from caller or DataService."""
        if ohlcv is not None:
            return ohlcv
        if self.data_service is None:
            raise ValueError(
                "Either provide ohlcv DataFrame or initialize RegimeService with a DataService"
            )
        start_date = date.today() - timedelta(
            days=int(self.config.training_lookback_years * 365)
        )
        return self.data_service.get_ohlcv(ticker, start_date=start_date)

    def _model_path(self, ticker: str) -> Path:
        return self.model_dir / f"{ticker}.joblib"

    def _get_trainer(self, ticker: str) -> HMMTrainer | None:
        """Get cached trainer, or try loading from disk."""
        if ticker in self._trainers:
            return self._trainers[ticker]

        path = self._model_path(ticker)
        if path.exists():
            trainer = HMMTrainer(self.config)
            trainer.load(path)
            self._trainers[ticker] = trainer
            return trainer

        return None

    def fit(self, ticker: str, ohlcv: pd.DataFrame | None = None) -> None:
        """Train/retrain HMM for a given instrument."""
        df = self._get_ohlcv(ticker, ohlcv)
        features = compute_features(df, self.feature_config)

        trainer = HMMTrainer(self.config)
        trainer.fit(features)
        trainer.save(self._model_path(ticker))
        self._trainers[ticker] = trainer

    def detect(
        self, ticker: str, ohlcv: pd.DataFrame | None = None
    ) -> RegimeResult:
        """Detect current regime for a single instrument.

        Auto-fits if no model exists. Auto-fetches OHLCV if data_service available.
        """
        df = self._get_ohlcv(ticker, ohlcv)

        trainer = self._get_trainer(ticker)
        if trainer is None:
            self.fit(ticker, df)
            trainer = self._trainers[ticker]

        features = compute_features(df, self.feature_config)
        inference = RegimeInference(trainer)
        return inference.predict(features, ticker)

    def detect_batch(
        self,
        tickers: list[str] | None = None,
        data: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, RegimeResult]:
        """Detect regimes for multiple instruments."""
        if tickers is None and data is None:
            raise ValueError("Provide either tickers list or data dict")

        results: dict[str, RegimeResult] = {}
        if data is not None:
            for ticker, df in data.items():
                results[ticker] = self.detect(ticker, df)
        elif tickers is not None:
            for ticker in tickers:
                results[ticker] = self.detect(ticker)

        return results

    def explain(
        self, ticker: str, ohlcv: pd.DataFrame | None = None
    ) -> RegimeExplanation:
        """Master inspection: features + model info + regime series + explanation."""
        df = self._get_ohlcv(ticker, ohlcv)

        # Ensure model is fitted
        trainer = self._get_trainer(ticker)
        if trainer is None:
            self.fit(ticker, df)
            trainer = self._trainers[ticker]

        # Compute features with inspection
        features, feature_inspection = compute_features_with_inspection(
            df, ticker, self.feature_config
        )

        # Run inference with explanation
        inference = RegimeInference(trainer)
        result, series, explanation_text = inference.predict_with_explanation(
            features, ticker, feature_inspection
        )

        model_info = trainer.get_model_info()

        return RegimeExplanation(
            regime_result=result,
            feature_inspection=feature_inspection,
            model_info=model_info,
            regime_series=series,
            explanation_text=explanation_text,
        )

    def get_model_info(self, ticker: str) -> HMMModelInfo:
        """Get model inspection for a ticker. Raises if no model fitted."""
        trainer = self._get_trainer(ticker)
        if trainer is None:
            raise RuntimeError(f"No fitted model for {ticker}. Call fit() first.")
        return trainer.get_model_info()

    def get_regime_history(
        self, ticker: str, ohlcv: pd.DataFrame | None = None
    ) -> RegimeTimeSeries:
        """Get regime classification for every date in the data."""
        df = self._get_ohlcv(ticker, ohlcv)

        trainer = self._get_trainer(ticker)
        if trainer is None:
            self.fit(ticker, df)
            trainer = self._trainers[ticker]

        features = compute_features(df, self.feature_config)
        inference = RegimeInference(trainer)
        return inference.predict_series(features, ticker)

    # --- Phase API ---

    def detect_phase(
        self, ticker: str, ohlcv: pd.DataFrame | None = None
    ) -> PhaseResult:
        """Detect Wyckoff phase for a single instrument.

        Requires regime series (auto-fits if needed) and OHLCV data.
        """
        df = self._get_ohlcv(ticker, ohlcv)
        regime_series = self.get_regime_history(ticker, df)
        detector = PhaseDetector()
        return detector.detect(ticker, df, regime_series)

    # --- Research API ---

    def research(
        self, ticker: str, ohlcv: pd.DataFrame | None = None
    ) -> TickerResearch:
        """Full interpreted regime research for a single ticker."""
        df = self._get_ohlcv(ticker, ohlcv)
        exp = self.explain(ticker, df)
        phase = PhaseDetector().detect(ticker, df, exp.regime_series)
        return _build_ticker_research(exp, phase)

    def research_batch(
        self,
        tickers: list[str] | None = None,
        data: dict[str, pd.DataFrame] | None = None,
    ) -> ResearchReport:
        """Full research for multiple tickers with cross-comparison."""
        if tickers is None and data is None:
            raise ValueError("Provide either tickers list or data dict")

        ticker_list = tickers if tickers is not None else list(data.keys())
        researches: list[TickerResearch] = []
        for t in ticker_list:
            ohlcv = data.get(t) if data is not None else None
            researches.append(self.research(t, ohlcv))

        comparison: list[CrossTickerEntry] | None = None
        if len(researches) >= 2:
            comparison = [
                CrossTickerEntry(
                    ticker=r.ticker,
                    regime=r.regime_result.regime,
                    trend_direction=r.regime_result.trend_direction,
                    confidence=r.regime_result.confidence,
                    regime_probabilities=r.regime_result.regime_probabilities,
                    strategy_comment=r.strategy_comment,
                    phase=r.phase_result.phase if r.phase_result else None,
                    phase_name=r.phase_result.phase_name if r.phase_result else None,
                )
                for r in researches
            ]

        return ResearchReport(tickers=researches, comparison=comparison)


# --- Private helpers (interpretation logic) ---

def _regime_names() -> dict[int, str]:
    from market_regime.config import get_settings
    return get_settings().regimes.names


def _regime_strategies() -> dict[int, str]:
    from market_regime.config import get_settings
    return get_settings().regimes.strategies


def _zscore_comment(name: str, val: float) -> str:
    """Interpret a z-score value in context of the feature."""
    from market_regime.config import get_settings

    zt = get_settings().interpretation.zscore_thresholds
    abs_v = abs(val)
    if abs_v < zt.normal:
        intensity = "normal"
    elif abs_v < zt.mild:
        intensity = "mild"
    elif abs_v < zt.elevated:
        intensity = "elevated"
    else:
        intensity = "extreme"

    if name in ("realized_vol", "atr_normalized"):
        direction = "high" if val > 0 else "low"
        return f"{intensity} vol ({direction})"
    elif name == "trend_strength":
        if abs_v < zt.normal:
            return "no trend"
        direction = "bullish" if val > 0 else "bearish"
        return f"{intensity} {direction} trend"
    elif name in ("log_return_1d", "log_return_5d"):
        direction = "up" if val > 0 else "down"
        period = "daily" if "1d" in name else "weekly"
        return f"{intensity} {period} move ({direction})"
    elif name == "volume_anomaly":
        if abs_v >= zt.normal:
            return f"{'above' if val > 0 else 'below'} avg volume"
        return "normal volume"
    return ""


def _transition_stability(stay_prob: float) -> str:
    from market_regime.config import get_settings

    st = get_settings().interpretation.stability_thresholds
    if stay_prob > st.very_sticky:
        return "very sticky"
    elif stay_prob > st.sticky:
        return "sticky"
    elif stay_prob > st.moderately_stable:
        return "moderately stable"
    return "unstable"


def _build_transition_matrix(info: HMMModelInfo) -> list[TransitionRow]:
    rows: list[TransitionRow] = []
    for i, row in enumerate(info.transition_matrix):
        regime = RegimeID(i + 1)
        stay_prob = row[i]
        to_probs = {j + 1: p for j, p in enumerate(row)}
        stability = _transition_stability(stay_prob)

        target: RegimeID | None = None
        if stability == "unstable":
            other = [(p, j) for j, p in enumerate(row) if j != i]
            other.sort(reverse=True)
            target = RegimeID(other[0][1] + 1)

        rows.append(
            TransitionRow(
                from_regime=regime,
                to_probabilities=to_probs,
                stay_probability=stay_prob,
                stability=stability,
                likely_transition_target=target,
            )
        )
    return rows


def _build_state_means(info: HMMModelInfo) -> list[StateMeansRow]:
    rows: list[StateMeansRow] = []
    vol_idx = (
        info.feature_names.index("realized_vol")
        if "realized_vol" in info.feature_names
        else None
    )
    trend_idx = (
        info.feature_names.index("trend_strength")
        if "trend_strength" in info.feature_names
        else None
    )

    for rid in sorted(info.state_means.keys()):
        means = info.state_means[rid]
        feature_means = {
            name: val for name, val in zip(info.feature_names, means)
        }
        vol_char = "high-vol" if (vol_idx is not None and means[vol_idx] > 0) else "low-vol"
        from market_regime.config import get_settings

        trend_boundary = get_settings().interpretation.trend_strength_boundary
        trend_char = (
            "trending"
            if (trend_idx is not None and abs(means[trend_idx]) > trend_boundary)
            else "mean-rev"
        )
        rows.append(
            StateMeansRow(
                regime=RegimeID(rid),
                feature_means=feature_means,
                vol_character=vol_char,
                trend_character=trend_char,
            )
        )
    return rows


def _build_label_alignment(info: HMMModelInfo) -> list[LabelAlignmentDetail]:
    align = info.label_alignment
    rows: list[LabelAlignmentDetail] = []
    for rid in sorted(align.per_state_vol_mean.keys()):
        v = align.per_state_vol_mean[rid]
        t = align.per_state_trend_mean[rid]
        rows.append(
            LabelAlignmentDetail(
                regime=RegimeID(rid),
                vol_mean=v,
                trend_mean=t,
                vol_side="low" if v < align.vol_threshold else "high",
                trend_side="MR" if abs(t) < abs(align.trend_threshold) else "trend",
                vol_threshold=align.vol_threshold,
                trend_threshold=align.trend_threshold,
            )
        )
    return rows


def _build_current_features(exp: RegimeExplanation) -> list[FeatureZScore]:
    last = exp.feature_inspection.normalized_features[-1]
    rows: list[FeatureZScore] = []
    for k, v in last.items():
        if k == "date":
            continue
        rows.append(
            FeatureZScore(
                feature=k,
                z_score=v,
                comment=_zscore_comment(k, v),
            )
        )
    return rows


def _build_recent_history(exp: RegimeExplanation) -> list[RegimeHistoryDay]:
    from market_regime.config import get_settings

    n = get_settings().interpretation.recent_history_days
    all_entries = exp.regime_series.entries
    entries = all_entries[-n:]
    rows: list[RegimeHistoryDay] = []
    for i, entry in enumerate(entries):
        if i == 0 and len(all_entries) > n:
            prev_regime = all_entries[-(n + 1)].regime
        elif i > 0:
            prev_regime = entries[i - 1].regime
        else:
            prev_regime = entry.regime

        changed_from = RegimeID(prev_regime) if entry.regime != prev_regime else None

        rows.append(
            RegimeHistoryDay(
                date=entry.date,
                regime=entry.regime,
                trend_direction=entry.trend_direction,
                confidence=entry.confidence,
                changed_from=changed_from,
            )
        )
    return rows


def _build_regime_distribution(exp: RegimeExplanation) -> list[RegimeDistributionEntry]:
    from market_regime.config import get_settings

    settings = get_settings()
    regime_names = settings.regimes.names
    rare_pct = settings.interpretation.rare_regime_pct

    counts: dict[int, int] = {}
    for entry in exp.regime_series.entries:
        counts[entry.regime] = counts.get(entry.regime, 0) + 1
    total = len(exp.regime_series.entries)

    dominant = max(counts, key=lambda r: counts[r])
    rows: list[RegimeDistributionEntry] = []
    for rid in sorted(counts.keys()):
        pct = counts[rid] / total * 100
        rows.append(
            RegimeDistributionEntry(
                regime=RegimeID(rid),
                name=regime_names.get(rid, f"R{rid}"),
                days=counts[rid],
                percentage=pct,
                is_dominant=(rid == dominant),
                is_rare=(pct < rare_pct),
            )
        )
    return rows


def _build_ticker_research(
    exp: RegimeExplanation,
    phase: PhaseResult | None = None,
) -> TickerResearch:
    result = exp.regime_result
    info = exp.model_info
    return TickerResearch(
        ticker=result.ticker,
        regime_result=result,
        explanation_text=exp.explanation_text,
        transition_matrix=_build_transition_matrix(info),
        state_means=_build_state_means(info),
        label_alignment=_build_label_alignment(info),
        current_features=_build_current_features(exp),
        recent_history=_build_recent_history(exp),
        regime_distribution=_build_regime_distribution(exp),
        strategy_comment=_regime_strategies().get(int(result.regime), ""),
        model_info=info,
        phase_result=phase,
    )
