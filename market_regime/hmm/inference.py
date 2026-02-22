"""Regime inference: predict current regime from features."""

from datetime import date

import numpy as np
import pandas as pd
from tabulate import tabulate

from market_regime.hmm.trainer import MODEL_VERSION, HMMTrainer
from market_regime.models.features import FeatureInspection
from market_regime.models.regime import (
    RegimeID,
    RegimeResult,
    RegimeTimeSeries,
    RegimeTimeSeriesEntry,
)


class RegimeInference:
    """Runs regime inference using a fitted HMM."""

    def __init__(self, trainer: HMMTrainer) -> None:
        if not trainer.is_fitted:
            raise RuntimeError("RegimeInference requires a fitted HMMTrainer")
        self._trainer = trainer

    @staticmethod
    def _trend_direction(regime: RegimeID, trend_val: float) -> str | None:
        """Derive trend direction from regime and trend_strength feature value.

        Returns "bullish"/"bearish" for trending regimes (R3/R4), None for MR.
        """
        if regime in (RegimeID.R3_LOW_VOL_TREND, RegimeID.R4_HIGH_VOL_TREND):
            return "bullish" if trend_val >= 0 else "bearish"
        return None

    def predict(self, feature_matrix: pd.DataFrame, ticker: str) -> RegimeResult:
        """Predict current regime from recent feature matrix.

        Uses Viterbi decode on full sequence, returns the last state's regime.
        """
        X = feature_matrix.values
        model = self._trainer.model
        lmap = self._trainer.label_map

        # Viterbi decode
        _, raw_states = model.decode(X, algorithm="viterbi")
        last_raw_state = raw_states[-1]

        # Posterior probabilities for the last observation
        log_prob, posteriors = model.score_samples(X)
        last_posteriors = posteriors[-1]

        # Map to RegimeID
        regime_id = RegimeID(lmap[last_raw_state])
        confidence = float(last_posteriors[last_raw_state])

        regime_probs = {
            lmap[s]: float(last_posteriors[s])
            for s in range(len(last_posteriors))
        }

        as_of = (
            feature_matrix.index[-1].date()
            if isinstance(feature_matrix.index, pd.DatetimeIndex)
            else date.today()
        )

        # Derive trend direction from trend_strength feature
        trend_val = 0.0
        if "trend_strength" in feature_matrix.columns:
            trend_val = float(feature_matrix["trend_strength"].iloc[-1])

        return RegimeResult(
            ticker=ticker,
            regime=regime_id,
            confidence=confidence,
            regime_probabilities=regime_probs,
            trend_direction=self._trend_direction(regime_id, trend_val),
            as_of_date=as_of,
            model_version=MODEL_VERSION,
        )

    def predict_series(
        self, feature_matrix: pd.DataFrame, ticker: str
    ) -> RegimeTimeSeries:
        """Predict regime for every date in the feature matrix."""
        X = feature_matrix.values
        model = self._trainer.model
        lmap = self._trainer.label_map

        _, raw_states = model.decode(X, algorithm="viterbi")
        _, posteriors = model.score_samples(X)

        # Get trend_strength column index for direction
        has_trend = "trend_strength" in feature_matrix.columns

        entries: list[RegimeTimeSeriesEntry] = []
        for i, (idx, raw_state) in enumerate(zip(feature_matrix.index, raw_states)):
            regime_id = RegimeID(lmap[raw_state])
            dt = idx.date() if isinstance(idx, pd.Timestamp) else idx

            probs = {lmap[s]: float(posteriors[i, s]) for s in range(len(posteriors[i]))}

            trend_val = float(feature_matrix["trend_strength"].iloc[i]) if has_trend else 0.0

            entries.append(
                RegimeTimeSeriesEntry(
                    date=dt,
                    regime=regime_id,
                    confidence=float(posteriors[i, raw_state]),
                    probabilities=probs,
                    trend_direction=self._trend_direction(regime_id, trend_val),
                )
            )

        return RegimeTimeSeries(ticker=ticker, entries=entries)

    def predict_with_explanation(
        self,
        feature_matrix: pd.DataFrame,
        ticker: str,
        feature_inspection: FeatureInspection,
    ) -> tuple[RegimeResult, RegimeTimeSeries, str]:
        """Predict regime and generate human-readable explanation."""
        from market_regime.config import get_settings

        result = self.predict(feature_matrix, ticker)
        series = self.predict_series(feature_matrix, ticker)

        # Build explanation text
        settings = get_settings()
        regime_names = settings.regimes.names

        regime_name = regime_names[result.regime]
        # Cap display — no model should claim perfect certainty
        confidence_pct = min(result.confidence * 100, settings.display.confidence_cap)

        # Get last normalized feature values
        last_features = feature_inspection.normalized_features[-1] if feature_inspection.normalized_features else {}

        direction_suffix = ""
        if result.trend_direction:
            direction_suffix = f", {result.trend_direction}"

        lines = [
            f"{ticker} classified as R{result.regime} ({regime_name}{direction_suffix}) with {confidence_pct:.1f}% confidence.",
            "",
            "Current features (normalized z-scores):",
        ]

        # Build feature table with comments
        feat_rows = []
        for name in feature_inspection.feature_names:
            val = last_features.get(name, "N/A")
            if isinstance(val, (int, float)):
                feat_rows.append({
                    "Feature": name,
                    "Z-Score": f"{val:+.2f}",
                    "Comment": self._zscore_comment(name, val),
                })
            else:
                feat_rows.append({"Feature": name, "Z-Score": str(val), "Comment": ""})

        lines.append(tabulate(feat_rows, headers="keys", tablefmt="simple", stralign="right"))

        # Explain quadrant placement
        vol_val = last_features.get("realized_vol", 0)
        trend_val = last_features.get("trend_strength", 0)
        vol_desc = "low-vol" if isinstance(vol_val, (int, float)) and vol_val < 0 else "high-vol"
        trend_boundary = settings.interpretation.trend_strength_boundary
        trend_desc = "trending" if isinstance(trend_val, (int, float)) and abs(trend_val) > trend_boundary else "mean-reverting"
        lines.append(f"\nThis places {ticker} in the {vol_desc} + {trend_desc} quadrant.")

        explanation_text = "\n".join(lines)
        return result, series, explanation_text

    @staticmethod
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
            return f"{'above' if val > 0 else 'below'} avg volume" if abs_v >= zt.normal else "normal volume"
        return ""
