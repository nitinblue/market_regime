"""HMM training: fit, label alignment, persist."""

from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from market_regime.models.regime import (
    HMMModelInfo,
    LabelAlignment,
    RegimeConfig,
    RegimeID,
)

MODEL_VERSION = "hmm-v1"


class HMMTrainer:
    """Trains and persists GaussianHMM models with label alignment."""

    def __init__(self, config: RegimeConfig = RegimeConfig()) -> None:
        self.config = config
        self._model: GaussianHMM | None = None
        self._label_map: dict[int, int] | None = None
        self._feature_names: list[str] | None = None
        self._alignment: LabelAlignment | None = None
        self._training_rows: int = 0
        self._training_date_range: tuple[date, date] | None = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None and self._label_map is not None

    @property
    def model(self) -> GaussianHMM:
        if self._model is None:
            raise RuntimeError("HMMTrainer has not been fitted yet")
        return self._model

    @property
    def label_map(self) -> dict[int, int]:
        if self._label_map is None:
            raise RuntimeError("HMMTrainer has not been fitted yet")
        return self._label_map

    def fit(self, feature_matrix: pd.DataFrame) -> None:
        """Fit GaussianHMM on feature matrix and align labels to R1-R4."""
        X = feature_matrix.values
        self._feature_names = list(feature_matrix.columns)
        self._training_rows = len(X)

        if isinstance(feature_matrix.index, pd.DatetimeIndex):
            self._training_date_range = (
                feature_matrix.index[0].date(),
                feature_matrix.index[-1].date(),
            )

        from market_regime.config import get_settings

        hmm_cfg = get_settings().hmm
        self._model = GaussianHMM(
            n_components=self.config.n_states,
            covariance_type=hmm_cfg.covariance_type,
            n_iter=hmm_cfg.n_iter,
            random_state=hmm_cfg.random_state,
        )
        self._model.fit(X)
        self._align_labels()

    def _align_labels(self) -> None:
        """Map arbitrary HMM states to R1-R4 using vol/trend 2x2 grid.

        Volatility axis: mean realized_vol per state (low vs high)
        Trend axis: mean |trend_strength| per state (mean-reverting vs trending)
        """
        means = self._model.means_
        feature_names = self._feature_names

        vol_idx = feature_names.index("realized_vol")
        trend_idx = feature_names.index("trend_strength")

        n_states = self.config.n_states
        vol_means = {s: float(means[s, vol_idx]) for s in range(n_states)}
        trend_means = {s: float(abs(means[s, trend_idx])) for s in range(n_states)}

        vol_threshold = float(np.median(list(vol_means.values())))
        trend_threshold = float(np.median(list(trend_means.values())))

        # Classify each state into the 2x2 grid
        # R1: low-vol, low-trend (mean reverting)
        # R2: high-vol, low-trend (mean reverting)
        # R3: low-vol, high-trend (trending)
        # R4: high-vol, high-trend (trending)
        buckets: dict[int, list[tuple[int, float]]] = {
            RegimeID.R1_LOW_VOL_MR: [],
            RegimeID.R2_HIGH_VOL_MR: [],
            RegimeID.R3_LOW_VOL_TREND: [],
            RegimeID.R4_HIGH_VOL_TREND: [],
        }

        for s in range(n_states):
            low_vol = vol_means[s] <= vol_threshold
            low_trend = trend_means[s] <= trend_threshold

            if low_vol and low_trend:
                regime = RegimeID.R1_LOW_VOL_MR
            elif not low_vol and low_trend:
                regime = RegimeID.R2_HIGH_VOL_MR
            elif low_vol and not low_trend:
                regime = RegimeID.R3_LOW_VOL_TREND
            else:
                regime = RegimeID.R4_HIGH_VOL_TREND

            # Distance from boundary for collision resolution
            vol_dist = abs(vol_means[s] - vol_threshold)
            trend_dist = abs(trend_means[s] - trend_threshold)
            distance = vol_dist + trend_dist
            buckets[regime].append((s, distance))

        # Resolve collisions: assign by distance priority
        state_to_regime: dict[int, int] = {}
        assigned_states: set[int] = set()
        unassigned_regimes: list[int] = []

        # First pass: assign highest-distance state to each bucket
        for regime_id in sorted(buckets.keys()):
            candidates = [
                (s, d) for s, d in buckets[regime_id] if s not in assigned_states
            ]
            if candidates:
                # Pick state with greatest distance from boundary (most certain)
                best = max(candidates, key=lambda x: x[1])
                state_to_regime[best[0]] = regime_id
                assigned_states.add(best[0])
            else:
                unassigned_regimes.append(regime_id)

        # Second pass: assign remaining states to unassigned regimes
        remaining_states = [s for s in range(n_states) if s not in assigned_states]
        for state, regime_id in zip(remaining_states, unassigned_regimes):
            state_to_regime[state] = regime_id

        self._label_map = state_to_regime
        self._alignment = LabelAlignment(
            state_to_regime=state_to_regime,
            per_state_vol_mean={
                state_to_regime[s]: vol_means[s] for s in range(n_states)
            },
            per_state_trend_mean={
                state_to_regime[s]: trend_means[s] for s in range(n_states)
            },
            vol_threshold=vol_threshold,
            trend_threshold=trend_threshold,
        )

    def get_model_info(self) -> HMMModelInfo:
        """Return full model inspection, remapped through label alignment."""
        if not self.is_fitted:
            raise RuntimeError("HMMTrainer has not been fitted yet")

        model = self._model
        lmap = self._label_map
        n_states = self.config.n_states

        # Remap means and covariances by RegimeID
        state_means: dict[int, list[float]] = {}
        state_covs: dict[int, list[list[float]]] = {}
        initial_probs: dict[int, float] = {}

        for raw_state in range(n_states):
            regime_id = lmap[raw_state]
            state_means[regime_id] = model.means_[raw_state].tolist()
            state_covs[regime_id] = model.covars_[raw_state].tolist()
            initial_probs[regime_id] = float(model.startprob_[raw_state])

        # Remap transition matrix: transmat[i,j] -> regime_i -> regime_j
        regime_order = sorted(lmap.values())
        raw_order = [
            next(s for s, r in lmap.items() if r == rid) for rid in regime_order
        ]
        transition_matrix = [
            [float(model.transmat_[i, j]) for j in raw_order] for i in raw_order
        ]

        return HMMModelInfo(
            state_means=state_means,
            state_covariances=state_covs,
            transition_matrix=transition_matrix,
            initial_probabilities=initial_probs,
            label_alignment=self._alignment,
            feature_names=self._feature_names,
            training_rows=self._training_rows,
            training_date_range=self._training_date_range,
        )

    def save(self, path: Path) -> None:
        """Persist fitted model to disk via joblib."""
        if not self.is_fitted:
            raise RuntimeError("HMMTrainer has not been fitted yet")

        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": self._model,
            "label_map": self._label_map,
            "feature_names": self._feature_names,
            "alignment": self._alignment,
            "config": self.config,
            "training_rows": self._training_rows,
            "training_date_range": self._training_date_range,
            "model_version": MODEL_VERSION,
        }
        joblib.dump(bundle, path)

    def load(self, path: Path) -> None:
        """Load a previously fitted model from disk."""
        if not path.exists():
            raise FileNotFoundError(f"No model file at {path}")

        bundle = joblib.load(path)
        self._model = bundle["model"]
        self._label_map = bundle["label_map"]
        self._feature_names = bundle["feature_names"]
        self._alignment = bundle["alignment"]
        self.config = bundle["config"]
        self._training_rows = bundle["training_rows"]
        self._training_date_range = bundle.get("training_date_range")
