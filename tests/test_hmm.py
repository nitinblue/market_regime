"""Tests for HMM training and inference."""

import numpy as np
import pandas as pd
import pytest

from market_analyzer.hmm.inference import RegimeInference
from market_analyzer.hmm.trainer import HMMTrainer, MODEL_VERSION
from market_analyzer.models.regime import (
    HMMModelInfo,
    RegimeConfig,
    RegimeID,
    RegimeResult,
    RegimeTimeSeries,
)


class TestHMMTrainerFit:
    def test_fit_produces_4_states(self, fitted_trainer):
        assert fitted_trainer.model.n_components == 4

    def test_is_fitted(self, fitted_trainer):
        assert fitted_trainer.is_fitted

    def test_not_fitted_raises(self):
        trainer = HMMTrainer()
        assert not trainer.is_fitted
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = trainer.model

    def test_label_map_covers_all_regimes(self, fitted_trainer):
        """Label map should map to all 4 regime IDs."""
        regimes = set(fitted_trainer.label_map.values())
        expected = {
            RegimeID.R1_LOW_VOL_MR,
            RegimeID.R2_HIGH_VOL_MR,
            RegimeID.R3_LOW_VOL_TREND,
            RegimeID.R4_HIGH_VOL_TREND,
        }
        assert regimes == expected

    def test_label_map_covers_all_states(self, fitted_trainer):
        """Label map should cover all 4 raw HMM states."""
        states = set(fitted_trainer.label_map.keys())
        assert states == {0, 1, 2, 3}


class TestHMMTrainerPersistence:
    def test_save_load_roundtrip(self, fitted_trainer, tmp_path):
        path = tmp_path / "test_model.joblib"
        fitted_trainer.save(path)

        loaded = HMMTrainer()
        loaded.load(path)

        assert loaded.is_fitted
        np.testing.assert_allclose(
            loaded.model.means_, fitted_trainer.model.means_, rtol=1e-10
        )
        assert loaded.label_map == fitted_trainer.label_map

    def test_save_not_fitted_raises(self, tmp_path):
        trainer = HMMTrainer()
        with pytest.raises(RuntimeError, match="not been fitted"):
            trainer.save(tmp_path / "model.joblib")

    def test_load_missing_file_raises(self, tmp_path):
        trainer = HMMTrainer()
        with pytest.raises(FileNotFoundError):
            trainer.load(tmp_path / "nonexistent.joblib")


class TestModelInfo:
    def test_get_model_info(self, fitted_trainer):
        info = fitted_trainer.get_model_info()
        assert isinstance(info, HMMModelInfo)

    def test_model_info_has_all_regimes(self, fitted_trainer):
        info = fitted_trainer.get_model_info()
        for rid in [1, 2, 3, 4]:
            assert rid in info.state_means
            assert rid in info.state_covariances
            assert rid in info.initial_probabilities

    def test_transition_matrix_rows_sum_to_one(self, fitted_trainer):
        info = fitted_trainer.get_model_info()
        for row in info.transition_matrix:
            np.testing.assert_allclose(sum(row), 1.0, atol=1e-6)

    def test_transition_matrix_shape(self, fitted_trainer):
        info = fitted_trainer.get_model_info()
        assert len(info.transition_matrix) == 4
        assert all(len(row) == 4 for row in info.transition_matrix)

    def test_feature_names_preserved(self, fitted_trainer):
        info = fitted_trainer.get_model_info()
        assert "realized_vol" in info.feature_names
        assert "trend_strength" in info.feature_names

    def test_not_fitted_raises(self):
        trainer = HMMTrainer()
        with pytest.raises(RuntimeError):
            trainer.get_model_info()


class TestRegimeInference:
    def test_predict_returns_regime_result(self, fitted_trainer, sample_feature_matrix):
        inference = RegimeInference(fitted_trainer)
        result = inference.predict(sample_feature_matrix, "TEST")
        assert isinstance(result, RegimeResult)

    def test_confidence_between_0_and_1(self, fitted_trainer, sample_feature_matrix):
        inference = RegimeInference(fitted_trainer)
        result = inference.predict(sample_feature_matrix, "TEST")
        assert 0.0 <= result.confidence <= 1.0

    def test_probabilities_sum_to_1(self, fitted_trainer, sample_feature_matrix):
        inference = RegimeInference(fitted_trainer)
        result = inference.predict(sample_feature_matrix, "TEST")
        total = sum(result.regime_probabilities.values())
        np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_regime_is_valid(self, fitted_trainer, sample_feature_matrix):
        inference = RegimeInference(fitted_trainer)
        result = inference.predict(sample_feature_matrix, "TEST")
        assert result.regime in list(RegimeID)

    def test_model_version(self, fitted_trainer, sample_feature_matrix):
        inference = RegimeInference(fitted_trainer)
        result = inference.predict(sample_feature_matrix, "TEST")
        assert result.model_version == MODEL_VERSION

    def test_predict_series(self, fitted_trainer, sample_feature_matrix):
        inference = RegimeInference(fitted_trainer)
        series = inference.predict_series(sample_feature_matrix, "TEST")
        assert isinstance(series, RegimeTimeSeries)
        assert len(series.entries) == len(sample_feature_matrix)

    def test_predict_series_entries_valid(self, fitted_trainer, sample_feature_matrix):
        inference = RegimeInference(fitted_trainer)
        series = inference.predict_series(sample_feature_matrix, "TEST")
        for entry in series.entries:
            assert entry.regime in list(RegimeID)
            assert 0.0 <= entry.confidence <= 1.0
            total = sum(entry.probabilities.values())
            np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_requires_fitted_trainer(self):
        trainer = HMMTrainer()
        with pytest.raises(RuntimeError, match="fitted"):
            RegimeInference(trainer)
