"""Tests for regime label alignment validation."""

import numpy as np
import pytest

from market_analyzer.hmm.trainer import HMMTrainer
from market_analyzer.models.regime import RegimeConfig, RegimeID


class TestLabelAlignmentSemantics:
    def test_r1_has_lowest_vol(self, fitted_trainer):
        """R1 (Low-Vol MR) should have the lowest vol among states."""
        info = fitted_trainer.get_model_info()
        vol_means = info.label_alignment.per_state_vol_mean
        r1_vol = vol_means[RegimeID.R1_LOW_VOL_MR]
        r2_vol = vol_means[RegimeID.R2_HIGH_VOL_MR]
        assert r1_vol < r2_vol

    def test_r4_has_highest_vol(self, fitted_trainer):
        """R4 (High-Vol Trend) should have higher vol than R3."""
        info = fitted_trainer.get_model_info()
        vol_means = info.label_alignment.per_state_vol_mean
        r3_vol = vol_means[RegimeID.R3_LOW_VOL_TREND]
        r4_vol = vol_means[RegimeID.R4_HIGH_VOL_TREND]
        assert r4_vol > r3_vol

    def test_r4_has_highest_trend(self, fitted_trainer):
        """R4 should have higher trend than R2."""
        info = fitted_trainer.get_model_info()
        trend_means = info.label_alignment.per_state_trend_mean
        r2_trend = trend_means[RegimeID.R2_HIGH_VOL_MR]
        r4_trend = trend_means[RegimeID.R4_HIGH_VOL_TREND]
        assert r4_trend > r2_trend

    def test_low_vol_regimes_below_threshold(self, fitted_trainer):
        """R1 and R3 vol means should be below the vol threshold."""
        info = fitted_trainer.get_model_info()
        alignment = info.label_alignment
        assert alignment.per_state_vol_mean[RegimeID.R1_LOW_VOL_MR] <= alignment.vol_threshold
        assert alignment.per_state_vol_mean[RegimeID.R3_LOW_VOL_TREND] <= alignment.vol_threshold

    def test_high_vol_regimes_above_threshold(self, fitted_trainer):
        """R2 and R4 vol means should be above the vol threshold."""
        info = fitted_trainer.get_model_info()
        alignment = info.label_alignment
        assert alignment.per_state_vol_mean[RegimeID.R2_HIGH_VOL_MR] >= alignment.vol_threshold
        assert alignment.per_state_vol_mean[RegimeID.R4_HIGH_VOL_TREND] >= alignment.vol_threshold


class TestDeterminism:
    def test_same_seed_same_labels(self, sample_feature_matrix):
        """Same data + same seed -> same labels."""
        trainer1 = HMMTrainer(RegimeConfig())
        trainer1.fit(sample_feature_matrix)

        trainer2 = HMMTrainer(RegimeConfig())
        trainer2.fit(sample_feature_matrix)

        assert trainer1.label_map == trainer2.label_map
        np.testing.assert_allclose(
            trainer1.model.means_, trainer2.model.means_, rtol=1e-10
        )
