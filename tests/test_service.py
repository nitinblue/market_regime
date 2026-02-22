"""Tests for RegimeService."""

import pandas as pd
import pytest

from market_analyzer.models.regime import (
    HMMModelInfo,
    RegimeExplanation,
    RegimeResult,
    RegimeTimeSeries,
)
from market_analyzer.service.regime_service import RegimeService


class TestDetect:
    def test_detect_with_ohlcv(self, sample_ohlcv_mixed, tmp_path):
        svc = RegimeService(model_dir=tmp_path / "models")
        result = svc.detect("TEST", ohlcv=sample_ohlcv_mixed)
        assert isinstance(result, RegimeResult)
        assert result.ticker == "TEST"

    def test_detect_raises_without_ohlcv_or_data_service(self):
        svc = RegimeService()
        with pytest.raises(ValueError, match="Either provide ohlcv"):
            svc.detect("TEST")


class TestFit:
    def test_fit_persists_model(self, sample_ohlcv_mixed, tmp_path):
        model_dir = tmp_path / "models"
        svc = RegimeService(model_dir=model_dir)
        svc.fit("TEST", ohlcv=sample_ohlcv_mixed)
        assert (model_dir / "TEST.joblib").exists()

    def test_fit_then_detect(self, sample_ohlcv_mixed, tmp_path):
        svc = RegimeService(model_dir=tmp_path / "models")
        svc.fit("TEST", ohlcv=sample_ohlcv_mixed)
        result = svc.detect("TEST", ohlcv=sample_ohlcv_mixed)
        assert isinstance(result, RegimeResult)


class TestDetectBatch:
    def test_batch_with_data(self, sample_ohlcv_trending, sample_ohlcv_choppy, tmp_path):
        svc = RegimeService(model_dir=tmp_path / "models")
        results = svc.detect_batch(
            data={"TREND": sample_ohlcv_trending, "CHOP": sample_ohlcv_choppy}
        )
        assert "TREND" in results
        assert "CHOP" in results
        assert isinstance(results["TREND"], RegimeResult)

    def test_batch_raises_no_args(self, tmp_path):
        svc = RegimeService(model_dir=tmp_path / "models")
        with pytest.raises(ValueError, match="Provide either"):
            svc.detect_batch()


class TestExplain:
    def test_explain_returns_full_explanation(self, sample_ohlcv_mixed, tmp_path):
        svc = RegimeService(model_dir=tmp_path / "models")
        explanation = svc.explain("TEST", ohlcv=sample_ohlcv_mixed)
        assert isinstance(explanation, RegimeExplanation)
        assert isinstance(explanation.regime_result, RegimeResult)
        assert isinstance(explanation.model_info, HMMModelInfo)
        assert isinstance(explanation.regime_series, RegimeTimeSeries)
        assert len(explanation.explanation_text) > 0

    def test_explain_has_feature_inspection(self, sample_ohlcv_mixed, tmp_path):
        svc = RegimeService(model_dir=tmp_path / "models")
        explanation = svc.explain("TEST", ohlcv=sample_ohlcv_mixed)
        assert explanation.feature_inspection.ticker == "TEST"
        assert explanation.feature_inspection.normalized_row_count > 0


class TestGetModelInfo:
    def test_get_model_info(self, sample_ohlcv_mixed, tmp_path):
        svc = RegimeService(model_dir=tmp_path / "models")
        svc.fit("TEST", ohlcv=sample_ohlcv_mixed)
        info = svc.get_model_info("TEST")
        assert isinstance(info, HMMModelInfo)

    def test_get_model_info_no_model_raises(self, tmp_path):
        svc = RegimeService(model_dir=tmp_path / "models")
        with pytest.raises(RuntimeError, match="No fitted model"):
            svc.get_model_info("MISSING")


class TestGetRegimeHistory:
    def test_regime_history(self, sample_ohlcv_mixed, tmp_path):
        svc = RegimeService(model_dir=tmp_path / "models")
        series = svc.get_regime_history("TEST", ohlcv=sample_ohlcv_mixed)
        assert isinstance(series, RegimeTimeSeries)
        assert len(series.entries) > 0
