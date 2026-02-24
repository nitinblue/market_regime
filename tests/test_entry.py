"""Tests for EntryService."""

import pytest

from market_analyzer.models.entry import EntryConfirmation, EntryTriggerType
from market_analyzer.service.entry import EntryService
from market_analyzer.service.technical import TechnicalService


class TestEntryService:
    def test_requires_technical_service(self):
        svc = EntryService()
        with pytest.raises(ValueError, match="requires a TechnicalService"):
            svc.confirm("TEST", EntryTriggerType.BREAKOUT_CONFIRMED)

    def test_requires_data(self):
        svc = EntryService(technical_service=TechnicalService())
        with pytest.raises(ValueError, match="Either provide ohlcv"):
            svc.confirm("TEST", EntryTriggerType.BREAKOUT_CONFIRMED)

    def test_breakout_confirmation(self, sample_ohlcv_trending):
        svc = EntryService(technical_service=TechnicalService())
        result = svc.confirm(
            "TEST",
            EntryTriggerType.BREAKOUT_CONFIRMED,
            ohlcv=sample_ohlcv_trending,
        )
        assert isinstance(result, EntryConfirmation)
        assert result.ticker == "TEST"
        assert result.trigger_type == EntryTriggerType.BREAKOUT_CONFIRMED
        assert 0.0 <= result.confidence <= 1.0
        assert result.conditions_met <= result.conditions_total
        assert result.suggested_entry_price is not None

    def test_pullback_confirmation(self, sample_ohlcv_choppy):
        svc = EntryService(technical_service=TechnicalService())
        result = svc.confirm(
            "TEST",
            EntryTriggerType.PULLBACK_TO_SUPPORT,
            ohlcv=sample_ohlcv_choppy,
        )
        assert isinstance(result, EntryConfirmation)
        assert result.trigger_type == EntryTriggerType.PULLBACK_TO_SUPPORT

    def test_momentum_confirmation(self, sample_ohlcv_trending):
        svc = EntryService(technical_service=TechnicalService())
        result = svc.confirm(
            "TEST",
            EntryTriggerType.MOMENTUM_CONTINUATION,
            ohlcv=sample_ohlcv_trending,
        )
        assert isinstance(result, EntryConfirmation)
        assert result.trigger_type == EntryTriggerType.MOMENTUM_CONTINUATION

    def test_mean_reversion_confirmation(self, sample_ohlcv_choppy):
        svc = EntryService(technical_service=TechnicalService())
        result = svc.confirm(
            "TEST",
            EntryTriggerType.MEAN_REVERSION_EXTREME,
            ohlcv=sample_ohlcv_choppy,
        )
        assert isinstance(result, EntryConfirmation)
        assert result.trigger_type == EntryTriggerType.MEAN_REVERSION_EXTREME

    def test_orb_confirmation(self, sample_ohlcv_trending):
        svc = EntryService(technical_service=TechnicalService())
        result = svc.confirm(
            "TEST",
            EntryTriggerType.ORB_BREAKOUT,
            ohlcv=sample_ohlcv_trending,
        )
        assert isinstance(result, EntryConfirmation)
        assert result.trigger_type == EntryTriggerType.ORB_BREAKOUT

    def test_all_trigger_types_handled(self, sample_ohlcv_trending):
        svc = EntryService(technical_service=TechnicalService())
        for trigger in EntryTriggerType:
            result = svc.confirm("TEST", trigger, ohlcv=sample_ohlcv_trending)
            assert isinstance(result, EntryConfirmation)
