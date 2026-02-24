"""ScreeningService: find setups across a universe of tickers."""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

from pydantic import BaseModel

from market_analyzer.features.screening import (
    screen_breakout,
    screen_income,
    screen_mean_reversion,
    screen_momentum,
)

if TYPE_CHECKING:
    from market_analyzer.data.service import DataService
    from market_analyzer.service.phase import PhaseService
    from market_analyzer.service.regime import RegimeService
    from market_analyzer.service.technical import TechnicalService

logger = logging.getLogger(__name__)


class ScreenCandidate(BaseModel):
    """A single screening hit."""

    ticker: str
    screen: str             # "breakout", "momentum", "mean_reversion", "income"
    score: float            # 0.0–1.0
    reason: str
    regime_id: int
    rsi: float
    atr_pct: float


class ScreeningResult(BaseModel):
    """Complete screening result for a universe."""

    as_of_date: date
    tickers_scanned: int
    candidates: list[ScreenCandidate]
    by_screen: dict[str, list[ScreenCandidate]]
    summary: str


# Available screens
AVAILABLE_SCREENS = ["breakout", "momentum", "mean_reversion", "income"]


class ScreeningService:
    """Scan a universe of tickers for trading setups.

    Runs configurable screens (breakout, momentum, mean_reversion, income)
    and returns candidates sorted by score.
    """

    def __init__(
        self,
        regime_service: RegimeService | None = None,
        technical_service: TechnicalService | None = None,
        phase_service: PhaseService | None = None,
        data_service: DataService | None = None,
    ) -> None:
        self.regime_service = regime_service
        self.technical_service = technical_service
        self.phase_service = phase_service
        self.data_service = data_service

    def scan(
        self,
        tickers: list[str],
        screens: list[str] | None = None,
    ) -> ScreeningResult:
        """Run screens across tickers, return candidates.

        Args:
            tickers: Universe to scan.
            screens: Which screens to run. None = all.
        """
        if self.regime_service is None or self.technical_service is None:
            raise ValueError("ScreeningService requires regime and technical services")

        today = date.today()
        active_screens = screens or AVAILABLE_SCREENS
        candidates: list[ScreenCandidate] = []

        screen_fns = {
            "breakout": screen_breakout,
            "momentum": screen_momentum,
            "mean_reversion": screen_mean_reversion,
            "income": screen_income,
        }

        for ticker in tickers:
            try:
                ohlcv = None
                if self.data_service is not None:
                    ohlcv = self.data_service.get_ohlcv(ticker)

                regime = self.regime_service.detect(ticker, ohlcv)
                technicals = self.technical_service.snapshot(ticker, ohlcv)

                for screen_name in active_screens:
                    fn = screen_fns.get(screen_name)
                    if fn is None:
                        continue
                    passes, score, reason = fn(regime, technicals)
                    if passes:
                        candidates.append(ScreenCandidate(
                            ticker=ticker,
                            screen=screen_name,
                            score=score,
                            reason=reason,
                            regime_id=int(regime.regime),
                            rsi=technicals.rsi.value,
                            atr_pct=technicals.atr_pct,
                        ))

            except Exception as exc:
                logger.warning("Screening failed for %s: %s", ticker, exc)

        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)

        # Group by screen
        by_screen: dict[str, list[ScreenCandidate]] = {}
        for c in candidates:
            by_screen.setdefault(c.screen, []).append(c)

        total = len(candidates)
        summary_parts = [f"Scanned {len(tickers)} tickers", f"{total} candidates found"]
        for s in active_screens:
            count = len(by_screen.get(s, []))
            if count:
                summary_parts.append(f"{s}: {count}")

        return ScreeningResult(
            as_of_date=today,
            tickers_scanned=len(tickers),
            candidates=candidates,
            by_screen=by_screen,
            summary=" | ".join(summary_parts),
        )
