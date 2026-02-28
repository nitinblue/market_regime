"""TastyTrade market data — DXLink streaming for quotes and Greeks.

Adapted from eTrading tastytrade_adapter.py.
Uses DXLinkStreamer exactly as implemented in cotrader.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd

from market_analyzer.broker.base import MarketDataProvider
from market_analyzer.models.quotes import OptionQuote

if TYPE_CHECKING:
    from market_analyzer.broker.tastytrade.session import TastyTradeBrokerSession
    from market_analyzer.models.opportunity import LegSpec

logger = logging.getLogger(__name__)

# Thread pool for running async code from sync context
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def _today_market_open() -> datetime:
    """Return today at 9:30 ET as a timezone-aware datetime (UTC)."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

    et = ZoneInfo("US/Eastern")
    now_et = datetime.now(et)
    open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    return open_et.astimezone(timezone.utc)


class TastyTradeMarketData(MarketDataProvider):
    """DXLink streaming quotes and Greeks — same pattern as cotrader adapter."""

    def __init__(self, session: TastyTradeBrokerSession) -> None:
        self._session = session
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def provider_name(self) -> str:
        return "tastytrade"

    # -- MarketDataProvider ABC --

    def get_option_chain(
        self, ticker: str, expiration: date | None = None,
    ) -> list[OptionQuote]:
        """Fetch full option chain via SDK + DXLink quotes/Greeks."""
        from tastytrade.instruments import Option

        chain = Option.get_option_chain(self._session.sdk_session, ticker)

        # chain is dict[date, list[Option]]
        all_options: list = []
        for exp_date, options in chain.items():
            if expiration and exp_date != expiration:
                continue
            all_options.extend(options)

        if not all_options:
            return []

        # Build streamer symbols for all options
        streamer_symbols = [o.streamer_symbol for o in all_options if o.streamer_symbol]

        # Fetch quotes and Greeks via DXLink
        quotes_map = self._run_async(self._fetch_quotes_via_dxlink(streamer_symbols))
        greeks_map = self._run_async(self._fetch_greeks_via_dxlink(streamer_symbols))

        result: list[OptionQuote] = []
        for opt in all_options:
            sym = opt.streamer_symbol
            if not sym:
                continue

            q = quotes_map.get(sym, {})
            g = greeks_map.get(sym, {})

            bid = q.get("bid", 0.0)
            ask = q.get("ask", 0.0)

            result.append(OptionQuote(
                ticker=ticker,
                expiration=opt.expiration_date,
                strike=float(opt.strike_price),
                option_type="call" if opt.option_type == "C" else "put",
                bid=bid,
                ask=ask,
                mid=round((bid + ask) / 2, 4),
                volume=q.get("volume", 0),
                open_interest=q.get("open_interest", 0),
                implied_volatility=g.get("iv"),
                delta=g.get("delta"),
                gamma=g.get("gamma"),
                theta=g.get("theta"),
                vega=g.get("vega"),
            ))

        return result

    def get_quotes(self, legs: list[LegSpec]) -> list[OptionQuote]:
        """Fetch quotes for specific legs via DXLink streaming."""
        symbols = [self._leg_to_streamer_symbol(leg) for leg in legs]
        symbols = [s for s in symbols if s]

        if not symbols:
            return []

        quotes_map = self._run_async(self._fetch_quotes_via_dxlink(symbols))
        greeks_map = self._run_async(self._fetch_greeks_via_dxlink(symbols))

        result: list[OptionQuote] = []
        for leg, sym in zip(legs, symbols):
            if not sym:
                continue
            q = quotes_map.get(sym, {})
            g = greeks_map.get(sym, {})

            bid = q.get("bid", 0.0)
            ask = q.get("ask", 0.0)

            result.append(OptionQuote(
                ticker=leg.strike_label.split()[1] if " " in leg.strike_label else "",
                expiration=leg.expiration,
                strike=leg.strike,
                option_type=leg.option_type,
                bid=bid,
                ask=ask,
                mid=round((bid + ask) / 2, 4),
                implied_volatility=g.get("iv"),
                delta=g.get("delta"),
                gamma=g.get("gamma"),
                theta=g.get("theta"),
                vega=g.get("vega"),
            ))

        return result

    def get_greeks(self, legs: list[LegSpec]) -> dict[str, dict]:
        """Fetch Greeks for specific legs via DXLink."""
        symbols = [self._leg_to_streamer_symbol(leg) for leg in legs]
        symbol_keys = [f"{leg.strike:.0f}{leg.option_type[0].upper()}" for leg in legs]
        symbols_clean = [s for s in symbols if s]

        greeks_map = self._run_async(self._fetch_greeks_via_dxlink(symbols_clean))

        result: dict[str, dict] = {}
        for key, sym in zip(symbol_keys, symbols):
            if sym and sym in greeks_map:
                result[key] = greeks_map[sym]

        return result

    # -- Intraday candles + underlying price --

    def get_intraday_candles(
        self, ticker: str, interval: str = "5m",
    ) -> pd.DataFrame:
        """Today's intraday OHLCV candles via DXLink Candle subscription."""
        try:
            return self._run_async(self._fetch_intraday_candles(ticker, interval))
        except Exception as e:
            logger.warning("Intraday candle fetch failed for %s: %s", ticker, e)
            return pd.DataFrame()

    def get_underlying_price(self, ticker: str) -> float | None:
        """Real-time underlying mid price via DXLink equity Quote."""
        try:
            return self._run_async(self._fetch_underlying_price(ticker))
        except Exception as e:
            logger.warning("Underlying price fetch failed for %s: %s", ticker, e)
            return None

    async def _fetch_intraday_candles(
        self, ticker: str, interval: str,
    ) -> pd.DataFrame:
        """Subscribe to DXLink Candle events for today's bars."""
        from tastytrade.dxfeed import Candle
        from tastytrade.streamer import DXLinkStreamer

        rows: dict[int, dict] = {}  # event.time → OHLCV row (dedup updates)

        try:
            async with DXLinkStreamer(self._session.data_session) as streamer:
                start_time = _today_market_open()
                await streamer.subscribe_candle([ticker], interval, start_time)

                end = asyncio.get_event_loop().time() + 10.0
                while asyncio.get_event_loop().time() < end:
                    try:
                        event = await asyncio.wait_for(
                            streamer.get_event(Candle), timeout=2.0,
                        )
                        ts = int(event.time) if event.time else 0
                        if ts > 0:
                            rows[ts] = {
                                "time": ts,
                                "Open": float(event.open),
                                "High": float(event.high),
                                "Low": float(event.low),
                                "Close": float(event.close),
                                "Volume": float(event.volume or 0),
                            }
                        if getattr(event, "snapshot_end", False):
                            break
                    except asyncio.TimeoutError:
                        break

        except Exception as e:
            logger.warning("DXLink candle fetch error: %s", e)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(list(rows.values()))
        df.index = pd.to_datetime(df.pop("time"), unit="ms", utc=True)
        df.index = df.index.tz_convert("US/Eastern").tz_localize(None)
        return df.sort_index()

    async def _fetch_underlying_price(self, ticker: str) -> float | None:
        """Subscribe to DXLink equity Quote for real-time bid/ask mid."""
        from tastytrade.dxfeed import Quote as DXQuote
        from tastytrade.streamer import DXLinkStreamer

        try:
            async with DXLinkStreamer(self._session.data_session) as streamer:
                await streamer.subscribe(DXQuote, [ticker])
                try:
                    event = await asyncio.wait_for(
                        streamer.get_event(DXQuote), timeout=5.0,
                    )
                    bid = float(event.bid_price or 0)
                    ask = float(event.ask_price or 0)
                    if bid > 0 and ask > 0:
                        return round((bid + ask) / 2, 2)
                    return float(event.last_price or 0) or None
                except asyncio.TimeoutError:
                    return None
        except Exception as e:
            logger.warning("DXLink underlying quote error: %s", e)
            return None

    # -- DXLink streaming (same pattern as cotrader) --

    async def _fetch_quotes_via_dxlink(self, streamer_symbols: list[str]) -> dict[str, dict]:
        """Fetch bid/ask quotes via DXLink streaming."""
        from tastytrade.dxfeed import Quote as DXQuote
        from tastytrade.streamer import DXLinkStreamer

        quotes: dict[str, dict] = {}
        if not streamer_symbols:
            return quotes

        try:
            async with DXLinkStreamer(self._session.data_session) as streamer:
                await streamer.subscribe(DXQuote, streamer_symbols)

                timeout = 5.0
                end_time = asyncio.get_event_loop().time() + timeout

                while asyncio.get_event_loop().time() < end_time:
                    try:
                        event = await asyncio.wait_for(
                            streamer.get_event(DXQuote), timeout=1.0,
                        )
                        if event:
                            quotes[event.event_symbol] = {
                                "bid": float(event.bid_price or 0),
                                "ask": float(event.ask_price or 0),
                            }
                            if len(quotes) >= len(streamer_symbols):
                                break
                    except asyncio.TimeoutError:
                        continue

        except Exception as e:
            logger.warning("DXLink quote fetch error: %s", e)

        return quotes

    async def _fetch_greeks_via_dxlink(self, streamer_symbols: list[str]) -> dict[str, dict]:
        """Fetch Greeks via DXLink streaming — same as cotrader adapter."""
        from tastytrade.dxfeed import Greeks as DXGreeks
        from tastytrade.streamer import DXLinkStreamer

        greeks: dict[str, dict] = {}
        if not streamer_symbols:
            return greeks

        symbols_needed = set(streamer_symbols)
        timeout_seconds = 15

        try:
            async with DXLinkStreamer(self._session.data_session) as streamer:
                await streamer.subscribe(DXGreeks, streamer_symbols)
                logger.info("Subscribed to Greeks for %d symbols", len(streamer_symbols))

                start_time = asyncio.get_event_loop().time()

                while symbols_needed and (asyncio.get_event_loop().time() - start_time) < timeout_seconds:
                    try:
                        event = await asyncio.wait_for(
                            streamer.get_event(DXGreeks), timeout=2.0,
                        )
                        sym = event.event_symbol
                        if sym in symbols_needed:
                            greeks[sym] = {
                                "delta": float(event.delta or 0),
                                "gamma": float(event.gamma or 0),
                                "theta": float(event.theta or 0),
                                "vega": float(event.vega or 0),
                                "iv": float(event.volatility or 0) if hasattr(event, "volatility") else None,
                            }
                            symbols_needed.remove(sym)
                            logger.info("Got Greeks for %s: delta=%.4f", sym, event.delta or 0)

                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.warning("Error getting Greeks event: %s", e)
                        continue

                if symbols_needed:
                    logger.warning("Timeout: missing Greeks for %d symbols", len(symbols_needed))

        except Exception as e:
            logger.error("DXLink Greeks streaming error: %s", e)

        return greeks

    def _run_async(self, coro):
        """Run async coroutine from sync context.

        Uses a persistent event loop so the tastytrade SDK's httpx AsyncClient
        doesn't accumulate stale connections across asyncio.run() calls.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an async context (e.g. FastAPI) — use thread pool
            future = _thread_pool.submit(asyncio.run, coro)
            return future.result(timeout=30)
        else:
            # Standalone: reuse persistent loop to keep httpx connections alive
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop.run_until_complete(coro)

    def _leg_to_streamer_symbol(self, leg: LegSpec) -> str | None:
        """Convert LegSpec to DXLink streamer symbol.

        Format: ``.{TICKER}{YYMMDD}{C|P}{STRIKE}``
        Example: ``.SPY260320P580``
        """
        try:
            exp_str = leg.expiration.strftime("%y%m%d")
            opt_char = "C" if leg.option_type == "call" else "P"
            strike_int = int(leg.strike)

            # Derive ticker from strike_label (e.g. "580 put" doesn't have ticker)
            # Use the trade's ticker context — caller should set it on legs
            ticker = leg.strike_label.split()[1] if " " in (leg.strike_label or "") else ""
            if not ticker or ticker.lower() in ("put", "call"):
                # Fallback: can't determine ticker from leg alone
                return None

            return f".{ticker}{exp_str}{opt_char}{strike_int}"
        except Exception:
            return None

    def leg_to_streamer_symbol_with_ticker(
        self, ticker: str, leg: LegSpec,
    ) -> str:
        """Convert LegSpec + known ticker to DXLink streamer symbol."""
        exp_str = leg.expiration.strftime("%y%m%d")
        opt_char = "C" if leg.option_type == "call" else "P"
        strike_int = int(leg.strike)
        return f".{ticker}{exp_str}{opt_char}{strike_int}"
