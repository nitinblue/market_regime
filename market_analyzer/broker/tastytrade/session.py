"""TastyTrade session — auth from YAML + env vars.

Adapted from eTrading tastytrade_adapter.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path

from market_analyzer.broker.base import BrokerSession

logger = logging.getLogger(__name__)


class TastyTradeBrokerSession(BrokerSession):
    """TastyTrade session with YAML + env var credentials.

    Credential file layout (``tastytrade_broker.yaml``)::

        broker:
          live:
            client_secret: ${TT_LIVE_SECRET}
            refresh_token: ${TT_LIVE_TOKEN}
          paper:
            client_secret: ${TT_PAPER_SECRET}
            refresh_token: ${TT_PAPER_TOKEN}
          data:                          # optional — DXLink always live
            client_secret: ${TT_LIVE_SECRET}
            refresh_token: ${TT_LIVE_TOKEN}
    """

    def __init__(
        self,
        config_path: str = "tastytrade_broker.yaml",
        is_paper: bool = False,
        account_number: str | None = None,
    ) -> None:
        self._config_path = config_path
        self._is_paper = is_paper
        self._account_number = account_number

        # SDK objects (set on connect)
        self._session = None       # trading session
        self._data_session = None  # DXLink session (always live)
        self._account = None
        self._accounts: dict = {}
        self._connected = False

        # Credentials (loaded lazily)
        self._client_secret: str = ""
        self._refresh_token: str = ""
        self._data_client_secret: str = ""
        self._data_refresh_token: str = ""

    # -- BrokerSession ABC --

    def connect(self) -> bool:
        """Authenticate and establish session. Returns True on success."""
        try:
            from tastytrade import Account, Session
        except ImportError:
            logger.error("tastytrade SDK not installed — pip install tastytrade-sdk")
            return False

        try:
            self._load_credentials()

            logger.info("Connecting to TastyTrade | %s", "PAPER" if self._is_paper else "LIVE")

            self._session = Session(
                self._client_secret,
                self._refresh_token,
                is_test=self._is_paper,
            )

            # Account.get() is async in tastytrade SDK v12+
            loop = asyncio.new_event_loop()
            try:
                accounts = loop.run_until_complete(Account.get(self._session))
            finally:
                loop.close()
            self._accounts = {a.account_number: a for a in accounts}

            if self._account_number:
                if self._account_number not in self._accounts:
                    raise ValueError(f"Account {self._account_number} not found")
                self._account = self._accounts[self._account_number]
            else:
                self._account = next(iter(self._accounts.values()))

            # Create a FRESH Session for DXLink streaming.
            # The auth session above may have stale httpx connections from
            # the event loop used for Account.get(). A fresh Session avoids
            # "Event loop is closed" errors in subsequent asyncio.run() calls.
            if self._is_paper:
                self._data_session = Session(
                    self._data_client_secret,
                    self._data_refresh_token,
                    is_test=False,
                )
            else:
                self._data_session = Session(
                    self._data_client_secret,
                    self._data_refresh_token,
                    is_test=False,
                )

            self._connected = True
            logger.info("Authenticated with TastyTrade (account %s)", self._account.account_number)
            return True

        except Exception:
            logger.exception("TastyTrade authentication failed")
            return False

    def disconnect(self) -> None:
        self._session = None
        self._data_session = None
        self._account = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def broker_name(self) -> str:
        return "tastytrade"

    # -- TastyTrade-specific (not in ABC) --

    @property
    def sdk_session(self):
        """Trading session (paper or live)."""
        if not self._session:
            raise RuntimeError("Not connected — call connect() first")
        return self._session

    @property
    def data_session(self):
        """DXLink session (always live, for market data streaming)."""
        if not self._data_session:
            raise RuntimeError("Not connected — call connect() first")
        return self._data_session

    @property
    def account(self):
        if not self._account:
            raise RuntimeError("Not connected — call connect() first")
        return self._account

    # -- Credential loading --

    def _load_credentials(self) -> None:
        """Load credentials from YAML file with env var resolution."""
        import yaml

        cred_path = self._find_config_file()
        if not cred_path:
            raise FileNotFoundError(
                f"Credentials file '{self._config_path}' not found. "
                "See tastytrade_broker.yaml.template for format."
            )

        with open(cred_path) as f:
            creds = yaml.safe_load(f)

        mode = "paper" if self._is_paper else "live"
        mode_creds = creds["broker"][mode]

        self._client_secret = _resolve_env(mode_creds["client_secret"])
        self._refresh_token = _resolve_env(mode_creds["refresh_token"])

        # DXLink data credentials — falls back to live
        data_section = creds["broker"].get("data") or creds["broker"]["live"]
        self._data_client_secret = _resolve_env(data_section["client_secret"])
        self._data_refresh_token = _resolve_env(data_section["refresh_token"])

    def _find_config_file(self) -> Path | None:
        """Search common locations for the credential YAML."""
        candidates = [
            Path(self._config_path),
            Path.home() / ".market_analyzer" / self._config_path,
            Path(__file__).parent / self._config_path,
            Path(__file__).parent.parent / self._config_path,
            Path(__file__).parent.parent.parent / self._config_path,
        ]
        for p in candidates:
            if p.exists():
                return p
        return None


def _resolve_env(value: str) -> str:
    """Resolve ``${ENV_VAR}`` or ``$ENV_VAR`` patterns in a credential value."""
    if not value:
        return value
    match = re.match(r"\$\{?([A-Z_][A-Z0-9_]*)\}?", value)
    if match:
        env_var = match.group(1)
        resolved = os.getenv(env_var)
        if not resolved:
            raise ValueError(f"Environment variable {env_var} not set")
        return resolved
    return value
