"""IBKR connection manager: handles connect, disconnect, reconnect with IB Gateway."""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from ib_insync import IB
from core.ibkr.event_bridge import AsyncEventBridge
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger("ibkr_connection")


class ConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class ConnectionStatus:
    state: ConnectionState = ConnectionState.DISCONNECTED
    message: str = ""
    last_connected: float = 0.0
    reconnect_attempts: int = 0
    server_version: int = 0
    account: str = ""


class IBKRConnectionManager:
    """Manages the lifecycle of an IB connection via AsyncEventBridge."""

    MAX_RECONNECT_ATTEMPTS = 5
    BASE_RETRY_DELAY = 5.0
    MAX_RETRY_DELAY = 60.0

    def __init__(self, bridge: AsyncEventBridge):
        self._bridge = bridge
        self._ib = IB()
        self._status = ConnectionStatus()
        self._subscriptions: list = []  # track active subscriptions for reconnect
        self._reconnecting = False

        # Wire disconnect event
        self._ib.disconnectedEvent += self._on_disconnected

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ib(self) -> IB:
        return self._ib

    @property
    def status(self) -> ConnectionStatus:
        return self._status

    @property
    def is_connected(self) -> bool:
        return self._ib.isConnected()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(
        self,
        host: str | None = None,
        port: int | None = None,
        client_id: int | None = None,
    ) -> bool:
        """Connect to IB Gateway (synchronous, called from Dash thread).
        
        Note: IBKR username and password are configured via environment variables
        (IBKR_USERNAME, IBKR_PASSWORD) and used by the IB Gateway Docker container.
        This method connects to the already-running IB Gateway instance.
        """
        host = host or settings.IBKR_HOST
        port = port or settings.IBKR_PORT
        client_id = client_id or settings.IBKR_CLIENT_ID

        self._status.state = ConnectionState.CONNECTING
        self._status.message = f"Connecting to {host}:{port}..."

        try:
            # Increased timeout for remote connections (from 15s to 30s)
            self._bridge.run_coroutine(
                self._async_connect(host, port, client_id),
                timeout=30.0,
            )
            self._status.state = ConnectionState.CONNECTED
            self._status.last_connected = time.time()
            self._status.reconnect_attempts = 0
            self._status.server_version = self._ib.client.serverVersion() if self._ib.client else 0

            # Get managed accounts with fallback methods
            accounts = self._ib.managedAccounts()
            if not accounts or len(accounts) == 0:
                # Try alternative method to get account
                try:
                    account_summary = self._ib.accountSummary()
                    if account_summary and len(account_summary) > 0:
                        self._status.account = account_summary[0].account
                    else:
                        self._status.account = "Unknown"
                except Exception as e:
                    logger.warning(f"Failed to get account info: {e}")
                    self._status.account = "Unknown"
            else:
                self._status.account = accounts[0]
            
            self._status.message = f"Connected (account: {self._status.account})"
            logger.info("Connected to IBKR at %s:%d, account=%s", host, port, self._status.account)
            return True

        except asyncio.TimeoutError as e:
            self._status.state = ConnectionState.ERROR
            self._status.message = f"Connection timed out after 30s. Please check network connectivity and IB Gateway status."
            logger.error(f"Connection to {host}:{port} timed out: {e}")
            return False
        except Exception as e:
            self._status.state = ConnectionState.ERROR
            self._status.message = f"Connection failed: {e}"
            logger.error("Failed to connect: %s", e)
            return False

    def disconnect(self):
        """Disconnect from IB Gateway."""
        self._reconnecting = False
        try:
            if self._ib.isConnected():
                self._bridge.run_coroutine(self._async_disconnect(), timeout=5.0)
        except Exception as e:
            logger.warning("Error during disconnect: %s", e)
        finally:
            self._status.state = ConnectionState.DISCONNECTED
            self._status.message = "Disconnected"
            logger.info("Disconnected from IBKR")

    def reconnect(self) -> bool:
        """Attempt to reconnect with exponential backoff."""
        if self._reconnecting:
            return False
        self._reconnecting = True
        self._status.state = ConnectionState.RECONNECTING

        for attempt in range(1, self.MAX_RECONNECT_ATTEMPTS + 1):
            if not self._reconnecting:
                break
            self._status.reconnect_attempts = attempt
            delay = min(self.BASE_RETRY_DELAY * (2 ** (attempt - 1)), self.MAX_RETRY_DELAY)
            self._status.message = f"Reconnecting (attempt {attempt}/{self.MAX_RECONNECT_ATTEMPTS})..."
            logger.info("Reconnect attempt %d/%d, delay=%.1fs", attempt, self.MAX_RECONNECT_ATTEMPTS, delay)
            time.sleep(delay)

            if self.connect():
                self._reconnecting = False
                return True

        self._reconnecting = False
        self._status.state = ConnectionState.ERROR
        self._status.message = "Reconnection failed after max attempts"
        logger.error("Reconnection failed after %d attempts", self.MAX_RECONNECT_ATTEMPTS)
        return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _async_connect(self, host: str, port: int, client_id: int):
        # Increased connection timeout for remote servers (from 10s to 20s)
        await self._ib.connectAsync(host, port, clientId=client_id, timeout=20)

    async def _async_disconnect(self):
        self._ib.disconnect()

    def _on_disconnected(self):
        """Called by ib_insync when connection drops."""
        if self._status.state == ConnectionState.DISCONNECTED:
            return  # intentional disconnect
        logger.warning("IBKR connection lost unexpectedly")
        self._status.state = ConnectionState.DISCONNECTED
        self._status.message = "Connection lost"
        # Auto-reconnect in background
        if not self._reconnecting:
            import threading
            threading.Thread(target=self.reconnect, daemon=True, name="ibkr-reconnect").start()
