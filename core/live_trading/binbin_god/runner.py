"""Entry point for the Binbin God live paper-trading worker."""

from __future__ import annotations

from app.services import set_services
from core.ibkr.connection import IBKRConnectionManager
from core.ibkr.data_client import IBKRDataClient
from core.ibkr.event_bridge import AsyncEventBridge
from core.ibkr.trading_client import IBKRTradingClient
from core.live_trading.binbin_god.repository import BinbinGodLiveRepository
from core.live_trading.binbin_god.service import BinbinGodLiveService
from core.live_trading.binbin_god.worker import BinbinGodLiveWorker
from core.market_data.cache import DataCache
from models.base import init_db
from utils.logger import setup_logger

logger = setup_logger("binbin_god_live_runner")


def main() -> None:
    """Start the standalone worker process."""
    init_db()

    bridge = AsyncEventBridge()
    bridge.start()
    cache = DataCache()
    conn_mgr = IBKRConnectionManager(bridge)
    data_client = IBKRDataClient(conn_mgr, bridge, cache)
    trading_client = IBKRTradingClient(conn_mgr, bridge)
    repo = BinbinGodLiveRepository()
    service = BinbinGodLiveService(repo)

    set_services(
        {
            "bridge": bridge,
            "cache": cache,
            "conn_mgr": conn_mgr,
            "data_client": data_client,
            "trading_client": trading_client,
            "binbin_god_live_repo": repo,
            "binbin_god_live_service": service,
        }
    )

    if not conn_mgr.is_connected:
        logger.info("Connecting live worker to IBKR %s:%s", conn_mgr.status.message or "", "")
        conn_mgr.connect()

    worker = BinbinGodLiveWorker(
        repository=repo,
        broker_client=trading_client,
        market_data_client=data_client,
    )
    worker.run_forever()


if __name__ == "__main__":
    main()
