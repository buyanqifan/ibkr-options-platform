"""Dash application entry point. Initializes services, wires up global callbacks."""

import os
import asyncio
import dash
import dash_bootstrap_components as dbc
from config.settings import settings
from core.ibkr.event_bridge import AsyncEventBridge
from core.ibkr.connection import IBKRConnectionManager
from core.ibkr.data_client import IBKRDataClient
from core.market_data.cache import DataCache
from models.base import init_db
from utils.logger import setup_logger
from app.services import get_services, set_services

logger = setup_logger("app")

# Prevent Dash from conflicting with IBKR's event loop
# Set environment variable to disable Dash's internal async handling
os.environ['DASH_ASYNC'] = 'false'


def _init_services() -> dict:
    """Create and wire all service singletons."""
    bridge = AsyncEventBridge()
    bridge.start()

    cache = DataCache()
    conn_mgr = IBKRConnectionManager(bridge)
    data_client = IBKRDataClient(conn_mgr, bridge, cache)

    # Lazy imports to avoid circular deps during page registration
    from core.screener.screener import StockScreener
    from core.backtesting.engine import BacktestEngine
    from core.ml.inference.predictor import VolatilityPredictor

    screener = StockScreener(data_client)
    
    # Initialize ML volatility predictor (loads existing model if available)
    vol_predictor = VolatilityPredictor()
    if vol_predictor.is_ready():
        logger.info("ML volatility predictor loaded successfully")
    else:
        logger.info("ML volatility predictor not available (model not trained)")
    
    backtest_engine = BacktestEngine(data_client, vol_predictor)

    return {
        "bridge": bridge,
        "cache": cache,
        "conn_mgr": conn_mgr,
        "data_client": data_client,
        "screener": screener,
        "backtest_engine": backtest_engine,
    }


# ---------------------------------------------------------------------------
# Create Dash app
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder=os.path.join(_THIS_DIR, "pages"),
    external_stylesheets=[
        dbc.themes.DARKLY,
        dbc.icons.BOOTSTRAP,
    ],
    suppress_callback_exceptions=True,
    title="IBKR Options Platform",
)

# Log registered pages for debugging
import logging
_log = logging.getLogger("app.pages")
_log.info("Registered pages: %s", list(dash.page_registry.keys()))

server = app.server  # Flask server for gunicorn

# Apply layout
from app.layout import create_layout  # noqa: E402
app.layout = create_layout()


# ---------------------------------------------------------------------------
# Global callback: update connection badge in navbar
# ---------------------------------------------------------------------------
@app.callback(
    dash.Output("connection-badge", "children"),
    dash.Input("global-interval", "n_intervals"),
)
def update_navbar_badge(n):
    from app.components.connection_status import connection_badge
    services = get_services()
    if not services:
        return connection_badge("disconnected")
    status = services["conn_mgr"].status
    return connection_badge(status.state.value, status.message)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("Initializing database...")
    init_db()

    logger.info("Initializing services...")
    services = _init_services()
    set_services(services)

    logger.info("Starting Dash app on %s:%d", settings.APP_HOST, settings.APP_PORT)
    app.run(
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        debug=settings.APP_DEBUG,
    )


if __name__ == "__main__":
    main()
