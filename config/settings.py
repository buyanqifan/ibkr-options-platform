"""Global configuration loaded from environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    # IBKR Connection
    IBKR_HOST: str = os.getenv("IBKR_HOST", "ibgateway")
    IBKR_PORT: int = int(os.getenv("IBKR_PORT", "8888"))
    IBKR_CLIENT_ID: int = int(os.getenv("IBKR_CLIENT_ID", "1"))
    IBKR_TRADING_MODE: str = os.getenv("IBKR_TRADING_MODE", "paper")
    
    # IBKR Credentials (used by Docker Compose for IB Gateway)
    IBKR_USERNAME: str = os.getenv("IBKR_USERNAME", "")
    IBKR_PASSWORD: str = os.getenv("IBKR_PASSWORD", "")

    # App
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8050"))
    APP_DEBUG: bool = os.getenv("APP_DEBUG", "false").lower() == "true"

    # Database
    DB_PATH: str = os.getenv("DB_PATH", str(BASE_DIR / "data" / "trading.db"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Cache TTL (seconds)
    REALTIME_CACHE_TTL: int = 2
    HISTORICAL_CACHE_TTL: int = 300
    FUNDAMENTALS_CACHE_TTL: int = 86400

    # Rate Limiter
    API_REQUESTS_PER_SECOND: int = 45


settings = Settings()
