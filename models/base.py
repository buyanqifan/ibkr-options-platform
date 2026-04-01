"""SQLAlchemy base and engine setup."""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool
from config.settings import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(
    f"sqlite:///{settings.DB_PATH}",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False,
)

SessionLocal = sessionmaker(bind=engine)


def init_db():
    """Create all tables."""
    import models.backtest  # noqa: F401
    import models.fundamentals  # noqa: F401
    import models.live_trading  # noqa: F401
    import models.market_data  # noqa: F401

    Base.metadata.create_all(bind=engine)
