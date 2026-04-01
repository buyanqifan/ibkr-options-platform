"""SQLAlchemy base and engine setup."""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool
from config.settings import settings


class Base(DeclarativeBase):
    pass


_engine = None
_engine_path = None
_session_factory = None
_session_factory_path = None


def _current_db_path() -> str:
    return os.getenv("DB_PATH", settings.DB_PATH)


def _build_engine(db_path: str):
    return create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )


def _get_engine():
    global _engine, _engine_path
    db_path = _current_db_path()
    if _engine is None or _engine_path != db_path:
        if _engine is not None:
            try:
                _engine.dispose()
            except Exception:
                pass
        _engine = _build_engine(db_path)
        _engine_path = db_path
    return _engine


def _get_session_factory():
    global _session_factory, _session_factory_path
    db_path = _current_db_path()
    if _session_factory is None or _session_factory_path != db_path:
        _session_factory = sessionmaker(bind=_get_engine())
        _session_factory_path = db_path
    return _session_factory


class _EngineProxy:
    def __getattr__(self, name):
        return getattr(_get_engine(), name)

    def __repr__(self) -> str:
        return repr(_get_engine())


class _SessionLocalProxy:
    def __call__(self, *args, **kwargs):
        return _get_session_factory()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(_get_session_factory(), name)

    @property
    def kw(self):
        return _get_session_factory().kw


engine = _EngineProxy()
SessionLocal = _SessionLocalProxy()


def _ensure_sqlite_column(table_name: str, column_name: str, ddl: str) -> None:
    bind = _get_engine()
    if bind.dialect.name != "sqlite":
        return
    with bind.begin() as conn:
        tables = {row[0] for row in conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if table_name not in tables:
            return
        columns = {row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table_name})").fetchall()}
        if column_name in columns:
            return
        conn.exec_driver_sql(f"ALTER TABLE {table_name} ADD COLUMN {ddl}")


def init_db():
    """Create all tables."""
    import models.backtest  # noqa: F401
    import models.fundamentals  # noqa: F401
    import models.live_trading  # noqa: F401
    import models.market_data  # noqa: F401

    bind = _get_engine()
    Base.metadata.create_all(bind=bind)
    _ensure_sqlite_column("backtest_trades", "strategy_phase", "strategy_phase VARCHAR(10)")
