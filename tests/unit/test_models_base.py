"""Tests for dynamic database binding and lightweight schema migration."""

from __future__ import annotations

import os
import sqlite3
import tempfile


def test_session_local_tracks_db_path_changes(monkeypatch):
    from models import base

    first_fd, first_path = tempfile.mkstemp(suffix=".db")
    second_fd, second_path = tempfile.mkstemp(suffix=".db")
    os.close(first_fd)
    os.close(second_fd)
    try:
        monkeypatch.setenv("DB_PATH", first_path)
        first_session = base.SessionLocal()
        try:
            assert first_session.bind.url.database == first_path
        finally:
            first_session.close()

        monkeypatch.setenv("DB_PATH", second_path)
        second_session = base.SessionLocal()
        try:
            assert second_session.bind.url.database == second_path
        finally:
            second_session.close()
    finally:
        if os.path.exists(first_path):
            os.unlink(first_path)
        if os.path.exists(second_path):
            os.unlink(second_path)


def test_init_db_migrates_backtest_trade_strategy_phase(monkeypatch):
    from models import base

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                """
                CREATE TABLE backtest_trades (
                    id INTEGER NOT NULL PRIMARY KEY,
                    backtest_id INTEGER NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    strategy_name VARCHAR(50),
                    trade_type VARCHAR(20),
                    entry_date VARCHAR(10) NOT NULL,
                    exit_date VARCHAR(10),
                    expiry VARCHAR(10),
                    strike FLOAT,
                    entry_price FLOAT,
                    exit_price FLOAT,
                    quantity INTEGER,
                    pnl FLOAT,
                    pnl_pct FLOAT,
                    exit_reason VARCHAR(50),
                    underlying_entry FLOAT,
                    underlying_exit FLOAT,
                    iv_at_entry FLOAT,
                    delta_at_entry FLOAT,
                    capital_at_entry FLOAT,
                    capital_at_exit FLOAT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

        monkeypatch.setenv("DB_PATH", db_path)
        base.init_db()

        conn = sqlite3.connect(db_path)
        try:
            columns = {
                row[1]: row
                for row in conn.execute("PRAGMA table_info(backtest_trades)").fetchall()
            }
        finally:
            conn.close()

        assert "strategy_phase" in columns
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)
