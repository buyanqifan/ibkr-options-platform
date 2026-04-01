"""Tests for Binbin God live-trading state, repository, and page wiring."""

from __future__ import annotations

import importlib
import sys
from datetime import datetime

import dash
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


def _load_live_page(monkeypatch):
    monkeypatch.setattr(dash, "register_page", lambda *args, **kwargs: None)
    sys.modules.pop("app.pages.binbin_god_live", None)
    return importlib.import_module("app.pages.binbin_god_live")


def _load_layout_module(monkeypatch):
    fake_registry = {
        "pages.dashboard": {"layout": lambda: "dashboard"},
        "pages.market_data": {"layout": lambda: "market-data"},
        "pages.screener": {"layout": lambda: "screener"},
        "pages.options_chain": {"layout": lambda: "options-chain"},
        "pages.backtester": {"layout": lambda: "backtester"},
        "pages.backtest_history": {"layout": lambda: "backtest-history"},
        "pages.binbin_god": {"layout": lambda: "binbin-god"},
        "pages.binbin_god_live": {"layout": lambda: "binbin-god-live"},
        "pages.settings": {"layout": lambda: "settings"},
    }
    monkeypatch.setattr(dash, "page_registry", fake_registry, raising=False)
    sys.modules.pop("app.layout", None)
    return importlib.import_module("app.layout")


def _collect_ids(node):
    ids = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current is None:
            continue
        if isinstance(current, (list, tuple)):
            stack.extend(current)
            continue
        if isinstance(current, (str, int, float, bool)):
            continue
        current_id = getattr(current, "id", None)
        if isinstance(current_id, str):
            ids.add(current_id)
        children = getattr(current, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None:
            stack.append(children)
    return ids


@pytest.fixture()
def live_db():
    from models.base import Base

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    return Session


def test_live_repository_stores_config_state_and_commands(live_db):
    from core.live_trading.binbin_god.models import CommandStatus, ControlCommandType, StrategyStatus
    from core.live_trading.binbin_god.repository import BinbinGodLiveRepository

    repo = BinbinGodLiveRepository(session_factory=live_db)
    saved = repo.save_config(
        {
            "stock_pool": ["MSFT", "AAPL"],
            "poll_interval_seconds": 60,
            "allow_new_entries": True,
        }
    )

    assert saved.params["stock_pool"] == ["MSFT", "AAPL"]
    assert saved.params["poll_interval_seconds"] == 60
    assert saved.status == StrategyStatus.STOPPED.value

    command = repo.enqueue_command(ControlCommandType.START)
    state = repo.upsert_state(
        status=StrategyStatus.STOPPED,
        heartbeat_at=datetime(2026, 4, 1, 9, 30),
        account_id="DU123456",
        detail={"phase_counts": {"cash": 2}},
    )

    pending = repo.list_pending_commands()
    assert [item.command_type for item in pending] == [ControlCommandType.START.value]
    assert state.account_id == "DU123456"

    repo.mark_command_processed(command.id, CommandStatus.COMPLETED)
    assert repo.list_pending_commands() == []


def test_state_reconstruction_detects_short_put_and_repair_call():
    from core.live_trading.binbin_god.reconstruction import reconstruct_live_state

    account_summary = {"NetLiquidation": 100000.0}
    positions = [
        {
            "symbol": "MSFT",
            "secType": "OPT",
            "expiry": "20260515",
            "strike": 380.0,
            "right": "P",
            "position": -1,
            "avgCost": 2.5,
            "marketValue": -210.0,
            "unrealizedPNL": 40.0,
        },
        {
            "symbol": "AAPL",
            "secType": "STK",
            "expiry": "",
            "strike": 0.0,
            "right": "",
            "position": 100,
            "avgCost": 22000.0,
            "marketValue": 20500.0,
            "unrealizedPNL": -1500.0,
        },
        {
            "symbol": "AAPL",
            "secType": "OPT",
            "expiry": "20260515",
            "strike": 210.0,
            "right": "C",
            "position": -1,
            "avgCost": 1.8,
            "marketValue": -150.0,
            "unrealizedPNL": 60.0,
        },
    ]

    snapshot = reconstruct_live_state(
        account_summary=account_summary,
        positions=positions,
        open_orders=[],
        fills=[],
        config={"stock_pool": ["MSFT", "AAPL"], "stock_inventory_block_threshold": 0.85},
    )

    assert snapshot["status"] == "ready"
    assert snapshot["symbols"]["MSFT"]["phase"] == "short_put"
    assert snapshot["symbols"]["AAPL"]["phase"] == "repair_call"
    assert snapshot["symbols"]["AAPL"]["blocked_reasons"] == []


def test_worker_start_guardrail_requires_paper_mode(live_db, monkeypatch):
    from core.live_trading.binbin_god.models import RecoveryStatus, StrategyStatus
    from core.live_trading.binbin_god.repository import BinbinGodLiveRepository
    from core.live_trading.binbin_god.worker import BinbinGodLiveWorker

    repo = BinbinGodLiveRepository(session_factory=live_db)
    repo.save_config({"stock_pool": ["MSFT"], "poll_interval_seconds": 60})

    monkeypatch.setattr("config.settings.settings.IBKR_TRADING_MODE", "live")
    worker = BinbinGodLiveWorker(repository=repo, broker_client=None, market_data_client=None)
    result = worker.apply_startup_recovery()

    state = repo.get_state()
    assert result["recovery_status"] == RecoveryStatus.BLOCKED.value
    assert state.status == StrategyStatus.STOPPED.value
    assert "paper" in state.last_error.lower()


def test_worker_run_cycle_submits_single_open_put_and_avoids_duplicate(live_db, monkeypatch):
    from core.live_trading.binbin_god.models import ControlCommandType
    from core.live_trading.binbin_god.repository import BinbinGodLiveRepository
    from core.live_trading.binbin_god.worker import BinbinGodLiveWorker

    class FakeBroker:
        def __init__(self):
            self.submissions = []

        def get_account_summary(self):
            return {"Account": "DU123", "NetLiquidation": 150000.0}

        def get_positions(self):
            return []

        def get_open_orders(self):
            return []

        def get_recent_fills(self):
            return []

        def submit_option_limit_order(self, request):
            self.submissions.append(request)
            return {"orderId": len(self.submissions), "permId": 1000 + len(self.submissions), "status": "Submitted"}

    class FakeMarketData:
        def get_option_chain_params(self, symbol):
            return [{"expirations": ["20260515"], "strikes": [380.0, 385.0, 390.0]}]

        def get_option_chain(self, symbol, expiry, strikes=None, right=""):
            return [
                {"symbol": symbol, "expiry": expiry, "strike": 380.0, "right": "P", "bid": 2.0, "ask": 2.2, "last": 2.1, "delta": -0.28},
                {"symbol": symbol, "expiry": expiry, "strike": 385.0, "right": "P", "bid": 2.6, "ask": 2.8, "last": 2.7, "delta": -0.32},
            ]

    repo = BinbinGodLiveRepository(session_factory=live_db)
    repo.save_config(
        {
            "stock_pool": ["MSFT"],
            "stock_pool_text": "MSFT",
            "put_delta": 0.30,
            "dte_min": 21,
            "dte_max": 60,
            "poll_interval_seconds": 60,
            "allow_new_entries": True,
        }
    )
    repo.enqueue_command(ControlCommandType.START)

    monkeypatch.setattr("config.settings.settings.IBKR_TRADING_MODE", "paper")
    worker = BinbinGodLiveWorker(repository=repo, broker_client=FakeBroker(), market_data_client=FakeMarketData())
    worker.apply_startup_recovery()

    first = worker.run_cycle()
    second = worker.run_cycle()

    audits = repo.list_order_audits()
    assert first["status"] == "running"
    assert len(worker.broker_client.submissions) == 1
    assert len(audits) == 1
    assert audits[0].action == "open_put"
    assert second["status"] == "running"
    assert len(worker.broker_client.submissions) == 1


def test_live_page_layout_exposes_control_and_recovery_sections(monkeypatch):
    page = _load_live_page(monkeypatch)

    layout = page.layout() if callable(page.layout) else page.layout
    ids = _collect_ids(layout)

    expected = {
        "bbg-live-status",
        "bbg-live-config-store",
        "bbg-live-start-btn",
        "bbg-live-pause-btn",
        "bbg-live-emergency-stop-btn",
        "bbg-live-cancel-entry-orders-btn",
        "bbg-live-save-config-btn",
        "bbg-live-load-config-btn",
        "bbg-live-reset-config-btn",
        "bbg-live-apply-config-btn",
        "bbg-live-orders-table",
        "bbg-live-fills-table",
        "bbg-live-positions-table",
        "bbg-live-events",
        "bbg-live-recovery-summary",
    }

    assert expected.issubset(ids)


def test_live_page_form_helpers_round_trip(monkeypatch):
    page = _load_live_page(monkeypatch)

    payload = page._build_live_form_payload(*[page.LIVE_FORM_DEFAULTS[field["default"]] for field in page.LIVE_FORM_FIELDS])
    values = page._form_values_from_live_params(payload)

    assert payload["stock_pool"] == ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
    assert payload["poll_interval_seconds"] == 60
    assert values[-4:] == (60, True, 2, True)


def test_layout_routes_binbin_god_live(monkeypatch):
    layout_module = _load_layout_module(monkeypatch)

    page = layout_module.display_page("/binbin-god-live")

    assert page == "binbin-god-live"
