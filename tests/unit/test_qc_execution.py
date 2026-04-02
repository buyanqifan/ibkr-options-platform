"""Focused tests for QuantConnect execution helpers."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[2]
QC_DIR = ROOT / "quantconnect"
if str(QC_DIR) not in sys.path:
    sys.path.insert(0, str(QC_DIR))

if "AlgorithmImports" not in sys.modules:
    algorithm_imports = types.ModuleType("AlgorithmImports")

    class _OptionRight:
        Put = "Put"
        Call = "Call"

    class _OrderStatus:
        Filled = "Filled"

    class _Resolution:
        Daily = "Daily"
        Minute = "Minute"

    class _SecurityType:
        Option = "Option"

    algorithm_imports.OptionRight = _OptionRight
    algorithm_imports.OrderStatus = _OrderStatus
    algorithm_imports.Resolution = _Resolution
    algorithm_imports.SecurityType = _SecurityType
    algorithm_imports.__all__ = [
        "OptionRight",
        "OrderStatus",
        "Resolution",
        "SecurityType",
    ]
    sys.modules["AlgorithmImports"] = algorithm_imports

import execution as qc_execution
from ml_integration import StrategySignal


def test_execute_signal_uses_rebalanced_delta_tolerance(monkeypatch):
    captured = {}
    algo = SimpleNamespace(
        equities={"NVDA": SimpleNamespace(Symbol="NVDA")},
        Securities={"NVDA": SimpleNamespace(Price=120.0)},
        Log=lambda *_args, **_kwargs: None,
        max_positions=20,
    )

    monkeypatch.setattr(qc_execution, "get_option_position_count", lambda _algo: 0)
    monkeypatch.setattr(qc_execution, "calculate_put_quantity", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(qc_execution, "safe_execute_option_order", lambda *_args, **_kwargs: SimpleNamespace(OrderId=7))
    monkeypatch.setattr(qc_execution, "_enqueue_open_order_metadata", lambda *_args, **_kwargs: None)

    def fake_find_option(*_args, **kwargs):
        captured["delta_tolerance"] = kwargs["delta_tolerance"]
        return {
            "option_symbol": "NVDA_PUT",
            "premium": 2.5,
            "strike": 110.0,
            "expiry": "2025-01-17",
            "delta": -0.30,
            "iv": 0.40,
        }

    signal = StrategySignal(
        action="SELL_PUT",
        symbol="NVDA",
        delta=0.30,
        dte_min=21,
        dte_max=60,
        num_contracts=1,
        expected_premium=2.5,
        expected_return=0.02,
        expected_risk=0.05,
        assignment_probability=0.20,
        confidence=0.8,
        reasoning="test",
    )

    qc_execution.execute_signal(algo, signal, fake_find_option)

    assert captured["delta_tolerance"] == pytest.approx(0.08)


def test_execute_close_records_pending_close_when_order_is_deferred(monkeypatch):
    algo = SimpleNamespace(
        pending_close_orders={},
        Time=datetime(2025, 1, 2),
    )
    position = {
        "symbol": "NVDA",
        "option_symbol": "NVDA_CALL",
        "expiry": datetime(2025, 1, 17),
        "strike": 120.0,
        "right": "C",
        "entry_price": 2.0,
        "quantity": -1,
    }

    monkeypatch.setattr(qc_execution, "safe_execute_option_order", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(qc_execution, "remove_position_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(qc_execution, "record_trade", lambda *_args, **_kwargs: None)

    qc_execution.execute_close(algo, qc_execution.make_signal("NVDA", "CLOSE"), existing_position=position)

    assert "NVDA_20250117_120_C" in algo.pending_close_orders


def test_execute_roll_records_pending_roll_when_close_ticket_is_not_filled(monkeypatch):
    algo = SimpleNamespace(
        pending_close_orders={},
        pending_roll_orders={},
        Time=datetime(2025, 1, 2),
        equities={"NVDA": SimpleNamespace(Symbol="NVDA")},
    )
    position = {
        "symbol": "NVDA",
        "option_symbol": "NVDA_PUT_OLD",
        "expiry": datetime(2025, 1, 17),
        "strike": 100.0,
        "right": "P",
        "entry_price": 2.5,
        "quantity": -1,
    }

    ticket = SimpleNamespace(Status="Submitted", OrderId=42)
    monkeypatch.setattr(qc_execution, "safe_execute_option_order", lambda *_args, **_kwargs: ticket)

    qc_execution.execute_roll(algo, qc_execution.make_signal("NVDA", "ROLL"), lambda *_args, **_kwargs: None, existing_position=position)

    assert "NVDA_20250117_100_P" in algo.pending_roll_orders
