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
        Equity = "Equity"

    class _DataNormalizationMode:
        Raw = "Raw"

    class _QCAlgorithm:
        pass

    algorithm_imports.OptionRight = _OptionRight
    algorithm_imports.OrderStatus = _OrderStatus
    algorithm_imports.Resolution = _Resolution
    algorithm_imports.SecurityType = _SecurityType
    algorithm_imports.DataNormalizationMode = _DataNormalizationMode
    algorithm_imports.QCAlgorithm = _QCAlgorithm
    algorithm_imports.__all__ = [
        "OptionRight",
        "OrderStatus",
        "Resolution",
        "SecurityType",
        "DataNormalizationMode",
        "QCAlgorithm",
    ]
    sys.modules["AlgorithmImports"] = algorithm_imports

import execution as qc_execution
import option_selector as qc_option_selector
from ml_integration import StrategySignal


class _SecurityDict(dict):
    def ContainsKey(self, key):
        return key in self


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


def test_safe_execute_option_order_enqueues_deferred_open_when_no_data():
    option_symbol = "NVDA 240621C01000000"
    security = SimpleNamespace(HasData=False, Price=0.0)
    signal = qc_execution.make_signal("NVDA", "SELL_CALL", confidence=0.9)
    selected = {
        "strike": 1000.0,
        "expiry": datetime(2024, 6, 21),
        "delta": 0.3,
        "iv": 0.4,
    }

    algo = SimpleNamespace(
        Securities=_SecurityDict({option_symbol: security}),
        Log=lambda *_args, **_kwargs: None,
        pending_open_orders={},
    )

    ticket = qc_execution.safe_execute_option_order(
        algo,
        option_symbol=option_symbol,
        quantity=-1,
        theoretical_price=1.25,
        deferred_context={
            "queue_key": "NVDA_20240621_1000_C_-1",
            "signal": signal,
            "selected": selected,
            "target_right": "Call",
        },
    )

    assert ticket is None
    assert "NVDA_20240621_1000_C_-1" in algo.pending_open_orders


def test_find_option_by_greeks_uses_fallback_selection_tiers(monkeypatch):
    class _ChainProvider:
        def __init__(self, contracts):
            self.contracts = contracts

        def GetOptionContractList(self, _equity_symbol, _time):
            return list(self.contracts)

    expiry = datetime(2024, 2, 16)
    contracts = [
        SimpleNamespace(ID=SimpleNamespace(OptionRight="Call", StrikePrice=110.0, Date=expiry)),
        SimpleNamespace(ID=SimpleNamespace(OptionRight="Call", StrikePrice=120.0, Date=expiry)),
    ]
    algo = SimpleNamespace(
        Time=datetime(2024, 1, 15),
        Securities={"NVDA": SimpleNamespace(Price=100.0)},
        OptionChainProvider=_ChainProvider(contracts),
        price_history={"NVDA": [{"close": 100.0}] * 25},
    )

    monkeypatch.setattr(qc_option_selector, "calculate_historical_vol", lambda *_args, **_kwargs: 0.25)
    monkeypatch.setattr(qc_option_selector, "bs_call_price", lambda *_args, **_kwargs: 1.25)

    selected = qc_option_selector.find_option_by_greeks(
        algo,
        symbol="NVDA",
        equity_symbol="NVDA",
        target_right="Call",
        target_delta=0.30,
        dte_min=21,
        dte_max=60,
        delta_tolerance=0.08,
        min_strike=118.0,
        selection_tiers=[
            {
                "label": "primary",
                "delta_tolerance": 0.08,
                "dte_min": 21,
                "dte_max": 60,
                "min_strike": 118.0,
            },
            {
                "label": "rescue_discount",
                "delta_tolerance": 0.15,
                "dte_min": 21,
                "dte_max": 60,
                "min_strike": 105.0,
            },
        ],
    )

    assert selected["strike"] == pytest.approx(110.0)
    assert selected["selection_tier"] == "rescue_discount"


def test_find_option_by_greeks_reaches_rescue_discount_tier(monkeypatch):
    class _ChainProvider:
        def __init__(self, contracts):
            self.contracts = contracts

        def GetOptionContractList(self, _equity_symbol, _time):
            return list(self.contracts)

    primary_expiry = datetime(2024, 2, 2)
    fallback_expiry = datetime(2024, 2, 14)
    contracts = [
        SimpleNamespace(ID=SimpleNamespace(OptionRight="Call", StrikePrice=146.0, Date=primary_expiry)),
        SimpleNamespace(ID=SimpleNamespace(OptionRight="Call", StrikePrice=129.0, Date=fallback_expiry)),
    ]
    algo = SimpleNamespace(
        Time=datetime(2024, 1, 15),
        Securities={"NVDA": SimpleNamespace(Price=120.0)},
        OptionChainProvider=_ChainProvider(contracts),
        price_history={"NVDA": [{"close": 120.0}] * 40},
    )

    monkeypatch.setattr(qc_option_selector, "calculate_historical_vol", lambda *_args, **_kwargs: 0.25)
    monkeypatch.setattr(
        qc_option_selector,
        "bs_call_price",
        lambda _underlying, strike, _time, _iv: 1.50 if strike == 129.0 else 0.01,
    )

    selected = qc_option_selector.find_option_by_greeks(
        algo,
        symbol="NVDA",
        equity_symbol="NVDA",
        target_right="Call",
        target_delta=0.35,
        dte_min=7,
        dte_max=21,
        delta_tolerance=0.08,
        min_strike=145.5,
        selection_tiers=[
            {"label": "primary", "delta_tolerance": 0.08, "dte_min": 7, "dte_max": 21, "min_strike": 145.5},
            {"label": "fallback_delta", "delta_tolerance": 0.12, "dte_min": 7, "dte_max": 21, "min_strike": 145.5},
            {"label": "fallback_dte", "delta_tolerance": 0.12, "dte_min": 14, "dte_max": 30, "min_strike": 145.5},
            {"label": "rescue_discount", "delta_tolerance": 0.15, "dte_min": 14, "dte_max": 30, "min_strike": 127.5},
        ],
    )

    assert selected["selection_tier"] == "rescue_discount"
    assert selected["strike"] == pytest.approx(129.0)
