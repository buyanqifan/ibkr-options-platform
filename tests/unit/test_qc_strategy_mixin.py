"""Focused tests for QC rebalance control flow."""

from pathlib import Path
from types import SimpleNamespace
import sys
import types


ROOT = Path(__file__).resolve().parents[2]
QC_DIR = ROOT / "quantconnect"
if str(QC_DIR) not in sys.path:
    sys.path.insert(0, str(QC_DIR))

if "AlgorithmImports" not in sys.modules:
    algorithm_imports = types.ModuleType("AlgorithmImports")

    class _OptionRight:
        Put = "Put"
        Call = "Call"

    class _SecurityType:
        Option = "Option"
        Equity = "Equity"

    class _OrderStatus:
        Filled = "Filled"

    class _Resolution:
        Daily = "Daily"
        Minute = "Minute"

    class _DataNormalizationMode:
        Raw = "Raw"

    class _QCAlgorithm:
        pass

    algorithm_imports.OptionRight = _OptionRight
    algorithm_imports.SecurityType = _SecurityType
    algorithm_imports.OrderStatus = _OrderStatus
    algorithm_imports.Resolution = _Resolution
    algorithm_imports.DataNormalizationMode = _DataNormalizationMode
    algorithm_imports.QCAlgorithm = _QCAlgorithm
    algorithm_imports.__all__ = [
        "OptionRight",
        "SecurityType",
        "OrderStatus",
        "Resolution",
        "DataNormalizationMode",
        "QCAlgorithm",
    ]
    sys.modules["AlgorithmImports"] = algorithm_imports

import strategy_mixin as qc_strategy_mixin


def _make_algo():
    logs = []
    return SimpleNamespace(
        IsWarmingUp=False,
        max_positions=2,
        ml_min_confidence=0.4,
        _last_selected_stock=None,
        _selection_count=0,
        _min_hold_cycles=3,
        _last_stock_scores={},
        Log=lambda msg: logs.append(msg),
        logs=logs,
    )


def test_rebalance_executes_cc_even_when_option_slots_are_full(monkeypatch):
    algo = _make_algo()
    cc_signal = SimpleNamespace(action="SELL_CALL", symbol="NVDA", delta=0.35, confidence=0.9)

    monkeypatch.setattr(qc_strategy_mixin, "calculate_dynamic_max_positions", lambda _algo: 2)
    monkeypatch.setattr(qc_strategy_mixin, "check_position_management", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(qc_strategy_mixin, "generate_ml_signals", lambda _algo: [cc_signal])

    open_counts = iter([2, 2])
    monkeypatch.setattr(qc_strategy_mixin, "get_option_position_count", lambda _algo: next(open_counts))

    executed = []
    monkeypatch.setattr(
        qc_strategy_mixin,
        "execute_signal",
        lambda _algo, signal, _finder: executed.append(signal.action),
    )

    qc_strategy_mixin.rebalance(algo)

    assert executed == ["SELL_CALL"]
    assert any("CC_SIGNAL: NVDA" in entry for entry in algo.logs)


def test_rebalance_keeps_short_puts_blocked_when_option_slots_are_full(monkeypatch):
    algo = _make_algo()
    cc_signal = SimpleNamespace(action="SELL_CALL", symbol="NVDA", delta=0.35, confidence=0.9)
    sp_signal = SimpleNamespace(action="SELL_PUT", symbol="META", delta=0.30, confidence=0.9)

    monkeypatch.setattr(qc_strategy_mixin, "calculate_dynamic_max_positions", lambda _algo: 2)
    monkeypatch.setattr(qc_strategy_mixin, "check_position_management", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(qc_strategy_mixin, "generate_ml_signals", lambda _algo: [cc_signal, sp_signal])

    open_counts = iter([2, 2, 2])
    monkeypatch.setattr(qc_strategy_mixin, "get_option_position_count", lambda _algo: next(open_counts))

    executed = []
    monkeypatch.setattr(
        qc_strategy_mixin,
        "execute_signal",
        lambda _algo, signal, _finder: executed.append(signal.action),
    )

    qc_strategy_mixin.rebalance(algo)

    assert executed == ["SELL_CALL"]
    assert not any("SP_SIGNAL:" in entry for entry in algo.logs)


def test_rebalance_executes_all_eligible_cc_signals_before_put_gating(monkeypatch):
    algo = _make_algo()
    algo.max_positions = 3
    cc_nvda = SimpleNamespace(action="SELL_CALL", symbol="NVDA", delta=0.35, confidence=0.95)
    cc_meta = SimpleNamespace(action="SELL_CALL", symbol="META", delta=0.32, confidence=0.80)
    sp_msft = SimpleNamespace(action="SELL_PUT", symbol="MSFT", delta=0.30, confidence=0.85)

    monkeypatch.setattr(qc_strategy_mixin, "calculate_dynamic_max_positions", lambda _algo: 3)
    monkeypatch.setattr(qc_strategy_mixin, "check_position_management", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(qc_strategy_mixin, "generate_ml_signals", lambda _algo: [cc_nvda, cc_meta, sp_msft])

    open_counts = iter([3, 3, 3, 3])
    monkeypatch.setattr(qc_strategy_mixin, "get_option_position_count", lambda _algo: next(open_counts))

    executed = []
    monkeypatch.setattr(
        qc_strategy_mixin,
        "execute_signal",
        lambda _algo, signal, _finder: executed.append((signal.action, signal.symbol)),
    )

    qc_strategy_mixin.rebalance(algo)

    assert executed == [("SELL_CALL", "NVDA"), ("SELL_CALL", "META")]
    assert not any(action == "SELL_PUT" for action, _symbol in executed)
