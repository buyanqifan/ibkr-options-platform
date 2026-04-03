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

_MISSING = object()


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


def _make_summary_algo(debug_counters=_MISSING):
    logs = []
    algo = SimpleNamespace(
        initial_capital=100000.0,
        total_trades=12,
        winning_trades=9,
        stock_pool=[],
        equities={},
        Portfolio=SimpleNamespace(
            TotalProfit=1250.0,
            TotalPortfolioValue=101250.0,
            Values=[],
            ContainsKey=lambda _symbol: False,
        ),
        ml_integration=SimpleNamespace(get_status_report=lambda: "STATUS"),
        Log=lambda msg: logs.append(msg),
        logs=logs,
    )
    if debug_counters is not _MISSING:
        algo.debug_counters = debug_counters
    return algo


def _summary_token_map(logs, prefix):
    line = next(
        (
            entry
            for entry in logs
            if entry.split(maxsplit=1) and entry.split(maxsplit=1)[0] == prefix
        ),
        None,
    )

    assert line is not None

    tokens = {}
    for token in line.split()[1:]:
        key, value = token.split("=", 1)
        tokens[key] = value
    return tokens


def test_on_end_of_algorithm_emits_summary_lines(monkeypatch):
    monkeypatch.setattr(qc_strategy_mixin, "get_symbols_with_holdings", lambda *_args, **_kwargs: [])

    algo = _make_summary_algo(
        {
            "holdings_seen": 2,
            "cc_signals": 1,
            "sp_signals": 4,
            "put_block": 5,
            "no_suitable_options": 6,
            "assigned_stock_track": 7,
            "immediate_cc": 8,
            "assigned_repair_attempt": 9,
            "assigned_repair_fail": 10,
            "assigned_stock_exit": 11,
            "stock_buy": 12,
            "stock_sell": 13,
            "sp_quality_block": 14,
            "sp_stock_block": 15,
            "sp_held_block": 16,
        }
    )

    qc_strategy_mixin.on_end_of_algorithm(algo)

    flow_tokens = _summary_token_map(algo.logs, "SUMMARY_FLOW:")
    assignment_tokens = _summary_token_map(algo.logs, "SUMMARY_ASSIGNMENT:")
    stock_fill_tokens = _summary_token_map(algo.logs, "SUMMARY_STOCK_FILLS:")

    assert flow_tokens["holdings_seen"] == "2"
    assert flow_tokens["cc_signals"] == "1"
    assert flow_tokens["sp_signals"] == "4"
    assert flow_tokens["put_block"] == "5"
    assert flow_tokens["no_suitable_options"] == "6"
    assert assignment_tokens["assigned_stock_track"] == "7"
    assert assignment_tokens["immediate_cc"] == "8"
    assert assignment_tokens["assigned_repair_attempt"] == "9"
    assert assignment_tokens["assigned_repair_fail"] == "10"
    assert assignment_tokens["assigned_stock_exit"] == "11"
    assert stock_fill_tokens["stock_buy"] == "12"
    assert stock_fill_tokens["stock_sell"] == "13"
    assert stock_fill_tokens["sp_quality_block"] == "14"
    assert stock_fill_tokens["sp_stock_block"] == "15"
    assert stock_fill_tokens["sp_held_block"] == "16"


def test_on_end_of_algorithm_defaults_missing_summary_counters_to_zero(monkeypatch):
    monkeypatch.setattr(qc_strategy_mixin, "get_symbols_with_holdings", lambda *_args, **_kwargs: [])

    algo = _make_summary_algo(
        {
            "holdings_seen": 2,
            "assigned_stock_track": 7,
            "stock_buy": 12,
        }
    )

    qc_strategy_mixin.on_end_of_algorithm(algo)

    flow_tokens = _summary_token_map(algo.logs, "SUMMARY_FLOW:")
    assignment_tokens = _summary_token_map(algo.logs, "SUMMARY_ASSIGNMENT:")
    stock_fill_tokens = _summary_token_map(algo.logs, "SUMMARY_STOCK_FILLS:")

    assert flow_tokens["holdings_seen"] == "2"
    assert flow_tokens["cc_signals"] == "0"
    assert flow_tokens["sp_signals"] == "0"
    assert flow_tokens["put_block"] == "0"
    assert flow_tokens["no_suitable_options"] == "0"
    assert assignment_tokens["assigned_stock_track"] == "7"
    assert assignment_tokens["immediate_cc"] == "0"
    assert assignment_tokens["assigned_repair_attempt"] == "0"
    assert assignment_tokens["assigned_repair_fail"] == "0"
    assert assignment_tokens["assigned_stock_exit"] == "0"
    assert stock_fill_tokens["stock_buy"] == "12"
    assert stock_fill_tokens["stock_sell"] == "0"
    assert stock_fill_tokens["sp_quality_block"] == "0"
    assert stock_fill_tokens["sp_stock_block"] == "0"
    assert stock_fill_tokens["sp_held_block"] == "0"


def test_on_end_of_algorithm_defaults_all_summary_counters_to_zero_when_debug_counters_missing(monkeypatch):
    monkeypatch.setattr(qc_strategy_mixin, "get_symbols_with_holdings", lambda *_args, **_kwargs: [])

    algo = _make_summary_algo()

    qc_strategy_mixin.on_end_of_algorithm(algo)

    flow_tokens = _summary_token_map(algo.logs, "SUMMARY_FLOW:")
    assignment_tokens = _summary_token_map(algo.logs, "SUMMARY_ASSIGNMENT:")
    stock_fill_tokens = _summary_token_map(algo.logs, "SUMMARY_STOCK_FILLS:")

    assert flow_tokens["holdings_seen"] == "0"
    assert flow_tokens["cc_signals"] == "0"
    assert flow_tokens["sp_signals"] == "0"
    assert flow_tokens["put_block"] == "0"
    assert flow_tokens["no_suitable_options"] == "0"
    assert assignment_tokens["assigned_stock_track"] == "0"
    assert assignment_tokens["immediate_cc"] == "0"
    assert assignment_tokens["assigned_repair_attempt"] == "0"
    assert assignment_tokens["assigned_repair_fail"] == "0"
    assert assignment_tokens["assigned_stock_exit"] == "0"
    assert stock_fill_tokens["stock_buy"] == "0"
    assert stock_fill_tokens["stock_sell"] == "0"
    assert stock_fill_tokens["sp_quality_block"] == "0"
    assert stock_fill_tokens["sp_stock_block"] == "0"
    assert stock_fill_tokens["sp_held_block"] == "0"


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


def test_rebalance_counts_cc_and_sp_signals(monkeypatch):
    algo = _make_algo()
    cc_signal = SimpleNamespace(action="SELL_CALL", symbol="NVDA", delta=0.35, confidence=0.9, ml_score_adjustment=0.0)
    sp_signal = SimpleNamespace(action="SELL_PUT", symbol="META", delta=0.30, confidence=0.9, ml_score_adjustment=0.0)

    monkeypatch.setattr(qc_strategy_mixin, "calculate_dynamic_max_positions", lambda _algo: 3)
    monkeypatch.setattr(qc_strategy_mixin, "check_position_management", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(qc_strategy_mixin, "generate_ml_signals", lambda _algo: [cc_signal, sp_signal])

    open_counts = iter([0, 0, 0])
    monkeypatch.setattr(qc_strategy_mixin, "get_option_position_count", lambda _algo: next(open_counts))
    monkeypatch.setattr(qc_strategy_mixin, "execute_signal", lambda *_args, **_kwargs: None)

    qc_strategy_mixin.rebalance(algo)

    assert algo.debug_counters["cc_signals"] == 1
    assert algo.debug_counters["sp_signals"] == 1


def test_rebalance_retries_pending_cc_opens_before_new_put_signals(monkeypatch):
    algo = _make_algo()
    algo.max_positions = 2
    algo.pending_open_orders = {
        "cc-1": {
            "signal": SimpleNamespace(action="SELL_CALL", symbol="NVDA", confidence=0.9),
            "option_symbol": "NVDA 240621C01000000",
            "quantity": -1,
            "theoretical_price": 1.25,
            "attempt_count": 0,
            "target_right": "Call",
            "selected": {"strike": 1000.0},
        }
    }

    order = []
    monkeypatch.setattr(qc_strategy_mixin, "calculate_dynamic_max_positions", lambda _algo: 2)
    monkeypatch.setattr(qc_strategy_mixin, "check_position_management", lambda *_args, **_kwargs: order.append("manage"))
    monkeypatch.setattr(
        qc_strategy_mixin,
        "retry_pending_open_orders",
        lambda _algo, _finder: order.append("retry") or ["cc-1"],
        raising=False,
    )
    monkeypatch.setattr(
        qc_strategy_mixin,
        "generate_ml_signals",
        lambda _algo: order.append("signals") or [SimpleNamespace(action="SELL_PUT", symbol="META", delta=0.3, confidence=0.9)],
    )
    monkeypatch.setattr(qc_strategy_mixin, "get_option_position_count", lambda _algo: 2)

    qc_strategy_mixin.rebalance(algo)

    assert order[:3] == ["manage", "retry", "signals"]
