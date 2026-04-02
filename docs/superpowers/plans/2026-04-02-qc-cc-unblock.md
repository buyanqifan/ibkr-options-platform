# QC Covered Call Unblock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure covered calls can still be generated and executed even when short-option slot usage has already reached `max_positions`, while keeping new short puts capped.

**Architecture:** This is a narrow control-flow fix in `quantconnect/strategy_mixin.py`. First, add focused regression tests that prove `CC` must still run when `open_count >= max_positions` and `SP` must remain blocked in that same condition. Then make the minimal rebalance ordering change so covered calls run before the short-put slot gate.

**Tech Stack:** Python, pytest, QuantConnect strategy modules

---

### Task 1: Add Rebalance Regression Tests For Covered-Call Unblocking

**Files:**
- Create: `tests/unit/test_qc_strategy_mixin.py`
- Test: `tests/unit/test_qc_strategy_mixin.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_qc_strategy_mixin.py` with two focused tests that monkeypatch the mixin dependencies and assert the desired rebalance behavior:

```python
from types import SimpleNamespace

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
    monkeypatch.setattr(qc_strategy_mixin, "execute_signal", lambda _algo, signal, _finder: executed.append(signal.action))

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
    monkeypatch.setattr(qc_strategy_mixin, "execute_signal", lambda _algo, signal, _finder: executed.append(signal.action))

    qc_strategy_mixin.rebalance(algo)

    assert executed == ["SELL_CALL"]
    assert not any("SP_SIGNAL:" in entry for entry in algo.logs)
```

- [ ] **Step 2: Run the new test file to verify RED**

Run:

```bash
pytest tests/unit/test_qc_strategy_mixin.py -q
```

Expected: at least the first test fails because `rebalance()` currently returns before signal generation when `open_count >= max_positions`.

- [ ] **Step 3: Commit the failing test only if your workflow requires it**

Do not commit yet unless you intentionally checkpoint red tests.

### Task 2: Make Covered Calls Run Before The Short-Put Slot Gate

**Files:**
- Modify: `quantconnect/strategy_mixin.py`
- Test: `tests/unit/test_qc_strategy_mixin.py`

- [ ] **Step 1: Write the minimal implementation**

Update `rebalance()` in `quantconnect/strategy_mixin.py` so covered calls are no longer blocked by the early slot return:

```python
def rebalance(algo):
    if algo.IsWarmingUp:
        return

    algo.max_positions = calculate_dynamic_max_positions(algo)
    check_position_management(algo, execute_signal, find_option_by_greeks)

    signals = generate_ml_signals(algo)
    if not signals:
        return

    sp_signals = [s for s in signals if s.action == "SELL_PUT"]
    cc_signals = [s for s in signals if s.action == "SELL_CALL"]

    open_count = get_option_position_count(algo)

    if cc_signals:
        best_cc = max(cc_signals, key=lambda x: x.confidence)
        algo.Log(f"CC_SIGNAL: {best_cc.symbol} delta={best_cc.delta:.2f}")
        if best_cc.confidence >= algo.ml_min_confidence:
            execute_signal(algo, best_cc, find_option_by_greeks)

    open_count = get_option_position_count(algo)
    if not sp_signals or open_count >= algo.max_positions:
        return

    available_slots = max(0, algo.max_positions - open_count)
    for sp_signal in _select_sp_candidates_for_execution(algo, sp_signals, available_slots):
        algo.Log(f"SP_SIGNAL: {sp_signal.symbol} delta={sp_signal.delta:.2f}")
        execute_signal(algo, sp_signal, find_option_by_greeks)
        open_count = get_option_position_count(algo)
        if open_count >= algo.max_positions:
            break
```

Do not modify:

- `execute_signal()` call coverage sizing
- `max_positions` semantics for new short puts
- signal-generation logic itself

- [ ] **Step 2: Run the focused tests to verify GREEN**

Run:

```bash
pytest tests/unit/test_qc_strategy_mixin.py -q
```

Expected: `2 passed`

- [ ] **Step 3: Run the existing regression suite**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py tests/unit/test_strategies.py tests/unit/test_roll_logic.py tests/unit/test_page_imports.py tests/unit/test_qc_execution.py tests/unit/test_qc_strategy_mixin.py -q
```

Expected: full selected suite passes.

- [ ] **Step 4: Commit**

```bash
git add quantconnect/strategy_mixin.py tests/unit/test_qc_strategy_mixin.py
git commit -m "fix(quantconnect): unblock covered calls at slot limit"
```
