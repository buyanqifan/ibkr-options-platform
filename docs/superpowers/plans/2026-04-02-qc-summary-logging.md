# QC Summary Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add low-volume QC summary counters so free-tier QuantConnect backtests still reveal whether holdings, covered calls, assignment tracking, and stock fills actually occurred.

**Architecture:** Introduce a lightweight `debug_counters` dictionary on the QC algorithm state, increment it at existing high-value behavior points, and emit a compact `SUMMARY_*` block from `on_end_of_algorithm()`. Keep detailed logs in place; the new summary is a durable fallback when detailed logs are truncated.

**Tech Stack:** Python, QuantConnect algorithm modules, pytest

---

## File Map

- Modify: `quantconnect/strategy_init.py`
  Purpose: initialize `algo.debug_counters`
- Modify: `quantconnect/main.py`
  Purpose: count stock buy/sell fills in `OnOrderEvent`
- Modify: `quantconnect/signal_generation.py`
  Purpose: count holdings visibility and SP block reasons
- Modify: `quantconnect/expiry.py`
  Purpose: count assignment tracking, repair attempts/failures, and immediate CC attempts
- Modify: `quantconnect/strategy_mixin.py`
  Purpose: count `CC_SIGNAL` / `SP_SIGNAL` and print `SUMMARY_*` lines at the end
- Modify: `tests/unit/test_qc_strategy_mixin.py`
  Purpose: verify summary emission and signal counters
- Modify: `tests/unit/test_binbin_qc_parity.py`
  Purpose: verify QC init creates debug counters and selected lifecycle paths increment them safely

### Task 1: Add Failing Tests for QC Debug Counter Initialization and Summary Output

**Files:**
- Modify: `tests/unit/test_qc_strategy_mixin.py`
- Modify: `tests/unit/test_binbin_qc_parity.py`
- Test: `tests/unit/test_qc_strategy_mixin.py`
- Test: `tests/unit/test_binbin_qc_parity.py`

- [ ] **Step 1: Write the failing initialization test**

Add this test near the existing QC init/default tests in `tests/unit/test_binbin_qc_parity.py`:

```python
def test_strategy_init_state_creates_debug_counters():
    algo = SimpleNamespace()
    algo.SetWarmUp = lambda days: setattr(algo, "_warmup_days", days)

    from strategy_init import init_state

    init_state(algo)

    assert isinstance(algo.debug_counters, dict)
    assert algo.debug_counters["holdings_seen"] == 0
    assert algo.debug_counters["cc_signals"] == 0
    assert algo.debug_counters["stock_buy"] == 0
```

- [ ] **Step 2: Run the initialization test to verify it fails**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py::test_strategy_init_state_creates_debug_counters -q
```

Expected: `FAIL` because `debug_counters` does not exist yet.

- [ ] **Step 3: Write the failing summary-output test**

Add this test to `tests/unit/test_qc_strategy_mixin.py`:

```python
def test_on_end_of_algorithm_emits_summary_counters():
    logs = []
    algo = SimpleNamespace(
        total_trades=10,
        winning_trades=6,
        initial_capital=100000,
        ml_integration=SimpleNamespace(get_status_report=lambda: "ml ok"),
        Portfolio=SimpleNamespace(
            TotalProfit=1234.0,
            TotalPortfolioValue=110000.0,
        ),
        debug_counters={
            "holdings_seen": 2,
            "cc_signals": 3,
            "sp_signals": 5,
            "put_block": 4,
            "no_suitable_options": 1,
            "assigned_stock_track": 1,
            "immediate_cc": 1,
            "assigned_repair_attempt": 0,
            "assigned_repair_fail": 0,
            "assigned_stock_exit": 0,
            "stock_buy": 2,
            "stock_sell": 1,
            "sp_quality_block": 0,
            "sp_stock_block": 0,
            "sp_held_block": 0,
        },
        Log=lambda msg: logs.append(msg),
        logs=logs,
    )

    qc_strategy_mixin.on_end_of_algorithm(algo)

    assert any("SUMMARY_FLOW:" in entry for entry in logs)
    assert any("SUMMARY_ASSIGNMENT:" in entry for entry in logs)
    assert any("SUMMARY_STOCK_FILLS:" in entry for entry in logs)
```

- [ ] **Step 4: Run the summary-output test to verify it fails**

Run:

```bash
pytest tests/unit/test_qc_strategy_mixin.py::test_on_end_of_algorithm_emits_summary_counters -q
```

Expected: `FAIL` because `on_end_of_algorithm()` does not emit `SUMMARY_*` lines yet.

- [ ] **Step 5: Commit the failing tests**

```bash
git add tests/unit/test_qc_strategy_mixin.py tests/unit/test_binbin_qc_parity.py
git commit -m "test(qc): add summary logging coverage"
```

### Task 2: Implement Debug Counter Initialization and Safe Increment Helpers

**Files:**
- Modify: `quantconnect/strategy_init.py`
- Modify: `quantconnect/strategy_mixin.py`
- Test: `tests/unit/test_binbin_qc_parity.py`

- [ ] **Step 1: Add a helper and default counter map**

In `quantconnect/strategy_mixin.py`, add a small helper near the top of the module:

```python
DEFAULT_DEBUG_COUNTERS = {
    "holdings_seen": 0,
    "cc_signals": 0,
    "sp_signals": 0,
    "put_block": 0,
    "sp_quality_block": 0,
    "sp_stock_block": 0,
    "sp_held_block": 0,
    "assigned_stock_track": 0,
    "assigned_repair_attempt": 0,
    "assigned_repair_fail": 0,
    "assigned_stock_exit": 0,
    "immediate_cc": 0,
    "stock_buy": 0,
    "stock_sell": 0,
    "no_suitable_options": 0,
}


def increment_debug_counter(algo, key, amount=1):
    counters = getattr(algo, "debug_counters", None)
    if counters is None:
        counters = dict(DEFAULT_DEBUG_COUNTERS)
        setattr(algo, "debug_counters", counters)
    counters[key] = counters.get(key, 0) + amount
```

- [ ] **Step 2: Initialize counters during QC state setup**

In `quantconnect/strategy_init.py:init_state()`, after the existing algorithm state fields are created, add:

```python
from strategy_mixin import DEFAULT_DEBUG_COUNTERS

algo.debug_counters = dict(DEFAULT_DEBUG_COUNTERS)
```

Keep the rest of `init_state()` unchanged.

- [ ] **Step 3: Run the initialization test to verify it passes**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py::test_strategy_init_state_creates_debug_counters -q
```

Expected: `PASS`

- [ ] **Step 4: Commit the helper and initialization change**

```bash
git add quantconnect/strategy_init.py quantconnect/strategy_mixin.py tests/unit/test_binbin_qc_parity.py
git commit -m "feat(qc): initialize debug summary counters"
```

### Task 3: Count High-Value Runtime Events

**Files:**
- Modify: `quantconnect/main.py`
- Modify: `quantconnect/signal_generation.py`
- Modify: `quantconnect/expiry.py`
- Modify: `quantconnect/strategy_mixin.py`
- Test: `tests/unit/test_qc_strategy_mixin.py`
- Test: `tests/unit/test_binbin_qc_parity.py`

- [ ] **Step 1: Write the failing stock-fill counter test**

Add this test to `tests/unit/test_binbin_qc_parity.py`:

```python
def test_on_order_event_counts_stock_buys_and_sells(monkeypatch):
    algo = SimpleNamespace(
        debug_counters={"stock_buy": 0, "stock_sell": 0},
        Log=lambda *_args, **_kwargs: None,
    )

    monkeypatch.setattr(qc_main, "handle_order_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(qc_main, "handle_assignment_order_event", lambda *_args, **_kwargs: None)

    equity_symbol = SimpleNamespace(SecurityType="Equity")
    buy_event = SimpleNamespace(Status="Filled", Symbol=equity_symbol, FillQuantity=100, FillPrice=10.0, IsAssignment=False)
    sell_event = SimpleNamespace(Status="Filled", Symbol=equity_symbol, FillQuantity=-100, FillPrice=11.0, IsAssignment=False)

    qc_main.BinbinGodStrategy().OnOrderEvent.__get__(algo, object)  # reference only if needed
```

Then simplify it to call the class method directly:

```python
    qc_main.BinbinGodStrategy.OnOrderEvent(algo, buy_event)
    qc_main.BinbinGodStrategy.OnOrderEvent(algo, sell_event)

    assert algo.debug_counters["stock_buy"] == 1
    assert algo.debug_counters["stock_sell"] == 1
```

- [ ] **Step 2: Run the stock-fill counter test to verify it fails**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py::test_on_order_event_counts_stock_buys_and_sells -q
```

Expected: `FAIL` because the counters are not incremented yet.

- [ ] **Step 3: Implement runtime counter increments**

Make these focused edits:

In `quantconnect/main.py`:

```python
from strategy_mixin import increment_debug_counter
```

and inside `OnOrderEvent()`:

```python
if symbol.SecurityType == SecurityType.Equity:
    action = "BUY" if qty > 0 else "SELL"
    increment_debug_counter(self, f"stock_{action.lower()}")
    self.Log(f"STOCK_{action}: {symbol} qty={qty} @ ${price:.2f}")
```

In `quantconnect/signal_generation.py`:

```python
from strategy_mixin import increment_debug_counter
```

then increment at the existing log points:

```python
if held_symbols:
    increment_debug_counter(algo, "holdings_seen")
    algo.Log(f"HOLDINGS: {shares_info}")
```

```python
increment_debug_counter(algo, "sp_held_block")
algo.Log(f"SP_HELD_BLOCK:{symbol}:shares={shares_held}")
```

```python
increment_debug_counter(algo, "sp_stock_block")
algo.Log(f"SP_STOCK_BLOCK:{symbol}:stock={stock_notional:.0f}:cap={inventory_cap:.0f}")
```

At the extreme block helper log site, increment:

```python
increment_debug_counter(algo, "sp_quality_block")
```

In `quantconnect/strategy_mixin.py`, before the existing `CC_SIGNAL` / `SP_SIGNAL` logs:

```python
increment_debug_counter(algo, "cc_signals")
increment_debug_counter(algo, "sp_signals")
```

In `quantconnect/expiry.py`:

```python
from strategy_mixin import increment_debug_counter
```

then add:

```python
increment_debug_counter(algo, "assigned_stock_track")
increment_debug_counter(algo, "immediate_cc")
increment_debug_counter(algo, "assigned_repair_attempt")
increment_debug_counter(algo, "assigned_repair_fail")
increment_debug_counter(algo, "assigned_stock_exit")
```

only at the exact existing behavior points that already log those events.

- [ ] **Step 4: Run focused tests to verify they pass**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py::test_on_order_event_counts_stock_buys_and_sells tests/unit/test_qc_strategy_mixin.py::test_rebalance_executes_all_eligible_cc_signals_before_put_gating -q
```

Expected: `2 passed`

- [ ] **Step 5: Commit runtime counter increments**

```bash
git add quantconnect/main.py quantconnect/signal_generation.py quantconnect/expiry.py quantconnect/strategy_mixin.py tests/unit/test_binbin_qc_parity.py tests/unit/test_qc_strategy_mixin.py
git commit -m "feat(qc): count summary debug events"
```

### Task 4: Emit End-of-Algorithm Summary Lines and Verify Full QC Test Slice

**Files:**
- Modify: `quantconnect/strategy_mixin.py`
- Optionally Modify: `quantconnect/execution.py`
- Modify: `tests/unit/test_qc_strategy_mixin.py`
- Modify: `tests/unit/test_binbin_qc_parity.py`

- [ ] **Step 1: Write the failing no-suitable-options counter test**

Add this focused test to `tests/unit/test_binbin_qc_parity.py`:

```python
def test_execute_signal_counts_no_suitable_options(monkeypatch):
    algo = SimpleNamespace(
        debug_counters={"no_suitable_options": 0},
        Log=lambda *_args, **_kwargs: None,
        Securities=SimpleNamespace(ContainsKey=lambda *_args, **_kwargs: False),
    )
```

Reuse the existing `qc_execution.execute_signal()` test scaffolding in this file and assert:

```python
assert algo.debug_counters["no_suitable_options"] == 1
```

- [ ] **Step 2: Run the no-suitable-options test to verify it fails**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py::test_execute_signal_counts_no_suitable_options -q
```

Expected: `FAIL` because the execution path does not increment that counter yet.

- [ ] **Step 3: Implement summary emission and final missing counter**

In `quantconnect/execution.py`, at the existing `No suitable options for ...` log point, add:

```python
from strategy_mixin import increment_debug_counter
increment_debug_counter(algo, "no_suitable_options")
```

In `quantconnect/strategy_mixin.py:on_end_of_algorithm()`, after the current result lines and before the final separator, add:

```python
counters = dict(DEFAULT_DEBUG_COUNTERS)
counters.update(getattr(algo, "debug_counters", {}) or {})

algo.Log(
    "SUMMARY_FLOW: "
    f"holdings_seen={counters['holdings_seen']} "
    f"cc_signals={counters['cc_signals']} "
    f"sp_signals={counters['sp_signals']} "
    f"put_block={counters['put_block']} "
    f"no_suitable_options={counters['no_suitable_options']}"
)
algo.Log(
    "SUMMARY_ASSIGNMENT: "
    f"assigned_stock_track={counters['assigned_stock_track']} "
    f"immediate_cc={counters['immediate_cc']} "
    f"assigned_repair_attempt={counters['assigned_repair_attempt']} "
    f"assigned_repair_fail={counters['assigned_repair_fail']} "
    f"assigned_stock_exit={counters['assigned_stock_exit']}"
)
algo.Log(
    "SUMMARY_STOCK_FILLS: "
    f"stock_buy={counters['stock_buy']} "
    f"stock_sell={counters['stock_sell']} "
    f"sp_quality_block={counters['sp_quality_block']} "
    f"sp_stock_block={counters['sp_stock_block']} "
    f"sp_held_block={counters['sp_held_block']}"
)
```

- [ ] **Step 4: Run the focused summary tests to verify they pass**

Run:

```bash
pytest tests/unit/test_qc_strategy_mixin.py::test_on_end_of_algorithm_emits_summary_counters tests/unit/test_binbin_qc_parity.py::test_execute_signal_counts_no_suitable_options -q
```

Expected: `2 passed`

- [ ] **Step 5: Run the full QC regression slice**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py tests/unit/test_qc_execution.py tests/unit/test_qc_strategy_mixin.py tests/unit/test_backtest_engine_qc_replay.py tests/unit/test_strategies.py tests/unit/test_roll_logic.py tests/unit/test_page_imports.py -q
```

Expected: all tests pass with no new failures.

- [ ] **Step 6: Commit the summary logging implementation**

```bash
git add quantconnect/execution.py quantconnect/strategy_mixin.py tests/unit/test_qc_strategy_mixin.py tests/unit/test_binbin_qc_parity.py
git commit -m "feat(qc): emit summary debug logging"
```
