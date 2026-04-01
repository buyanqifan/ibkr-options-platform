# QC Assigned Stock Fail-Safe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an assignment-aware covered-call repair path plus a stock fail-safe so QC can raise utilization without allowing assignment-driven stock losses to dominate returns.

**Architecture:** Extend the existing QC wheel flow instead of adding a separate subsystem. Assignment detection in `expiry.py` creates per-symbol state, `signal_generation.py` reads that state to produce more aggressive repair calls, and `position_management.py` enforces the final stock exit only after repeated uncovered failures.

**Tech Stack:** Python, QuantConnect algorithm modules, pytest

---

### Task 1: Parameter And State Wiring

**Files:**
- Modify: `quantconnect/strategy_init.py`
- Test: `tests/unit/test_binbin_qc_parity.py`

- [ ] **Step 1: Write the failing test**

Add a test that validates the new runtime defaults and state initialization:

```python
def test_strategy_init_sets_assigned_stock_fail_safe_defaults():
    defaults = _extract_strategy_init_parameter_defaults()

    assert defaults["assigned_stock_fail_safe_enabled"] is True
    assert defaults["assigned_stock_drawdown_pct"] == pytest.approx(0.12)
    assert defaults["assigned_stock_repair_attempt_limit"] == 3
    assert defaults["assigned_stock_min_days_held"] == 5
    assert defaults["assigned_stock_force_exit_pct"] == pytest.approx(1.0)
    assert defaults["assigned_stock_repair_delta_boost"] == pytest.approx(0.10)
    assert defaults["assigned_stock_repair_dte_min"] == 7
    assert defaults["assigned_stock_repair_dte_max"] == 14
    assert defaults["assigned_stock_repair_max_discount_pct"] == pytest.approx(0.12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_binbin_qc_parity.py::test_strategy_init_sets_assigned_stock_fail_safe_defaults -q`
Expected: FAIL because the defaults do not yet exist in `strategy_init.py`

- [ ] **Step 3: Write minimal implementation**

In `quantconnect/strategy_init.py`:

- add the new parameters inside `init_parameters(algo)`
- add `algo.assigned_stock_state = {}` inside `init_state(algo)`
- keep names exactly:
  - `assigned_stock_fail_safe_enabled`
  - `assigned_stock_drawdown_pct`
  - `assigned_stock_repair_attempt_limit`
  - `assigned_stock_min_days_held`
  - `assigned_stock_force_exit_pct`
  - `assigned_stock_repair_delta_boost`
  - `assigned_stock_repair_dte_min`
  - `assigned_stock_repair_dte_max`
  - `assigned_stock_repair_max_discount_pct`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_binbin_qc_parity.py::test_strategy_init_sets_assigned_stock_fail_safe_defaults -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add quantconnect/strategy_init.py tests/unit/test_binbin_qc_parity.py
git commit -m "feat(quantconnect): add assigned stock fail-safe defaults"
```

### Task 2: Assignment Tracking In Expiry Handling

**Files:**
- Modify: `quantconnect/expiry.py`
- Test: `tests/unit/test_roll_logic.py`

- [ ] **Step 1: Write the failing test**

Add a focused test for put-assignment tracking:

```python
def test_check_expired_options_tracks_put_assignment_state(monkeypatch):
    algo = _make_assignment_algo()

    check_expired_options(algo)

    state = algo.assigned_stock_state["NVDA"]
    assert state["source"] == "put_assignment"
    assert state["repair_failures"] == 0
    assert state["assignment_cost_basis"] == pytest.approx(120.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_roll_logic.py::test_check_expired_options_tracks_put_assignment_state -q`
Expected: FAIL because `assigned_stock_state` is not populated by expiry logic

- [ ] **Step 3: Write minimal implementation**

In `quantconnect/expiry.py`:

- when a put assignment is detected, initialize or refresh `algo.assigned_stock_state[symbol]`
- store:
  - `source`
  - `assignment_date`
  - `assignment_cost_basis`
  - `repair_failures`
  - `last_repair_attempt`
  - `force_exit_triggered`
- do this before `try_sell_cc_immediately(algo, symbol)`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_roll_logic.py::test_check_expired_options_tracks_put_assignment_state -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add quantconnect/expiry.py tests/unit/test_roll_logic.py
git commit -m "feat(quantconnect): track assignment stock state"
```

### Task 3: Assignment-Aware Repair Call Signals

**Files:**
- Modify: `quantconnect/signal_generation.py`
- Test: `tests/unit/test_roll_logic.py`

- [ ] **Step 1: Write the failing test**

Add a signal-generation test that demonstrates stronger repair behavior for assignment-tagged stock:

```python
def test_generate_signal_for_symbol_uses_assigned_stock_repair_overrides(monkeypatch):
    algo = _make_assigned_stock_algo()

    signal = generate_signal_for_symbol(algo, "NVDA", "CC", get_portfolio_state(algo))

    assert signal.delta >= pytest.approx(0.45)
    assert signal.dte_min == 7
    assert signal.dte_max == 14
    assert signal.min_strike > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_roll_logic.py::test_generate_signal_for_symbol_uses_assigned_stock_repair_overrides -q`
Expected: FAIL because assignment state does not yet change CC repair settings

- [ ] **Step 3: Write minimal implementation**

In `quantconnect/signal_generation.py`:

- detect `assigned_stock_state` for the symbol
- when generating `CC`, force enhanced repair mode for assignment-tagged stock
- boost delta using `assigned_stock_repair_delta_boost`
- constrain DTE to `assigned_stock_repair_dte_min/max`
- allow the larger discount using `assigned_stock_repair_max_discount_pct`
- keep the existing `CC_REPAIR` flow compatible

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_roll_logic.py::test_generate_signal_for_symbol_uses_assigned_stock_repair_overrides -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add quantconnect/signal_generation.py tests/unit/test_roll_logic.py
git commit -m "feat(quantconnect): prioritize repair calls for assigned stock"
```

### Task 4: Stock Fail-Safe Enforcement

**Files:**
- Modify: `quantconnect/position_management.py`
- Modify: `quantconnect/qc_portfolio.py`
- Test: `tests/unit/test_roll_logic.py`

- [ ] **Step 1: Write the failing test**

Add two stock-management tests:

```python
def test_assignment_fail_safe_skips_when_call_is_already_covering_shares():
    algo = _make_assigned_stock_algo(with_covering_call=True)

    check_position_management(algo, execute_signal_func=lambda *a, **k: None, find_option_func=lambda *a, **k: None)

    assert algo.assigned_stock_state["NVDA"]["repair_failures"] == 0


def test_assignment_fail_safe_force_exits_stock_after_repeated_failures():
    algo = _make_assigned_stock_algo(with_covering_call=False, days_held=8, drawdown_pct=0.16, failures=2)

    check_position_management(algo, execute_signal_func=lambda *a, **k: None, find_option_func=lambda *a, **k: None)

    assert algo.market_orders == [("NVDA", -100)]
    assert algo.assigned_stock_state["NVDA"]["force_exit_triggered"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_roll_logic.py::test_assignment_fail_safe_skips_when_call_is_already_covering_shares tests/unit/test_roll_logic.py::test_assignment_fail_safe_force_exits_stock_after_repeated_failures -q`
Expected: FAIL because stock-level fail-safe logic does not yet exist

- [ ] **Step 3: Write minimal implementation**

In `quantconnect/qc_portfolio.py`:

- add a helper that returns active call coverage for a symbol in shares or contracts, reusing existing option position inspection

In `quantconnect/position_management.py`:

- add a stock-level management pass before option loop
- for each symbol in `assigned_stock_state`:
  - clear state if no shares remain
  - skip and reset failures if covered by an active call
  - compute days held and drawdown versus `assignment_cost_basis`
  - increment `repair_failures` when uncovered and below threshold after `min_days_held`
  - `algo.MarketOrder(equity.Symbol, -shares_to_sell)` when failures hit limit
  - mark `force_exit_triggered`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_roll_logic.py::test_assignment_fail_safe_skips_when_call_is_already_covering_shares tests/unit/test_roll_logic.py::test_assignment_fail_safe_force_exits_stock_after_repeated_failures -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add quantconnect/position_management.py quantconnect/qc_portfolio.py tests/unit/test_roll_logic.py
git commit -m "feat(quantconnect): add assigned stock fail-safe exits"
```

### Task 5: Regression Verification

**Files:**
- Modify: `tests/unit/test_binbin_qc_parity.py`
- Modify: `tests/unit/test_roll_logic.py`

- [ ] **Step 1: Add any missing regression assertions**

Ensure regressions cover:

```python
def test_assignment_state_clears_when_shares_are_gone():
    ...

def test_assignment_fail_safe_respects_min_days_held():
    ...
```

- [ ] **Step 2: Run targeted suite**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py tests/unit/test_roll_logic.py -q
```

Expected: PASS

- [ ] **Step 3: Run broader QC-related suite**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py tests/unit/test_binbin_god_page.py tests/unit/test_strategies.py tests/unit/test_roll_logic.py tests/unit/test_page_imports.py -q
```

Expected: PASS

- [ ] **Step 4: Commit final regression adjustments**

```bash
git add tests/unit/test_binbin_qc_parity.py tests/unit/test_roll_logic.py
git commit -m "test(quantconnect): cover assigned stock fail-safe flow"
```
