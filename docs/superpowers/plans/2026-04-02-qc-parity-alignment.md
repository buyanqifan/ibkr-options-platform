# QC And Parity Lifecycle Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align QC live strategy behavior and local QC parity replay around multi-symbol covered-call execution, event-first assignment handling, asynchronous-safe roll/close flow, and portfolio-value-based dynamic capacity.

**Architecture:** Implement this as four narrow behavior changes across `quantconnect/` and parity replay. Use test-first changes to pin each lifecycle rule independently, then wire parity defaults and replay logic to the same semantics. Keep the execution-state additions minimal and purpose-built rather than introducing a broad new order framework.

**Tech Stack:** Python, pytest, QuantConnect strategy modules, local QC parity backtest engine

---

### Task 1: Add Regression Tests For Multi-Symbol Covered Call Fan-Out

**Files:**
- Modify: `tests/unit/test_qc_strategy_mixin.py`
- Test: `tests/unit/test_qc_strategy_mixin.py`

- [ ] **Step 1: Write the failing QC test for multi-symbol covered calls**

Append this test to `tests/unit/test_qc_strategy_mixin.py`:

```python
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
```

- [ ] **Step 2: Run the focused mixin test file to verify RED**

Run:

```bash
pytest tests/unit/test_qc_strategy_mixin.py -q
```

Expected: the new test fails because `rebalance()` currently only executes one best covered call.

- [ ] **Step 3: Implement minimal QC fan-out behavior**

Update `quantconnect/strategy_mixin.py` so the covered-call section sorts all `cc_signals` by confidence descending and attempts each one:

```python
    if cc_signals:
        for cc_signal in sorted(cc_signals, key=lambda x: x.confidence, reverse=True):
            algo.Log(f"CC_SIGNAL: {cc_signal.symbol} delta={cc_signal.delta:.2f}")
            if cc_signal.confidence >= algo.ml_min_confidence:
                execute_signal(algo, cc_signal, find_option_by_greeks)
```

Keep the existing short-put slot gate below it unchanged except for recomputing `open_count` after the CC loop.

- [ ] **Step 4: Run the focused mixin test file to verify GREEN**

Run:

```bash
pytest tests/unit/test_qc_strategy_mixin.py -q
```

Expected: all tests in the file pass.

- [ ] **Step 5: Commit the covered-call fan-out change**

```bash
git add quantconnect/strategy_mixin.py tests/unit/test_qc_strategy_mixin.py
git commit -m "fix(quantconnect): fan out covered calls per rebalance"
```

### Task 2: Make Assignment Processing Event-First And Deduplicated

**Files:**
- Modify: `quantconnect/expiry.py`
- Modify: `quantconnect/main.py`
- Modify: `quantconnect/strategy_init.py`
- Modify: `tests/unit/test_binbin_qc_parity.py`
- Test: `tests/unit/test_binbin_qc_parity.py`

- [ ] **Step 1: Write the failing assignment dedupe tests**

Add two focused tests to `tests/unit/test_binbin_qc_parity.py`:

```python
def test_assignment_event_marks_option_as_processed(monkeypatch):
    algo = _make_assignment_algo()
    event = _make_assignment_event(symbol="NVDA", right="P", strike=100.0, expiry=datetime(2025, 1, 17), quantity=-1)

    tracked = []
    monkeypatch.setattr(qc_expiry, "try_sell_cc_immediately", lambda *_args, **_kwargs: tracked.append("cc"))

    qc_expiry.handle_assignment_order_event(algo, event)

    assert tracked == ["cc"]
    assert len(algo.processed_assignment_keys) == 1


def test_expiry_scan_skips_assignment_already_processed_by_event(monkeypatch):
    algo = _make_assignment_algo()
    processed_key = "NVDA_20250117_100_P"
    algo.processed_assignment_keys = {processed_key}

    monkeypatch.setattr(
        qc_expiry,
        "get_option_positions",
        lambda _algo: {
            processed_key: {
                "symbol": "NVDA",
                "option_symbol": SimpleNamespace(),
                "right": "P",
                "strike": 100.0,
                "expiry": datetime(2025, 1, 17),
                "quantity": -1,
                "entry_price": 2.5,
            }
        },
    )

    tracked = []
    monkeypatch.setattr(qc_expiry, "_track_assigned_stock", lambda *_args, **_kwargs: tracked.append("track"))
    monkeypatch.setattr(qc_expiry, "record_trade", lambda *_args, **_kwargs: tracked.append("record"))

    qc_expiry.check_expired_options(algo)

    assert tracked == []
```

Use or extend existing test helpers in the file to build `_make_assignment_algo()` and `_make_assignment_event()` if they do not already exist.

- [ ] **Step 2: Run the assignment-focused tests to verify RED**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py -k "assignment_event_marks_option_as_processed or expiry_scan_skips_assignment_already_processed_by_event" -q
```

Expected: both tests fail because no dedupe ledger exists yet.

- [ ] **Step 3: Implement minimal QC assignment dedupe state**

Update `quantconnect/strategy_init.py` to initialize:

```python
    algo.processed_assignment_keys = set()
```

Add a helper in `quantconnect/expiry.py`:

```python
def _build_assignment_key(symbol: str, expiry, strike: float, right: str) -> str:
    return f"{symbol}_{expiry.strftime('%Y%m%d')}_{strike:.0f}_{right}"
```

Update `handle_assignment_order_event()` to:

- build the key
- return early if already processed
- add the key before side effects

Update `check_expired_options()` to:

- reuse the same key shape
- skip processing when the key is already in `processed_assignment_keys`
- only add the key when fallback processing really handles an assignment

- [ ] **Step 4: Run the assignment-focused tests to verify GREEN**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py -k "assignment_event_marks_option_as_processed or expiry_scan_skips_assignment_already_processed_by_event" -q
```

Expected: both tests pass.

- [ ] **Step 5: Commit the event-first assignment change**

```bash
git add quantconnect/expiry.py quantconnect/main.py quantconnect/strategy_init.py tests/unit/test_binbin_qc_parity.py
git commit -m "fix(quantconnect): dedupe assignment handling"
```

### Task 3: Make QC Roll And Close Flows Pending-State Safe

**Files:**
- Modify: `quantconnect/execution.py`
- Modify: `quantconnect/main.py`
- Modify: `quantconnect/strategy_init.py`
- Modify: `tests/unit/test_qc_execution.py`
- Test: `tests/unit/test_qc_execution.py`

- [ ] **Step 1: Write failing tests for deferred close and roll tracking**

Add focused tests to `tests/unit/test_qc_execution.py`:

```python
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
```

- [ ] **Step 2: Run the focused execution tests to verify RED**

Run:

```bash
pytest tests/unit/test_qc_execution.py -k "pending_close or pending_roll" -q
```

Expected: failures because deferred/non-filled close and roll currently just return.

- [ ] **Step 3: Implement minimal pending close/roll tracking**

Update `quantconnect/strategy_init.py` state initialization with:

```python
    algo.pending_close_orders = {}
    algo.pending_roll_orders = {}
```

In `quantconnect/execution.py`:

- add a helper to build the position key:

```python
def _build_position_key(symbol: str, expiry, strike: float, right: str) -> str:
    return f"{symbol}_{expiry.strftime('%Y%m%d')}_{strike:.0f}_{right}"
```

- update `execute_close()` so that if `close_ticket` is missing or not immediately filled it stores a minimal pending-close record under `pending_close_orders[position_key]` and returns
- update `execute_roll()` so that if the close ticket is missing or not immediately filled it stores a minimal pending-roll record under `pending_roll_orders[position_key]` and returns

Do not implement the full finalization path in this task; only prevent the intent from disappearing.

- [ ] **Step 4: Run the focused execution tests to verify GREEN**

Run:

```bash
pytest tests/unit/test_qc_execution.py -k "pending_close or pending_roll" -q
```

Expected: the new tests pass.

- [ ] **Step 5: Commit the pending-state execution change**

```bash
git add quantconnect/execution.py quantconnect/main.py quantconnect/strategy_init.py tests/unit/test_qc_execution.py
git commit -m "fix(quantconnect): retain pending close and roll state"
```

### Task 4: Switch Dynamic Max Positions To Portfolio-Value Compounding

**Files:**
- Modify: `quantconnect/execution.py`
- Modify: `core/backtesting/qc_parity.py`
- Modify: `tests/unit/test_binbin_qc_parity.py`
- Test: `tests/unit/test_binbin_qc_parity.py`

- [ ] **Step 1: Write failing tests for portfolio-value-based capacity**

Add or update tests in `tests/unit/test_binbin_qc_parity.py`:

```python
def test_calculate_dynamic_max_positions_from_prices_uses_portfolio_value_budget():
    config = BinbinGodParityConfig.from_params({"parity_mode": "qc", "initial_capital": 100000})
    config.target_margin_utilization = 0.58
    result_small = calculate_dynamic_max_positions_from_prices([100.0, 100.0, 100.0], config, portfolio_value=100000.0)
    result_large = calculate_dynamic_max_positions_from_prices([100.0, 100.0, 100.0], config, portfolio_value=160000.0)

    assert result_large > result_small


def test_qc_dynamic_max_positions_uses_total_portfolio_value():
    algo = SimpleNamespace(
        stock_pool=["MSFT", "AAPL"],
        equities={"MSFT": SimpleNamespace(Symbol="MSFT"), "AAPL": SimpleNamespace(Symbol="AAPL")},
        Securities=_make_security_map({"MSFT": 100.0, "AAPL": 100.0}),
        Portfolio=SimpleNamespace(TotalPortfolioValue=160000.0),
        target_margin_utilization=0.58,
        max_positions_ceiling=20,
    )

    result = qc_execution.calculate_dynamic_max_positions(algo)

    assert result > 5
```

If `_make_security_map()` does not exist, add a small local helper in the test file that mimics `ContainsKey()` and item lookup.

- [ ] **Step 2: Run the focused parity/capacity tests to verify RED**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py -k "dynamic_max_positions" -q
```

Expected: failures because the formula still uses initial capital and the parity helper signature has not yet been extended.

- [ ] **Step 3: Implement the compounding formula in QC and parity**

In `quantconnect/execution.py`, change:

```python
    margin_budget = algo.initial_capital * algo.target_margin_utilization
```

to:

```python
    portfolio_value = max(getattr(algo.Portfolio, "TotalPortfolioValue", algo.initial_capital), 0.0)
    margin_budget = portfolio_value * algo.target_margin_utilization
```

In `core/backtesting/qc_parity.py`, extend `calculate_dynamic_max_positions_from_prices()` to accept an optional `portfolio_value` parameter:

```python
def calculate_dynamic_max_positions_from_prices(prices, config, portfolio_value=None):
    ...
    capital_base = portfolio_value if portfolio_value is not None else config.initial_capital
    margin_budget = capital_base * config.target_margin_utilization
```

Update its callers in parity replay to pass current portfolio value when available.

- [ ] **Step 4: Run the focused parity/capacity tests to verify GREEN**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py -k "dynamic_max_positions" -q
```

Expected: the updated tests pass.

- [ ] **Step 5: Commit the compounding-capacity change**

```bash
git add quantconnect/execution.py core/backtesting/qc_parity.py tests/unit/test_binbin_qc_parity.py
git commit -m "feat(parity): compound dynamic slot capacity"
```

### Task 5: Mirror Multi-CC And Assignment Dedupe In QC Replay

**Files:**
- Modify: `core/backtesting/strategies/binbin_god.py`
- Modify: `core/backtesting/engine.py`
- Modify: `tests/unit/test_binbin_qc_parity.py`
- Modify: `tests/unit/test_backtest_engine_qc_replay.py`
- Test: `tests/unit/test_binbin_qc_parity.py`
- Test: `tests/unit/test_backtest_engine_qc_replay.py`

- [ ] **Step 1: Write failing parity tests**

Add focused tests to `tests/unit/test_binbin_qc_parity.py`:

```python
def test_qc_parity_generates_multiple_cc_signals_for_multiple_stock_holdings(monkeypatch):
    strategy = BinbinGodStrategy({"parity_mode": "qc"})
    strategy.set_parity_context(
        {
            "portfolio_state": {
                "positions": [
                    {"symbol": "NVDA", "quantity": 100, "market_value": 12000.0},
                    {"symbol": "META", "quantity": 100, "market_value": 14000.0},
                ],
                "cost_basis": {"NVDA": 125.0, "META": 145.0},
            },
            "dynamic_max_positions": 2,
        }
    )

    monkeypatch.setattr(strategy, "_generate_qc_parity_signal_for_symbol", lambda symbol, phase, *_args, **_kwargs: Signal(
        symbol=symbol,
        trade_type="SELL_CALL",
        right="C",
        strike=150.0,
        expiry="2025-01-17",
        premium=2.0,
        confidence=0.8,
    ) if phase == "CC" else None)

    signals = strategy._generate_qc_parity_signals(current_date="2024-03-01", data_by_symbol={})

    assert [signal.symbol for signal in signals if signal.trade_type == "SELL_CALL"] == ["NVDA", "META"]
```

Also add a focused replay-engine test in `tests/unit/test_backtest_engine_qc_replay.py` asserting parity current-portfolio value is passed into dynamic slot calculation if that call path exists there.

- [ ] **Step 2: Run the focused parity tests to verify RED**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py tests/unit/test_backtest_engine_qc_replay.py -k "multiple_cc or dynamic_max_positions" -q
```

Expected: failures because replay currently still behaves as single-best or lacks the updated capacity wiring.

- [ ] **Step 3: Implement replay alignment**

In `core/backtesting/strategies/binbin_god.py`:

- update the QC replay signal path so multiple covered-call signals can coexist for multiple held symbols
- ensure put slot gating remains separate from covered-call eligibility

In `core/backtesting/engine.py`, if dynamic max-position calculation happens there, pass current portfolio value through to the parity helper.

If replay assignment reconciliation needs dedupe state, add the smallest state container required rather than a new framework.

- [ ] **Step 4: Run the focused parity tests to verify GREEN**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py tests/unit/test_backtest_engine_qc_replay.py -k "multiple_cc or dynamic_max_positions" -q
```

Expected: the focused tests pass.

- [ ] **Step 5: Commit the parity-alignment replay change**

```bash
git add core/backtesting/strategies/binbin_god.py core/backtesting/engine.py tests/unit/test_binbin_qc_parity.py tests/unit/test_backtest_engine_qc_replay.py
git commit -m "fix(parity): align replay lifecycle with QC"
```

### Task 6: Run Full Targeted Regression Suite And Finalize

**Files:**
- Verify only

- [ ] **Step 1: Run the full targeted regression suite**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py tests/unit/test_qc_execution.py tests/unit/test_qc_strategy_mixin.py tests/unit/test_backtest_engine_qc_replay.py tests/unit/test_strategies.py tests/unit/test_roll_logic.py tests/unit/test_page_imports.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Inspect git status**

Run:

```bash
git status --short --branch
```

Expected: only intended tracked changes are present.

- [ ] **Step 3: Commit any remaining integration adjustments**

```bash
git add quantconnect core/backtesting tests/unit docs/superpowers/plans/2026-04-02-qc-parity-alignment.md
git commit -m "feat(quantconnect): align QC and parity lifecycle"
```

Only do this step if there are integration edits not already captured by earlier commits.
