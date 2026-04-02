# QC Return Rebalance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise QC return potential by moderately reopening risk budgets and widening option matching tolerance, while keeping the recent assignment fail-safe and signal-quality protections intact.

**Architecture:** This change stays deliberately narrow. Update `quantconnect/strategy_init.py` runtime defaults first, then mirror the same defaults into `core/backtesting/qc_parity.py` so QC replay and page defaults remain aligned, and finally widen the execution-layer `delta_tolerance` in `quantconnect/execution.py` with a focused unit test that locks the new behavior in place.

**Tech Stack:** Python, pytest, QuantConnect strategy modules, Dash QC parity defaults

---

### Task 1: Raise QC Runtime Defaults In `strategy_init.py`

**Files:**
- Modify: `quantconnect/strategy_init.py`
- Test: `tests/unit/test_strategies.py`

- [ ] **Step 1: Write the failing test**

Update `tests/unit/test_strategies.py` inside `test_native_defaults_follow_qc_default_config` so the expected QC-aligned defaults match the approved "return rebalance" spec:

```python
def test_native_defaults_follow_qc_default_config(self):
    strategy = BinbinGodStrategy({"symbol": "NVDA"})

    assert strategy.initial_capital == 300000
    assert strategy.max_positions == 20
    assert strategy.profit_target_pct == 70
    assert strategy.margin_buffer_pct == 0.40
    assert strategy.target_margin_utilization == pytest.approx(0.58)
    assert strategy.position_aggressiveness == pytest.approx(1.35)
    assert strategy.symbol_assignment_base_cap == pytest.approx(0.36)
    assert strategy.stock_inventory_base_cap == pytest.approx(0.24)
    assert strategy.stock_inventory_block_threshold == pytest.approx(0.92)
    assert strategy.max_risk_per_trade == pytest.approx(0.03)
    assert strategy.max_assignment_risk_per_trade == pytest.approx(0.25)
    assert strategy.max_new_puts_per_day == 3
    assert strategy._is_qc_parity_enabled() is True
    assert strategy.contract_universe_mode == "qc_emulated_lattice"
```

- [ ] **Step 2: Run the targeted test to confirm it fails**

Run:

```bash
pytest tests/unit/test_strategies.py::TestBinbinGodStrategy::test_native_defaults_follow_qc_default_config -q
```

Expected: `FAIL` because `strategy_init.py` still uses `0.50`, `1.2`, `0.28`, `0.17`, `0.85`, and `0.20`.

- [ ] **Step 3: Write the minimal implementation**

Update the default literals in `quantconnect/strategy_init.py`:

```python
algo.max_assignment_risk_per_trade = _as_float(_get_param(algo, "max_assignment_risk_per_trade", 0.25), 0.25)
algo.position_aggressiveness = _clamp(_as_float(_get_param(algo, "position_aggressiveness", 1.35), 1.35), 0.3, 2.0)

default_symbol_assignment_cap = 0.36
algo.symbol_assignment_base_cap = _clamp(
    _as_float(_get_param(algo, "symbol_assignment_base_cap", default_symbol_assignment_cap), default_symbol_assignment_cap),
    0.05,
    1.5,
)
algo.stock_inventory_base_cap = _clamp(_as_float(_get_param(algo, "stock_inventory_base_cap", 0.24), 0.24), 0.05, 1.0)
algo.stock_inventory_block_threshold = _clamp(_as_float(_get_param(algo, "stock_inventory_block_threshold", 0.92), 0.92), 0.50, 1.20)
algo.target_margin_utilization = _as_float(_get_param(algo, "target_margin_utilization", 0.58), 0.58)
```

Do not change:

- assigned-stock fail-safe defaults
- signal-quality filter thresholds
- scoring weights

- [ ] **Step 4: Re-run the targeted test to verify it passes**

Run:

```bash
pytest tests/unit/test_strategies.py::TestBinbinGodStrategy::test_native_defaults_follow_qc_default_config -q
```

Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add quantconnect/strategy_init.py tests/unit/test_strategies.py
git commit -m "feat(quantconnect): raise QC runtime defaults"
```

### Task 2: Sync Parity And Page Defaults To The New QC Values

**Files:**
- Modify: `core/backtesting/qc_parity.py`
- Modify: `tests/unit/test_binbin_qc_parity.py`
- Modify: `tests/unit/test_binbin_god_page.py`

- [ ] **Step 1: Write the failing tests**

Update the QC parity expectations in `tests/unit/test_binbin_qc_parity.py`:

```python
def test_qc_parity_config_uses_qc_defaults():
    config = BinbinGodParityConfig.from_params({"parity_mode": "qc"})
    assert config.position_aggressiveness == pytest.approx(1.35)
    assert config.target_margin_utilization == pytest.approx(0.58)
    assert config.max_assignment_risk_per_trade == pytest.approx(0.25)
    assert config.stock_inventory_base_cap == pytest.approx(0.24)
    assert config.stock_inventory_block_threshold == pytest.approx(0.92)
    assert config.symbol_assignment_base_cap == pytest.approx(0.36)

def test_extract_strategy_init_parameter_defaults_reads_qc_source_defaults():
    defaults = _extract_strategy_init_parameter_defaults()
    assert defaults["target_margin_utilization"] == pytest.approx(0.58)
    assert defaults["max_assignment_risk_per_trade"] == pytest.approx(0.25)
    assert defaults["symbol_assignment_base_cap"] == pytest.approx(0.36)
    assert defaults["stock_inventory_base_cap"] == pytest.approx(0.24)
    assert defaults["stock_inventory_block_threshold"] == pytest.approx(0.92)

def test_qc_parameter_defaults_merge_config_and_strategy_init_sources():
    assert QC_PARAMETER_DEFAULTS["max_assignment_risk_per_trade"] == pytest.approx(0.25)
    assert QC_PARAMETER_DEFAULTS["target_margin_utilization"] == pytest.approx(0.58)
    assert QC_PARAMETER_DEFAULTS["stock_inventory_base_cap"] == pytest.approx(0.24)
    assert QC_PARAMETER_DEFAULTS["symbol_assignment_base_cap"] == pytest.approx(0.36)
```

Update page-default expectations in `tests/unit/test_binbin_god_page.py`:

```python
def _default_form_inputs() -> dict[str, Any]:
    return {
        "target_margin_utilization": 0.58,
        "position_aggressiveness": 1.35,
        "symbol_assignment_base_cap": 0.36,
        "stock_inventory_base_cap": 0.24,
        "stock_inventory_block_threshold": 0.92,
        ...
    }

def test_build_binbin_backtest_params_uses_qc_defaults(monkeypatch):
    params = page.build_binbin_backtest_params(_default_form_inputs())
    assert params["target_margin_utilization"] == 0.58
    assert params["position_aggressiveness"] == 1.35
    assert params["symbol_assignment_base_cap"] == 0.36
    assert params["stock_inventory_base_cap"] == 0.24
    assert params["stock_inventory_block_threshold"] == 0.92
```

- [ ] **Step 2: Run the parity and page tests to confirm they fail**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py tests/unit/test_binbin_god_page.py -q
```

Expected: `FAIL` on the old default-value assertions.

- [ ] **Step 3: Write the minimal implementation**

Update `_QC_PARAMETER_FALLBACKS` in `core/backtesting/qc_parity.py` so fallback defaults match the new QC runtime values:

```python
_QC_PARAMETER_FALLBACKS = {
    ...
    "target_margin_utilization": 0.58,
    "position_aggressiveness": 1.35,
    "max_assignment_risk_per_trade": 0.25,
    "symbol_assignment_base_cap": 0.36,
    "stock_inventory_base_cap": 0.24,
    "stock_inventory_block_threshold": 0.92,
    ...
}
```

Do not rewrite the `QC_BINBIN_DEFAULTS` plumbing. It already fans out from `QC_PARAMETER_DEFAULTS`; the fallback values are the only part that needs explicit sync here.

- [ ] **Step 4: Re-run the parity and page tests to verify they pass**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py tests/unit/test_binbin_god_page.py -q
```

Expected: all selected tests pass with the new defaults reflected in parity config and page payload expectations.

- [ ] **Step 5: Commit**

```bash
git add core/backtesting/qc_parity.py tests/unit/test_binbin_qc_parity.py tests/unit/test_binbin_god_page.py
git commit -m "feat(backtesting): sync QC return defaults"
```

### Task 3: Widen QC Execution Delta Tolerance And Lock It With A Focused Test

**Files:**
- Modify: `quantconnect/execution.py`
- Create: `tests/unit/test_qc_execution.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_qc_execution.py` with a focused test that proves `execute_signal()` passes `delta_tolerance=0.08` to the option finder:

```python
from types import SimpleNamespace

import pytest

from quantconnect.execution import execute_signal
from quantconnect.ml_integration import StrategySignal
import quantconnect.execution as qc_execution


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

    execute_signal(algo, signal, fake_find_option)

    assert captured["delta_tolerance"] == pytest.approx(0.08)
```

- [ ] **Step 2: Run the new test to confirm it fails**

Run:

```bash
pytest tests/unit/test_qc_execution.py::test_execute_signal_uses_rebalanced_delta_tolerance -q
```

Expected: `FAIL` because `execute_signal()` still passes `0.05`.

- [ ] **Step 3: Write the minimal implementation**

Change the execution call site in `quantconnect/execution.py`:

```python
selected = find_option_func(
    algo,
    symbol=signal.symbol,
    equity_symbol=equity.Symbol,
    target_right=target_right,
    target_delta=target_delta,
    dte_min=signal.dte_min,
    dte_max=signal.dte_max,
    delta_tolerance=0.08,
    min_strike=min_strike if min_strike > 0 else None,
)
```

Do not change log messages or signal-generation logic in this task.

- [ ] **Step 4: Run focused and broad verification**

Run:

```bash
pytest tests/unit/test_qc_execution.py::test_execute_signal_uses_rebalanced_delta_tolerance -q
pytest tests/unit/test_binbin_qc_parity.py tests/unit/test_binbin_god_page.py tests/unit/test_strategies.py tests/unit/test_roll_logic.py tests/unit/test_page_imports.py tests/unit/test_qc_execution.py -q
```

Expected:

- first command: `1 passed`
- second command: full selected suite passes

- [ ] **Step 5: Commit**

```bash
git add quantconnect/execution.py tests/unit/test_qc_execution.py
git commit -m "feat(quantconnect): widen QC option matching tolerance"
```
