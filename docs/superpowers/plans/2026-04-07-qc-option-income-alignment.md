# QC Option-Income Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make QC and `binbin_god / parity` generate and keep covered-call income more reliably by wiring the real four-tier CC ladder into the active execution path, adding a repair-mode rules fallback, loosening over-eager assignment stock exits, and removing redundant CC selection code.

**Architecture:** The QuantConnect runtime will use one authoritative CC ladder builder plus a narrow rules-based fallback for repair / assigned-stock contexts. Parity replay will mirror the same ladder semantics and relaxed repair timing defaults so return composition stays aligned. Redundant two-tier helpers will be removed or made to delegate to the authoritative path.

**Tech Stack:** Python, QuantConnect algorithm modules, local parity backtest engine, pytest

---

### Task 1: Lock in the new QC repair-call behavior with failing tests

**Files:**
- Modify: `tests/unit/test_qc_strategy_mixin.py`
- Modify: `tests/unit/test_qc_execution.py`
- Modify: `tests/unit/test_binbin_qc_parity.py`
- Modify: `tests/unit/test_strategies.py`

- [ ] **Step 1: Add a failing QC test proving repair-mode CC can bypass the ML confidence gate**

```python
def test_rebalance_executes_repair_cc_even_below_ml_gate(monkeypatch):
    algo = _make_algo()
    cc_signal = SimpleNamespace(
        action="SELL_CALL",
        symbol="NVDA",
        delta=0.35,
        confidence=0.10,
        reasoning="rules fallback",
        metadata={"inventory_mode": "repair", "rules_fallback": True},
    )
    algo.ml_min_confidence = 0.45

    monkeypatch.setattr(qc_strategy_mixin, "calculate_dynamic_max_positions", lambda _algo: 3)
    monkeypatch.setattr(qc_strategy_mixin, "check_position_management", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(qc_strategy_mixin, "retry_pending_open_orders", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(qc_strategy_mixin, "generate_ml_signals", lambda _algo: [cc_signal])
    monkeypatch.setattr(qc_strategy_mixin, "get_option_position_count", lambda _algo: 0)

    executed = []
    monkeypatch.setattr(
        qc_strategy_mixin,
        "execute_signal",
        lambda _algo, signal, _finder: executed.append((signal.action, signal.symbol, signal.confidence)),
    )

    qc_strategy_mixin.rebalance(algo)

    assert executed == [("SELL_CALL", "NVDA", 0.10)]
```

- [ ] **Step 2: Add a failing QC execution test proving the four-tier ladder reaches rescue tier**

```python
def test_find_option_by_greeks_reaches_rescue_discount_tier(monkeypatch):
    algo = _make_algo()
    equity_symbol = Symbol.Create("NVDA", SecurityType.Equity, Market.USA)

    def fake_find_constraints(
        algo,
        symbol,
        equity_symbol,
        target_right,
        target_delta,
        dte_min,
        dte_max,
        delta_tolerance,
        min_strike,
    ):
        if dte_min == 14 and delta_tolerance == 0.15:
            return {
                "option_symbol": Symbol.CreateOption("NVDA", Market.USA, OptionStyle.American, OptionRight.Call, 128, datetime(2024, 2, 16)),
                "strike": 128.0,
                "expiry": datetime(2024, 2, 16),
                "dte": 24,
                "delta": 0.27,
                "premium": 1.35,
            }
        return None

    monkeypatch.setattr(qc_execution, "_find_option_by_constraints", fake_find_constraints)

    selected = qc_execution.find_option_by_greeks(
        algo,
        symbol="NVDA",
        equity_symbol=equity_symbol,
        target_right=OptionRight.Call,
        target_delta=0.35,
        dte_min=7,
        dte_max=21,
        delta_tolerance=0.08,
        min_strike=140.0,
        selection_tiers=[
            {"label": "primary", "delta_tolerance": 0.08, "dte_min": 7, "dte_max": 21, "min_strike": 140.0},
            {"label": "fallback_delta", "delta_tolerance": 0.12, "dte_min": 7, "dte_max": 21, "min_strike": 140.0},
            {"label": "fallback_dte", "delta_tolerance": 0.12, "dte_min": 14, "dte_max": 30, "min_strike": 140.0},
            {"label": "rescue_discount", "delta_tolerance": 0.15, "dte_min": 14, "dte_max": 30, "min_strike": 128.0},
        ],
    )

    assert selected["selection_tier"] == "rescue_discount"
    assert selected["strike"] == pytest.approx(128.0)
```

- [ ] **Step 3: Add a failing parity test proving the active CC ladder exposes all four tiers**

```python
def test_build_cc_selection_tiers_qc_exposes_four_tiers():
    config = BinbinGodParityConfig.from_params({})
    tiers = build_cc_selection_tiers_qc(
        config=config,
        underlying_price=120.0,
        cost_basis=150.0,
        primary_dte_min=7,
        primary_dte_max=21,
        primary_delta_tolerance=0.08,
        primary_min_strike=145.5,
    )

    assert [tier["label"] for tier in tiers] == [
        "primary",
        "fallback_delta",
        "fallback_dte",
        "rescue_discount",
    ]
    assert tiers[-1]["min_strike"] == pytest.approx(max(120.0 * 1.01, 150.0 * 0.85))
```

- [ ] **Step 4: Add a failing fail-safe timing regression**

```python
def test_assigned_stock_fail_safe_does_not_exit_on_first_cc_miss(monkeypatch):
    algo = _make_algo()
    algo.assigned_stock_fail_safe_enabled = True
    algo.assigned_stock_cc_miss_limit = 3
    algo.assigned_stock_state = {
        "NVDA": {
            "assignment_date": datetime(2024, 1, 10),
            "assignment_cost_basis": 150.0,
            "repair_deadline": datetime(2024, 1, 20),
            "cc_miss_count": 0,
            "force_exit_triggered": False,
            "inventory_mode": "repair",
        }
    }

    monkeypatch.setattr(position_management, "get_shares_held", lambda *_args, **_kwargs: 200)
    monkeypatch.setattr(position_management, "get_call_position_contracts", lambda *_args, **_kwargs: 0)

    orders = []
    algo.MarketOrder = lambda symbol, quantity: orders.append((symbol, quantity))

    position_management._manage_assigned_stock_fail_safe(algo)

    assert orders == []
    assert algo.assigned_stock_state["NVDA"]["cc_miss_count"] == 1
    assert algo.assigned_stock_state["NVDA"]["force_exit_triggered"] is False
```

- [ ] **Step 5: Run the focused slice and verify RED**

Run:

```bash
pytest tests/unit/test_qc_strategy_mixin.py tests/unit/test_qc_execution.py tests/unit/test_binbin_qc_parity.py tests/unit/test_strategies.py -q
```

Expected:

```text
FAIL test_rebalance_executes_repair_cc_even_below_ml_gate
FAIL test_find_option_by_greeks_reaches_rescue_discount_tier
FAIL test_build_cc_selection_tiers_qc_exposes_four_tiers
FAIL test_assigned_stock_fail_safe_does_not_exit_on_first_cc_miss
```

- [ ] **Step 6: Commit the failing-test scaffold**

```bash
git add tests/unit/test_qc_strategy_mixin.py tests/unit/test_qc_execution.py tests/unit/test_binbin_qc_parity.py tests/unit/test_strategies.py
git commit -m "test: cover QC option-income alignment regressions"
```

### Task 2: Wire one authoritative four-tier CC ladder into QC

**Files:**
- Modify: `quantconnect/signals.py`
- Modify: `quantconnect/signal_generation.py`
- Modify: `quantconnect/strategy_init.py`
- Modify: `tests/unit/test_qc_execution.py`
- Modify: `tests/unit/test_qc_strategy_mixin.py`

- [ ] **Step 1: Move the ladder builder to `quantconnect/signals.py` and make it authoritative**

```python
def build_cc_selection_tiers(
    *,
    underlying_price: float,
    cost_basis: float,
    primary_dte_min: int,
    primary_dte_max: int,
    primary_delta_tolerance: float,
    primary_min_strike: Optional[float],
    fallback_delta_tolerance_1: float,
    fallback_delta_tolerance_2: float,
    fallback_dte_min: int,
    fallback_dte_max: int,
    fallback_min_cost_basis_ratio: float,
) -> List[Dict[str, float]]:
    if cost_basis <= 0 or primary_min_strike is None:
        return []

    rescue_min_strike = max(underlying_price * 1.01, cost_basis * fallback_min_cost_basis_ratio)
    return [
        {"label": "primary", "delta_tolerance": primary_delta_tolerance, "dte_min": primary_dte_min, "dte_max": primary_dte_max, "min_strike": primary_min_strike},
        {"label": "fallback_delta", "delta_tolerance": fallback_delta_tolerance_1, "dte_min": primary_dte_min, "dte_max": primary_dte_max, "min_strike": primary_min_strike},
        {"label": "fallback_dte", "delta_tolerance": fallback_delta_tolerance_1, "dte_min": fallback_dte_min, "dte_max": fallback_dte_max, "min_strike": primary_min_strike},
        {"label": "rescue_discount", "delta_tolerance": fallback_delta_tolerance_2, "dte_min": fallback_dte_min, "dte_max": fallback_dte_max, "min_strike": rescue_min_strike},
    ]
```

- [ ] **Step 2: Remove the duplicate two-tier builder from `quantconnect/signal_generation.py` and replace it with a call into `signals.build_cc_selection_tiers`**

```python
from signals import build_cc_selection_tiers
from signals import score_single_stock
from debug_counters import increment_debug_counter

if strategy_phase == "CC":
    signal.selection_tiers = build_cc_selection_tiers(
        underlying_price=underlying_price,
        cost_basis=cost_basis,
        primary_dte_min=signal.dte_min,
        primary_dte_max=signal.dte_max,
        primary_delta_tolerance=cc_mode["primary_tolerance"] if cc_mode else 0.08,
        primary_min_strike=cc_min_strike,
        fallback_delta_tolerance_1=algo.cc_fallback_delta_tolerance_1,
        fallback_delta_tolerance_2=algo.cc_fallback_delta_tolerance_2,
        fallback_dte_min=algo.cc_fallback_dte_min,
        fallback_dte_max=algo.cc_fallback_dte_max,
        fallback_min_cost_basis_ratio=algo.cc_fallback_min_cost_basis_ratio,
    )
```

- [ ] **Step 3: Add the ladder defaults to `strategy_init.py` and log them**

```python
algo.cc_fallback_delta_tolerance_1 = _clamp(_as_float(_get_param(algo, "cc_fallback_delta_tolerance_1", 0.12), 0.12), 0.08, 0.30)
algo.cc_fallback_delta_tolerance_2 = _clamp(_as_float(_get_param(algo, "cc_fallback_delta_tolerance_2", 0.15), 0.15), algo.cc_fallback_delta_tolerance_1, 0.40)
algo.cc_fallback_dte_min = _as_int(_get_param(algo, "cc_fallback_dte_min", 14), 14)
algo.cc_fallback_dte_max = _as_int(_get_param(algo, "cc_fallback_dte_max", 30), 30)
algo.cc_fallback_min_cost_basis_ratio = _clamp(_as_float(_get_param(algo, "cc_fallback_min_cost_basis_ratio", 0.85), 0.85), 0.50, 1.0)
```

- [ ] **Step 4: Run the focused QC tests and verify GREEN**

Run:

```bash
pytest tests/unit/test_qc_execution.py tests/unit/test_qc_strategy_mixin.py tests/unit/test_binbin_qc_parity.py -q
```

Expected:

```text
PASS
```

- [ ] **Step 5: Commit the ladder integration**

```bash
git add quantconnect/signals.py quantconnect/signal_generation.py quantconnect/strategy_init.py tests/unit/test_qc_execution.py tests/unit/test_qc_strategy_mixin.py tests/unit/test_binbin_qc_parity.py
git commit -m "feat(qc): wire four-tier covered call ladder"
```

### Task 3: Add rules-based repair-call fallback and relax QC assignment exits

**Files:**
- Modify: `quantconnect/signal_generation.py`
- Modify: `quantconnect/strategy_mixin.py`
- Modify: `quantconnect/expiry.py`
- Modify: `quantconnect/position_management.py`
- Modify: `quantconnect/strategy_init.py`
- Modify: `tests/unit/test_qc_strategy_mixin.py`
- Modify: `tests/unit/test_strategies.py`

- [ ] **Step 1: Add a narrow rules fallback path for repair / assigned-stock CC generation**

```python
def _build_rules_fallback_cc_signal(algo, symbol: str, cc_mode: Dict, cost_basis: float, underlying_price: float) -> StrategySignal:
    signal = StrategySignal(
        symbol=symbol,
        action="SELL_CALL",
        delta=float(cc_mode["delta"]),
        dte_min=int(cc_mode["dte_min"]),
        dte_max=int(cc_mode["dte_max"]),
        confidence=0.0,
        reasoning="rules fallback repair call",
        expected_premium=0.0,
        expected_return=0.0,
        expected_risk=0.0,
        assignment_probability=0.0,
    )
    signal.min_strike = cc_mode["min_strike"]
    signal.selection_tiers = build_cc_selection_tiers(
        underlying_price=underlying_price,
        cost_basis=cost_basis,
        primary_dte_min=int(cc_mode["dte_min"]),
        primary_dte_max=int(cc_mode["dte_max"]),
        primary_delta_tolerance=cc_mode.get("primary_tolerance", 0.08),
        primary_min_strike=cc_mode["min_strike"],
        fallback_delta_tolerance_1=algo.cc_fallback_delta_tolerance_1,
        fallback_delta_tolerance_2=algo.cc_fallback_delta_tolerance_2,
        fallback_dte_min=algo.cc_fallback_dte_min,
        fallback_dte_max=algo.cc_fallback_dte_max,
        fallback_min_cost_basis_ratio=algo.cc_fallback_min_cost_basis_ratio,
    )
    signal.metadata = {"inventory_mode": cc_mode["label"], "rules_fallback": True}
    return signal
```

- [ ] **Step 2: Use the rules fallback in `generate_signal_for_symbol` when `strategy_phase == "CC"` and ML does not return a usable signal**

```python
if strategy_phase == "CC" and signal is None and cc_mode and cc_mode["label"] in {"repair", "income"}:
    signal = _build_rules_fallback_cc_signal(algo, symbol, cc_mode, cost_basis, underlying_price)
```

- [ ] **Step 3: Allow repair-mode / rules-fallback CCs to bypass the normal ML confidence gate in `strategy_mixin.rebalance`**

```python
def _can_execute_cc_signal(algo, signal) -> bool:
    metadata = getattr(signal, "metadata", {}) or {}
    if metadata.get("rules_fallback") or metadata.get("inventory_mode") in {"repair", "income"}:
        return True
    return signal.confidence >= algo.ml_min_confidence

if cc_signals:
    for cc_signal in sorted(cc_signals, key=lambda x: x.confidence, reverse=True):
        increment_debug_counter(algo, "cc_signals")
        if _can_execute_cc_signal(algo, cc_signal):
            execute_signal(algo, cc_signal, find_option_by_greeks)
        else:
            increment_debug_counter(algo, "cc_confidence_block")
```

- [ ] **Step 4: Relax the assignment repair timing defaults and fail-safe logic**

```python
algo.assigned_stock_max_repair_days = max(1, _as_int(_get_param(algo, "assigned_stock_max_repair_days", 7), 7))
algo.assigned_stock_cc_miss_limit = max(1, _as_int(_get_param(algo, "assigned_stock_cc_miss_limit", 3), 3))
```

```python
elif state["cc_miss_count"] >= getattr(algo, "assigned_stock_cc_miss_limit", 3):
    if days_held >= getattr(algo, "assigned_stock_min_days_held", 5):
        should_exit = True
        exit_reason = "cc_miss_limit"
```

- [ ] **Step 5: Run the QC repair-flow tests**

Run:

```bash
pytest tests/unit/test_qc_strategy_mixin.py tests/unit/test_strategies.py -q
```

Expected:

```text
PASS
```

- [ ] **Step 6: Commit the repair-flow changes**

```bash
git add quantconnect/signal_generation.py quantconnect/strategy_mixin.py quantconnect/expiry.py quantconnect/position_management.py quantconnect/strategy_init.py tests/unit/test_qc_strategy_mixin.py tests/unit/test_strategies.py
git commit -m "feat(qc): add repair call rules fallback"
```

### Task 4: Sync parity / replay and remove redundant CC helpers

**Files:**
- Modify: `core/backtesting/qc_parity.py`
- Modify: `core/backtesting/strategies/binbin_god.py`
- Modify: `tests/unit/test_binbin_qc_parity.py`
- Modify: `tests/unit/test_backtest_engine_qc_replay.py`
- Modify: `tests/unit/test_strategies.py`

- [ ] **Step 1: Expand `build_cc_selection_tiers_qc` in parity to the same four-tier ladder and sync drifted defaults**

```python
def build_cc_selection_tiers_qc(
    *,
    config: BinbinGodParityConfig,
    underlying_price: float,
    cost_basis: float,
    primary_dte_min: int,
    primary_dte_max: int,
    primary_delta_tolerance: float,
    primary_min_strike: float | None,
) -> List[Dict[str, float]]:
    rescue_min_strike = max(underlying_price * 1.01, cost_basis * config.cc_fallback_min_cost_basis_ratio)
    return [
        {"label": "primary", "delta_tolerance": primary_delta_tolerance, "dte_min": primary_dte_min, "dte_max": primary_dte_max, "min_strike": primary_min_strike},
        {"label": "fallback_delta", "delta_tolerance": config.cc_fallback_delta_tolerance_1, "dte_min": primary_dte_min, "dte_max": primary_dte_max, "min_strike": primary_min_strike},
        {"label": "fallback_dte", "delta_tolerance": config.cc_fallback_delta_tolerance_1, "dte_min": config.cc_fallback_dte_min, "dte_max": config.cc_fallback_dte_max, "min_strike": primary_min_strike},
        {"label": "rescue_discount", "delta_tolerance": config.cc_fallback_delta_tolerance_2, "dte_min": config.cc_fallback_dte_min, "dte_max": config.cc_fallback_dte_max, "min_strike": rescue_min_strike},
    ]
```

- [ ] **Step 2: Add the same ladder defaults and relaxed repair timing defaults to `QC_BINBIN_DEFAULTS` / `BinbinGodParityConfig`**

```python
"cc_fallback_delta_tolerance_1": 0.12,
"cc_fallback_delta_tolerance_2": 0.15,
"cc_fallback_dte_min": 14,
"cc_fallback_dte_max": 30,
"cc_fallback_min_cost_basis_ratio": 0.85,
"assigned_stock_max_repair_days": 7,
"assigned_stock_cc_miss_limit": 3,
```

- [ ] **Step 3: Make parity replay use the same active ladder and remove stale two-tier assumptions from tests**

```python
selection_tiers = build_cc_selection_tiers_qc(
    config=self.parity_config,
    underlying_price=underlying_price,
    cost_basis=cost_basis,
    primary_dte_min=dte_window_min,
    primary_dte_max=dte_window_max,
    primary_delta_tolerance=0.08,
    primary_min_strike=min_strike,
)
selected_contract = select_contract_from_lattice(
    symbol=symbol,
    current_date=current_date,
    underlying_price=underlying_price,
    iv=iv,
    target_right="C",
    target_delta=abs(final_delta),
    dte_min=dte_window_min,
    dte_max=dte_window_max,
    delta_tolerance=0.08,
    min_strike=min_strike,
    selection_tiers=selection_tiers or None,
)
```

- [ ] **Step 4: Run the parity regression slice**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py tests/unit/test_backtest_engine_qc_replay.py tests/unit/test_strategies.py -q
```

Expected:

```text
PASS
```

- [ ] **Step 5: Commit the parity alignment**

```bash
git add core/backtesting/qc_parity.py core/backtesting/strategies/binbin_god.py tests/unit/test_binbin_qc_parity.py tests/unit/test_backtest_engine_qc_replay.py tests/unit/test_strategies.py
git commit -m "feat(parity): align covered call repair with QC"
```

### Task 5: Run the full regression slice and finalize

**Files:**
- Verify only

- [ ] **Step 1: Run the full targeted regression slice**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py tests/unit/test_qc_execution.py tests/unit/test_qc_strategy_mixin.py tests/unit/test_backtest_engine_qc_replay.py tests/unit/test_strategies.py tests/unit/test_roll_logic.py tests/unit/test_page_imports.py -q
```

Expected:

```text
PASS
```

- [ ] **Step 2: Inspect the final branch state**

Run:

```bash
git status --short
git log --oneline -5
```

Expected:

```text
Working tree clean except intentional local artifacts
Recent commits include the test scaffold, QC ladder wiring, QC repair fallback, and parity alignment
```

- [ ] **Step 3: Hand off for completion**

After the regression slice is green, use the `finishing-a-development-branch` skill and present the standard four integration options.
