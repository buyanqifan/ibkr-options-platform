# QC Covered Call Fallback Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tiered covered-call fallback ladder so underwater stock can still produce bounded rescue calls in QC and `binbin_god` parity replay.

**Architecture:** Extend QC covered-call selection from a single constraint set into an ordered ladder of progressively looser tiers. Keep QC and parity aligned by introducing the same fallback parameters in `strategy_init.py`, `qc_parity.py`, and replay call-generation logic, then verify both paths with focused red-green tests.

**Tech Stack:** Python, QuantConnect modules, parity backtest helpers, pytest

---

## File Map

- Modify: `quantconnect/strategy_init.py`
  Purpose: define and log the new covered-call fallback defaults.
- Modify: `quantconnect/option_selector.py`
  Purpose: evaluate a sequence of covered-call candidate tiers and return the winning tier.
- Modify: `quantconnect/signal_generation.py`
  Purpose: build the CC fallback ladder for QC holdings and repair-call flows.
- Modify: `core/backtesting/qc_parity.py`
  Purpose: mirror fallback defaults and config parsing for parity.
- Modify: `core/backtesting/strategies/binbin_god.py`
  Purpose: apply the same fallback ladder semantics to replay covered-call generation.
- Modify: `tests/unit/test_qc_execution.py`
  Purpose: QC-focused selection-tier tests.
- Modify: `tests/unit/test_binbin_qc_parity.py`
  Purpose: parity default sync and replay ladder regression tests.
- Modify: `tests/unit/test_qc_strategy_mixin.py`
  Purpose: keep rebalance / CC execution regressions green if needed after selector changes.

### Task 1: Add Red Tests for QC Covered-Call Fallback Tiers

**Files:**
- Modify: `tests/unit/test_qc_execution.py`
- Modify: `tests/unit/test_binbin_qc_parity.py`

- [ ] **Step 1: Add a failing QC selector-tier test**

Append this test to `tests/unit/test_qc_execution.py`:

```python
def test_find_option_by_greeks_uses_fallback_tier_when_primary_cc_constraints_fail(monkeypatch):
    class _ID:
        def __init__(self, strike, expiry, right):
            self.StrikePrice = strike
            self.Date = expiry
            self.OptionRight = right

    class _Symbol:
        def __init__(self, strike, expiry, right):
            self.ID = _ID(strike, expiry, right)

    expiry = datetime(2024, 6, 21)
    option_chain = [
        _Symbol(80.0, expiry, qc_execution.OptionRight.Call),
        _Symbol(95.0, expiry, qc_execution.OptionRight.Call),
    ]

    algo = SimpleNamespace(
        Securities={"NVDA": SimpleNamespace(Price=100.0)},
        OptionChainProvider=SimpleNamespace(GetOptionContractList=lambda *_args: option_chain),
        price_history={"NVDA": []},
        Time=datetime(2024, 6, 1),
    )

    monkeypatch.setattr(qc_execution, "calculate_historical_vol", lambda *_args, **_kwargs: 0.30)
    monkeypatch.setattr(qc_execution, "filter_option_by_itm_protection", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        qc_execution,
        "estimate_delta_from_moneyness",
        lambda strike, _price, _right: 0.31 if strike == 80.0 else 0.34,
    )

    result = qc_execution.find_option_by_greeks(
        algo,
        symbol="NVDA",
        equity_symbol="NVDA",
        target_right=qc_execution.OptionRight.Call,
        target_delta=0.30,
        dte_min=7,
        dte_max=21,
        delta_tolerance=0.08,
        min_strike=90.0,
        selection_tiers=[
            {"delta_tolerance": 0.08, "dte_min": 7, "dte_max": 21, "min_strike": 90.0, "tier": "primary"},
            {"delta_tolerance": 0.15, "dte_min": 7, "dte_max": 21, "min_strike": 80.0, "tier": "rescue"},
        ],
    )

    assert result["strike"] == 80.0
    assert result["selection_tier"] == "rescue"
```

- [ ] **Step 2: Run the selector-tier test to verify it fails**

Run:

```bash
pytest tests/unit/test_qc_execution.py::test_find_option_by_greeks_uses_fallback_tier_when_primary_cc_constraints_fail -q
```

Expected: `FAIL` because `find_option_by_greeks()` does not accept `selection_tiers` or return `selection_tier` yet.

- [ ] **Step 3: Add a failing parity-default sync test**

Append this test to `tests/unit/test_binbin_qc_parity.py`:

```python
def test_qc_parity_config_exposes_cc_fallback_defaults():
    config = BinbinGodParityConfig.from_params({"parity_mode": "qc"})

    assert config.cc_fallback_delta_tolerance_1 == pytest.approx(0.12)
    assert config.cc_fallback_delta_tolerance_2 == pytest.approx(0.15)
    assert config.cc_fallback_dte_min == 14
    assert config.cc_fallback_dte_max == 30
    assert config.cc_fallback_min_cost_basis_ratio == pytest.approx(0.85)
```

- [ ] **Step 4: Run the parity-default sync test to verify it fails**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py::test_qc_parity_config_exposes_cc_fallback_defaults -q
```

Expected: `FAIL` because parity config does not expose the new fallback fields yet.

- [ ] **Step 5: Add a failing replay rescue-call test**

Append this test to `tests/unit/test_binbin_qc_parity.py`:

```python
def test_replay_call_signal_uses_rescue_tier_when_near_cost_call_is_unavailable(monkeypatch):
    strategy = BinbinGodStrategy(
        initial_capital=100000,
        symbol="NVDA",
        config={
            "parity_mode": "qc",
            "contract_universe_mode": "qc_emulated_lattice",
            "cc_fallback_delta_tolerance_1": 0.12,
            "cc_fallback_delta_tolerance_2": 0.15,
            "cc_fallback_dte_min": 14,
            "cc_fallback_dte_max": 30,
            "cc_fallback_min_cost_basis_ratio": 0.85,
        },
    )
    strategy.stock_holding.add_shares("NVDA", 100, 120.0)

    rescue_contract = SimpleNamespace(
        strike=102.0,
        expiry=datetime(2024, 6, 21),
        dte=20,
        premium=1.5,
        delta=0.34,
        to_dict=lambda: {"selection_tier": "rescue"},
    )

    calls = []

    def fake_select_contract_from_lattice(**kwargs):
        calls.append(kwargs)
        if kwargs.get("min_strike", 0) >= 110:
            return None
        return rescue_contract

    monkeypatch.setattr(binbin_god_module, "select_contract_from_lattice", fake_select_contract_from_lattice)

    signals = strategy._generate_backtest_call_signal(
        symbol="NVDA",
        current_date="2024-06-01",
        underlying_price=100.0,
        iv=0.30,
        shares_available=100,
        cost_basis=120.0,
    )

    assert signals
    assert signals[0].metadata["selection_tier"] == "rescue"
```

- [ ] **Step 6: Run the replay rescue-call test to verify it fails**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py::test_replay_call_signal_uses_rescue_tier_when_near_cost_call_is_unavailable -q
```

Expected: `FAIL` because replay call generation only tries a single near-cost constraint set today.

- [ ] **Step 7: Commit the red tests**

```bash
git add tests/unit/test_qc_execution.py tests/unit/test_binbin_qc_parity.py
git commit -m "test(qc): cover covered call fallback ladder"
```

### Task 2: Implement QC Covered-Call Fallback Defaults and Selector Ladder

**Files:**
- Modify: `quantconnect/strategy_init.py`
- Modify: `quantconnect/option_selector.py`
- Modify: `quantconnect/signal_generation.py`
- Test: `tests/unit/test_qc_execution.py`

- [ ] **Step 1: Add fallback defaults to QC runtime parameters**

In `quantconnect/strategy_init.py:init_parameters()`, add:

```python
algo.cc_fallback_delta_tolerance_1 = _as_float(_get_param(algo, "cc_fallback_delta_tolerance_1", 0.12), 0.12)
algo.cc_fallback_delta_tolerance_2 = _as_float(_get_param(algo, "cc_fallback_delta_tolerance_2", 0.15), 0.15)
algo.cc_fallback_dte_min = _as_int(_get_param(algo, "cc_fallback_dte_min", 14), 14)
algo.cc_fallback_dte_max = _as_int(_get_param(algo, "cc_fallback_dte_max", 30), 30)
algo.cc_fallback_min_cost_basis_ratio = _as_float(_get_param(algo, "cc_fallback_min_cost_basis_ratio", 0.85), 0.85)
```

Also add them to the `EFFECTIVE_PARAMS:` log line in `log_effective_parameters()`.

- [ ] **Step 2: Extend `find_option_by_greeks()` to accept a tier list**

In `quantconnect/option_selector.py`, change the signature to:

```python
def find_option_by_greeks(
    algo,
    symbol: str,
    equity_symbol,
    target_right,
    target_delta: float,
    dte_min: int,
    dte_max: int,
    delta_tolerance: float = 0.10,
    min_strike: float = None,
    selection_tiers: Optional[list[dict]] = None,
) -> Optional[Dict]:
```

Build default tiers from the legacy arguments when `selection_tiers is None`, then evaluate each tier in order. When a contract matches, return:

```python
return {
    **best_contract,
    "selection_tier": tier_name,
}
```

- [ ] **Step 3: Build the QC CC ladder in `signal_generation.py`**

In `generate_signal_for_symbol()` under the `strategy_phase == "CC"` path, assemble:

```python
cc_selection_tiers = [
    {
        "tier": "primary",
        "delta_tolerance": 0.08,
        "dte_min": signal.dte_min,
        "dte_max": signal.dte_max,
        "min_strike": cc_min_strike,
    },
    {
        "tier": "delta_fallback",
        "delta_tolerance": algo.cc_fallback_delta_tolerance_1,
        "dte_min": signal.dte_min,
        "dte_max": signal.dte_max,
        "min_strike": cc_min_strike,
    },
    {
        "tier": "dte_fallback",
        "delta_tolerance": algo.cc_fallback_delta_tolerance_1,
        "dte_min": algo.cc_fallback_dte_min,
        "dte_max": algo.cc_fallback_dte_max,
        "min_strike": cc_min_strike,
    },
]
```

If repair mode is active and `cost_basis > 0`, add the rescue tier:

```python
rescue_min_strike = max(
    underlying_price * 1.01,
    cost_basis * algo.cc_fallback_min_cost_basis_ratio,
)
cc_selection_tiers.append(
    {
        "tier": "rescue",
        "delta_tolerance": algo.cc_fallback_delta_tolerance_2,
        "dte_min": algo.cc_fallback_dte_min,
        "dte_max": algo.cc_fallback_dte_max,
        "min_strike": rescue_min_strike,
    }
)
signal.selection_tiers = cc_selection_tiers
```

- [ ] **Step 4: Pass ladder tiers from execution to option selection**

In `quantconnect/execution.py:execute_signal()`, change the `find_option_func(...)` call to pass:

```python
selection_tiers=getattr(signal, "selection_tiers", None),
```

After selection, if the result includes a tier, log it:

```python
if target_right == OptionRight.Call and selected.get("selection_tier"):
    algo.Log(f"CC_SELECTION_TIER:{signal.symbol}:{selected['selection_tier']}")
```

- [ ] **Step 5: Run focused QC tests**

Run:

```bash
pytest tests/unit/test_qc_execution.py::test_find_option_by_greeks_uses_fallback_tier_when_primary_cc_constraints_fail -q
```

Expected: `PASS`

- [ ] **Step 6: Commit the QC ladder implementation**

```bash
git add quantconnect/strategy_init.py quantconnect/option_selector.py quantconnect/signal_generation.py quantconnect/execution.py tests/unit/test_qc_execution.py
git commit -m "feat(qc): add covered call fallback ladder"
```

### Task 3: Mirror the Ladder in Parity / BinbinGod Replay

**Files:**
- Modify: `core/backtesting/qc_parity.py`
- Modify: `core/backtesting/strategies/binbin_god.py`
- Modify: `tests/unit/test_binbin_qc_parity.py`

- [ ] **Step 1: Add parity defaults and config parsing**

In `core/backtesting/qc_parity.py`, add the new keys to:

```python
_QC_PARAMETER_FALLBACKS
QC_BINBIN_DEFAULTS
BinbinGodParityConfig
BinbinGodParityConfig.from_params()
BinbinGodParityConfig.to_params()
```

Use the same defaults as QC runtime:

```python
"cc_fallback_delta_tolerance_1": 0.12,
"cc_fallback_delta_tolerance_2": 0.15,
"cc_fallback_dte_min": 14,
"cc_fallback_dte_max": 30,
"cc_fallback_min_cost_basis_ratio": 0.85,
```

- [ ] **Step 2: Add replay fallback-tier selection**

In `core/backtesting/strategies/binbin_god.py:_generate_backtest_call_signal()`, replace the single parity contract lookup with an ordered tier loop:

```python
selection_tiers = [
    {"tier": "primary", "delta_tolerance": 0.05, "dte_min": dte_window_min, "dte_max": dte_window_max, "min_strike": min_strike},
    {"tier": "delta_fallback", "delta_tolerance": self.cc_fallback_delta_tolerance_1, "dte_min": dte_window_min, "dte_max": dte_window_max, "min_strike": min_strike},
    {"tier": "dte_fallback", "delta_tolerance": self.cc_fallback_delta_tolerance_1, "dte_min": self.cc_fallback_dte_min, "dte_max": self.cc_fallback_dte_max, "min_strike": min_strike},
    {"tier": "rescue", "delta_tolerance": self.cc_fallback_delta_tolerance_2, "dte_min": self.cc_fallback_dte_min, "dte_max": self.cc_fallback_dte_max, "min_strike": rescue_min_strike},
]
```

Try each tier in order. On success:

```python
contract_metadata = {**selected_contract.to_dict(), "selection_tier": tier["tier"]}
```

On total failure, keep the existing deferred record, but include:

```python
"reason": "no_lattice_contract_after_fallback",
```

- [ ] **Step 3: Run focused parity tests**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py::test_qc_parity_config_exposes_cc_fallback_defaults tests/unit/test_binbin_qc_parity.py::test_replay_call_signal_uses_rescue_tier_when_near_cost_call_is_unavailable -q
```

Expected: `2 passed`

- [ ] **Step 4: Run the full regression slice**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py tests/unit/test_qc_execution.py tests/unit/test_qc_strategy_mixin.py tests/unit/test_backtest_engine_qc_replay.py tests/unit/test_strategies.py tests/unit/test_roll_logic.py tests/unit/test_page_imports.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the parity sync**

```bash
git add core/backtesting/qc_parity.py core/backtesting/strategies/binbin_god.py tests/unit/test_binbin_qc_parity.py
git commit -m "feat(parity): align covered call fallback ladder"
```
