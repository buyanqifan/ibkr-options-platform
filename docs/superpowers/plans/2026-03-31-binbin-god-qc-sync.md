# Binbin God QC Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sync current QuantConnect BinbinGod behavior, parameters, and parity sizing rules into the self-built `binbin_god` strategy without rewriting the whole strategy architecture.

**Architecture:** Extend the existing self-built strategy in place by adding QC-aligned parameters, state helpers, signal/roll behaviors, and parity sizing helpers. Keep the public strategy interfaces stable while bringing decision logic closer to the current `quantconnect` implementation.

**Tech Stack:** Python, pytest, existing backtesting engine and strategy modules

---

### Task 1: Add Failing Tests For New QC Sync Behaviors

**Files:**
- Modify: `tests/unit/test_binbin_qc_parity.py`
- Modify: `tests/unit/test_roll_logic.py`
- Modify: `tests/unit/test_strategies.py`

- [ ] **Step 1: Write failing parity and behavior tests**

Add tests that cover:
- parity config exposes new QC parameters
- parity put sizing applies stock inventory and symbol risk limits
- strategy cooldown blocks SP signals
- strategy repair call logic tightens call signal selection inputs
- defensive put roll is preferred before legacy roll logic

- [ ] **Step 2: Run targeted tests to verify they fail**

Run: `pytest tests/unit/test_binbin_qc_parity.py tests/unit/test_roll_logic.py tests/unit/test_strategies.py -q`
Expected: FAIL because new QC-synced parameters/helpers/behaviors do not yet exist in the self-built engine.

- [ ] **Step 3: Keep the failing cases minimal and deterministic**

Use direct unit-level inputs and stubs instead of full engine runs wherever possible so failures point to missing behavior, not fixture setup noise.

- [ ] **Step 4: Re-run the same targeted test subset**

Run: `pytest tests/unit/test_binbin_qc_parity.py tests/unit/test_roll_logic.py tests/unit/test_strategies.py -q`
Expected: Still FAIL, with clear assertions tied to the missing QC-sync features.

### Task 2: Sync QC Parameters And Helper State Into Self-Built Strategy

**Files:**
- Modify: `core/backtesting/strategies/binbin_god.py`

- [ ] **Step 1: Add QC-aligned parameters to strategy initialization**

Sync these parameters and defaults from `quantconnect/strategy_init.py` into `BinbinGodStrategy.__init__`:
- `repair_call_threshold_pct`
- `repair_call_delta`
- `repair_call_dte_min`
- `repair_call_dte_max`
- `repair_call_max_discount_pct`
- `defensive_put_roll_enabled`
- `defensive_put_roll_loss_pct`
- `defensive_put_roll_itm_buffer_pct`
- `defensive_put_roll_min_dte`
- `defensive_put_roll_max_dte`
- `defensive_put_roll_dte_min`
- `defensive_put_roll_dte_max`
- `defensive_put_roll_delta`
- `assignment_cooldown_days`
- `large_loss_cooldown_days`
- `large_loss_cooldown_pct`
- `volatility_cap_floor`
- `volatility_cap_ceiling`
- `volatility_lookback`
- `dynamic_symbol_risk_enabled`
- `symbol_state_cap_floor`
- `symbol_state_cap_ceiling`
- `symbol_drawdown_lookback`
- `symbol_drawdown_sensitivity`
- `symbol_downtrend_sensitivity`
- `symbol_volatility_sensitivity`
- `symbol_exposure_sensitivity`
- `symbol_assignment_base_cap`
- `stock_inventory_cap_enabled`
- `stock_inventory_base_cap`
- `stock_inventory_cap_floor`
- `stock_inventory_block_threshold`

- [ ] **Step 2: Add cooldown state and helper methods**

Add strategy-local helpers analogous to QC:
- set symbol cooldown
- read remaining cooldown
- check whether a symbol is on cooldown

- [ ] **Step 3: Run targeted tests**

Run: `pytest tests/unit/test_strategies.py -q`
Expected: parameter/state tests move from FAIL to PASS, while behavior tests may still fail.

### Task 3: Sync QC Signal Gating And Repair Call Behavior

**Files:**
- Modify: `core/backtesting/strategies/binbin_god.py`
- Test: `tests/unit/test_strategies.py`

- [ ] **Step 1: Update SP signal generation**

Before generating SP entries, add:
- cooldown blocking
- stock inventory cap blocking

- [ ] **Step 2: Update CC signal generation**

When stock is sufficiently below cost basis, apply QC-style repair behavior:
- raise minimum delta
- constrain DTE into repair band
- set a minimum strike floor derived from underlying and cost basis

- [ ] **Step 3: Run targeted tests**

Run: `pytest tests/unit/test_strategies.py -q`
Expected: SP gating and repair-call tests PASS.

### Task 4: Sync Defensive Put Roll And Cooldown Triggers

**Files:**
- Modify: `core/backtesting/strategies/binbin_god.py`
- Test: `tests/unit/test_roll_logic.py`

- [ ] **Step 1: Add a QC-style defensive short-put roll helper**

Mirror the QC conditions:
- only consider puts
- require DTE in configured window
- trigger on ITM buffer breach or large option loss

- [ ] **Step 2: Integrate defensive roll ahead of legacy roll rules**

Update strategy roll generation/management so short puts check defensive roll before existing ML/legacy roll behavior.

- [ ] **Step 3: Add cooldown side effects**

Set cooldown when:
- put assignment occurs
- defensive roll occurs
- put roll/close happens after large loss threshold breach

- [ ] **Step 4: Run targeted tests**

Run: `pytest tests/unit/test_roll_logic.py tests/unit/test_strategies.py -q`
Expected: defensive roll and cooldown-trigger tests PASS.

### Task 5: Sync QC Parity Sizing Helpers

**Files:**
- Modify: `core/backtesting/qc_parity.py`
- Test: `tests/unit/test_binbin_qc_parity.py`

- [ ] **Step 1: Extend parity defaults and config**

Add the new QC sizing-related parameters to `QC_BINBIN_DEFAULTS` and `BinbinGodParityConfig`.

- [ ] **Step 2: Add QC-aligned parity helper functions**

Implement parity-side equivalents for:
- volatility-weighted symbol cap
- symbol state risk multiplier
- stock inventory cap

- [ ] **Step 3: Update `calculate_put_quantity_qc()`**

Thread the new helper outputs into the final contract limit calculation so parity mode mirrors current QC sizing constraints more closely.

- [ ] **Step 4: Run targeted parity tests**

Run: `pytest tests/unit/test_binbin_qc_parity.py -q`
Expected: parity config and sizing tests PASS.

### Task 6: Verify Full Sync Surface And Touch UI Only If Needed

**Files:**
- Modify only if required: `app/pages/binbin_god.py`
- Modify only if required: related config/UI plumbing files
- Test: existing impacted unit tests

- [ ] **Step 1: Check whether new parameters are already tolerated by current UI/config flow**

If current UI passes unknown or optional params safely, do not modify UI.

- [ ] **Step 2: Add only minimal compatibility changes if required**

Limit any UI change to parameter exposure or safe defaults necessary for the synced strategy to be configurable.

- [ ] **Step 3: Run verification suite**

Run: `pytest tests/unit/test_binbin_qc_parity.py tests/unit/test_roll_logic.py tests/unit/test_strategies.py tests/unit/test_cc_sp_mode.py -q`
Expected: PASS with no new failures in the QC sync surface.

