# QC Option-Income Alignment Design

## Goal

Shift more realized return from stock-leg P&L back to option-income P&L by making covered-call repair more reliable in QuantConnect and keeping `binbin_god / parity` behavior aligned. The change should reduce cases where assigned stock sits uncovered or is force-exited before the strategy has a realistic chance to monetize it with calls.

## Problem

Recent QC changes improved assignment handling, deferred order retry, and covered-call routing, but the live runtime still has several structural gaps:

- the true four-tier covered-call fallback ladder is not wired into the main QC selection path
- QC and parity still use older two-tier covered-call helpers in their active execution paths
- assignment fail-safe defaults are aggressive enough to exit stock before the repair-call path is given enough time
- covered-call generation is still strongly gated by ML output and confidence, which can leave stock holdings with no call even when a rules-based call would be acceptable
- QC and parity defaults have drifted in key assignment-risk settings, so return composition differs across the two paths

The result is a strategy that still behaves too often like:

- short puts
- occasional assignment
- stock-leg P&L deciding the outcome

instead of:

- short puts
- assignment when needed
- recurring covered-call income and repair behavior

## Recommended Approach

Implement a single coherent covered-call repair flow across QC and parity with three pillars:

1. **Use one real covered-call fallback ladder everywhere**
   Replace active two-tier helpers with the intended four-tier ladder:
   - primary repair selection
   - wider delta tolerance
   - wider DTE window
   - bounded rescue strike discount

2. **Add a rules-based covered-call fallback when ML does not produce a usable call**
   For repair / assigned-stock contexts, the strategy should still attempt a covered call even if:
   - ML emits no `SELL_CALL` signal
   - ML confidence is below the standard confidence gate

   The intent is not to remove ML from covered calls entirely, but to stop ML from being a hard blocker in the exact scenario where the wheel most needs a call.

3. **Loosen assignment stock fail-safe timing**
   The fail-safe should remain as downside protection, but it should no longer be easy for stock to exit after the first missed call cycle. The default posture should be:
   - try to sell repair calls first
   - only force-exit after repeated failure or sufficient elapsed time / drawdown

## Scope

### QuantConnect Runtime

Modify:

- `quantconnect/signals.py`
- `quantconnect/signal_generation.py`
- `quantconnect/strategy_mixin.py`
- `quantconnect/position_management.py`
- `quantconnect/strategy_init.py`

Responsibilities:

- centralize the real four-tier covered-call ladder
- remove redundant ladder helpers that are no longer authoritative
- let repair / assigned-stock covered calls fall back to a rules-based path when ML is absent or too weak
- relax assignment fail-safe defaults and/or logic so the first missed cycle does not immediately force a stock exit
- keep runtime logs and summary counters meaningful for debugging

### Parity / Replay

Modify:

- `core/backtesting/qc_parity.py`
- `core/backtesting/strategies/binbin_god.py`

Responsibilities:

- mirror the same covered-call ladder semantics and defaults
- mirror the same assignment repair timing assumptions
- remove replay-side redundant covered-call helpers that would drift from QC

### Tests

Modify and/or extend:

- `tests/unit/test_binbin_qc_parity.py`
- `tests/unit/test_qc_strategy_mixin.py`
- `tests/unit/test_qc_execution.py`
- `tests/unit/test_backtest_engine_qc_replay.py`
- `tests/unit/test_strategies.py`

Add focused tests if needed, but prefer extending the existing QC / parity suites unless a new file materially improves clarity.

## Architecture

### 1. Single Covered-Call Ladder

The authoritative covered-call ladder should live in one shared QC-side helper and one parity-side mirror with the same tier order and parameter semantics.

Required tiers:

1. `primary`
   - existing repair delta / DTE / strike floor
2. `fallback_delta`
   - same DTE and strike floor, wider delta tolerance
3. `fallback_dte`
   - same strike floor, longer DTE window
4. `rescue_discount`
   - longer DTE, wider tolerance, bounded lower strike floor

Bounded rescue floor:

- `max(underlying_price * 1.01, cost_basis * cc_fallback_min_cost_basis_ratio)`

This keeps rescue calls OTM and prevents an excessively discounted strike.

### 2. Rules-Based Covered-Call Fallback

Covered-call generation should use ML when available, but repair/assigned-stock scenarios need a deterministic fallback:

- if ML returns a covered-call signal, use it
- if ML returns no signal, generate a rules-based covered-call candidate using the current repair context
- if ML returns a signal below normal confidence, repair/assigned-stock mode may still execute a call under the rules-based path

This fallback should be limited to situations where:

- stock is already held
- the symbol is in repair / assignment inventory mode, or otherwise clearly in a covered-call context

This is intentionally narrower than “ignore ML everywhere.”

### 3. Assignment Fail-Safe Rebalance

Current defaults and logic make the fail-safe too eager. The redesigned behavior should be:

- holding stock with an active covered call resets miss counters
- lack of a covered call alone should not trigger an immediate exit on the first cycle
- repeated misses, repair timeout, and/or drawdown should remain valid fail-safe exits

The most important behavioral shift is:

- “first miss” should mean “try again,” not “exit stock now”

### 4. QC / Parity Consistency

All of the following must stay aligned between QC and replay:

- covered-call ladder tier order
- ladder parameter names and defaults
- repair-call strike floor logic
- repair fallback behavior when ML is missing
- assignment fail-safe timing defaults that materially affect whether stock exits or transitions into covered-call repair

## Default Parameter Changes

The defaults should be updated in QC runtime and synchronized into parity:

- keep existing covered-call ladder defaults introduced earlier:
  - `cc_fallback_delta_tolerance_1 = 0.12`
  - `cc_fallback_delta_tolerance_2 = 0.15`
  - `cc_fallback_dte_min = 14`
  - `cc_fallback_dte_max = 30`
  - `cc_fallback_min_cost_basis_ratio = 0.85`

- relax assignment repair timing defaults from the current aggressive posture:
  - increase `assigned_stock_cc_miss_limit` above `1`
  - increase `assigned_stock_max_repair_days` above `3`

Recommended new defaults:

- `assigned_stock_cc_miss_limit = 3`
- `assigned_stock_max_repair_days = 7`

These values give the repair-call path multiple chances without removing the eventual fail-safe.

### Alignment Fixes

The following settings should be reviewed and synchronized where drift currently exists:

- `symbol_assignment_base_cap`
- `max_assignment_risk_per_trade`
- any covered-call repair defaults consumed by both QC and parity

The goal is not broad parameter retuning, only removing default drift that changes return composition between runtimes.

## Redundant Code Removal

This change should explicitly remove or retire redundant covered-call selection code where it is no longer authoritative.

Expected cleanup:

- remove QC-side duplicate covered-call ladder builders once the shared path is wired
- remove parity-side duplicate / outdated two-tier helpers once the four-tier ladder is the single active path
- avoid keeping “reference” implementations that are no longer called by production flow

If a helper must remain for backwards compatibility, it should delegate to the authoritative implementation rather than carry separate business logic.

## Testing Strategy

### QC Tests

Verify:

- covered-call repair still works when ML emits a normal signal
- repair / assigned-stock covered calls are still generated when ML emits no signal
- repair / assigned-stock covered calls can still execute when ML confidence is below the standard gate
- the four-tier ladder actually reaches `fallback_dte` and `rescue_discount`
- rescue tier never chooses strikes below `cost_basis * cc_fallback_min_cost_basis_ratio`
- rescue tier still respects the OTM floor
- assignment fail-safe does not exit stock on the first missed call cycle

### Parity Tests

Verify:

- replay uses the same four-tier ladder semantics
- replay call generation can produce fallback / rescue calls in the same underwater scenarios as QC
- replay defaults for repair timing match QC defaults

### Regression Slice

Keep these suites green:

- `tests/unit/test_binbin_qc_parity.py`
- `tests/unit/test_qc_execution.py`
- `tests/unit/test_qc_strategy_mixin.py`
- `tests/unit/test_backtest_engine_qc_replay.py`
- `tests/unit/test_strategies.py`
- `tests/unit/test_roll_logic.py`
- `tests/unit/test_page_imports.py`

## Non-Goals

This change does not:

- increase put-side capacity or target margin utilization
- redesign the ML models
- redesign the UI
- remove assignment fail-safe protection entirely
- add new broker / order-routing abstractions

## Risks and Mitigations

### Risk: Rules fallback sells too many low-quality calls

Mitigation:

- restrict the fallback to repair / assigned-stock covered-call scenarios
- preserve the bounded rescue strike floor
- preserve OTM protection

### Risk: Looser fail-safe reintroduces large stock-leg drawdowns

Mitigation:

- keep fail-safe logic active
- only relax timing and miss-count thresholds
- continue using drawdown-based exits as the final safety net

### Risk: QC and parity drift again

Mitigation:

- treat one ladder implementation per side as authoritative
- synchronize defaults in `qc_parity.py`
- add parity tests that assert the same fallback behavior on the same inventory context
