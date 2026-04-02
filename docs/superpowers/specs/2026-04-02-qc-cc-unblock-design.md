# QC Covered Call Unblock Design

## Summary

Fix a strategy control-flow issue that prevents covered calls from being generated and executed whenever short-option slot usage has already reached `max_positions`.

This change is intentionally narrow:

- covered calls should still obey share coverage limits
- new short puts should still obey `max_positions`
- no parameter tuning in this change
- no change to assignment fail-safe thresholds

## Problem

Recent QC runs show:

- stock inventory is present in trades
- `CC_SIGNAL=0`
- `IMMEDIATE_CC=0`
- `SP_SIGNAL` and `PUT_BLOCK` remain active

The current `rebalance()` flow in `quantconnect/strategy_mixin.py` does this:

1. run position management
2. compute `open_count`
3. return immediately if `open_count >= max_positions`
4. only then generate signals

Because covered calls are generated after that early return, stock inventory can remain uncovered whenever short-put slots are already full. That makes the strategy behave more like "short puts plus unmonetized stock exposure" instead of a full wheel.

## Goals

- always allow covered-call generation for existing stock inventory
- keep `max_positions` as a cap for new short puts
- preserve existing covered-call share coverage constraints
- keep the change localized to the rebalance control flow

## Non-Goals

- no new parameter knobs
- no change to call quantity calculation
- no separate CC slot accounting system
- no scoring or ML changes
- no additional capacity increase in this phase

## Recommended Approach

Adjust `rebalance()` so it no longer returns before covered-call processing.

New high-level flow:

1. run position management
2. generate signals
3. execute best covered-call signal if present
4. recompute open option count
5. allow new short puts only if `open_count < max_positions`

This preserves current behavior for short puts while removing the accidental gate on covered calls.

## Alternatives Considered

### 1. Recommended: reorder rebalance gating

Pros:

- smallest change
- directly addresses the observed issue
- keeps current slot model intact

Cons:

- still relies on one shared slot model conceptually

### 2. Separate CC and SP slot accounting

Pros:

- clearer long-term semantics

Cons:

- more invasive
- requires more tests and more behavioral changes

### 3. Assignment-only CC bypass

Pros:

- narrower than full CC unblock

Cons:

- misses non-assignment stock inventory
- adds branching complexity for limited benefit

## Design

### 1. Rebalance Flow Change

In `quantconnect/strategy_mixin.py`:

- remove the early `return` that fires before signal generation when `open_count >= algo.max_positions`
- always generate signals after position management
- execute covered calls first if any exist
- recompute `open_count`
- only gate the short-put loop using `open_count < algo.max_positions`

### 2. Keep Existing Execution Constraints

Do not modify `quantconnect/execution.py` covered-call sizing logic:

- call quantity remains capped by `shares_available // 100`
- existing covered calls still reduce available share coverage
- no new leverage or notional risk is introduced by this fix

### 3. Testing

Add regression coverage showing:

- when `get_option_position_count(algo) >= algo.max_positions`, a covered-call signal is still executed
- in the same condition, new short-put execution remains blocked

Minimum verification command:

```bash
pytest tests/unit/test_binbin_qc_parity.py tests/unit/test_strategies.py tests/unit/test_roll_logic.py tests/unit/test_page_imports.py tests/unit/test_qc_execution.py -q
```

## Success Criteria

The next QC run should show:

- `CC_SIGNAL > 0` when stock inventory exists
- covered-call activity reappearing without increasing naked stock risk
- short puts still respecting `max_positions`
