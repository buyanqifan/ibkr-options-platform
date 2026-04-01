# QC Assignment Stock Fail-Safe Design

## Summary

Raise `quantconnect` strategy return potential without giving back the recent drawdown improvements by adding a targeted fail-safe for post-assignment stock exposure.

The core idea is:

- keep the recent utilization increases
- make assigned-stock covered-call repair more aggressive
- only force-exit stock after repeated repair failure and continued adverse price action

This is intentionally narrower than a general stock stop-loss system. It only applies to stock inventory created by short-put assignment.

## Problem

Recent QC runs improved drawdown materially, but returns remain far below the earlier high-return versions. The latest reviewed run shows:

- total return around `32%`, far below the earlier `~160%`
- maximum drawdown still around `11.6%`
- a small number of stock-leg losses still dominate downside, especially assignment-driven equity exposure
- recent quality filters are not the main return drag in this sample; `SP_QUALITY_BLOCK=0`

The current strategy already:

- caps put sizing and assignment notional
- cools down symbols after assignment
- attempts an immediate covered call after put assignment

But it does not have a final protection layer for stock inventory that remains uncovered and continues falling after assignment. That leaves a gap where:

1. put is assigned
2. immediate or next-cycle repair call does not get established
3. stock continues falling
4. large equity-leg loss overwhelms premium income

## Goals

- preserve the recent drawdown improvements
- recover part of the lost return by allowing higher utilization and more assertive repair behavior
- stop a small number of assignment-driven stock collapses from dominating the whole backtest
- keep the strategy logic understandable and close to the current QC wheel structure

## Non-Goals

- no general-purpose stop-loss framework for all equity holdings
- no partial liquidation ladder in this change
- no changes to the self-built backtester UI in this phase
- no redesign of QC scheduling, ML architecture, or option selection model

## Recommended Approach

Use a three-stage assignment stock protection flow:

1. mark stock that came from put assignment
2. treat that stock as a higher-priority, more aggressive repair-call candidate
3. if repair repeatedly fails and the stock remains deeply underwater, force-exit the stock

This is preferred over an immediate stock stop-loss because it gives the wheel strategy a defined repair window before taking the equity-leg loss.

## Alternatives Considered

### 1. Immediate stock stop-loss after assignment

Pros:

- simplest implementation
- strongest drawdown control

Cons:

- exits many positions that a repair call could have monetized
- likely hurts return too much
- changes wheel behavior too abruptly

### 2. Recommended: repair first, then fail-safe exit

Pros:

- balances return recovery and tail-risk control
- only intervenes after repeated repair failure
- fits current covered-call workflow

Cons:

- requires new per-symbol assignment state
- slightly more moving parts than a pure stop-loss

### 3. Partial stock reduction ladder

Pros:

- smoother than all-or-nothing exit
- may reduce regret in choppy recoveries

Cons:

- more complex to reason about
- adds more stock-leg transactions and more tuning surface
- not needed for the first version

## Design

### 1. New Parameters

Add these QC parameters in `quantconnect/strategy_init.py` and mirror defaults into parity config if needed later:

- `assigned_stock_fail_safe_enabled = true`
- `assigned_stock_drawdown_pct = 0.12`
- `assigned_stock_repair_attempt_limit = 3`
- `assigned_stock_min_days_held = 5`
- `assigned_stock_force_exit_pct = 1.0`
- `assigned_stock_repair_delta_boost = 0.10`
- `assigned_stock_repair_dte_min = 7`
- `assigned_stock_repair_dte_max = 14`
- `assigned_stock_repair_max_discount_pct = 0.12`

Interpretation:

- `drawdown_pct`: stock drawdown versus assignment cost basis required before fail-safe tracking advances
- `repair_attempt_limit`: number of rebalance cycles allowed without an active covered call before forced exit
- `min_days_held`: avoids instantly exiting freshly assigned shares
- `force_exit_pct`: allows full exit now, while leaving room for future partial exits
- repair parameters override normal repair-call behavior for assigned stock

### 2. New State

Add an `assigned_stock_state` dictionary on `algo` keyed by symbol. Each entry stores:

- `source`: always `put_assignment`
- `assignment_date`
- `assignment_cost_basis`
- `repair_failures`
- `last_repair_attempt`
- `force_exit_triggered`

Lifecycle:

- created on put assignment
- updated during rebalance checks
- cleared when shares are fully gone
- cleared when symbol no longer has assignment-derived stock risk

### 3. Assignment Hook

In `quantconnect/expiry.py`, when put assignment is detected:

- initialize or refresh `assigned_stock_state[symbol]`
- set `assignment_date`
- capture current cost basis from QC portfolio
- reset `repair_failures` to `0`
- keep the existing cooldown behavior
- keep the existing immediate covered-call attempt

This keeps the current assignment flow intact while adding explicit state for downstream protection.

### 4. More Aggressive Repair Calls For Assigned Stock

In `quantconnect/signal_generation.py`, when generating `CC` for a symbol with active `assigned_stock_state`:

- treat it as repair mode even if the normal repair threshold would not yet trigger
- increase target delta by `assigned_stock_repair_delta_boost`
- constrain DTE to the tighter assignment-repair window
- allow slightly more discount versus cost basis using `assigned_stock_repair_max_discount_pct`

Guardrails:

- still require strike above spot by a minimal amount
- do not produce invalid strike constraints
- only apply this enhanced mode to assignment-tagged stock

Expected effect:

- more frequent covered-call establishment after assignment
- faster premium collection and stock-risk handoff

### 5. Fail-Safe Check During Position Management

In `quantconnect/position_management.py`, add a stock-level management pass before normal option position handling:

For each symbol in `assigned_stock_state`:

1. confirm shares are still held; if not, clear state
2. check whether there is already active call coverage; if yes, reset or pause repair failure accumulation
3. compute:
   - days held since assignment
   - drawdown versus assignment cost basis
4. if:
   - days held >= `assigned_stock_min_days_held`
   - drawdown >= `assigned_stock_drawdown_pct`
   - no active covered call exists
   then increment `repair_failures`
5. if `repair_failures >= assigned_stock_repair_attempt_limit`, force-exit some or all shares

The forced exit should use a direct equity market order for:

- `shares_to_sell = int(current_shares * assigned_stock_force_exit_pct)`

For v1, defaults imply a full exit.

### 6. Logging

Add explicit logs for observability:

- `ASSIGNED_STOCK_TRACK:<symbol>:...`
- `ASSIGNED_REPAIR_ATTEMPT:<symbol>:...`
- `ASSIGNED_REPAIR_FAIL:<symbol>:failures=N:...`
- `ASSIGNED_STOCK_EXIT:<symbol>:shares=N:drawdown=X%:days=Y`

These logs are important because recent tuning has depended heavily on interpreting QC logs.

## Data Flow

1. short put expires ITM and is assigned
2. expiry handler records assignment state and tries immediate CC
3. later rebalances generate more aggressive CC signals for that assigned stock
4. if CC remains absent while drawdown deepens, fail-safe counter increments
5. once limit is hit, stock is exited

## Error Handling

- missing cost basis: skip fail-safe for that cycle and log once
- stale or missing equity price: skip fail-safe for that cycle
- zero shares after assignment state exists: clear state silently
- failed market order submission: log and retry next rebalance unless shares are gone

## Testing

Add or extend unit tests to cover:

- assignment state is initialized on put assignment
- assigned stock gets more aggressive CC parameters than normal repair mode
- fail-safe does not trigger before `min_days_held`
- fail-safe does not trigger when shares are already covered by a call
- fail-safe increments failure count when uncovered assigned stock stays below threshold
- fail-safe exits stock after the configured failure limit
- state clears when shares are gone

Regression focus:

- existing QC parity tests stay green
- recent signal-quality and risk-control tests stay green

## Risks

### Risk: return drops again because stock exit is too eager

Mitigation:

- use `min_days_held`
- require repeated failures, not one miss
- keep default force-exit threshold moderate at `12%`

### Risk: repair mode becomes so aggressive that calls cap upside too hard

Mitigation:

- only apply the stronger repair settings to assignment-tagged stock
- keep delta boost small and bounded

### Risk: assignment state drifts from portfolio reality

Mitigation:

- always reconcile against actual QC share holdings and active call positions
- clear state automatically when stock disappears

## Rollout Plan

Implement in one focused pass:

1. add parameters and state initialization
2. add assignment tracking in expiry handling
3. add assigned-stock-aware CC repair logic
4. add stock fail-safe management pass
5. add tests and verify current QC-related suites

## Success Criteria

This change is successful if follow-up backtests show:

- fewer very large assignment-driven stock losses
- return improves relative to the current `~32%` run
- drawdown does not materially exceed the recent post-tuning range
- logs clearly show when assignment fail-safe logic engages
