# QC And Parity Lifecycle Alignment Design

## Summary

This spec aligns the live `quantconnect/` strategy and the local `binbin_god` QC replay path around four behaviors that are currently drifting:

1. Covered calls should be attempted for all eligible held symbols, not just one best signal per rebalance.
2. Assignment processing should be event-first, with expiry scans acting only as deduplicated fallback.
3. Roll and close flows should be asynchronous-safe instead of assuming immediate fills.
4. Dynamic position capacity should compound off current portfolio value by default, and parity defaults should mirror that behavior.

The goal is not to redesign the strategy. The goal is to make the QC strategy and parity replay tell the same story about inventory repair, assignment handling, and capital expansion.

## Scope

In scope:

- `quantconnect/` strategy lifecycle and execution behavior
- `core/backtesting/strategies/binbin_god.py` QC replay behavior
- `core/backtesting/qc_parity.py` defaults and helper formulas
- Engine/parity glue only where needed to preserve semantic alignment
- Regression tests for QC and parity behavior

Out of scope:

- UI layout changes
- New strategy parameters unrelated to these four behaviors
- Large architectural rewrites of the backtest engine
- Replacing current option pricing or contract lattice models

## Approved Decisions

- Dynamic position expansion will default to compounding off `TotalPortfolioValue`, not initial capital.
- Assignment handling will use `OnOrderEvent` as the single primary path.
- Expiry scanning remains only as a fallback for missed events and must deduplicate against already-processed assignments.
- All changes must also be synchronized to `binbin_god` / QC parity replay so QC and local replay stay aligned.

## Problem Statement

The current system still has several structural mismatches:

- `quantconnect/strategy_mixin.py` now allows one covered call even at slot limit, but still only executes a single best covered call. Multi-symbol assigned inventory remains under-covered.
- `quantconnect/expiry.py` contains both event-based and expiry-scan assignment handling. Both paths can track assigned stock, update ML, and attempt immediate CC, which risks duplicate side effects.
- `quantconnect/execution.py` uses `safe_execute_option_order()` and then treats `ticket.Status == Filled` as if fills are synchronous. In QC this is not reliable for roll/close state progression.
- Dynamic max positions are still calculated from initial capital rather than current portfolio value, which suppresses intended compounding and leaves parity behavior under-aligned with the desired strategy semantics.
- Parity replay has accumulated QC-like behavior in pieces, but does not yet mirror these lifecycle decisions as one coherent model.

## Design Overview

The implementation will align QC and parity around a shared behavioral model:

- Covered call generation remains holdings-driven, but execution becomes per-symbol rather than single-best.
- Assignment state becomes event-driven first. Once an assignment is processed, later lifecycle passes must treat it as already handled.
- Roll/close transitions become ticket-submission flows with later fill reconciliation, rather than synchronous success assumptions.
- Dynamic capacity becomes portfolio-value aware in both QC and replay.

The design deliberately prefers small, targeted changes over introducing a new shared runtime abstraction layer.

## Workstream 1: Covered Call Fan-Out

### QC behavior

`quantconnect/strategy_mixin.py` will change from:

- generate all signals
- pick one best `SELL_CALL`
- then use remaining slots for `SELL_PUT`

to:

- generate all signals
- sort `SELL_CALL` signals by confidence descending
- attempt each covered call in order
- recompute option slot count after covered-call execution attempts
- only gate new `SELL_PUT` entries using `max_positions`

Important constraints:

- Covered calls still obey share coverage limits in `execute_signal()`
- No new leverage is introduced by this change
- `max_positions` continues to restrict only new short-put exposure

### Parity behavior

The QC replay path in `core/backtesting/strategies/binbin_god.py` will adopt the same rule:

- all eligible `SELL_CALL` signals are emitted and/or attempted
- slot gating applies only to new `SELL_PUT` entries

This keeps replay from understating repair income whenever multiple assigned symbols exist at once.

## Workstream 2: Event-First Assignment Processing

### QC behavior

`quantconnect/main.py` and `quantconnect/expiry.py` will be aligned so that:

- `OnOrderEvent -> handle_assignment_order_event()` is the only primary assignment path
- assignment processing creates or updates a dedupe record keyed by assigned option identity
- the primary path performs:
  - assigned stock tracking
  - cooldown updates
  - ML assignment update
  - immediate covered-call attempt
  - metadata cleanup

`check_expired_options()` remains as a fallback only when an assignment event was not seen. Before processing a candidate assignment or expiry, it must consult dedupe state and skip already-processed option identities.

### Dedupe state

QC state will add a small assignment-processing ledger, for example:

- processed assignment keys by option identity
- optional processed close/expiry keys if needed for symmetry

This ledger is lifecycle support state, not trading logic.

### Parity behavior

Parity replay will mirror the same semantic rule:

- one assignment event per option identity
- one stock tracking update
- one ML assignment learning update
- one immediate repair/CC opportunity

If the replay engine currently synthesizes assignment effects from closed trades, it must also guard against double-processing during later expiry-style reconciliation.

## Workstream 3: Asynchronous-Safe Roll And Close

### QC behavior

`quantconnect/execution.py` currently assumes that a close or roll order can be submitted and immediately inspected as `Filled`. This will be changed so that:

- roll/close submission records intent and pending metadata
- close/roll completion is finalized from later order events or other explicit fill reconciliation points
- if data is unavailable and order submission is deferred, state remains pending rather than silently disappearing

This does not require introducing a full generic order state machine. It requires enough pending-operation state to avoid losing roll/close intent just because the ticket was not synchronously filled.

Potential state:

- pending close requests keyed by order id or position id
- pending roll requests keyed by source position id
- metadata needed to finalize trade recording after fill

`safe_execute_option_order()` may continue to defer when a contract has no data, but the surrounding roll/close logic must no longer treat defer or non-immediate fills as terminal no-ops.

### Parity behavior

The replay side should mirror the same semantics at its abstraction level:

- close/roll transitions should not rely on immediate synthetic success unless the replay engine explicitly decides the fill occurred
- pending operation intent should survive until replay processing resolves it

Because parity replay is local and deterministic, this may be implemented more simply than QC, but the resulting behavioral semantics should match QC.

## Workstream 4: Portfolio-Value-Based Dynamic Capacity

### QC behavior

`quantconnect/execution.py::calculate_dynamic_max_positions()` will switch from:

- `initial_capital * target_margin_utilization`

to:

- `Portfolio.TotalPortfolioValue * target_margin_utilization`

while still respecting:

- `max_positions_ceiling`
- average stock-price-based margin-per-contract estimate
- minimum of one slot

This means the strategy compounds when equity grows and contracts capacity when equity declines.

### Parity behavior

`core/backtesting/qc_parity.py::calculate_dynamic_max_positions_from_prices()` and any replay callers will use the same portfolio-value-based formula.

Parity config defaults will be updated so this is the default semantic model, not an optional divergence.

## File-Level Responsibilities

### QC files

- `quantconnect/strategy_mixin.py`
  - fan out covered-call execution across all eligible CC signals
  - keep short-put slot gating intact

- `quantconnect/expiry.py`
  - centralize assignment dedupe and fallback behavior
  - ensure immediate CC and ML updates happen only once per assignment

- `quantconnect/main.py`
  - keep order-event routing as primary assignment entrypoint
  - integrate any pending roll/close fill reconciliation hooks if needed

- `quantconnect/execution.py`
  - introduce pending roll/close state handling
  - stop relying on synchronous `Filled`
  - switch dynamic capacity to portfolio-value based formula

- `quantconnect/strategy_init.py`
  - initialize any new pending-operation or assignment-dedupe state
  - log any new effective runtime defaults if necessary

### Parity files

- `core/backtesting/qc_parity.py`
  - align defaults and dynamic max-position formula
  - expose any parity-side constants/state needed for lifecycle alignment

- `core/backtesting/strategies/binbin_god.py`
  - mirror multi-symbol covered-call behavior
  - mirror assignment dedupe / repair / pending close semantics at replay level

- `core/backtesting/engine.py`
  - only if needed to support parity-side pending lifecycle resolution cleanly

## Testing Strategy

### New or expanded QC tests

- covered calls execute for multiple held symbols in one rebalance cycle
- short puts remain blocked at slot limit while covered calls still execute
- assignment order events process exactly once
- expiry fallback skips already-processed assignments
- roll/close requests no longer disappear when tickets are not immediately filled
- dynamic max positions expand and contract with portfolio value

### New or expanded parity tests

- replay path emits or attempts multiple covered calls for multiple stock holdings
- parity slot gating still applies only to short puts
- parity assignment tracking is deduplicated
- parity dynamic max positions use portfolio value semantics
- parity defaults reflect the QC defaults for compounding capacity

### Regression suites to keep green

- `tests/unit/test_binbin_qc_parity.py`
- `tests/unit/test_qc_execution.py`
- `tests/unit/test_qc_strategy_mixin.py`
- `tests/unit/test_backtest_engine_qc_replay.py`
- any new focused QC lifecycle tests added for assignment and order handling

## Risks And Mitigations

- Risk: Multi-CC fan-out could overstate covered-call frequency if replay and QC differ on share coverage.
  - Mitigation: preserve existing share-based contract limits and add tests for multi-symbol, not multi-call-over-coverage.

- Risk: Assignment dedupe could suppress legitimate fallback processing.
  - Mitigation: key dedupe by option identity and only suppress work already completed by the event path.

- Risk: Pending roll/close state could become stale.
  - Mitigation: keep state minimal, keyed by concrete position/order identities, and clear it on fill/cancel/expiry resolution.

- Risk: Portfolio-value-based capacity can materially change backtest behavior.
  - Mitigation: make this explicit in effective parameter logging and parity defaults so the new semantics are observable and mirrored.

## Success Criteria

The change is successful when all of the following are true:

- QC can monetize multiple held symbols with covered calls in the same rebalance window.
- Assignment handling produces one lifecycle update per assignment, not duplicate logging/learning/repair attempts.
- Roll/close state no longer depends on instantaneous fills to progress safely.
- Dynamic capacity compounds off current equity by default.
- Local QC replay remains behaviorally aligned with QC for these lifecycle decisions.
