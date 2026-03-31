# Binbin God QC Warmup Alignment Design

## Summary

Align the local `binbin_god` QC replay path with QuantConnect's warmup lifecycle semantics without changing any other strategy. The local replay should treat the first 60 bars as a non-trading warmup window, then run a one-time `OnWarmupFinished` equivalent for ML pretraining, and only begin normal QC replay trading after warmup completes.

## Goals

- Match QC's "warm up first, trade later" behavior for local `binbin_god` QC replay.
- Keep the change scoped only to the `binbin_god` QC replay path.
- Avoid introducing a global warmup framework for all strategies.
- Preserve the existing QC replay signal, sizing, and bookkeeping path after warmup ends.

## Non-Goals

- Changing non-`binbin_god` strategies.
- Refactoring the entire engine into a generic warmup lifecycle framework.
- Rewriting QuantConnect algorithm code.
- Perfectly simulating every QC platform callback beyond the warmup behavior needed here.

## Chosen Approach

Implement the warmup lifecycle directly inside `BacktestEngine._run_binbin_god_qc_parity()`.

This approach keeps the scope tight:

- warmup applies only to `binbin_god`
- no UI changes are required
- no non-`binbin_god` engine paths are touched
- QC replay remains the only active trading path after warmup completes

## Behavioral Design

### Warmup Window

The first 60 bars of the replay act as warmup bars.

During this period, the replay may accumulate and maintain context already needed by the strategy, but it must not:

- generate entry signals
- open positions
- close positions via position management
- process expiry/assignment bookkeeping
- emit immediate covered calls

In practical terms, the QC parity trading loop should be skipped during warmup.

### Warmup Completion

When the warmup window ends:

- run a one-time local equivalent of QC `OnWarmupFinished`
- if `ml_enabled` is false, do nothing
- if ML has already been pretrained, do nothing
- otherwise, pretrain once using the accumulated historical bars for each symbol in the stock pool, matching the existing QC behavior that requires at least 60 bars

After this one-time step, the replay transitions into the current QC parity trading flow.

### Post-Warmup Replay

Once warmup is complete, the existing QC replay logic continues unchanged:

- position management
- signal generation
- order opening
- expiry handling
- immediate CC after put assignment
- parity event tracing
- daily portfolio snapshots

The key change is timing, not strategy logic.

## Files To Modify

- Modify: `core/backtesting/engine.py`
- Possibly modify: `core/backtesting/strategies/binbin_god.py`
- Modify: `tests/unit/test_backtest_engine_qc_replay.py`
- Possibly modify: `tests/unit/test_binbin_qc_parity.py` or `tests/unit/test_strategies.py` if small strategy state assertions are useful

## Implementation Notes

### `core/backtesting/engine.py`

Add a small warmup state flow inside `_run_binbin_god_qc_parity()`:

- define a warmup length of 60 bars
- iterate bars as today
- while `i < warmup_bars`, skip the trading steps
- when warmup completes for the first time, run a helper for ML pretraining
- only then allow the rest of the QC replay loop to execute

To keep the code readable, use a small helper method rather than inlining all warmup logic in the middle of the runner if the loop gets too dense.

### `core/backtesting/strategies/binbin_god.py`

Only add state if needed. Examples:

- `_ml_pretrained`
- `warmup_complete`

Do not broaden this into a reusable framework unless a small state field is required for clarity.

## Testing Strategy

### Engine Warmup Tests

Add focused QC replay warmup tests that verify:

1. With fewer than 60 bars, no trades are opened.
2. With more than 60 bars, trades only appear after the warmup boundary.
3. ML pretraining runs once at warmup completion when ML is enabled.
4. ML pretraining does not run when ML is disabled.

### Regression Tests

Keep these passing:

- `tests/unit/test_page_imports.py`
- `tests/unit/test_binbin_god_page.py`
- `tests/unit/test_binbin_qc_parity.py`
- `tests/unit/test_backtest_engine_qc_replay.py`
- `tests/unit/test_strategies.py`

## Risks

### Risk: Off-by-one warmup behavior

The most likely bug is letting trading begin one bar too early or one bar too late.

Mitigation:

- write explicit tests around a 59-bar, 60-bar, and 61+ bar dataset

### Risk: ML pretraining duplication

If pretraining is triggered in more than one place, replay behavior may become inconsistent.

Mitigation:

- centralize the one-time warmup completion pretraining call in the QC replay runner
- protect it with a one-time flag

### Risk: Hidden assumptions in current replay loop

Some bookkeeping may assume trading starts at bar 0.

Mitigation:

- keep the post-warmup replay loop unchanged as much as possible
- only guard entry into it during the warmup window

## Success Criteria

- Local `binbin_god` QC replay does not trade during the first 60 bars.
- A one-time warmup-finished ML pretraining step occurs after warmup when enabled.
- Trading begins only after warmup completes.
- Existing QC replay regressions remain green.
