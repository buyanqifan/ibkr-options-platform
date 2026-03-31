# Binbin God QC Replay Design

## Summary

Rework the `binbin_god` backtest experience so it becomes a QC-only replay entrypoint instead of a hybrid local/QC simulator. The page, parameter mapping, engine routing, and strategy execution path should all align around one enforced mode: QC replay using the emulated lattice contract universe. Legacy native execution paths and UI affordances that imply local theoretical backtesting should be removed from the `binbin_god` page flow.

## Goals

- Make `binbin_god` backtests execute a single QC replay path end-to-end.
- Remove old UI controls and payload branches that expose native/local execution semantics.
- Ensure QC defaults remain the single source of truth for page defaults and replay config.
- Reduce result drift between QuantConnect and local `binbin_god` runs by eliminating legacy execution paths from the page-triggered flow.

## Non-Goals

- Renaming the `binbin_god` route or strategy identifier.
- Refactoring unrelated strategies or pages.
- Rewriting QuantConnect source files.
- Guaranteeing exact one-to-one parity with QC fills or exchange microstructure beyond the existing local QC replay model.

## Recommended Approach

Adopt a strict QC-only design for the `binbin_god` page and execution chain.

### Alternative A: Recommended

Convert `binbin_god` into a pure QC replay entrypoint:

- UI exposes only QC-backed parameters.
- Payload always forces `parity_mode="qc"`.
- Payload always forces `contract_universe_mode="qc_emulated_lattice"`.
- Engine always routes `binbin_god` runs into the QC parity runner.
- Native/theoretical branches remain unreachable from the page flow.

Benefits:

- Clear semantics and fewer hidden mismatches.
- Lower long-term maintenance cost.
- Best chance of reducing large QC vs local performance drift.

Tradeoff:

- Removes previous native compatibility for the `binbin_god` page.

### Alternative B: Hidden Compatibility Layer

Keep native code paths internally but hide them from the UI.

Benefits:

- Easier rollback if parity runner has issues.

Tradeoff:

- Leaves dual execution semantics in the codebase.
- High risk of future drift and accidental regression.

### Alternative C: New Dedicated QC Replay Page

Create a separate QC replay page while leaving `binbin_god` as-is.

Benefits:

- Clean conceptual split.

Tradeoff:

- More routing, docs, and maintenance overhead.
- Not necessary to achieve the current goal.

This design chooses Alternative A.

## Architecture

The final request path should become:

`Binbin God UI -> QC-only params -> BinbinGodParityConfig -> engine QC replay runner -> QC parity strategy signal chain`

### Layer 1: Page

`app/pages/binbin_god.py` becomes a QC-only backtest form. It continues to surface QC parameters, but it no longer exposes run-mode switching or any payload branch that suggests native/local execution.

### Layer 2: Config Resolution

`core/backtesting/qc_parity.py` remains the default/config authority for:

- QC-backed defaults loaded from `quantconnect/config.json`
- parity config resolution
- forced QC contract universe mode

Its role shifts from “optional parity overlay” to “formal replay config interpreter” for `binbin_god`.

### Layer 3: Engine Routing

`core/backtesting/engine.py` treats `binbin_god` as a QC replay strategy. Instead of conditionally entering the parity runner, it should route `binbin_god` requests directly to the QC replay execution path.

### Layer 4: Strategy Execution

`core/backtesting/strategies/binbin_god.py` should use QC replay signal generation as the active backtest path. Legacy non-parity/native branches may remain temporarily only if needed for low-risk refactoring, but they must no longer be reachable from the `binbin_god` page flow or engine routing.

## UI Design

The page should remain grouped around QC concepts:

- Run setup
- Core wheel parameters
- ML
- Covered call / repair
- Defensive put / cooldown
- Symbol risk / inventory

The page should remove these concepts entirely:

- `run_mode`
- user-facing native/local mode selection
- any local theoretical mode wording
- any payload branching for native execution

### Page Submission Rules

When the page builds params:

- Always set `strategy="binbin_god"`
- Always set `parity_mode="qc"`
- Always set `contract_universe_mode="qc_emulated_lattice"`
- Continue normalizing `stock_pool`
- Continue deriving `symbol`
- Continue fanning out `ml_enabled` to the four internal ML flags

No UI control should allow users to turn QC replay off.

## Backend Design

### `core/backtesting/qc_parity.py`

Required changes:

- Keep loading defaults from `quantconnect/config.json`
- Continue exposing QC-aligned defaults for page and strategy use
- Treat QC replay as the expected configuration for `binbin_god`
- Ensure `apply_to_params()` produces a fully resolved QC replay config, including enforced `contract_universe_mode="qc_emulated_lattice"` when used for `binbin_god`

### `core/backtesting/engine.py`

Required changes:

- For `strategy == "binbin_god"`, resolve parity config immediately
- Always route into `_run_binbin_god_qc_parity()`
- Remove dependence on `parity_config.enabled` as a gate for whether QC replay should be used
- Keep the generic engine loop for other strategies unchanged

### `core/backtesting/strategies/binbin_god.py`

Required changes:

- Make QC parity signal generation the active backtest path
- Ensure `generate_signals()` for `binbin_god` replay uses `_generate_qc_parity_signals()`
- Prevent the page-triggered path from falling back to legacy non-parity sizing or contract selection
- Eliminate or isolate old native-only behaviors that affect replay semantics, especially:
  - legacy theoretical contract path
  - local-only SP-in-CC enhancements not present in QC replay
  - position sizing fallbacks based on `strike * 100` in the active replay path

Where direct deletion is safe, remove dead branches. Where deletion is risky, keep helper functions only as inactive/internal leftovers, but ensure they are unreachable from the `binbin_god` page path.

## Data and Behavior Alignment

The purpose of this change is not merely to rename parameters. It is to align execution semantics:

- QC replay contract selection should use the emulated lattice path
- QC replay sizing should use parity sizing logic
- QC replay trade flow should go through the parity runner bookkeeping
- UI defaults should continue to mirror QC defaults

This does not claim perfect live-market parity. It does ensure the local `binbin_god` entrypoint stops mixing QC-style parameters with legacy local-theoretical execution semantics.

## Error Handling

- If QC config defaults cannot be loaded, `qc_parity.py` should continue to use safe fallback defaults.
- If a QC replay run cannot resolve valid contracts from the emulated lattice, the engine should produce a normal empty-signal behavior rather than falling back to legacy local-theoretical selection.
- The page should not silently reintroduce native behavior when a QC-specific field is missing.

## Testing Strategy

Add or update tests that prove the page and engine are QC-only for `binbin_god`.

### Page Tests

Update `tests/unit/test_binbin_god_page.py` to verify:

- removed UI controls are absent, especially `run_mode`
- built params always include `parity_mode="qc"`
- built params always include `contract_universe_mode="qc_emulated_lattice"`
- default payload contains only QC replay semantics

### Engine Tests

Add or update tests around `core/backtesting/engine.py` to verify:

- `binbin_god` always routes into the QC parity runner
- `binbin_god` no longer falls into the generic/native simulation loop

### Strategy Tests

Add or update tests around `core/backtesting/strategies/binbin_god.py` to verify:

- `generate_signals()` uses QC replay semantics for page-triggered runs
- no active replay path uses legacy `strike * 100` sizing logic
- QC replay contract universe mode is the emulated lattice mode

### Regression Coverage

Keep these suites passing:

- `tests/unit/test_page_imports.py`
- `tests/unit/test_binbin_god_page.py`
- `tests/unit/test_binbin_qc_parity.py`
- `tests/unit/test_strategies.py`

Add targeted engine-routing tests if current coverage does not already prove the forced QC replay path.

## Risks

### Risk: Hidden native dependencies

Some helper methods in `binbin_god.py` may still assume legacy usage patterns.

Mitigation:

- remove or isolate only the parts reachable from the page flow
- add routing tests so future changes cannot silently restore the old path

### Risk: UI and backend drift

If the page forces QC replay but backend defaults remain optional, drift may reappear.

Mitigation:

- centralize forced replay semantics in `qc_parity.py`
- assert the forced fields in page tests

### Risk: Over-deleting code

Directly deleting all native helper functions in one pass may create avoidable churn.

Mitigation:

- prioritize removing reachability first
- prune dead code where confidence is high

## Files Expected To Change

- Modify: `app/pages/binbin_god.py`
- Modify: `core/backtesting/qc_parity.py`
- Modify: `core/backtesting/engine.py`
- Modify: `core/backtesting/strategies/binbin_god.py`
- Modify: `tests/unit/test_binbin_god_page.py`
- Modify: `tests/unit/test_binbin_qc_parity.py`
- Modify: `tests/unit/test_strategies.py`
- Possibly add: a focused engine-routing unit test if current coverage is insufficient

## Success Criteria

- Running `binbin_god` from the UI always executes QC replay semantics
- UI no longer exposes native/local execution controls
- Engine does not route `binbin_god` into the generic/native simulation loop
- Active replay path does not use legacy theoretical sizing/selection behavior
- QC default config remains the source of truth for defaults
- Targeted unit tests prove the enforced QC replay path
