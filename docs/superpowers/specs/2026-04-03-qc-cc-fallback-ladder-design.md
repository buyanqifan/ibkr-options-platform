# QC Covered Call Fallback Ladder Design

## Goal

Improve covered-call execution for underwater assigned stock by adding a tiered fallback ladder that progressively relaxes covered-call selection constraints when the preferred repair contract is unavailable. The change must apply to both QuantConnect runtime code and `binbin_god` / parity replay so the two paths stay aligned.

## Problem

Recent QC backtests still show zero covered-call orders even after deferred-open retry was added. The closed-trade and order data indicate the strategy is behaving like short puts plus occasional stock legs rather than a full wheel. The most likely cause is not order retry anymore, but covered-call contract selection being too strict for underwater holdings:

- repair calls require a narrow delta target
- repair calls use short DTE windows
- repair calls enforce a strike near cost basis
- deep underwater stock may have no contracts satisfying all three constraints simultaneously

That leads to long-lived stock positions with no call income, which materially hurts returns and leaves stock-leg P&L to dominate outcomes.

## Recommended Approach

Implement a four-tier covered-call fallback ladder.

### Tier 0: Existing Repair Rules

Try the current repair-call selection as-is:

- current repair delta target
- current repair DTE window
- current minimum strike logic
- existing delta tolerance

### Tier 1: Wider Delta Tolerance

If Tier 0 finds no contract, keep the same DTE and strike floor but widen delta tolerance.

Recommended default:

- `cc_fallback_delta_tolerance_1 = 0.12`

### Tier 2: Longer DTE Window

If Tier 1 still fails, keep the strike floor but widen the repair DTE window.

Recommended defaults:

- `cc_fallback_dte_min = 14`
- `cc_fallback_dte_max = 30`

### Tier 3: Limited Rescue Strike Relaxation

If Tier 2 still fails, allow a rescue covered call with a lower strike floor, but never too far below cost basis.

Recommended default:

- `cc_fallback_min_cost_basis_ratio = 0.85`

Effective floor:

- `max(underlying_price * 1.01, cost_basis * cc_fallback_min_cost_basis_ratio)`

This preserves two constraints:

- no obviously ITM covered calls
- no excessively discounted rescue calls

If Tier 3 also fails, the strategy should log a final ladder failure and skip the call for that rebalance cycle.

## Scope

### QuantConnect Runtime

Modify:

- `quantconnect/strategy_init.py`
- `quantconnect/option_selector.py`
- `quantconnect/signal_generation.py`
- related QC tests

Responsibilities:

- add new fallback ladder defaults
- teach the selector to evaluate a sequence of candidate constraints
- build the tier sequence for CC / repair-call selection
- expose which fallback tier matched for debugging

### BinbinGod / Parity Replay

Modify:

- `core/backtesting/qc_parity.py`
- `core/backtesting/strategies/binbin_god.py`
- related parity tests

Responsibilities:

- mirror the QC defaults
- apply the same ladder semantics in replay call generation
- ensure parity backtests do not diverge from QC because of different CC selection rules

## Architecture

### Shared Ladder Semantics

The same conceptual ladder must exist in both paths:

1. preferred repair contract
2. wider delta tolerance
3. longer DTE
4. limited rescue strike relaxation

The implementation does not need to literally share code across QC and parity, but the tier order, parameter meaning, and default values must stay aligned.

### Selector Contract

`quantconnect/option_selector.py` should support evaluating multiple candidate selection tiers in order and return both:

- the selected option result
- metadata about which tier matched

The same pattern should be reflected in replay selection logic, either via a helper or equivalent inline tier loop.

### Logging / Observability

When a fallback tier is used, emit an informative log token so future backtests can show whether calls are being sourced from:

- Tier 0
- Tier 1
- Tier 2
- Tier 3
- no tier matched

This should be low-volume and compatible with the existing summary-driven debugging approach.

## New Defaults

Add these defaults to QC runtime and parity config:

- `cc_fallback_delta_tolerance_1 = 0.12`
- `cc_fallback_delta_tolerance_2 = 0.15`
- `cc_fallback_dte_min = 14`
- `cc_fallback_dte_max = 30`
- `cc_fallback_min_cost_basis_ratio = 0.85`

Notes:

- Tier 0 continues using current defaults (`delta_tolerance=0.08`, existing repair DTE, existing min strike logic)
- Tier 3 uses `cc_fallback_delta_tolerance_2`
- Tier 3 also keeps the ITM guard via `underlying_price * 1.01`

## Testing Strategy

### QC Tests

Add or update tests to verify:

- Tier 0 still wins when a normal repair-call contract exists
- Tier 1 can win when only a wider delta tolerance makes a contract eligible
- Tier 2 can win when only a longer DTE contract exists
- Tier 3 can win when only a lower-but-bounded strike exists
- Tier 3 never allows strikes below `cost_basis * 0.85`
- CC selection still rejects clearly ITM calls

### Parity Tests

Add or update tests to verify:

- replay call generation uses the same fallback ordering
- parity defaults expose the same ladder parameters
- the same underwater holding scenario produces a CC candidate in replay when a rescue-tier contract exists

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

- increase put-side capacity
- change assignment fail-safe thresholds
- redesign UI layout
- remove the deferred-open retry queue

## Risks and Mitigations

### Risk: Rescue calls sacrifice too much upside

Mitigation:

- hard floor at `cost_basis * 0.85`
- still require OTM via `underlying_price * 1.01`
- rescue mode only activates after higher-quality tiers fail

### Risk: QC and parity drift again

Mitigation:

- add explicit parity default tests
- mirror the same ladder parameter names and semantics in `qc_parity.py`
- add replay-side regression around underwater call generation

### Risk: More complexity in selector logic

Mitigation:

- keep tiers explicit and ordered
- log which tier matched
- do not add unrelated scoring logic in this change
