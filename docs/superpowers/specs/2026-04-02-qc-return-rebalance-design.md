# QC Return Rebalance Design

## Summary

Raise `quantconnect` strategy return potential after the recent drawdown-focused tuning by moderately reopening capital utilization and high-premium symbol capacity.

This change is intentionally narrow:

- increase effective capital deployment
- relax assignment and stock inventory caps for premium-rich symbols
- improve option match rate by widening delta tolerance
- keep the recently added assignment stock fail-safe in place
- keep the recent signal-quality hard filter in place

## Problem

Recent tuning successfully reduced drawdown, but return remains far below the earlier high-return runs.

From the latest reviewed QC run:

- total return recovered only to about `32%`
- drawdown stayed around `11.6%`
- `SP_QUALITY_BLOCK=0`, so the new quality filter is not the current bottleneck
- `PUT_BLOCK=16`, mostly on `NVDA`, `META`, and `AMZN`
- the dominant block reasons are symbol notional cap, stock inventory cap, and assignment risk cap
- `No suitable options for = 11`, indicating contract matching is still leaving opportunities unused

The current version is therefore not underperforming because it rejects too many low-quality setups. It is underperforming because it is still too conservative when turning acceptable setups into actual risk.

## Goals

- raise return potential without fully reverting to the earlier aggressive risk posture
- preserve the new assignment stock fail-safe
- preserve the recent signal-quality filter
- improve effective position utilization and contract fill opportunity
- keep changes small enough that backtest differences are attributable

## Non-Goals

- no removal of the assigned-stock fail-safe
- no removal of the `SP_QUALITY_BLOCK` signal-quality filter
- no redesign of scoring, ML, scheduling, or position management
- no changes to self-built backtester UI in this phase
- no attempt to fully restore the earlier `~160%` profile in one jump

## Recommended Approach

Apply a measured "return rebalance" in two places:

1. relax the main QC default risk budgets that are currently constraining high-premium names
2. widen option selection tolerance so more valid contracts convert from signal to trade

This is preferred over reverting the recent defensive changes because the logs show the current bottleneck is capacity, not the new hard filters.

## Alternatives Considered

### 1. Revert the recent defensive tuning

Pros:

- fastest way to increase gross exposure
- likely produces higher raw returns quickly

Cons:

- throws away recent drawdown improvements
- makes it harder to identify which new protections were actually valuable
- likely reintroduces the same assignment-driven tail losses

### 2. Recommended: moderate utilization and cap increase

Pros:

- targets the actual bottlenecks shown in logs
- keeps recent protection layers intact
- small enough to attribute in the next backtest

Cons:

- may still not be enough to approach the earlier peak-return regime
- could increase drawdown if premium-rich names dominate too quickly

### 3. Add more symbol-specific overrides now

Pros:

- could target `NVDA` or `META` more precisely
- might improve return/risk balance further

Cons:

- adds complexity before the simpler utilization explanation is tested
- harder to reason about if the next run changes materially

## Design

### 1. Parameter Changes

Update QC defaults in `quantconnect/strategy_init.py` to:

- `target_margin_utilization = 0.58`
- `symbol_assignment_base_cap = 0.36`
- `stock_inventory_base_cap = 0.24`
- `stock_inventory_block_threshold = 0.92`
- `max_assignment_risk_per_trade = 0.25`
- `position_aggressiveness = 1.35`

Rationale:

- `target_margin_utilization` directly raises deployable capital
- `symbol_assignment_base_cap` reopens premium-rich symbol capacity
- `stock_inventory_base_cap` and `stock_inventory_block_threshold` reduce overblocking after partial stock accumulation
- `max_assignment_risk_per_trade` allows slightly larger individual put exposures where overall portfolio controls already pass
- `position_aggressiveness` lifts derived contract caps without manually hardcoding more contract count knobs

### 2. Execution Change

In `quantconnect/execution.py`, widen the option matching tolerance used by signal execution:

- `delta_tolerance = 0.08` instead of `0.05`

Rationale:

- current logs show repeated `No suitable options for` outcomes
- a modestly wider tolerance should improve conversion from accepted signal to actual trade
- this is a bounded change and does not alter the signal-generation layer itself

### 3. Parity Sync

Mirror the default changes into `core/backtesting/qc_parity.py` so the self-built QC replay path stays aligned with QC defaults.

This keeps:

- page defaults
- parity config defaults
- QC runtime defaults

in the same range after the change.

### 4. No Changes In This Phase

Explicitly leave these areas unchanged:

- assigned-stock fail-safe parameters and logic
- signal-quality hard filter thresholds
- scoring weights
- covered-call repair logic
- cooldown logic

The goal is to test whether capacity reopening alone is enough to materially improve return.

## Data Flow Impact

After the change:

1. more symbols pass position sizing with non-zero quantity
2. more high-premium names can use additional assignment-risk budget
3. more accepted signals find executable contracts
4. existing downstream protections still apply if positions deteriorate later

## Risks

- drawdown may rise if the extra capacity concentrates too quickly in a few premium-heavy names
- widening delta tolerance could slightly reduce contract quality in exchange for higher trade conversion
- if the latest backtests were actually run on stale code, return changes may be confounded by environment drift rather than just these defaults

## Testing

Update and run targeted tests covering:

- QC default parameter values in `quantconnect/strategy_init.py`
- parity default alignment in `core/backtesting/qc_parity.py`
- any page/default propagation tests that assert QC defaults
- execution behavior tests that depend on delta tolerance, if present

Minimum verification command:

```bash
pytest tests/unit/test_binbin_qc_parity.py tests/unit/test_binbin_god_page.py tests/unit/test_strategies.py tests/unit/test_roll_logic.py tests/unit/test_page_imports.py -q
```

## Success Criteria

The next QC run should show:

- higher effective trade conversion than the latest reviewed run
- fewer `PUT_BLOCK` events caused by `symnot`, `stockcap`, and `assigntrade`
- fewer `No suitable options for` events
- materially higher total return than the recent `~32%` run

without needing to remove the newly added protection layers.
