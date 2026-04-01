# QC Parity Margin And Divergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the local BinbinGod QC parity path with `quantconnect/` for margin accounting and make parity output identify the first event/snapshot divergence.

**Architecture:** Keep the existing parity engine flow, but change the opened put-leg allocation to use the QC-estimated margin instead of `strike * 100`, stop double-counting stock assignment as new margin, and enrich `parity_report` with the first mismatching event plus nearby portfolio snapshots. This keeps the current replay structure intact while removing the biggest accounting distortion and improving traceability.

**Tech Stack:** Python, pytest, existing parity helpers in `core/backtesting/qc_parity.py`, parity engine in `core/backtesting/engine.py`

---

### Task 1: Lock the target behavior with tests

**Files:**
- Modify: `tests/unit/test_binbin_qc_parity.py`
- Test: `tests/unit/test_binbin_qc_parity.py`

- [ ] Add a test that opens a parity-mode short put and asserts the allocated margin matches the QC estimate, not `strike * 100`.
- [ ] Add a test that builds a mismatch parity report and asserts it exposes the first mismatching event and the corresponding snapshot context.
- [ ] Run the targeted parity unit tests and confirm the new assertions fail before implementation.

### Task 2: Fix parity margin and stock ledger behavior

**Files:**
- Modify: `core/backtesting/qc_parity.py`
- Modify: `core/backtesting/engine.py`

- [ ] Add a reusable QC-estimated put margin helper in parity utilities so engine and tests share the same formula.
- [ ] Use that helper when opening parity short puts instead of `strike * 100`.
- [ ] Stop allocating extra stock “margin” on assignment in parity bookkeeping so stock inventory is reflected through portfolio value rather than duplicated margin reservations.
- [ ] Keep current call handling unchanged unless required by the new tests.

### Task 3: Enrich first-divergence diagnostics

**Files:**
- Modify: `core/backtesting/qc_parity.py`

- [ ] Extend `EventTracer.build_parity_report()` to compare a richer key set, including `strike` and `expiry`.
- [ ] When a mismatch happens, include the first mismatching actual/expected event and the same-date portfolio snapshots.
- [ ] When trace lengths differ, report the first missing/extra event with snapshot context.

### Task 4: Verify the behavior

**Files:**
- Test: `tests/unit/test_binbin_qc_parity.py`

- [ ] Run the focused parity test file.
- [ ] If environment dependencies block the full file, run the narrowest reachable tests and record the blocker clearly.
