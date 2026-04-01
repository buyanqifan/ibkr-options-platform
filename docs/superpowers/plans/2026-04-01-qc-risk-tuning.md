# QC Risk Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tighten BinbinGod QC defaults so lower nominal position sizing also reduces tail risk, and reduce scoring bias toward extreme-volatility names.

**Architecture:** Update QC default parameters in `quantconnect/strategy_init.py`, then let downstream parity/UI tests inherit the new defaults automatically. Adjust `quantconnect/scoring.py` so extreme realized volatility reduces stock-selection attractiveness instead of being purely rewarded.

**Tech Stack:** Python, pytest, QuantConnect strategy modules

---

### Task 1: Lock in new default expectations

**Files:**
- Modify: `tests/unit/test_binbin_qc_parity.py`
- Modify: `tests/unit/test_binbin_god_page.py`
- Modify: `tests/unit/test_strategies.py`

- [ ] Step 1: Write failing assertions for stricter defaults.
- [ ] Step 2: Run focused pytest commands and verify the new assertions fail.
- [ ] Step 3: Keep the failures isolated to the changed defaults.

### Task 2: Tighten QC risk-control defaults

**Files:**
- Modify: `quantconnect/strategy_init.py`
- Test: `tests/unit/test_binbin_qc_parity.py`
- Test: `tests/unit/test_binbin_god_page.py`
- Test: `tests/unit/test_strategies.py`

- [ ] Step 1: Lower single-symbol assignment and stock-inventory defaults.
- [ ] Step 2: Move defensive put roll defaults earlier in the loss / ITM / DTE lifecycle.
- [ ] Step 3: Re-run focused tests until they pass.

### Task 3: Penalize extreme volatility in stock scoring

**Files:**
- Modify: `tests/unit/test_binbin_qc_parity.py`
- Modify: `quantconnect/scoring.py`

- [ ] Step 1: Add a failing unit test showing extreme-volatility bars score worse than moderate-volatility bars.
- [ ] Step 2: Implement the minimal scoring penalty that makes the test pass.
- [ ] Step 3: Re-run scoring/parity tests and confirm no unrelated regressions.
