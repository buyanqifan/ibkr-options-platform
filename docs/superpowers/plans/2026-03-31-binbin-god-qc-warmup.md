# Binbin God QC Warmup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align local `binbin_god` QC replay warmup behavior with QuantConnect so the first 60 bars are non-trading warmup bars and ML pretraining happens once when warmup completes.

**Architecture:** The change stays inside the `binbin_god` QC replay path. `BacktestEngine._run_binbin_god_qc_parity()` will gain a small warmup lifecycle gate and one-time warmup-finished pretraining hook. Tests will prove no trading occurs during warmup and that pretraining is triggered once when enabled.

**Tech Stack:** Python, pytest, local backtesting engine

---

### Task 1: Add failing QC warmup regression tests

**Files:**
- Modify: `tests/unit/test_backtest_engine_qc_replay.py`
- Test: `tests/unit/test_backtest_engine_qc_replay.py`

- [ ] **Step 1: Write the failing tests**

Add focused tests that assert:

```python
def test_qc_replay_does_not_trade_before_60_bars(...):
    ...
    assert result["trades"] == []

def test_qc_replay_only_trades_after_warmup(...):
    ...
    assert all(trade["entry_date"] > warmup_end_date for trade in result["trades"])

def test_qc_replay_warmup_pretrains_ml_once(...):
    ...
    assert called["count"] == 1
```

Use monkeypatching for `_get_historical_data`, `_rolling_hv`, and `strategy.pretrain_ml_model` or the instantiated strategy class so the tests are deterministic.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/unit/test_backtest_engine_qc_replay.py -q
```

Expected: FAIL because QC replay currently trades from the first bar and has no explicit warmup-finished hook.

- [ ] **Step 3: Write minimal implementation**

No implementation in this task.

- [ ] **Step 4: Re-run to confirm failures are still targeted**

Run:

```bash
pytest tests/unit/test_backtest_engine_qc_replay.py -q
```

Expected: same targeted failures, no unrelated breakage.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_backtest_engine_qc_replay.py
git commit -m "test(backtesting): cover binbin god QC warmup behavior"
```

### Task 2: Implement warmup lifecycle in QC replay runner

**Files:**
- Modify: `core/backtesting/engine.py`
- Possibly modify: `core/backtesting/strategies/binbin_god.py`
- Test: `tests/unit/test_backtest_engine_qc_replay.py`

- [ ] **Step 1: Implement the smallest warmup gate**

Update `core/backtesting/engine.py` so `_run_binbin_god_qc_parity()`:

- defines a warmup length of 60 bars
- skips trading actions while `i < warmup_bars`
- runs a one-time warmup-finished ML pretrain hook immediately after warmup
- then resumes the existing QC replay loop unchanged

If strategy state is needed, add only minimal fields such as:

```python
self._ml_pretrained = False
```

in `core/backtesting/strategies/binbin_god.py`.

- [ ] **Step 2: Run targeted warmup tests**

Run:

```bash
pytest tests/unit/test_backtest_engine_qc_replay.py -q
```

Expected: PASS

- [ ] **Step 3: Run parity regression tests**

Run:

```bash
pytest tests/unit/test_binbin_qc_parity.py tests/unit/test_backtest_engine_qc_replay.py -q
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add core/backtesting/engine.py core/backtesting/strategies/binbin_god.py tests/unit/test_backtest_engine_qc_replay.py
git commit -m "feat(backtesting): add QC warmup lifecycle to binbin god replay"
```

### Task 3: Run full targeted regression verification

**Files:**
- Modify: any touched file above only if regression cleanup is needed

- [ ] **Step 1: Run full targeted regression suite**

Run:

```bash
pytest tests/unit/test_page_imports.py tests/unit/test_binbin_god_page.py tests/unit/test_binbin_qc_parity.py tests/unit/test_backtest_engine_qc_replay.py tests/unit/test_strategies.py -q
```

Expected: PASS

- [ ] **Step 2: Inspect final branch diff**

Run:

```bash
git status --short
git diff --stat main...HEAD
```

Expected: only warmup-alignment files and tests are changed.

- [ ] **Step 3: Final cleanup commit if needed**

If regression cleanup changed files after Task 2:

```bash
git add core/backtesting/engine.py core/backtesting/strategies/binbin_god.py tests/unit/test_backtest_engine_qc_replay.py tests/unit/test_binbin_qc_parity.py tests/unit/test_strategies.py
git commit -m "test(backtesting): verify binbin god QC warmup alignment"
```

- [ ] **Step 4: Prepare for branch finishing**

After tests are green, use the branch-finishing workflow to merge, push, or preserve the worktree.
