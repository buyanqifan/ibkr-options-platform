# Binbin God Last Result Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the most recent successful `binbin_god` backtest result on the server and automatically restore it when the page loads again.

**Architecture:** Add a focused JSON-backed persistence helper under `core/backtesting/` and wire the `binbin_god` page to save successful results and restore them on first load. Reuse the existing page result rendering path so restored state matches freshly-run state and keep the feature isolated from `BacktestStorage` history.

**Tech Stack:** Python, Dash callbacks, pytest, JSON file persistence

---

### Task 1: Add the lightweight server-side last-result store

**Files:**
- Create: `core/backtesting/last_result_store.py`
- Test: `tests/unit/test_last_result_store.py`

- [ ] **Step 1: Write the failing persistence tests**

```python
from core.backtesting.last_result_store import (
    load_last_binbin_god_result,
    save_last_binbin_god_result,
)


def test_save_and_load_last_binbin_god_result(tmp_path, monkeypatch):
    monkeypatch.setattr("core.backtesting.last_result_store._LAST_RESULT_PATH", tmp_path / "last.json")

    save_last_binbin_god_result({"strategy": "binbin_god"}, {"metrics": {"total_return_pct": 12.3}})

    payload = load_last_binbin_god_result()

    assert payload["params"]["strategy"] == "binbin_god"
    assert payload["result"]["metrics"]["total_return_pct"] == 12.3


def test_load_last_binbin_god_result_returns_none_for_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr("core.backtesting.last_result_store._LAST_RESULT_PATH", tmp_path / "missing.json")

    assert load_last_binbin_god_result() is None


def test_load_last_binbin_god_result_returns_none_for_invalid_json(tmp_path, monkeypatch):
    path = tmp_path / "broken.json"
    path.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr("core.backtesting.last_result_store._LAST_RESULT_PATH", path)

    assert load_last_binbin_god_result() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_last_result_store.py -q`
Expected: FAIL with `ModuleNotFoundError` or missing function errors for `core.backtesting.last_result_store`

- [ ] **Step 3: Write the minimal persistence module**

```python
import json
from datetime import datetime
from pathlib import Path

from utils.logger import setup_logger

logger = setup_logger("last_result_store")
_LAST_RESULT_PATH = Path("data/binbin_god_last_result.json")


def save_last_binbin_god_result(params: dict, result: dict) -> None:
    _LAST_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": datetime.utcnow().isoformat(),
        "params": params,
        "result": result,
    }
    _LAST_RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")


def load_last_binbin_god_result() -> dict | None:
    if not _LAST_RESULT_PATH.exists():
        return None
    try:
        payload = json.loads(_LAST_RESULT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load last Binbin God result: %s", exc)
        return None
    if not isinstance(payload, dict) or "params" not in payload or "result" not in payload:
        return None
    return payload
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_last_result_store.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/backtesting/last_result_store.py tests/unit/test_last_result_store.py
git commit -m "feat(backtesting): persist last binbin god result"
```

### Task 2: Save and restore the last result in the Binbin God page

**Files:**
- Modify: `app/pages/binbin_god.py`
- Test: `tests/unit/test_binbin_god_page.py`

- [ ] **Step 1: Write the failing page tests**

```python
def test_run_binbin_backtest_saves_last_result(monkeypatch):
    page = _load_binbin_god_page(monkeypatch)
    saved = {}

    monkeypatch.setattr(page, "save_last_binbin_god_result", lambda params, result: saved.update({"params": params, "result": result}))
    monkeypatch.setattr(page, "get_services_cached", lambda: {"backtest_engine": StubEngine()})

    page.run_binbin_backtest(n_clicks=1, **_default_form_inputs())

    assert saved["params"]["strategy"] == "binbin_god"
    assert saved["result"]["metrics"]["total_return_pct"] == 1.0


def test_restore_binbin_backtest_restores_saved_result(monkeypatch):
    page = _load_binbin_god_page(monkeypatch)
    monkeypatch.setattr(
        page,
        "load_last_binbin_god_result",
        lambda: {
            "params": page.build_binbin_backtest_params(_default_form_inputs()),
            "result": StubEngine().run({}),
        },
    )

    restored = page.restore_binbin_backtest_on_load("/binbin-god")

    assert restored[0]["metrics"]["total_return_pct"] == 1.0
    assert restored[3] != dash.no_update
    assert restored[5] == "d-block mt-2"
```

- [ ] **Step 2: Run page tests to verify they fail**

Run: `pytest tests/unit/test_binbin_god_page.py -q`
Expected: FAIL because the save/load hooks and restore callback do not exist yet

- [ ] **Step 3: Implement save/restore wiring in the page**

```python
from core.backtesting.last_result_store import (
    load_last_binbin_god_result,
    save_last_binbin_god_result,
)


def _build_binbin_results_view(result: dict, params: dict):
    ...
    return mag7_section, content, {"display": "none"}, "d-block mt-2"


@callback(
    Output("binbin-results-store", "data", allow_duplicate=True),
    Output("binbin-params-store", "data", allow_duplicate=True),
    Output("binbin-mag7-analysis", "children", allow_duplicate=True),
    Output("binbin-results-container", "children", allow_duplicate=True),
    Output("bbg-loading-indicator", "style", allow_duplicate=True),
    Output("bbg-export-container", "className", allow_duplicate=True),
    Input("bbg-page-url", "pathname"),
    prevent_initial_call=False,
)
def restore_binbin_backtest_on_load(pathname):
    if pathname != "/binbin-god":
        return no_update, no_update, no_update, no_update, no_update, no_update
    payload = load_last_binbin_god_result()
    if not payload:
        return no_update, no_update, no_update, no_update, no_update, no_update
    params = payload["params"]
    result = payload["result"]
    mag7_section, content, loading_style, export_class = _build_binbin_results_view(result, params)
    return result, params, mag7_section, content, loading_style, export_class
```

- [ ] **Step 4: Run page tests to verify they pass**

Run: `pytest tests/unit/test_binbin_god_page.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/pages/binbin_god.py tests/unit/test_binbin_god_page.py
git commit -m "feat(ui): restore last binbin god backtest result"
```

### Task 3: Restore form values and run the targeted regression suite

**Files:**
- Modify: `app/pages/binbin_god.py`
- Test: `tests/unit/test_binbin_god_page.py`
- Test: `tests/unit/test_page_imports.py`

- [ ] **Step 1: Write the failing form-restore test**

```python
def test_restore_binbin_backtest_restores_form_values(monkeypatch):
    page = _load_binbin_god_page(monkeypatch)
    params = page.build_binbin_backtest_params(_default_form_inputs())
    monkeypatch.setattr(page, "load_last_binbin_god_result", lambda: {"params": params, "result": StubEngine().run({})})

    restored = page.restore_binbin_backtest_form("/binbin-god")

    assert restored[0] == params["start_date"]
    assert restored[2] == params["initial_capital"]
    assert restored[3] == ",".join(params["stock_pool"])
```

- [ ] **Step 2: Run the form-restore test to verify it fails**

Run: `pytest tests/unit/test_binbin_god_page.py::test_restore_binbin_backtest_restores_form_values -q`
Expected: FAIL because the form restore callback does not exist yet

- [ ] **Step 3: Implement form restoration and keep layout import-safe**

```python
layout = dbc.Container(
    [
        dcc.Location(id="bbg-page-url"),
        ...
    ],
    fluid=True,
)


@callback(
    Output("bbg-start", "value"),
    Output("bbg-end", "value"),
    Output("bbg-initial-capital", "value"),
    Output("bbg-stock-pool-text", "value"),
    ...,
    Input("bbg-page-url", "pathname"),
    prevent_initial_call=False,
)
def restore_binbin_backtest_form(pathname):
    if pathname != "/binbin-god":
        return tuple(no_update for _ in FORM_OUTPUT_IDS)
    payload = load_last_binbin_god_result()
    if not payload:
        return tuple(no_update for _ in FORM_OUTPUT_IDS)
    return _form_values_from_params(payload["params"])
```

- [ ] **Step 4: Run the targeted regression suite**

Run: `pytest tests/unit/test_last_result_store.py tests/unit/test_binbin_god_page.py tests/unit/test_page_imports.py -q`
Expected: PASS

- [ ] **Step 5: Run the broader Binbin God regression suite**

Run: `pytest tests/unit/test_page_imports.py tests/unit/test_binbin_god_page.py tests/unit/test_binbin_qc_parity.py tests/unit/test_backtest_engine_qc_replay.py tests/unit/test_strategies.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/pages/binbin_god.py tests/unit/test_binbin_god_page.py tests/unit/test_page_imports.py
git commit -m "test(ui): cover binbin god last result restore"
```
