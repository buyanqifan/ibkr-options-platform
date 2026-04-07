# UI Routing Language Debug Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make app-shell language selection persist across navigation/refresh, make the mobile navbar toggler functional, and show full traceback details in-page for route render failures.

**Architecture:** Keep the current manual Dash routing model and full-page navigation, but move language state into browser-local storage as the source of truth for navbar rendering. Add one focused navbar toggle callback and upgrade route error rendering to include traceback details instead of only the exception message.

**Tech Stack:** Dash, dash-bootstrap-components, pytest

---

### Task 1: Add failing app-shell regression tests

**Files:**
- Create: `tests/unit/test_navbar.py`
- Modify: `tests/unit/test_layout_routing.py`
- Test: `tests/unit/test_navbar.py`
- Test: `tests/unit/test_layout_routing.py`

- [ ] **Step 1: Write the failing navbar persistence and toggler tests**

```python
"""Tests for navbar language persistence and mobile toggling."""

from __future__ import annotations

import importlib
import sys

import dash


def _load_navbar_module():
    sys.modules.pop("app.components.navbar", None)
    return importlib.import_module("app.components.navbar")


def _find_by_id(node, target_id):
    stack = [node]
    while stack:
        current = stack.pop()
        if current is None:
            continue
        if isinstance(current, (list, tuple)):
            stack.extend(current)
            continue
        if isinstance(current, (str, int, float, bool)):
            continue
        if getattr(current, "id", None) == target_id:
            return current
        children = getattr(current, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None:
            stack.append(children)
    return None


def test_update_navbar_language_uses_persisted_language_value(monkeypatch):
    navbar = _load_navbar_module()

    nav = navbar.update_navbar_language("zh")
    selector = _find_by_id(nav, "language-selector")

    assert selector.value == "zh"
    assert any("仪表盘" in str(child) for child in nav.children)


def test_toggle_navbar_collapses_and_expands():
    navbar = _load_navbar_module()

    assert navbar.toggle_navbar(None, False) is False
    assert navbar.toggle_navbar(1, False) is True
    assert navbar.toggle_navbar(2, True) is False
```

- [ ] **Step 2: Write the failing layout persistence and traceback tests**

```python
def test_create_layout_uses_local_language_store(monkeypatch):
    layout_module = _load_layout_module(monkeypatch)

    layout = layout_module.create_layout()
    stores = _collect_components_by_type(layout, "Store")
    language_store = next(store for store in stores if store.id == "language-store")

    assert language_store.storage_type == "local"


def test_display_page_includes_traceback_details_for_render_errors(monkeypatch):
    layout_module = _load_layout_module(monkeypatch)

    def broken_layout():
        raise RuntimeError("boom")

    monkeypatch.setattr(
        dash,
        "page_registry",
        {"pages.dashboard": {"layout": broken_layout}},
        raising=False,
    )
    monkeypatch.setattr(layout_module, "_ROUTE_KEYS", {"/": "pages.dashboard"})

    page = layout_module.display_page("/")

    assert "RuntimeError" in str(page.children)
    assert "boom" in str(page.children)
    assert "Traceback" in str(page.children)
```

- [ ] **Step 3: Run the focused tests to verify they fail**

Run:

```bash
pytest tests/unit/test_navbar.py tests/unit/test_layout_routing.py -q
```

Expected:

```text
FAIL test_update_navbar_language_uses_persisted_language_value
FAIL test_toggle_navbar_collapses_and_expands
FAIL test_create_layout_uses_local_language_store
FAIL test_display_page_includes_traceback_details_for_render_errors
```

- [ ] **Step 4: Commit the failing-test scaffold**

```bash
git add tests/unit/test_navbar.py tests/unit/test_layout_routing.py
git commit -m "test(ui): cover shell routing and language regressions"
```

### Task 2: Implement persisted language state in the app shell

**Files:**
- Modify: `app/layout.py`
- Modify: `app/components/navbar.py`
- Test: `tests/unit/test_navbar.py`
- Test: `tests/unit/test_layout_routing.py`

- [ ] **Step 1: Update the shell store to persist language locally**

```python
# app/layout.py
dcc.Store(id="language-store", data="en", storage_type="local")
```

- [ ] **Step 2: Keep the layout callback focused on storing the selector value**

```python
# app/layout.py
@callback(
    Output("language-store", "data"),
    Input("language-selector", "value"),
    prevent_initial_call=True,
)
def update_language(selected_lang):
    return selected_lang if selected_lang else "en"
```

- [ ] **Step 3: Make navbar item generation accept the selected language explicitly**

```python
# app/components/navbar.py
def create_navbar_items(lang="en", selected_lang=None):
    selected_lang = selected_lang or lang
    ...
    dcc.Dropdown(
        id="language-selector",
        options=[
            {"label": "🇺🇸 English", "value": "en"},
            {"label": "🇨🇳 中文", "value": "zh"},
        ],
        value=selected_lang,
        clearable=False,
        style={"width": "150px"},
    )
```

- [ ] **Step 4: Rebuild navbar content from `language-store` instead of from the dropdown directly**

```python
# app/components/navbar.py
@callback(
    Output("navbar-collapse", "children"),
    Input("language-store", "data"),
    prevent_initial_call=False,
)
def update_navbar_language(lang):
    lang = lang or "en"
    navbar_items = create_navbar_items(lang, selected_lang=lang)
    return dbc.Nav(navbar_items, className="ms-auto", navbar=True)
```

- [ ] **Step 5: Run the focused tests to verify the persistence implementation passes**

Run:

```bash
pytest tests/unit/test_navbar.py tests/unit/test_layout_routing.py -q
```

Expected:

```text
PASS
```

- [ ] **Step 6: Commit the language persistence change**

```bash
git add app/layout.py app/components/navbar.py tests/unit/test_navbar.py tests/unit/test_layout_routing.py
git commit -m "fix(ui): persist navbar language selection"
```

### Task 3: Implement the mobile navbar toggler

**Files:**
- Modify: `app/components/navbar.py`
- Test: `tests/unit/test_navbar.py`

- [ ] **Step 1: Add the navbar toggle callback**

```python
# app/components/navbar.py
from dash import State


@callback(
    Output("navbar-collapse", "is_open"),
    Input("navbar-toggler", "n_clicks"),
    State("navbar-collapse", "is_open"),
    prevent_initial_call=False,
)
def toggle_navbar(n_clicks, is_open):
    if not n_clicks:
        return is_open
    return not is_open
```

- [ ] **Step 2: Run the navbar tests to verify toggling works**

Run:

```bash
pytest tests/unit/test_navbar.py -q
```

Expected:

```text
PASS
```

- [ ] **Step 3: Commit the toggler fix**

```bash
git add app/components/navbar.py tests/unit/test_navbar.py
git commit -m "fix(ui): wire mobile navbar toggler"
```

### Task 4: Show full traceback details for route render failures

**Files:**
- Modify: `app/layout.py`
- Modify: `tests/unit/test_layout_routing.py`

- [ ] **Step 1: Upgrade `display_page()` to render the full traceback**

```python
# app/layout.py
import traceback


def display_page(pathname):
    try:
        return _render_page(pathname)
    except Exception as exc:
        trace = traceback.format_exc()
        return dbc.Alert(
            [
                html.H5("Page render error", className="alert-heading"),
                html.P(f"{exc.__class__.__name__}: {exc}"),
                html.Pre(trace, className="small mb-0", style={"whiteSpace": "pre-wrap", "overflowX": "auto"}),
            ],
            color="danger",
            className="mt-4",
        )
```

- [ ] **Step 2: Run the routing tests to verify traceback output**

Run:

```bash
pytest tests/unit/test_layout_routing.py -q
```

Expected:

```text
PASS
```

- [ ] **Step 3: Commit the traceback debug view**

```bash
git add app/layout.py tests/unit/test_layout_routing.py
git commit -m "fix(ui): show full route traceback in-page"
```

### Task 5: Run the regression slice and finalize

**Files:**
- Modify: `tests/unit/test_loading_interaction.py` (only if needed for assertion updates)
- Verify: `app/layout.py`
- Verify: `app/components/navbar.py`

- [ ] **Step 1: Run the targeted regression suite**

Run:

```bash
pytest tests/unit/test_navbar.py tests/unit/test_layout_routing.py tests/unit/test_loading_interaction.py tests/unit/test_page_imports.py tests/unit/test_binbin_god_page.py -q
```

Expected:

```text
All tests pass with 0 failures
```

- [ ] **Step 2: If `test_loading_interaction.py` needs updates, keep them minimal and rerun**

```python
# Only adjust assertions if the shell structure changed while preserving:
# - non-blocking loading wrappers
# - existing pointer-events override
```

- [ ] **Step 3: Commit any final test adjustments**

```bash
git add app/layout.py app/components/navbar.py tests/unit/test_navbar.py tests/unit/test_layout_routing.py tests/unit/test_loading_interaction.py
git commit -m "test(ui): lock shell routing regressions"
```

- [ ] **Step 4: Prepare integration**

Run:

```bash
git status --short
git log --oneline -5
```

Expected:

```text
Working tree clean except for intended plan/spec docs
Recent commits show the shell fixes in order
```
