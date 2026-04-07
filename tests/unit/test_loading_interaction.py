"""Regression tests for loading wrappers blocking page interaction."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import dash


PAGE_MODULES = [
    "app.pages.market_data",
    "app.pages.options_chain",
    "app.pages.screener",
    "app.pages.backtester",
    "app.pages.backtest_history",
    "app.pages.binbin_god",
]


def _load_page(monkeypatch, module_name: str):
    monkeypatch.setattr(dash, "register_page", lambda *args, **kwargs: None)
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def _collect_components_by_type(node: Any, target_type: str) -> list[Any]:
    found: list[Any] = []
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
        if current.__class__.__name__ == target_type:
            found.append(current)
        children = getattr(current, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None:
            stack.append(children)
    return found


def _layout_for(page):
    layout = page.layout
    return layout() if callable(layout) else layout


def test_page_loading_wrappers_do_not_show_before_user_action(monkeypatch):
    for module_name in PAGE_MODULES:
        page = _load_page(monkeypatch, module_name)
        layout = _layout_for(page)
        loading_components = _collect_components_by_type(layout, "Loading")

        for loading in loading_components:
            props = loading.to_plotly_json()["props"]
            assert props.get("show_initially") is False, module_name


def test_root_layout_includes_non_blocking_global_loading_style(monkeypatch):
    fake_registry = {
        "pages.dashboard": {"layout": "dashboard"},
        "pages.market_data": {"layout": "market-data"},
        "pages.screener": {"layout": "screener"},
        "pages.options_chain": {"layout": "options-chain"},
        "pages.backtester": {"layout": "backtester"},
        "pages.backtest_history": {"layout": "backtest-history"},
        "pages.binbin_god": {"layout": "binbin-god"},
        "pages.binbin_god_live": {"layout": "binbin-god-live"},
        "pages.settings": {"layout": "settings"},
    }
    monkeypatch.setattr(dash, "page_registry", fake_registry, raising=False)
    sys.modules.pop("app.layout", None)
    layout_module = importlib.import_module("app.layout")

    layout = layout_module.create_layout()
    markdown_components = _collect_components_by_type(layout, "Markdown")

    styles = [component.children for component in markdown_components if isinstance(component.children, str)]
    assert any(
        "_dash-loading-callback" in style and "pointer-events: none" in style
        for style in styles
    )
