"""Tests for top-level Dash routing behavior."""

from __future__ import annotations

import importlib
import sys

import dash
from dash import html
import dash.dcc as dcc


def _load_layout_module(monkeypatch):
    fake_registry = {
        "pages.dashboard": {"layout": html.Div("dashboard")},
        "pages.market_data": {"layout": html.Div("market-data")},
        "pages.screener": {"layout": html.Div("screener")},
        "pages.options_chain": {"layout": html.Div("options-chain")},
        "pages.backtester": {"layout": html.Div("backtester")},
        "pages.backtest_history": {"layout": html.Div("backtest-history")},
        "pages.binbin_god": {"layout": lambda: html.Div("binbin-god")},
        "pages.binbin_god_live": {"layout": html.Div("binbin-god-live")},
        "pages.settings": {"layout": html.Div("settings")},
    }
    monkeypatch.setattr(dash, "page_registry", fake_registry, raising=False)
    sys.modules.pop("app.layout", None)
    return importlib.import_module("app.layout")


def test_display_page_calls_callable_layout(monkeypatch):
    layout_module = _load_layout_module(monkeypatch)

    page = layout_module.display_page("/binbin-god")

    assert not callable(page)
    assert page.children == "binbin-god"


def test_display_page_returns_404_for_unknown_route(monkeypatch):
    layout_module = _load_layout_module(monkeypatch)

    page = layout_module.display_page("/does-not-exist")

    assert not callable(page)
    assert "404" in str(page.children)


def _collect_components_by_type(node, component_type):
    found = []
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
        if current.__class__.__name__ == component_type:
            found.append(current)
        children = getattr(current, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None:
            stack.append(children)
    return found


def test_create_layout_uses_local_language_store(monkeypatch):
    layout_module = _load_layout_module(monkeypatch)

    layout = layout_module.create_layout()
    stores = _collect_components_by_type(layout, dcc.Store.__name__)
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

    text = str(page)
    assert "RuntimeError" in text
    assert "boom" in text
    assert "Traceback" in text
