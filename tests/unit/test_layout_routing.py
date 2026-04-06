"""Tests for top-level Dash routing behavior."""

from __future__ import annotations

import importlib
import sys

import dash
from dash import html


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
