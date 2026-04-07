"""Tests for navbar language persistence and mobile toggling."""

from __future__ import annotations

import importlib
import sys


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


def test_update_navbar_language_uses_persisted_language_value():
    navbar = _load_navbar_module()

    nav = navbar.update_navbar_language("zh")
    selector = _find_by_id(nav, "language-selector")

    assert selector is not None
    assert selector.value == "zh"
    assert "仪表盘" in str(nav)


def test_toggle_navbar_collapses_and_expands():
    navbar = _load_navbar_module()

    assert navbar.toggle_navbar(None, False) is False
    assert navbar.toggle_navbar(1, False) is True
    assert navbar.toggle_navbar(2, True) is False
