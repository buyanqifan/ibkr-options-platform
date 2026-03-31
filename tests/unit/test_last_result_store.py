"""Tests for the lightweight Binbin God last-result persistence store."""

from __future__ import annotations


def test_save_and_load_last_binbin_god_result(tmp_path, monkeypatch):
    from core.backtesting import last_result_store

    monkeypatch.setattr(last_result_store, "_LAST_RESULT_PATH", tmp_path / "last.json")

    last_result_store.save_last_binbin_god_result(
        {"strategy": "binbin_god"},
        {"metrics": {"total_return_pct": 12.3}},
    )

    payload = last_result_store.load_last_binbin_god_result()

    assert payload["params"]["strategy"] == "binbin_god"
    assert payload["result"]["metrics"]["total_return_pct"] == 12.3
    assert payload["saved_at"]


def test_load_last_binbin_god_result_returns_none_for_missing_file(tmp_path, monkeypatch):
    from core.backtesting import last_result_store

    monkeypatch.setattr(last_result_store, "_LAST_RESULT_PATH", tmp_path / "missing.json")

    assert last_result_store.load_last_binbin_god_result() is None


def test_load_last_binbin_god_result_returns_none_for_invalid_json(tmp_path, monkeypatch):
    from core.backtesting import last_result_store

    path = tmp_path / "broken.json"
    path.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(last_result_store, "_LAST_RESULT_PATH", path)

    assert last_result_store.load_last_binbin_god_result() is None
