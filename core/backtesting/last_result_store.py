"""Lightweight persistence for the latest Binbin God backtest result."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from utils.logger import setup_logger

logger = setup_logger("last_result_store")
_LAST_RESULT_PATH = Path("data/binbin_god_last_result.json")


def save_last_binbin_god_result(params: dict, result: dict) -> None:
    """Persist the latest successful Binbin God run to a JSON file."""
    _LAST_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": datetime.utcnow().isoformat(),
        "params": params,
        "result": result,
    }
    _LAST_RESULT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def load_last_binbin_god_result() -> dict | None:
    """Load the latest persisted Binbin God run, if available."""
    if not _LAST_RESULT_PATH.exists():
        return None

    try:
        payload = json.loads(_LAST_RESULT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load last Binbin God result: %s", exc)
        return None

    if not isinstance(payload, dict):
        return None
    if "params" not in payload or "result" not in payload:
        return None
    return payload
