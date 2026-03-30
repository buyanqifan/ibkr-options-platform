"""Utilities to adapt QC exports/logs into the local parity trace format."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


def _normalize_trace_item(item: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(item)
    if "type" in normalized and "event_type" not in normalized:
        normalized["event_type"] = normalized.pop("type")
    if "quantity" in normalized and "qty" not in normalized:
        normalized["qty"] = normalized["quantity"]
    return normalized


def adapt_qc_trace(source: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Adapt JSON, dict, or raw log text into a parity trace payload."""
    if source is None:
        return {"event_trace": [], "portfolio_snapshots": []}

    if isinstance(source, dict):
        return {
            "event_trace": [_normalize_trace_item(item) for item in source.get("event_trace", [])],
            "portfolio_snapshots": list(source.get("portfolio_snapshots", [])),
        }

    if isinstance(source, (str, Path)):
        text = Path(source).read_text() if Path(str(source)).exists() else str(source)
        stripped = text.strip()
        if stripped.startswith("{"):
            payload = json.loads(stripped)
            return adapt_qc_trace(payload)
        return _adapt_qc_log_text(text)

    raise TypeError(f"Unsupported QC trace source: {type(source)!r}")


def _adapt_qc_log_text(text: str) -> Dict[str, List[Dict[str, Any]]]:
    event_trace: List[Dict[str, Any]] = []
    portfolio_snapshots: List[Dict[str, Any]] = []

    signal_pattern = re.compile(r"(?P<kind>SP_SIGNAL|CC_SIGNAL):\s*(?P<symbol>[A-Z]+)\s+delta=(?P<delta>[-+]?\d+(?:\.\d+)?)")
    assign_put = re.compile(r"Put assigned:\s*\+(?P<qty>\d+)\s+(?P<symbol>[A-Z]+)\s+@\s+\$(?P<strike>\d+(?:\.\d+)?)")
    assign_call = re.compile(r"Call assigned:\s*-(?P<qty>\d+)\s+(?P<symbol>[A-Z]+)\s+@\s+\$(?P<strike>\d+(?:\.\d+)?)")
    portfolio_pattern = re.compile(r"Final Portfolio:\s*\$(?P<value>[-+]?\d[\d,]*(?:\.\d+)?)")

    for index, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        match = signal_pattern.search(line)
        if match:
            event_trace.append(
                {
                    "seq": index,
                    "event_type": "signal_generated",
                    "symbol": match.group("symbol"),
                    "action": "SELL_PUT" if match.group("kind") == "SP_SIGNAL" else "SELL_CALL",
                    "delta": float(match.group("delta")),
                }
            )
            continue
        match = assign_put.search(line)
        if match:
            event_trace.append(
                {
                    "seq": index,
                    "event_type": "assigned_put",
                    "symbol": match.group("symbol"),
                    "qty": int(match.group("qty")),
                    "assignment": True,
                    "right": "P",
                    "strike": float(match.group("strike")),
                }
            )
            continue
        match = assign_call.search(line)
        if match:
            event_trace.append(
                {
                    "seq": index,
                    "event_type": "assigned_call",
                    "symbol": match.group("symbol"),
                    "qty": int(match.group("qty")),
                    "assignment": True,
                    "right": "C",
                    "strike": float(match.group("strike")),
                }
            )
            continue
        match = portfolio_pattern.search(line)
        if match:
            portfolio_snapshots.append(
                {
                    "phase": "final",
                    "portfolio_value": float(match.group("value").replace(",", "")),
                }
            )

    return {"event_trace": event_trace, "portfolio_snapshots": portfolio_snapshots}
