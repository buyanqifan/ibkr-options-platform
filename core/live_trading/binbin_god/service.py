"""Service layer for the Binbin God live page."""

from __future__ import annotations

from typing import Any

from .models import ControlCommandType
from .repository import BinbinGodLiveRepository


class BinbinGodLiveService:
    """High-level helpers for page callbacks and worker control."""

    def __init__(self, repository: BinbinGodLiveRepository):
        self.repository = repository

    def get_dashboard_data(self) -> dict[str, Any]:
        config = self.repository.get_or_create_config()
        state = self.repository.get_state()
        orders = self.repository.list_order_audits(limit=20)
        events = self.repository.list_events(limit=20)
        detail = state.detail if state else {}
        recovery = state.recovery_summary if state else {}
        return {
            "config": config.params,
            "config_status": config.status,
            "allow_new_entries": config.allow_new_entries,
            "state": {
                "status": state.status if state else "stopped",
                "account_id": state.account_id if state else None,
                "heartbeat_at": state.heartbeat_at.isoformat() if state and state.heartbeat_at else None,
                "last_error": state.last_error if state else None,
                "detail": detail,
                "recovery_status": state.recovery_status if state else None,
                "recovery_summary": recovery,
            },
            "orders": [
                {
                    "symbol": row.symbol,
                    "action": row.action,
                    "right": row.right,
                    "strike": row.strike,
                    "expiry": row.expiry,
                    "quantity": row.quantity,
                    "limit_price": row.limit_price,
                    "status": row.status,
                    "reason": row.reason,
                }
                for row in orders
            ],
            "fills": detail.get("recent_fills", []),
            "positions": detail.get("broker_positions", []),
            "events": [
                {
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "severity": row.severity,
                    "event_type": row.event_type,
                    "message": row.message,
                }
                for row in events
            ],
        }

    def save_config(self, params: dict[str, Any]) -> dict[str, Any]:
        row = self.repository.save_config(params)
        return row.params

    def enqueue_command(self, command_type: ControlCommandType, payload: dict[str, Any] | None = None) -> None:
        self.repository.enqueue_command(command_type, payload)
