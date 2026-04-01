"""Persistence helpers for Binbin God live trading."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from models.base import Base, SessionLocal
from models.live_trading import (
    LiveControlCommand,
    LiveExecutionEvent,
    LiveOrderAudit,
    LiveStrategyConfig,
    LiveStrategyState,
    LiveWorkerLease,
)

from .defaults import build_live_defaults
from .models import CommandStatus, ControlCommandType, RecoveryStatus, StrategyStatus


STRATEGY_NAME = "binbin_god_live"
SCHEMA_VERSION = "v1"


class BinbinGodLiveRepository:
    """Storage API shared by the worker and the Dash page."""

    def __init__(self, session_factory=SessionLocal):
        self.session_factory = session_factory
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        bind = self.session_factory.kw.get("bind")
        if bind is not None:
            Base.metadata.create_all(bind=bind)

    def get_config(self) -> LiveStrategyConfig | None:
        session = self.session_factory()
        try:
            row = session.query(LiveStrategyConfig).filter_by(strategy_name=STRATEGY_NAME).first()
            if row is not None:
                session.expunge(row)
            return row
        finally:
            session.close()

    def save_config(self, params: dict[str, Any]) -> LiveStrategyConfig:
        session = self.session_factory()
        try:
            row = session.query(LiveStrategyConfig).filter_by(strategy_name=STRATEGY_NAME).first()
            defaults = build_live_defaults()
            merged = {**defaults, **(row.params if row else {}), **params}
            if "stock_pool_text" not in merged and "stock_pool" in merged:
                merged["stock_pool_text"] = ",".join(merged["stock_pool"])
            if row is None:
                row = LiveStrategyConfig(
                    strategy_name=STRATEGY_NAME,
                    schema_version=SCHEMA_VERSION,
                    status=StrategyStatus.STOPPED.value,
                    allow_new_entries=bool(merged.get("allow_new_entries", True)),
                    params=merged,
                )
                session.add(row)
            else:
                row.params = merged
                row.allow_new_entries = bool(merged.get("allow_new_entries", True))
                row.schema_version = SCHEMA_VERSION
            session.commit()
            session.refresh(row)
            session.expunge(row)
            return row
        finally:
            session.close()

    def get_or_create_config(self) -> LiveStrategyConfig:
        existing = self.get_config()
        if existing is not None:
            return existing
        return self.save_config({})

    def get_state(self) -> LiveStrategyState | None:
        session = self.session_factory()
        try:
            row = session.query(LiveStrategyState).filter_by(strategy_name=STRATEGY_NAME).first()
            if row is not None:
                session.expunge(row)
            return row
        finally:
            session.close()

    def upsert_state(
        self,
        *,
        status: StrategyStatus,
        heartbeat_at: datetime | None = None,
        account_id: str | None = None,
        detail: dict[str, Any] | None = None,
        last_error: str | None = None,
        recovery_status: RecoveryStatus | str | None = None,
        recovery_summary: dict[str, Any] | None = None,
        code_version: str | None = None,
    ) -> LiveStrategyState:
        session = self.session_factory()
        try:
            row = session.query(LiveStrategyState).filter_by(strategy_name=STRATEGY_NAME).first()
            if row is None:
                row = LiveStrategyState(
                    strategy_name=STRATEGY_NAME,
                    schema_version=SCHEMA_VERSION,
                    status=status.value,
                    detail=detail or {},
                    recovery_summary=recovery_summary or {},
                )
                session.add(row)
            else:
                row.status = status.value
                row.detail = detail or row.detail or {}
                row.recovery_summary = recovery_summary or row.recovery_summary or {}
            row.schema_version = SCHEMA_VERSION
            row.heartbeat_at = heartbeat_at or datetime.utcnow()
            row.account_id = account_id
            row.last_error = last_error
            row.recovery_status = recovery_status.value if isinstance(recovery_status, RecoveryStatus) else recovery_status
            row.code_version = code_version or row.code_version
            session.commit()
            session.refresh(row)
            session.expunge(row)
            return row
        finally:
            session.close()

    def enqueue_command(
        self,
        command_type: ControlCommandType,
        payload: dict[str, Any] | None = None,
    ) -> LiveControlCommand:
        session = self.session_factory()
        try:
            row = LiveControlCommand(
                strategy_name=STRATEGY_NAME,
                command_type=command_type.value,
                status=CommandStatus.PENDING.value,
                payload=payload or {},
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            session.expunge(row)
            return row
        finally:
            session.close()

    def list_pending_commands(self) -> list[LiveControlCommand]:
        session = self.session_factory()
        try:
            rows = (
                session.query(LiveControlCommand)
                .filter_by(strategy_name=STRATEGY_NAME, status=CommandStatus.PENDING.value)
                .order_by(LiveControlCommand.id.asc())
                .all()
            )
            for row in rows:
                session.expunge(row)
            return rows
        finally:
            session.close()

    def mark_command_processed(
        self,
        command_id: int,
        status: CommandStatus,
        error_message: str | None = None,
    ) -> None:
        session = self.session_factory()
        try:
            row = session.query(LiveControlCommand).filter_by(id=command_id).first()
            if row is None:
                return
            row.status = status.value
            row.error_message = error_message
            row.processed_at = datetime.utcnow()
            session.commit()
        finally:
            session.close()

    def upsert_lease(self, instance_id: str, code_version: str) -> LiveWorkerLease:
        session = self.session_factory()
        try:
            row = session.query(LiveWorkerLease).filter_by(strategy_name=STRATEGY_NAME).first()
            if row is None:
                row = LiveWorkerLease(
                    strategy_name=STRATEGY_NAME,
                    instance_id=instance_id,
                    code_version=code_version,
                )
                session.add(row)
            else:
                row.instance_id = instance_id
                row.code_version = code_version
                row.heartbeat_at = datetime.utcnow()
            session.commit()
            session.refresh(row)
            session.expunge(row)
            return row
        finally:
            session.close()

    def add_event(
        self,
        event_type: str,
        message: str,
        *,
        severity: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> LiveExecutionEvent:
        session = self.session_factory()
        try:
            row = LiveExecutionEvent(
                strategy_name=STRATEGY_NAME,
                event_type=event_type,
                severity=severity,
                message=message,
                payload=payload or {},
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            session.expunge(row)
            return row
        finally:
            session.close()

    def list_events(self, limit: int = 50) -> list[LiveExecutionEvent]:
        session = self.session_factory()
        try:
            rows = (
                session.query(LiveExecutionEvent)
                .filter_by(strategy_name=STRATEGY_NAME)
                .order_by(LiveExecutionEvent.created_at.desc())
                .limit(limit)
                .all()
            )
            for row in rows:
                session.expunge(row)
            return rows
        finally:
            session.close()

    def add_order_audit(self, **kwargs: Any) -> LiveOrderAudit:
        session = self.session_factory()
        try:
            row = LiveOrderAudit(strategy_name=STRATEGY_NAME, **kwargs)
            session.add(row)
            session.commit()
            session.refresh(row)
            session.expunge(row)
            return row
        finally:
            session.close()

    def list_order_audits(self, limit: int = 50) -> list[LiveOrderAudit]:
        session = self.session_factory()
        try:
            rows = (
                session.query(LiveOrderAudit)
                .filter_by(strategy_name=STRATEGY_NAME)
                .order_by(LiveOrderAudit.created_at.desc())
                .limit(limit)
                .all()
            )
            for row in rows:
                session.expunge(row)
            return rows
        finally:
            session.close()
