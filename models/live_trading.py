"""Database models for live trading strategy control and audit."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base


class LiveStrategyConfig(Base):
    __tablename__ = "live_strategy_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    schema_version: Mapped[str] = mapped_column(String(20), nullable=False, default="v1")
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="stopped")
    allow_new_entries: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    params: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class LiveStrategyState(Base):
    __tablename__ = "live_strategy_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    schema_version: Mapped[str] = mapped_column(String(20), nullable=False, default="v1")
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="stopped")
    account_id: Mapped[str | None] = mapped_column(String(50))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime)
    last_error: Mapped[str | None] = mapped_column(Text)
    detail: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    recovery_status: Mapped[str | None] = mapped_column(String(30))
    recovery_summary: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    code_version: Mapped[str | None] = mapped_column(String(50))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class LiveControlCommand(Base):
    __tablename__ = "live_control_commands"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False)
    command_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime)


class LiveOrderAudit(Base):
    __tablename__ = "live_order_audits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    broker_order_id: Mapped[str | None] = mapped_column(String(50))
    action: Mapped[str] = mapped_column(String(30), nullable=False)
    right: Mapped[str | None] = mapped_column(String(1))
    strike: Mapped[float | None] = mapped_column(Float)
    expiry: Mapped[str | None] = mapped_column(String(20))
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    limit_price: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="submitted")
    intent_key: Mapped[str | None] = mapped_column(String(120))
    reason: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class LiveExecutionEvent(Base):
    __tablename__ = "live_execution_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False, default="info")
    message: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class LiveWorkerLease(Base):
    __tablename__ = "live_worker_leases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    instance_id: Mapped[str] = mapped_column(String(80), nullable=False)
    code_version: Mapped[str] = mapped_column(String(50), nullable=False, default="unknown")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    heartbeat_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
