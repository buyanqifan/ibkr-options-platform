"""Domain models for Binbin God live trading."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class StrategyStatus(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"
    RECOVERING = "recovering"


class RecoveryStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    BLOCKED = "blocked"


class ControlCommandType(str, Enum):
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    EMERGENCY_STOP = "emergency_stop"
    BLOCK_NEW_ENTRIES = "block_new_entries"
    ALLOW_NEW_ENTRIES = "allow_new_entries"
    APPLY_CONFIG = "apply_config"
    CANCEL_OPEN_ENTRY_ORDERS = "cancel_open_entry_orders"
    REFRESH_NOW = "refresh_now"


class CommandStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ActionType(str, Enum):
    OPEN_PUT = "open_put"
    SELL_CALL = "sell_call"
    REPAIR_CALL = "repair_call"
    ROLL_PUT = "roll_put"
    ROLL_CALL = "roll_call"
    HOLD = "hold"
    DEFER = "defer"


@dataclass(slots=True)
class StrategyAction:
    action_type: ActionType
    symbol: str
    reason: str
    right: str | None = None
    quantity: int = 0
    expiry: str | None = None
    strike: float | None = None
    target_delta: float | None = None
    metadata: dict = field(default_factory=dict)
