"""Background worker for Binbin God live paper trading."""

from __future__ import annotations

import os
import socket
import time
import uuid
from datetime import datetime
from typing import Any

from config.settings import settings
from core.ibkr.trading_client import OrderRequest

from .models import ActionType, CommandStatus, ControlCommandType, RecoveryStatus, StrategyAction, StrategyStatus
from .reconstruction import reconstruct_live_state


class BinbinGodLiveWorker:
    """Long-running worker that syncs IBKR state and updates strategy state."""

    def __init__(
        self,
        *,
        repository,
        broker_client,
        market_data_client,
        code_version: str | None = None,
    ):
        self.repository = repository
        self.broker_client = broker_client
        self.market_data_client = market_data_client
        self.code_version = code_version or os.getenv("APP_VERSION", "dev")
        self.instance_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"

    def apply_startup_recovery(self) -> dict[str, Any]:
        """Recover durable state after restart without resuming trading automatically."""
        self.repository.upsert_lease(self.instance_id, self.code_version)
        config = self.repository.get_or_create_config()
        if getattr(config, "schema_version", "v1") != "v1":
            message = f"Unsupported config schema: {config.schema_version}"
            self.repository.add_event("recovery_blocked", message, severity="danger")
            self.repository.upsert_state(
                status=StrategyStatus.STOPPED,
                heartbeat_at=datetime.utcnow(),
                detail={},
                last_error=message,
                recovery_status=RecoveryStatus.BLOCKED,
                recovery_summary={"result": RecoveryStatus.BLOCKED.value, "anomaly_count": 1},
                code_version=self.code_version,
            )
            return {"recovery_status": RecoveryStatus.BLOCKED.value, "message": message}

        if settings.IBKR_TRADING_MODE.lower() != "paper":
            message = "Binbin God live worker only supports paper trading mode."
            self.repository.add_event("recovery_blocked", message, severity="danger")
            self.repository.upsert_state(
                status=StrategyStatus.STOPPED,
                heartbeat_at=datetime.utcnow(),
                detail={},
                last_error=message,
                recovery_status=RecoveryStatus.BLOCKED,
                recovery_summary={"result": RecoveryStatus.BLOCKED.value, "anomaly_count": 1},
                code_version=self.code_version,
            )
            return {"recovery_status": RecoveryStatus.BLOCKED.value, "message": message}

        try:
            account_summary = self._broker_call("get_account_summary", default={})
            positions = self._broker_call("get_positions", default=[])
            open_orders = self._broker_call("get_open_orders", default=[])
            fills = self._broker_call("get_recent_fills", default=[])
            snapshot = reconstruct_live_state(
                account_summary=account_summary,
                positions=positions,
                open_orders=open_orders,
                fills=fills,
                config=config.params,
            )
            snapshot["broker_positions"] = positions
            snapshot["recent_fills"] = fills
            recovery_result = RecoveryStatus.SUCCESS.value
            message = "Recovery completed"
        except Exception as exc:
            snapshot = {"status": "error", "symbols": {}, "phase_counts": {}, "anomaly_count": 1}
            recovery_result = RecoveryStatus.PARTIAL.value
            message = f"Recovery partial: {exc}"

        self.repository.add_event("recovery_completed", message, payload=snapshot)
        state = self.repository.upsert_state(
            status=StrategyStatus.STOPPED,
            heartbeat_at=datetime.utcnow(),
            account_id=(account_summary or {}).get("Account") if isinstance(account_summary, dict) else None,
            detail=snapshot,
            last_error=None if recovery_result == RecoveryStatus.SUCCESS.value else message,
            recovery_status=recovery_result,
            recovery_summary={
                "recovered_at": datetime.utcnow().isoformat(),
                "result": recovery_result,
                "open_orders_count": len(open_orders) if "open_orders" in locals() else 0,
                "positions_count": len(positions) if "positions" in locals() else 0,
                "anomaly_count": snapshot.get("anomaly_count", 0),
                "code_version": self.code_version,
            },
            code_version=self.code_version,
        )
        return {"recovery_status": recovery_result, "state_id": state.id}

    def run_cycle(self) -> dict[str, Any]:
        """Process commands and refresh worker heartbeat."""
        config = self.repository.get_or_create_config()
        state = self.repository.get_state() or self.repository.upsert_state(status=StrategyStatus.STOPPED)
        self.repository.upsert_lease(self.instance_id, self.code_version)

        for command in self.repository.list_pending_commands():
            self._process_command(command, config)

        state = self.repository.get_state() or state
        if state.status != StrategyStatus.RUNNING.value:
            return {"status": state.status, "actions": []}

        account_summary = self._broker_call("get_account_summary", default={})
        positions = self._broker_call("get_positions", default=[])
        open_orders = self._broker_call("get_open_orders", default=[])
        fills = self._broker_call("get_recent_fills", default=[])
        snapshot = reconstruct_live_state(
            account_summary=account_summary,
            positions=positions,
            open_orders=open_orders,
            fills=fills,
            config=config.params,
        )
        snapshot["broker_positions"] = positions
        snapshot["recent_fills"] = fills
        actions = self._generate_actions(snapshot, config.params)
        self._execute_actions(actions, config.params)
        self.repository.upsert_state(
            status=StrategyStatus.RUNNING,
            heartbeat_at=datetime.utcnow(),
            account_id=(account_summary or {}).get("Account") if isinstance(account_summary, dict) else None,
            detail=snapshot,
            recovery_status=state.recovery_status,
            recovery_summary=state.recovery_summary,
            code_version=self.code_version,
        )
        return {"status": StrategyStatus.RUNNING.value, "actions": [action.action_type.value for action in actions]}

    def run_forever(self) -> None:
        """Keep running until the process is terminated."""
        self.apply_startup_recovery()
        while True:
            config = self.repository.get_or_create_config()
            self.run_cycle()
            interval = int(config.params.get("poll_interval_seconds", 60))
            time.sleep(max(interval, 5))

    def _process_command(self, command, config) -> None:
        try:
            command_type = ControlCommandType(command.command_type)
            if command_type == ControlCommandType.START:
                if settings.IBKR_TRADING_MODE.lower() != "paper":
                    raise ValueError("Paper trading mode is required before starting")
                self.repository.upsert_state(status=StrategyStatus.RUNNING, heartbeat_at=datetime.utcnow())
                self.repository.add_event("command_start", "Strategy started by operator")
            elif command_type == ControlCommandType.PAUSE:
                self.repository.upsert_state(status=StrategyStatus.PAUSED, heartbeat_at=datetime.utcnow())
                self.repository.add_event("command_pause", "Strategy paused by operator")
            elif command_type == ControlCommandType.RESUME:
                self.repository.upsert_state(status=StrategyStatus.RUNNING, heartbeat_at=datetime.utcnow())
                self.repository.add_event("command_resume", "Strategy resumed by operator")
            elif command_type == ControlCommandType.EMERGENCY_STOP:
                self.repository.upsert_state(status=StrategyStatus.EMERGENCY_STOP, heartbeat_at=datetime.utcnow())
                self.repository.add_event("emergency_stop", "Emergency stop activated", severity="warning")
            elif command_type == ControlCommandType.BLOCK_NEW_ENTRIES:
                params = dict(config.params)
                params["allow_new_entries"] = False
                self.repository.save_config(params)
                self.repository.add_event("block_new_entries", "New entries blocked")
            elif command_type == ControlCommandType.ALLOW_NEW_ENTRIES:
                params = dict(config.params)
                params["allow_new_entries"] = True
                self.repository.save_config(params)
                self.repository.add_event("allow_new_entries", "New entries re-enabled")
            elif command_type == ControlCommandType.APPLY_CONFIG:
                self.repository.save_config(command.payload or {})
                self.repository.add_event("apply_config", "Live config applied", payload=command.payload or {})
            elif command_type == ControlCommandType.CANCEL_OPEN_ENTRY_ORDERS:
                cancelled = self._broker_call("cancel_open_entry_orders", default=0)
                self.repository.add_event(
                    "cancel_open_entry_orders",
                    f"Cancelled {cancelled} open entry orders",
                    severity="warning",
                )
            elif command_type == ControlCommandType.REFRESH_NOW:
                self.repository.add_event("refresh_now", "Manual refresh requested")
            self.repository.mark_command_processed(command.id, CommandStatus.COMPLETED)
        except Exception as exc:
            self.repository.mark_command_processed(command.id, CommandStatus.FAILED, str(exc))
            self.repository.add_event("command_failed", str(exc), severity="danger", payload={"command_id": command.id})

    def _broker_call(self, method_name: str, *, default):
        if self.broker_client is None:
            return default
        method = getattr(self.broker_client, method_name, None)
        if method is None:
            return default
        result = method()
        return default if result is None else result

    def _generate_actions(self, snapshot: dict[str, Any], config: dict[str, Any]) -> list[StrategyAction]:
        actions: list[StrategyAction] = []
        if not config.get("allow_new_entries", True):
            return actions
        for symbol in config.get("stock_pool", []):
            state = snapshot.get("symbols", {}).get(symbol, {})
            phase = state.get("phase")
            if phase == "cash" and not state.get("blocked_reasons"):
                actions.append(
                    StrategyAction(
                        action_type=ActionType.OPEN_PUT,
                        symbol=symbol,
                        reason="cash_phase_entry",
                        right="P",
                        quantity=1,
                        target_delta=float(config.get("put_delta", 0.3)),
                    )
                )
                break
            if phase == "assigned_stock":
                actions.append(
                    StrategyAction(
                        action_type=ActionType.SELL_CALL,
                        symbol=symbol,
                        reason="assigned_stock_income",
                        right="C",
                        quantity=1,
                        target_delta=float(config.get("call_delta", 0.3)),
                    )
                )
            if phase == "repair_call":
                actions.append(
                    StrategyAction(
                        action_type=ActionType.REPAIR_CALL,
                        symbol=symbol,
                        reason="underwater_stock_repair",
                        right="C",
                        quantity=1,
                        target_delta=float(config.get("call_delta", 0.3)),
                    )
                )
        return actions

    def _execute_actions(self, actions: list[StrategyAction], config: dict[str, Any]) -> None:
        if self.broker_client is None or self.market_data_client is None:
            return
        existing_intents = {
            row.intent_key
            for row in self.repository.list_order_audits(limit=200)
            if row.status.lower() in {"submitted", "working", "pending"}
        }
        for action in actions:
            contract = self._select_contract(
                symbol=action.symbol,
                right=action.right or "P",
                target_delta=action.target_delta or 0.3,
                dte_min=int(config.get("dte_min", 21)),
                dte_max=int(config.get("dte_max", 60)),
            )
            if contract is None:
                self.repository.add_event(
                    "order_deferred",
                    f"No suitable contract found for {action.symbol} {action.action_type.value}",
                    severity="warning",
                )
                continue
            intent_key = f"{action.action_type.value}:{action.symbol}:{contract['expiry']}:{contract['strike']}:{contract['right']}"
            if intent_key in existing_intents:
                continue
            limit_price = self._mid_price(contract)
            request = OrderRequest(
                symbol=action.symbol,
                right=contract["right"],
                expiry=contract["expiry"],
                strike=float(contract["strike"]),
                action="SELL",
                quantity=action.quantity,
                limit_price=limit_price,
                order_ref=intent_key,
            )
            try:
                result = self.broker_client.submit_option_limit_order(request)
                self.repository.add_order_audit(
                    symbol=action.symbol,
                    broker_order_id=str(result.get("orderId") or result.get("permId") or ""),
                    action=action.action_type.value,
                    right=request.right,
                    strike=request.strike,
                    expiry=request.expiry,
                    quantity=request.quantity,
                    limit_price=request.limit_price,
                    status=str(result.get("status", "Submitted")).lower(),
                    intent_key=intent_key,
                    reason=action.reason,
                    payload={"metadata": action.metadata},
                )
                self.repository.add_event(
                    "order_submitted",
                    f"Submitted {action.action_type.value} for {action.symbol}",
                    payload={"intent_key": intent_key},
                )
                existing_intents.add(intent_key)
            except Exception as exc:
                self.repository.add_order_audit(
                    symbol=action.symbol,
                    broker_order_id=None,
                    action=action.action_type.value,
                    right=request.right,
                    strike=request.strike,
                    expiry=request.expiry,
                    quantity=request.quantity,
                    limit_price=request.limit_price,
                    status="failed",
                    intent_key=intent_key,
                    reason=str(exc),
                    payload={"metadata": action.metadata},
                )
                self.repository.add_event(
                    "order_failed",
                    f"{action.action_type.value} failed for {action.symbol}: {exc}",
                    severity="danger",
                    payload={"intent_key": intent_key},
                )

    def _select_contract(
        self,
        *,
        symbol: str,
        right: str,
        target_delta: float,
        dte_min: int,
        dte_max: int,
    ) -> dict[str, Any] | None:
        params = self.market_data_client.get_option_chain_params(symbol)
        if not params:
            return None
        expirations = params[0].get("expirations", [])
        if not expirations:
            return None
        expiry = expirations[0]
        chain = self.market_data_client.get_option_chain(symbol, expiry, right=right)
        if not chain:
            return None
        return min(
            chain,
            key=lambda item: abs(abs(float(item.get("delta", 0) or 0)) - target_delta),
        )

    @staticmethod
    def _mid_price(contract: dict[str, Any]) -> float:
        bid = float(contract.get("bid", 0) or 0)
        ask = float(contract.get("ask", 0) or 0)
        last = float(contract.get("last", 0) or 0)
        if bid > 0 and ask > 0:
            return round((bid + ask) / 2, 2)
        return round(last or bid or ask or 0.01, 2)
