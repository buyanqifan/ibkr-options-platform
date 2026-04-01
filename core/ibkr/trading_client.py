"""Trading client wrappers around ib_insync for live paper trading."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ib_insync import Contract, LimitOrder, Option, Stock

from core.ibkr.connection import IBKRConnectionManager
from core.ibkr.event_bridge import AsyncEventBridge
from utils.logger import setup_logger

logger = setup_logger("ibkr_trading")


@dataclass(slots=True)
class OrderRequest:
    symbol: str
    right: str
    expiry: str
    strike: float
    action: str
    quantity: int
    limit_price: float
    order_ref: str | None = None


class IBKRTradingClient:
    """Synchronous trading API for use by the live worker."""

    def __init__(self, conn_mgr: IBKRConnectionManager, bridge: AsyncEventBridge):
        self._conn = conn_mgr
        self._bridge = bridge

    def _option(self, symbol: str, expiry: str, strike: float, right: str) -> Option:
        return Option(symbol, expiry, strike, right, "SMART", currency="USD")

    def _qualify(self, contract: Contract) -> Contract:
        qualified = self._bridge.run_coroutine(self._conn.ib.qualifyContractsAsync(contract), timeout=10)
        if not qualified:
            raise ValueError(f"Unable to qualify contract {contract}")
        return qualified[0]

    def get_account_summary(self) -> dict:
        if not self._conn.is_connected:
            return {}
        summary = self._conn.ib.accountSummary()
        result = {"Account": self._conn.status.account}
        for item in summary:
            if item.tag in ("NetLiquidation", "TotalCashValue", "BuyingPower", "UnrealizedPnL", "GrossPositionValue"):
                result[item.tag] = float(item.value) if item.value else 0.0
        return result

    def get_positions(self) -> list[dict]:
        if not self._conn.is_connected:
            return []
        rows = []
        for item in self._conn.ib.portfolio():
            contract = item.contract
            rows.append(
                {
                    "symbol": contract.symbol,
                    "secType": contract.secType,
                    "expiry": getattr(contract, "lastTradeDateOrContractMonth", ""),
                    "strike": getattr(contract, "strike", 0.0),
                    "right": getattr(contract, "right", ""),
                    "position": item.position,
                    "avgCost": item.averageCost,
                    "marketValue": item.marketValue,
                    "unrealizedPNL": item.unrealizedPNL,
                    "realizedPNL": item.realizedPNL,
                }
            )
        return rows

    def get_open_orders(self) -> list[dict]:
        if not self._conn.is_connected:
            return []
        rows = []
        for trade in self._conn.ib.openTrades():
            contract = trade.contract
            order = trade.order
            rows.append(
                {
                    "symbol": contract.symbol,
                    "secType": contract.secType,
                    "expiry": getattr(contract, "lastTradeDateOrContractMonth", ""),
                    "strike": getattr(contract, "strike", 0.0),
                    "right": getattr(contract, "right", ""),
                    "action": order.action,
                    "orderId": getattr(order, "orderId", None),
                    "permId": getattr(order, "permId", None),
                    "quantity": getattr(order, "totalQuantity", 0),
                    "limitPrice": getattr(order, "lmtPrice", None),
                    "status": getattr(trade.orderStatus, "status", ""),
                }
            )
        return rows

    def get_recent_fills(self) -> list[dict]:
        if not self._conn.is_connected:
            return []
        rows = []
        for fill in self._conn.ib.fills():
            contract = fill.contract
            execution = fill.execution
            rows.append(
                {
                    "symbol": contract.symbol,
                    "secType": contract.secType,
                    "expiry": getattr(contract, "lastTradeDateOrContractMonth", ""),
                    "strike": getattr(contract, "strike", 0.0),
                    "right": getattr(contract, "right", ""),
                    "side": execution.side,
                    "shares": execution.shares,
                    "price": execution.price,
                    "time": execution.time.isoformat() if isinstance(execution.time, datetime) else str(execution.time),
                }
            )
        return rows

    def submit_option_limit_order(self, request: OrderRequest) -> dict:
        if not self._conn.is_connected:
            raise RuntimeError("IBKR is not connected")
        contract = self._qualify(self._option(request.symbol, request.expiry, request.strike, request.right))
        order = LimitOrder(request.action, request.quantity, request.limit_price, orderRef=request.order_ref or "")
        trade = self._conn.ib.placeOrder(contract, order)
        logger.info(
            "Submitted order %s %s %s %s x%s @ %s",
            request.action,
            request.symbol,
            request.expiry,
            request.right,
            request.quantity,
            request.limit_price,
        )
        return {
            "orderId": getattr(trade.order, "orderId", None),
            "permId": getattr(trade.order, "permId", None),
            "status": getattr(trade.orderStatus, "status", "Submitted"),
        }

    def cancel_open_entry_orders(self) -> int:
        if not self._conn.is_connected:
            return 0
        cancelled = 0
        for trade in self._conn.ib.openTrades():
            contract = trade.contract
            if contract.secType != "OPT":
                continue
            if trade.order.action not in {"SELL", "BUY"}:
                continue
            self._conn.ib.cancelOrder(trade.order)
            cancelled += 1
        return cancelled
