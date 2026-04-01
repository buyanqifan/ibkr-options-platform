"""QC parity helpers for BinbinGod backtests."""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from core.backtesting.pricing import OptionsPricer


_REPO_ROOT = Path(__file__).resolve().parents[2]
_QC_CONFIG_PATH = _REPO_ROOT / "quantconnect" / "config.json"
_QC_STRATEGY_INIT_PATH = _REPO_ROOT / "quantconnect" / "strategy_init.py"

_QC_PARAMETER_FALLBACKS = {
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 100000.0,
    "stock_pool": "MSFT,AAPL,NVDA,GOOGL,AMZN,META,TSLA",
    "max_positions_ceiling": 20,
    "profit_target_pct": 50.0,
    "stop_loss_pct": 999999.0,
    "margin_buffer_pct": 0.50,
    "margin_rate_per_contract": 0.25,
    "target_margin_utilization": 0.50,
    "position_aggressiveness": 1.0,
    "max_risk_per_trade": 0.02,
    "max_assignment_risk_per_trade": 0.20,
    "symbol_assignment_base_cap": 0.28,
    "stock_inventory_base_cap": 0.17,
    "stock_inventory_block_threshold": 0.85,
    "max_new_puts_per_day": 3,
    "defensive_put_roll_loss_pct": 85.0,
    "defensive_put_roll_itm_buffer_pct": 0.04,
    "defensive_put_roll_max_dte": 21,
    "ml_enabled": True,
    "dte_min": 21,
    "dte_max": 60,
    "put_delta": 0.30,
    "call_delta": 0.30,
}


def _evaluate_qc_default_expr(node: ast.AST, env: Dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.List):
        return [_evaluate_qc_default_expr(elt, env) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_evaluate_qc_default_expr(elt, env) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _evaluate_qc_default_expr(key, env): _evaluate_qc_default_expr(value, env)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_evaluate_qc_default_expr(node.operand, env)
    if isinstance(node, ast.BinOp):
        left = _evaluate_qc_default_expr(node.left, env)
        right = _evaluate_qc_default_expr(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and isinstance(node.func.value, ast.Constant)
        and isinstance(node.func.value.value, str)
        and len(node.args) == 1
    ):
        return node.func.value.value.join(_evaluate_qc_default_expr(node.args[0], env))
    raise ValueError(f"Unsupported QC default expression: {ast.dump(node)}")


def _extract_strategy_init_parameter_defaults() -> Dict[str, Any]:
    try:
        module = ast.parse(_QC_STRATEGY_INIT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

    env: Dict[str, Any] = {}
    defaults: Dict[str, Any] = {}
    for node in module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                env[node.targets[0].id] = _evaluate_qc_default_expr(node.value, env)
            except Exception:
                continue

    for node in module.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "init_parameters":
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                try:
                    env[stmt.targets[0].id] = _evaluate_qc_default_expr(stmt.value, env)
                except Exception:
                    pass
            for child in ast.walk(stmt):
                if not isinstance(child, ast.Call):
                    continue
                if not isinstance(child.func, ast.Name) or child.func.id != "_get_param":
                    continue
                if len(child.args) < 3:
                    continue
                try:
                    param_name = _evaluate_qc_default_expr(child.args[1], env)
                    param_default = _evaluate_qc_default_expr(child.args[2], env)
                except Exception:
                    continue
                defaults[str(param_name)] = param_default
        break

    return defaults


def _load_quantconnect_parameter_defaults() -> Dict[str, Any]:
    defaults = dict(_QC_PARAMETER_FALLBACKS)
    defaults.update(_extract_strategy_init_parameter_defaults())
    try:
        raw = json.loads(_QC_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return defaults

    parameters = raw.get("parameters", {})
    if isinstance(parameters, dict):
        defaults.update(parameters)
    return defaults


QC_PARAMETER_DEFAULTS = _load_quantconnect_parameter_defaults()

QC_BINBIN_DEFAULTS = {
    "initial_capital": float(QC_PARAMETER_DEFAULTS["initial_capital"]),
    "max_positions_ceiling": int(QC_PARAMETER_DEFAULTS["max_positions_ceiling"]),
    "profit_target_pct": float(QC_PARAMETER_DEFAULTS["profit_target_pct"]),
    "stop_loss_pct": 999999.0,
    "margin_buffer_pct": float(QC_PARAMETER_DEFAULTS["margin_buffer_pct"]),
    "margin_rate_per_contract": float(QC_PARAMETER_DEFAULTS["margin_rate_per_contract"]),
    "target_margin_utilization": float(QC_PARAMETER_DEFAULTS["target_margin_utilization"]),
    "position_aggressiveness": float(QC_PARAMETER_DEFAULTS["position_aggressiveness"]),
    "max_risk_per_trade": float(QC_PARAMETER_DEFAULTS["max_risk_per_trade"]),
    "max_assignment_risk_per_trade": float(QC_PARAMETER_DEFAULTS["max_assignment_risk_per_trade"]),
    "max_leverage": 1.0,
    "ml_enabled": bool(QC_PARAMETER_DEFAULTS["ml_enabled"]),
    "ml_min_confidence": 0.40,
    "dte_min": int(QC_PARAMETER_DEFAULTS["dte_min"]),
    "dte_max": int(QC_PARAMETER_DEFAULTS["dte_max"]),
    "put_delta": float(QC_PARAMETER_DEFAULTS["put_delta"]),
    "call_delta": float(QC_PARAMETER_DEFAULTS["call_delta"]),
    "repair_call_threshold_pct": float(QC_PARAMETER_DEFAULTS["repair_call_threshold_pct"]),
    "repair_call_delta": float(QC_PARAMETER_DEFAULTS["repair_call_delta"]),
    "repair_call_dte_min": int(QC_PARAMETER_DEFAULTS["repair_call_dte_min"]),
    "repair_call_dte_max": int(QC_PARAMETER_DEFAULTS["repair_call_dte_max"]),
    "repair_call_max_discount_pct": float(QC_PARAMETER_DEFAULTS["repair_call_max_discount_pct"]),
    "defensive_put_roll_enabled": bool(QC_PARAMETER_DEFAULTS["defensive_put_roll_enabled"]),
    "defensive_put_roll_loss_pct": float(QC_PARAMETER_DEFAULTS["defensive_put_roll_loss_pct"]),
    "defensive_put_roll_itm_buffer_pct": float(QC_PARAMETER_DEFAULTS["defensive_put_roll_itm_buffer_pct"]),
    "defensive_put_roll_min_dte": int(QC_PARAMETER_DEFAULTS["defensive_put_roll_min_dte"]),
    "defensive_put_roll_max_dte": int(QC_PARAMETER_DEFAULTS["defensive_put_roll_max_dte"]),
    "defensive_put_roll_dte_min": int(QC_PARAMETER_DEFAULTS["defensive_put_roll_dte_min"]),
    "defensive_put_roll_dte_max": int(QC_PARAMETER_DEFAULTS["defensive_put_roll_dte_max"]),
    "defensive_put_roll_delta": float(QC_PARAMETER_DEFAULTS["defensive_put_roll_delta"]),
    "assignment_cooldown_days": int(QC_PARAMETER_DEFAULTS["assignment_cooldown_days"]),
    "large_loss_cooldown_days": int(QC_PARAMETER_DEFAULTS["large_loss_cooldown_days"]),
    "large_loss_cooldown_pct": float(QC_PARAMETER_DEFAULTS["large_loss_cooldown_pct"]),
    "volatility_cap_floor": float(QC_PARAMETER_DEFAULTS["volatility_cap_floor"]),
    "volatility_cap_ceiling": float(QC_PARAMETER_DEFAULTS["volatility_cap_ceiling"]),
    "volatility_lookback": int(QC_PARAMETER_DEFAULTS["volatility_lookback"]),
    "dynamic_symbol_risk_enabled": bool(QC_PARAMETER_DEFAULTS["dynamic_symbol_risk_enabled"]),
    "symbol_state_cap_floor": float(QC_PARAMETER_DEFAULTS["symbol_state_cap_floor"]),
    "symbol_state_cap_ceiling": float(QC_PARAMETER_DEFAULTS["symbol_state_cap_ceiling"]),
    "symbol_drawdown_lookback": int(QC_PARAMETER_DEFAULTS["symbol_drawdown_lookback"]),
    "symbol_drawdown_sensitivity": float(QC_PARAMETER_DEFAULTS["symbol_drawdown_sensitivity"]),
    "symbol_downtrend_sensitivity": float(QC_PARAMETER_DEFAULTS["symbol_downtrend_sensitivity"]),
    "symbol_volatility_sensitivity": float(QC_PARAMETER_DEFAULTS["symbol_volatility_sensitivity"]),
    "symbol_exposure_sensitivity": float(QC_PARAMETER_DEFAULTS["symbol_exposure_sensitivity"]),
    "symbol_assignment_base_cap": float(QC_PARAMETER_DEFAULTS["symbol_assignment_base_cap"]),
    "stock_inventory_cap_enabled": bool(QC_PARAMETER_DEFAULTS["stock_inventory_cap_enabled"]),
    "stock_inventory_base_cap": float(QC_PARAMETER_DEFAULTS["stock_inventory_base_cap"]),
    "stock_inventory_cap_floor": float(QC_PARAMETER_DEFAULTS["stock_inventory_cap_floor"]),
    "stock_inventory_block_threshold": float(QC_PARAMETER_DEFAULTS["stock_inventory_block_threshold"]),
    "max_new_puts_per_day": int(QC_PARAMETER_DEFAULTS["max_new_puts_per_day"]),
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


@dataclass
class BinbinGodParityConfig:
    """Resolved QC-style parity configuration."""

    parity_mode: str = "qc"
    contract_universe_mode: str = "qc_emulated_lattice"
    ml_confidence_gate: float = 0.40
    initial_capital: float = QC_BINBIN_DEFAULTS["initial_capital"]
    max_positions_ceiling: int = QC_BINBIN_DEFAULTS["max_positions_ceiling"]
    profit_target_pct: float = QC_BINBIN_DEFAULTS["profit_target_pct"]
    stop_loss_pct: float = QC_BINBIN_DEFAULTS["stop_loss_pct"]
    margin_buffer_pct: float = QC_BINBIN_DEFAULTS["margin_buffer_pct"]
    margin_rate_per_contract: float = QC_BINBIN_DEFAULTS["margin_rate_per_contract"]
    target_margin_utilization: float = QC_BINBIN_DEFAULTS["target_margin_utilization"]
    position_aggressiveness: float = QC_BINBIN_DEFAULTS["position_aggressiveness"]
    max_risk_per_trade: float = QC_BINBIN_DEFAULTS["max_risk_per_trade"]
    max_assignment_risk_per_trade: float = QC_BINBIN_DEFAULTS["max_assignment_risk_per_trade"]
    max_leverage: float = QC_BINBIN_DEFAULTS["max_leverage"]
    ml_enabled: bool = bool(QC_BINBIN_DEFAULTS["ml_enabled"])
    ml_min_confidence: float = QC_BINBIN_DEFAULTS["ml_min_confidence"]
    dte_min: int = QC_BINBIN_DEFAULTS["dte_min"]
    dte_max: int = QC_BINBIN_DEFAULTS["dte_max"]
    put_delta: float = QC_BINBIN_DEFAULTS["put_delta"]
    call_delta: float = QC_BINBIN_DEFAULTS["call_delta"]
    max_put_contracts_per_symbol: int = 1
    max_put_contracts_total: int = 2
    max_contracts_per_trade: int = 1
    repair_call_threshold_pct: float = QC_BINBIN_DEFAULTS["repair_call_threshold_pct"]
    repair_call_delta: float = QC_BINBIN_DEFAULTS["repair_call_delta"]
    repair_call_dte_min: int = QC_BINBIN_DEFAULTS["repair_call_dte_min"]
    repair_call_dte_max: int = QC_BINBIN_DEFAULTS["repair_call_dte_max"]
    repair_call_max_discount_pct: float = QC_BINBIN_DEFAULTS["repair_call_max_discount_pct"]
    defensive_put_roll_enabled: bool = bool(QC_BINBIN_DEFAULTS["defensive_put_roll_enabled"])
    defensive_put_roll_loss_pct: float = QC_BINBIN_DEFAULTS["defensive_put_roll_loss_pct"]
    defensive_put_roll_itm_buffer_pct: float = QC_BINBIN_DEFAULTS["defensive_put_roll_itm_buffer_pct"]
    defensive_put_roll_min_dte: int = QC_BINBIN_DEFAULTS["defensive_put_roll_min_dte"]
    defensive_put_roll_max_dte: int = QC_BINBIN_DEFAULTS["defensive_put_roll_max_dte"]
    defensive_put_roll_dte_min: int = QC_BINBIN_DEFAULTS["defensive_put_roll_dte_min"]
    defensive_put_roll_dte_max: int = QC_BINBIN_DEFAULTS["defensive_put_roll_dte_max"]
    defensive_put_roll_delta: float = QC_BINBIN_DEFAULTS["defensive_put_roll_delta"]
    assignment_cooldown_days: int = QC_BINBIN_DEFAULTS["assignment_cooldown_days"]
    large_loss_cooldown_days: int = QC_BINBIN_DEFAULTS["large_loss_cooldown_days"]
    large_loss_cooldown_pct: float = QC_BINBIN_DEFAULTS["large_loss_cooldown_pct"]
    volatility_cap_floor: float = QC_BINBIN_DEFAULTS["volatility_cap_floor"]
    volatility_cap_ceiling: float = QC_BINBIN_DEFAULTS["volatility_cap_ceiling"]
    volatility_lookback: int = QC_BINBIN_DEFAULTS["volatility_lookback"]
    dynamic_symbol_risk_enabled: bool = bool(QC_BINBIN_DEFAULTS["dynamic_symbol_risk_enabled"])
    symbol_state_cap_floor: float = QC_BINBIN_DEFAULTS["symbol_state_cap_floor"]
    symbol_state_cap_ceiling: float = QC_BINBIN_DEFAULTS["symbol_state_cap_ceiling"]
    symbol_drawdown_lookback: int = QC_BINBIN_DEFAULTS["symbol_drawdown_lookback"]
    symbol_drawdown_sensitivity: float = QC_BINBIN_DEFAULTS["symbol_drawdown_sensitivity"]
    symbol_downtrend_sensitivity: float = QC_BINBIN_DEFAULTS["symbol_downtrend_sensitivity"]
    symbol_volatility_sensitivity: float = QC_BINBIN_DEFAULTS["symbol_volatility_sensitivity"]
    symbol_exposure_sensitivity: float = QC_BINBIN_DEFAULTS["symbol_exposure_sensitivity"]
    symbol_assignment_base_cap: float = QC_BINBIN_DEFAULTS["symbol_assignment_base_cap"]
    stock_inventory_cap_enabled: bool = bool(QC_BINBIN_DEFAULTS["stock_inventory_cap_enabled"])
    stock_inventory_base_cap: float = QC_BINBIN_DEFAULTS["stock_inventory_base_cap"]
    stock_inventory_cap_floor: float = QC_BINBIN_DEFAULTS["stock_inventory_cap_floor"]
    stock_inventory_block_threshold: float = QC_BINBIN_DEFAULTS["stock_inventory_block_threshold"]
    max_new_puts_per_day: int = QC_BINBIN_DEFAULTS["max_new_puts_per_day"]

    @property
    def enabled(self) -> bool:
        return self.parity_mode == "qc"

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "BinbinGodParityConfig":
        force_qc_replay = True
        parity_mode = str(params.get("parity_mode", "qc") or "qc").lower()
        if force_qc_replay:
            parity_mode = "qc"
        contract_universe_mode = str(
            params.get(
                "contract_universe_mode",
                "qc_emulated_lattice",
            )
            or "qc_emulated_lattice"
        ).lower()
        if force_qc_replay:
            contract_universe_mode = "qc_emulated_lattice"
        defaults = dict(QC_BINBIN_DEFAULTS if parity_mode == "qc" else {})
        merged = {**defaults, **params}

        if parity_mode == "qc":
            max_positions_ceiling = _to_int(
                merged.get("max_positions_ceiling", QC_BINBIN_DEFAULTS["max_positions_ceiling"]),
                QC_BINBIN_DEFAULTS["max_positions_ceiling"],
            )
        else:
            max_positions_ceiling = _to_int(
                merged.get("max_positions_ceiling", merged.get("max_positions", QC_BINBIN_DEFAULTS["max_positions_ceiling"])),
                QC_BINBIN_DEFAULTS["max_positions_ceiling"],
            )
        position_aggressiveness = _clamp(
            _to_float(merged.get("position_aggressiveness", QC_BINBIN_DEFAULTS["position_aggressiveness"]), QC_BINBIN_DEFAULTS["position_aggressiveness"]),
            0.3,
            2.0,
        )
        symbol_cap_factor = 0.30 + 0.30 * position_aggressiveness
        total_cap_factor = 1.20 + 0.80 * position_aggressiveness
        trade_cap_factor = 0.60 + 0.40 * position_aggressiveness
        ml_min_confidence = _to_float(
            merged.get("ml_min_confidence", QC_BINBIN_DEFAULTS["ml_min_confidence"]),
            QC_BINBIN_DEFAULTS["ml_min_confidence"],
        )

        return cls(
            parity_mode=parity_mode,
            contract_universe_mode=contract_universe_mode,
            ml_confidence_gate=_to_float(merged.get("ml_confidence_gate", ml_min_confidence), ml_min_confidence),
            initial_capital=_to_float(merged.get("initial_capital", QC_BINBIN_DEFAULTS["initial_capital"]), QC_BINBIN_DEFAULTS["initial_capital"]),
            max_positions_ceiling=max_positions_ceiling,
            profit_target_pct=_to_float(merged.get("profit_target_pct", QC_BINBIN_DEFAULTS["profit_target_pct"]), QC_BINBIN_DEFAULTS["profit_target_pct"]),
            stop_loss_pct=_to_float(merged.get("stop_loss_pct", QC_BINBIN_DEFAULTS["stop_loss_pct"]), QC_BINBIN_DEFAULTS["stop_loss_pct"]),
            margin_buffer_pct=_to_float(merged.get("margin_buffer_pct", QC_BINBIN_DEFAULTS["margin_buffer_pct"]), QC_BINBIN_DEFAULTS["margin_buffer_pct"]),
            margin_rate_per_contract=_to_float(merged.get("margin_rate_per_contract", QC_BINBIN_DEFAULTS["margin_rate_per_contract"]), QC_BINBIN_DEFAULTS["margin_rate_per_contract"]),
            target_margin_utilization=_to_float(
                merged.get("target_margin_utilization", QC_BINBIN_DEFAULTS["target_margin_utilization"]),
                QC_BINBIN_DEFAULTS["target_margin_utilization"],
            ),
            position_aggressiveness=position_aggressiveness,
            max_risk_per_trade=_to_float(merged.get("max_risk_per_trade", QC_BINBIN_DEFAULTS["max_risk_per_trade"]), QC_BINBIN_DEFAULTS["max_risk_per_trade"]),
            max_assignment_risk_per_trade=_to_float(merged.get("max_assignment_risk_per_trade", QC_BINBIN_DEFAULTS["max_assignment_risk_per_trade"]), QC_BINBIN_DEFAULTS["max_assignment_risk_per_trade"]),
            max_leverage=_to_float(merged.get("max_leverage", QC_BINBIN_DEFAULTS["max_leverage"]), QC_BINBIN_DEFAULTS["max_leverage"]),
            ml_enabled=bool(merged.get("ml_enabled", QC_BINBIN_DEFAULTS["ml_enabled"])),
            ml_min_confidence=ml_min_confidence,
            dte_min=_to_int(merged.get("dte_min", QC_BINBIN_DEFAULTS["dte_min"]), QC_BINBIN_DEFAULTS["dte_min"]),
            dte_max=_to_int(merged.get("dte_max", QC_BINBIN_DEFAULTS["dte_max"]), QC_BINBIN_DEFAULTS["dte_max"]),
            put_delta=_to_float(merged.get("put_delta", QC_BINBIN_DEFAULTS["put_delta"]), QC_BINBIN_DEFAULTS["put_delta"]),
            call_delta=_to_float(merged.get("call_delta", QC_BINBIN_DEFAULTS["call_delta"]), QC_BINBIN_DEFAULTS["call_delta"]),
            max_put_contracts_per_symbol=max(1, int(max_positions_ceiling * symbol_cap_factor)),
            max_put_contracts_total=max(2, int(max_positions_ceiling * total_cap_factor)),
            max_contracts_per_trade=max(1, int(max(1, int(max_positions_ceiling * symbol_cap_factor)) * trade_cap_factor)),
            repair_call_threshold_pct=_to_float(merged.get("repair_call_threshold_pct", QC_BINBIN_DEFAULTS["repair_call_threshold_pct"]), QC_BINBIN_DEFAULTS["repair_call_threshold_pct"]),
            repair_call_delta=_clamp(_to_float(merged.get("repair_call_delta", QC_BINBIN_DEFAULTS["repair_call_delta"]), QC_BINBIN_DEFAULTS["repair_call_delta"]), 0.20, 0.60),
            repair_call_dte_min=_to_int(merged.get("repair_call_dte_min", QC_BINBIN_DEFAULTS["repair_call_dte_min"]), QC_BINBIN_DEFAULTS["repair_call_dte_min"]),
            repair_call_dte_max=_to_int(merged.get("repair_call_dte_max", QC_BINBIN_DEFAULTS["repair_call_dte_max"]), QC_BINBIN_DEFAULTS["repair_call_dte_max"]),
            repair_call_max_discount_pct=_to_float(merged.get("repair_call_max_discount_pct", QC_BINBIN_DEFAULTS["repair_call_max_discount_pct"]), QC_BINBIN_DEFAULTS["repair_call_max_discount_pct"]),
            defensive_put_roll_enabled=bool(merged.get("defensive_put_roll_enabled", QC_BINBIN_DEFAULTS["defensive_put_roll_enabled"])),
            defensive_put_roll_loss_pct=_to_float(merged.get("defensive_put_roll_loss_pct", QC_BINBIN_DEFAULTS["defensive_put_roll_loss_pct"]), QC_BINBIN_DEFAULTS["defensive_put_roll_loss_pct"]),
            defensive_put_roll_itm_buffer_pct=_to_float(merged.get("defensive_put_roll_itm_buffer_pct", QC_BINBIN_DEFAULTS["defensive_put_roll_itm_buffer_pct"]), QC_BINBIN_DEFAULTS["defensive_put_roll_itm_buffer_pct"]),
            defensive_put_roll_min_dte=_to_int(merged.get("defensive_put_roll_min_dte", QC_BINBIN_DEFAULTS["defensive_put_roll_min_dte"]), QC_BINBIN_DEFAULTS["defensive_put_roll_min_dte"]),
            defensive_put_roll_max_dte=_to_int(merged.get("defensive_put_roll_max_dte", QC_BINBIN_DEFAULTS["defensive_put_roll_max_dte"]), QC_BINBIN_DEFAULTS["defensive_put_roll_max_dte"]),
            defensive_put_roll_dte_min=_to_int(merged.get("defensive_put_roll_dte_min", QC_BINBIN_DEFAULTS["defensive_put_roll_dte_min"]), QC_BINBIN_DEFAULTS["defensive_put_roll_dte_min"]),
            defensive_put_roll_dte_max=_to_int(merged.get("defensive_put_roll_dte_max", QC_BINBIN_DEFAULTS["defensive_put_roll_dte_max"]), QC_BINBIN_DEFAULTS["defensive_put_roll_dte_max"]),
            defensive_put_roll_delta=_clamp(_to_float(merged.get("defensive_put_roll_delta", QC_BINBIN_DEFAULTS["defensive_put_roll_delta"]), QC_BINBIN_DEFAULTS["defensive_put_roll_delta"]), 0.10, 0.40),
            assignment_cooldown_days=_to_int(merged.get("assignment_cooldown_days", QC_BINBIN_DEFAULTS["assignment_cooldown_days"]), QC_BINBIN_DEFAULTS["assignment_cooldown_days"]),
            large_loss_cooldown_days=_to_int(merged.get("large_loss_cooldown_days", QC_BINBIN_DEFAULTS["large_loss_cooldown_days"]), QC_BINBIN_DEFAULTS["large_loss_cooldown_days"]),
            large_loss_cooldown_pct=_to_float(merged.get("large_loss_cooldown_pct", QC_BINBIN_DEFAULTS["large_loss_cooldown_pct"]), QC_BINBIN_DEFAULTS["large_loss_cooldown_pct"]),
            volatility_cap_floor=_clamp(_to_float(merged.get("volatility_cap_floor", QC_BINBIN_DEFAULTS["volatility_cap_floor"]), QC_BINBIN_DEFAULTS["volatility_cap_floor"]), 0.10, 1.0),
            volatility_cap_ceiling=_clamp(_to_float(merged.get("volatility_cap_ceiling", QC_BINBIN_DEFAULTS["volatility_cap_ceiling"]), QC_BINBIN_DEFAULTS["volatility_cap_ceiling"]), 1.0, 3.0),
            volatility_lookback=_to_int(merged.get("volatility_lookback", QC_BINBIN_DEFAULTS["volatility_lookback"]), QC_BINBIN_DEFAULTS["volatility_lookback"]),
            dynamic_symbol_risk_enabled=bool(merged.get("dynamic_symbol_risk_enabled", QC_BINBIN_DEFAULTS["dynamic_symbol_risk_enabled"])),
            symbol_state_cap_floor=_clamp(_to_float(merged.get("symbol_state_cap_floor", QC_BINBIN_DEFAULTS["symbol_state_cap_floor"]), QC_BINBIN_DEFAULTS["symbol_state_cap_floor"]), 0.05, 1.0),
            symbol_state_cap_ceiling=_clamp(_to_float(merged.get("symbol_state_cap_ceiling", QC_BINBIN_DEFAULTS["symbol_state_cap_ceiling"]), QC_BINBIN_DEFAULTS["symbol_state_cap_ceiling"]), 0.50, 1.0),
            symbol_drawdown_lookback=_to_int(merged.get("symbol_drawdown_lookback", QC_BINBIN_DEFAULTS["symbol_drawdown_lookback"]), QC_BINBIN_DEFAULTS["symbol_drawdown_lookback"]),
            symbol_drawdown_sensitivity=_to_float(merged.get("symbol_drawdown_sensitivity", QC_BINBIN_DEFAULTS["symbol_drawdown_sensitivity"]), QC_BINBIN_DEFAULTS["symbol_drawdown_sensitivity"]),
            symbol_downtrend_sensitivity=_to_float(merged.get("symbol_downtrend_sensitivity", QC_BINBIN_DEFAULTS["symbol_downtrend_sensitivity"]), QC_BINBIN_DEFAULTS["symbol_downtrend_sensitivity"]),
            symbol_volatility_sensitivity=_to_float(merged.get("symbol_volatility_sensitivity", QC_BINBIN_DEFAULTS["symbol_volatility_sensitivity"]), QC_BINBIN_DEFAULTS["symbol_volatility_sensitivity"]),
            symbol_exposure_sensitivity=_to_float(merged.get("symbol_exposure_sensitivity", QC_BINBIN_DEFAULTS["symbol_exposure_sensitivity"]), QC_BINBIN_DEFAULTS["symbol_exposure_sensitivity"]),
            symbol_assignment_base_cap=_clamp(
                _to_float(
                    merged.get("symbol_assignment_base_cap", QC_BINBIN_DEFAULTS["symbol_assignment_base_cap"]),
                    QC_BINBIN_DEFAULTS["symbol_assignment_base_cap"],
                ),
                0.05,
                1.5,
            ),
            stock_inventory_cap_enabled=bool(merged.get("stock_inventory_cap_enabled", QC_BINBIN_DEFAULTS["stock_inventory_cap_enabled"])),
            stock_inventory_base_cap=_clamp(_to_float(merged.get("stock_inventory_base_cap", QC_BINBIN_DEFAULTS["stock_inventory_base_cap"]), QC_BINBIN_DEFAULTS["stock_inventory_base_cap"]), 0.05, 1.0),
            stock_inventory_cap_floor=_clamp(_to_float(merged.get("stock_inventory_cap_floor", QC_BINBIN_DEFAULTS["stock_inventory_cap_floor"]), QC_BINBIN_DEFAULTS["stock_inventory_cap_floor"]), 0.10, 1.0),
            stock_inventory_block_threshold=_clamp(_to_float(merged.get("stock_inventory_block_threshold", QC_BINBIN_DEFAULTS["stock_inventory_block_threshold"]), QC_BINBIN_DEFAULTS["stock_inventory_block_threshold"]), 0.50, 1.20),
            max_new_puts_per_day=max(1, _to_int(merged.get("max_new_puts_per_day", QC_BINBIN_DEFAULTS["max_new_puts_per_day"]), QC_BINBIN_DEFAULTS["max_new_puts_per_day"])),
        )

    def apply_to_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Overlay resolved QC-style values onto params."""
        resolved = dict(params)
        resolved.update(
            {
                "parity_mode": self.parity_mode,
                "contract_universe_mode": self.contract_universe_mode,
                "ml_confidence_gate": self.ml_confidence_gate,
                "initial_capital": self.initial_capital,
                "max_positions_ceiling": self.max_positions_ceiling,
                "profit_target_pct": self.profit_target_pct,
                "stop_loss_pct": self.stop_loss_pct,
                "margin_buffer_pct": self.margin_buffer_pct,
                "margin_rate_per_contract": self.margin_rate_per_contract,
                "target_margin_utilization": self.target_margin_utilization,
                "position_aggressiveness": self.position_aggressiveness,
                "max_risk_per_trade": self.max_risk_per_trade,
                "max_assignment_risk_per_trade": self.max_assignment_risk_per_trade,
                "max_leverage": self.max_leverage,
                "ml_enabled": self.ml_enabled,
                "ml_min_confidence": self.ml_min_confidence,
                "dte_min": self.dte_min,
                "dte_max": self.dte_max,
                "put_delta": self.put_delta,
                "call_delta": self.call_delta,
                "repair_call_threshold_pct": self.repair_call_threshold_pct,
                "repair_call_delta": self.repair_call_delta,
                "repair_call_dte_min": self.repair_call_dte_min,
                "repair_call_dte_max": self.repair_call_dte_max,
                "repair_call_max_discount_pct": self.repair_call_max_discount_pct,
                "defensive_put_roll_enabled": self.defensive_put_roll_enabled,
                "defensive_put_roll_loss_pct": self.defensive_put_roll_loss_pct,
                "defensive_put_roll_itm_buffer_pct": self.defensive_put_roll_itm_buffer_pct,
                "defensive_put_roll_min_dte": self.defensive_put_roll_min_dte,
                "defensive_put_roll_max_dte": self.defensive_put_roll_max_dte,
                "defensive_put_roll_dte_min": self.defensive_put_roll_dte_min,
                "defensive_put_roll_dte_max": self.defensive_put_roll_dte_max,
                "defensive_put_roll_delta": self.defensive_put_roll_delta,
                "assignment_cooldown_days": self.assignment_cooldown_days,
                "large_loss_cooldown_days": self.large_loss_cooldown_days,
                "large_loss_cooldown_pct": self.large_loss_cooldown_pct,
                "volatility_cap_floor": self.volatility_cap_floor,
                "volatility_cap_ceiling": self.volatility_cap_ceiling,
                "volatility_lookback": self.volatility_lookback,
                "dynamic_symbol_risk_enabled": self.dynamic_symbol_risk_enabled,
                "symbol_state_cap_floor": self.symbol_state_cap_floor,
                "symbol_state_cap_ceiling": self.symbol_state_cap_ceiling,
                "symbol_drawdown_lookback": self.symbol_drawdown_lookback,
                "symbol_drawdown_sensitivity": self.symbol_drawdown_sensitivity,
                "symbol_downtrend_sensitivity": self.symbol_downtrend_sensitivity,
                "symbol_volatility_sensitivity": self.symbol_volatility_sensitivity,
                "symbol_exposure_sensitivity": self.symbol_exposure_sensitivity,
                "symbol_assignment_base_cap": self.symbol_assignment_base_cap,
                "stock_inventory_cap_enabled": self.stock_inventory_cap_enabled,
                "stock_inventory_base_cap": self.stock_inventory_base_cap,
                "stock_inventory_cap_floor": self.stock_inventory_cap_floor,
                "stock_inventory_block_threshold": self.stock_inventory_block_threshold,
                "max_new_puts_per_day": self.max_new_puts_per_day,
                "max_positions": self.max_positions_ceiling if self.enabled else params.get("max_positions", self.max_positions_ceiling),
            }
        )
        return resolved

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LatticeContract:
    """Deterministic local contract used to emulate QC option selection."""

    symbol: str
    right: str
    strike: float
    expiry: datetime
    dte: int
    delta: float
    iv: float
    premium: float
    delta_diff: float
    bid: float
    ask: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "right": self.right,
            "strike": self.strike,
            "expiry": self.expiry,
            "dte": self.dte,
            "delta": self.delta,
            "iv": self.iv,
            "premium": self.premium,
            "delta_diff": self.delta_diff,
            "bid": self.bid,
            "ask": self.ask,
        }


def get_strike_increment(underlying_price: float) -> float:
    return 1.0 if underlying_price >= 100 else 0.5


def _round_to_increment(value: float, increment: float) -> float:
    steps = round(value / increment)
    return round(steps * increment, 4)


def _iter_strikes(low: float, high: float, increment: float) -> Iterable[float]:
    current = _round_to_increment(low, increment)
    if current < low:
        current += increment
    while current <= high + 1e-9:
        yield round(current, 4)
        current += increment


def generate_weekly_expiries(current_date: str | datetime | date, dte_min: int, dte_max: int) -> List[datetime]:
    if isinstance(current_date, str):
        current_dt = datetime.strptime(current_date, "%Y-%m-%d")
    elif isinstance(current_date, date) and not isinstance(current_date, datetime):
        current_dt = datetime.combine(current_date, datetime.min.time())
    else:
        current_dt = current_date

    expiries: List[datetime] = []
    for dte in range(max(dte_min, 0), max(dte_max, 0) + 1):
        candidate = current_dt + timedelta(days=dte)
        if candidate.weekday() == 4:
            expiries.append(candidate.replace(hour=0, minute=0, second=0, microsecond=0))
    return expiries


def filter_option_by_itm_protection(strike: float, underlying_price: float, right: str, itm_buffer_pct: float = 0.01) -> bool:
    if right == "C":
        if strike < underlying_price:
            return False
        if strike < underlying_price * (1 + itm_buffer_pct):
            return False
    else:
        if strike > underlying_price:
            return False
        if strike > underlying_price * (1 - itm_buffer_pct):
            return False
    return True


def estimate_delta_from_moneyness(strike: float, underlying_price: float, right: str) -> Optional[float]:
    if underlying_price <= 0:
        return None
    moneyness = strike / underlying_price
    if right == "P":
        if moneyness < 0.99:
            return -max(0.10, min(0.45, (1 - moneyness) * 3))
    else:
        if moneyness > 1.01:
            return max(0.10, min(0.45, (moneyness - 1) * 3))
    return None


def build_contract_lattice(
    symbol: str,
    current_date: str,
    underlying_price: float,
    iv: float,
    target_right: str,
    target_delta: float,
    dte_min: int,
    dte_max: int,
    delta_tolerance: float = 0.05,
    min_strike: float | None = None,
) -> List[LatticeContract]:
    right = "P" if str(target_right).upper().startswith("P") else "C"
    expiries = generate_weekly_expiries(current_date, dte_min, dte_max)
    increment = get_strike_increment(underlying_price)
    low = underlying_price * 0.8
    high = underlying_price * 1.2
    contracts: List[LatticeContract] = []

    for expiry in expiries:
        dte = (expiry - datetime.strptime(current_date, "%Y-%m-%d")).days
        if dte < dte_min or dte > dte_max:
            continue
        T = max(dte / 365.0, 1 / 365.0)
        for strike in _iter_strikes(low, high, increment):
            if min_strike is not None and strike < min_strike:
                continue
            if not filter_option_by_itm_protection(strike, underlying_price, right):
                continue
            delta = estimate_delta_from_moneyness(strike, underlying_price, right)
            if delta is None or abs(delta - target_delta) > delta_tolerance:
                continue
            premium = OptionsPricer.put_price(underlying_price, strike, T, iv) if right == "P" else OptionsPricer.call_price(underlying_price, strike, T, iv)
            if premium <= 0.10:
                continue
            contracts.append(
                LatticeContract(
                    symbol=symbol,
                    right=right,
                    strike=strike,
                    expiry=expiry,
                    dte=dte,
                    delta=delta,
                    iv=iv,
                    premium=premium,
                    delta_diff=abs(delta - target_delta),
                    bid=premium * 0.99,
                    ask=premium * 1.01,
                )
            )

    contracts.sort(key=lambda item: (item.delta_diff, item.dte, item.strike))
    return contracts


def select_contract_from_lattice(
    symbol: str,
    current_date: str,
    underlying_price: float,
    iv: float,
    target_right: str,
    target_delta: float,
    dte_min: int,
    dte_max: int,
    delta_tolerance: float = 0.05,
    min_strike: float | None = None,
) -> Optional[LatticeContract]:
    contracts = build_contract_lattice(
        symbol=symbol,
        current_date=current_date,
        underlying_price=underlying_price,
        iv=iv,
        target_right=target_right,
        target_delta=target_delta,
        dte_min=dte_min,
        dte_max=dte_max,
        delta_tolerance=delta_tolerance,
        min_strike=min_strike,
    )
    return contracts[0] if contracts else None


def calculate_dynamic_max_positions_from_prices(prices: Sequence[float], config: BinbinGodParityConfig) -> int:
    valid_prices = [price for price in prices if price and price > 0]
    if not valid_prices:
        return config.max_positions_ceiling
    avg_price = sum(valid_prices) / len(valid_prices)
    margin_budget = config.initial_capital * config.target_margin_utilization
    margin_per_contract = avg_price * 100 * 0.20
    dynamic_max = int(margin_budget / margin_per_contract) if margin_per_contract > 0 else config.max_positions_ceiling
    return max(1, min(dynamic_max, config.max_positions_ceiling))


def _calculate_historical_vol_from_bars(bars: Sequence[Dict[str, Any]], window: int = 20) -> float:
    if not bars or len(bars) < window + 1:
        return 0.25
    closes = [float(bar.get("close", 0.0)) for bar in bars[-(window + 1):] if bar.get("close", 0)]
    if len(closes) < window + 1:
        return 0.25
    returns = []
    for idx in range(1, len(closes)):
        prev = closes[idx - 1]
        if prev <= 0:
            continue
        returns.append((closes[idx] - prev) / prev)
    if len(returns) < 2:
        return 0.25
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / len(returns)
    return max(variance ** 0.5 * (252 ** 0.5), 0.25)


def calculate_volatility_weighted_symbol_cap_qc(
    config: BinbinGodParityConfig,
    symbol: str,
    symbol_history_bars: Sequence[Dict[str, Any]] | None,
    pool_history_bars: Dict[str, Sequence[Dict[str, Any]]] | None,
) -> int:
    base_cap = max(1, int(config.max_put_contracts_per_symbol))
    lookback = max(5, int(config.volatility_lookback))
    symbol_vol = _calculate_historical_vol_from_bars(symbol_history_bars or [], window=lookback)
    if symbol_vol <= 0:
        return base_cap

    pool_vols = []
    for bars in (pool_history_bars or {}).values():
        if bars and len(bars) >= lookback + 1:
            vol = _calculate_historical_vol_from_bars(bars, window=lookback)
            if vol > 0:
                pool_vols.append(vol)
    if not pool_vols:
        return base_cap

    avg_pool_vol = sum(pool_vols) / len(pool_vols)
    raw_multiplier = avg_pool_vol / symbol_vol if symbol_vol > 0 else 1.0
    multiplier = max(
        config.volatility_cap_floor,
        min(raw_multiplier, min(1.0, config.volatility_cap_ceiling)),
    )
    return max(1, int(round(base_cap * multiplier)))


def calculate_symbol_state_risk_multiplier_qc(
    config: BinbinGodParityConfig,
    symbol_history_bars: Sequence[Dict[str, Any]] | None,
    pool_history_bars: Dict[str, Sequence[Dict[str, Any]]] | None,
    underlying_price: float,
    symbol_put_notional: float,
    symbol_stock_notional: float,
    portfolio_value: float,
) -> tuple[float, Dict[str, float]]:
    if not config.dynamic_symbol_risk_enabled:
        return 1.0, {
            "vol_ratio": 1.0,
            "drawdown": 0.0,
            "momentum_20d": 0.0,
            "exposure_ratio": 0.0,
        }

    lookback = max(5, int(config.volatility_lookback))
    symbol_bars = list(symbol_history_bars or [])
    symbol_vol = _calculate_historical_vol_from_bars(symbol_bars, window=lookback) if len(symbol_bars) >= lookback + 1 else 0.25

    pool_vols = []
    for bars in (pool_history_bars or {}).values():
        if bars and len(bars) >= lookback + 1:
            pool_vols.append(_calculate_historical_vol_from_bars(bars, window=lookback))
    avg_pool_vol = (sum(pool_vols) / len(pool_vols)) if pool_vols else max(symbol_vol, 0.25)
    vol_ratio = symbol_vol / avg_pool_vol if avg_pool_vol > 0 else 1.0

    closes = [float(bar.get("close", 0.0)) for bar in symbol_bars if bar.get("close", 0)]
    momentum_20d = 0.0
    if len(closes) >= 20 and closes[-20] > 0:
        momentum_20d = underlying_price / closes[-20] - 1.0

    dd_lookback = max(20, int(config.symbol_drawdown_lookback))
    recent_closes = closes[-dd_lookback:] if closes else []
    peak_price = max(recent_closes) if recent_closes else max(underlying_price, 1.0)
    drawdown = max(0.0, (peak_price - underlying_price) / peak_price) if peak_price > 0 else 0.0

    assignment_exposure = symbol_put_notional + symbol_stock_notional
    exposure_ratio = assignment_exposure / portfolio_value if portfolio_value > 0 else 0.0

    volatility_penalty = max(0.35, 1.0 - max(0.0, vol_ratio - 1.0) * config.symbol_volatility_sensitivity)
    downtrend_penalty = max(0.35, 1.0 - max(0.0, -momentum_20d) * config.symbol_downtrend_sensitivity)
    drawdown_penalty = max(0.25, 1.0 - drawdown * config.symbol_drawdown_sensitivity)
    exposure_penalty = max(0.20, 1.0 - exposure_ratio * config.symbol_exposure_sensitivity)

    raw_multiplier = volatility_penalty * downtrend_penalty * drawdown_penalty * exposure_penalty
    multiplier = max(config.symbol_state_cap_floor, min(raw_multiplier, config.symbol_state_cap_ceiling))
    return multiplier, {
        "vol_ratio": vol_ratio,
        "drawdown": drawdown,
        "momentum_20d": momentum_20d,
        "exposure_ratio": exposure_ratio,
    }


def calculate_stock_inventory_cap_qc(
    config: BinbinGodParityConfig,
    portfolio_value: float,
    symbol_state_multiplier: float,
) -> float:
    if not config.stock_inventory_cap_enabled:
        return portfolio_value
    base_cap = portfolio_value * config.stock_inventory_base_cap
    dynamic_multiplier = max(config.stock_inventory_cap_floor, min(1.0, symbol_state_multiplier))
    return max(0.0, base_cap * dynamic_multiplier)


def calculate_put_quantity_qc(
    config: BinbinGodParityConfig,
    selected_contract: LatticeContract,
    current_positions: int,
    underlying_price: float,
    symbol: str,
    portfolio_value: float,
    margin_remaining: float,
    total_margin_used: float,
    stock_holdings_value: float,
    stock_holding_count: int,
    open_option_positions: Sequence[Any],
    shares_held: int,
    dynamic_max_positions: int,
    symbol_history_bars: Sequence[Dict[str, Any]] | None = None,
    pool_history_bars: Dict[str, Sequence[Dict[str, Any]]] | None = None,
) -> tuple[int, Dict[str, float]]:
    strike = selected_contract.strike
    premium = selected_contract.premium
    otm_amount = max(0.0, underlying_price - strike)
    margin_method_1 = 0.20 * underlying_price * 100 - otm_amount * 100
    margin_method_2 = 0.10 * strike * 100
    estimated_margin_per_contract = max(margin_method_1, margin_method_2) + premium * 100
    fallback_margin = strike * 100 * config.margin_rate_per_contract
    estimated_margin_per_contract = max(estimated_margin_per_contract, fallback_margin)

    usable_margin = max(0.0, margin_remaining * (1 - config.margin_buffer_pct))
    adjusted_max_positions = dynamic_max_positions - stock_holding_count
    stock_value_ratio = stock_holdings_value / config.initial_capital if config.initial_capital > 0 else 0
    if stock_value_ratio > 0.30:
        reduction_factor = min(0.5, stock_value_ratio)
        adjusted_max_positions = max(1, int(adjusted_max_positions * (1 - reduction_factor)))

    total_put_contracts = 0
    symbol_put_contracts = 0
    total_put_notional = 0.0
    symbol_put_notional = 0.0
    for pos in open_option_positions:
        if getattr(pos, "right", "") != "P":
            continue
        contracts = abs(int(getattr(pos, "quantity", 0)))
        strike_h = float(getattr(pos, "strike", 0.0))
        notional = contracts * strike_h * 100
        total_put_contracts += contracts
        total_put_notional += notional
        if getattr(pos, "symbol", "") == symbol:
            symbol_put_contracts += contracts
            symbol_put_notional += notional

    symbol_stock_notional = shares_held * max(underlying_price, 0.0)
    symbol_cap = calculate_volatility_weighted_symbol_cap_qc(
        config,
        symbol,
        symbol_history_bars,
        pool_history_bars,
    )
    symbol_state_multiplier, symbol_state = calculate_symbol_state_risk_multiplier_qc(
        config,
        symbol_history_bars,
        pool_history_bars,
        underlying_price,
        symbol_put_notional,
        symbol_stock_notional,
        portfolio_value,
    )
    symbol_cap = max(1, int(symbol_cap * symbol_state_multiplier))
    max_by_symbol_contracts = max(0, symbol_cap - symbol_put_contracts)
    max_by_total_contracts = max(0, config.max_put_contracts_total - total_put_contracts)
    max_by_trade_cap = max(0, min(config.max_contracts_per_trade, symbol_cap))
    max_by_margin = int(usable_margin / estimated_margin_per_contract) if estimated_margin_per_contract > 0 else 0
    max_by_limit = max(0, adjusted_max_positions - current_positions)

    margin_budget = portfolio_value * config.target_margin_utilization
    remaining_budget = max(0.0, margin_budget - total_margin_used)
    max_by_budget = int(remaining_budget / estimated_margin_per_contract) if estimated_margin_per_contract > 0 else 0

    leverage_budget = portfolio_value * config.max_leverage
    remaining_leverage_budget = max(0.0, leverage_budget - total_margin_used)
    max_by_leverage = int(remaining_leverage_budget / estimated_margin_per_contract) if estimated_margin_per_contract > 0 else 0

    aggr = config.position_aggressiveness
    base_symbol_notional_cap = portfolio_value * config.symbol_assignment_base_cap
    per_symbol_notional_cap = base_symbol_notional_cap * symbol_state_multiplier
    total_notional_cap = portfolio_value * (0.70 + 0.90 * aggr)
    candidate_notional = strike * 100
    remaining_symbol_notional = max(0.0, per_symbol_notional_cap - (symbol_put_notional + symbol_stock_notional))
    remaining_total_notional = max(0.0, total_notional_cap - (total_put_notional + stock_holdings_value))
    max_by_symbol_notional = int(remaining_symbol_notional / candidate_notional) if candidate_notional > 0 else 0
    max_by_total_notional = int(remaining_total_notional / candidate_notional) if candidate_notional > 0 else 0

    stock_inventory_cap = calculate_stock_inventory_cap_qc(config, portfolio_value, symbol_state_multiplier)
    remaining_stock_inventory = max(0.0, stock_inventory_cap - symbol_stock_notional)
    max_by_stock_inventory = int(remaining_stock_inventory / candidate_notional) if candidate_notional > 0 else 0
    assignment_trade_cap = portfolio_value * config.max_assignment_risk_per_trade
    max_by_assignment_trade = int(assignment_trade_cap / candidate_notional) if candidate_notional > 0 else 0
    max_by_risk = max_by_trade_cap
    if premium > 0 and portfolio_value > 0 and config.max_risk_per_trade > 0:
        max_by_risk = int((portfolio_value * config.max_risk_per_trade) / (premium * 100))

    diagnostics = {
        "margin": max_by_margin,
        "slots": max_by_limit,
        "budget": max_by_budget,
        "leverage": max_by_leverage,
        "symbol_contracts": max_by_symbol_contracts,
        "total_contracts": max_by_total_contracts,
        "trade_cap": max_by_trade_cap,
        "symbol_notional": max_by_symbol_notional,
        "total_notional": max_by_total_notional,
        "stock_inventory": max_by_stock_inventory,
        "assignment_trade": max_by_assignment_trade,
        "risk": max_by_risk,
        "symbol_cap": symbol_cap,
        "state_multiplier": round(symbol_state_multiplier, 4),
        "vol_ratio": round(symbol_state["vol_ratio"], 4),
        "drawdown": round(symbol_state["drawdown"], 4),
        "momentum_20d": round(symbol_state["momentum_20d"], 4),
        "exposure_ratio": round(symbol_state["exposure_ratio"], 4),
    }
    limit_keys = (
        "margin",
        "slots",
        "budget",
        "leverage",
        "symbol_contracts",
        "total_contracts",
        "trade_cap",
        "symbol_notional",
        "total_notional",
        "stock_inventory",
        "assignment_trade",
        "risk",
    )
    quantity = min(diagnostics[key] for key in limit_keys) if diagnostics else 0
    return max(0, quantity), diagnostics


def estimate_put_margin_qc(
    *,
    strike: float,
    premium: float,
    underlying_price: float,
    margin_rate_per_contract: float,
) -> float:
    """Mirror QC's estimated short-put margin requirement for one contract."""
    otm_amount = max(0.0, underlying_price - strike)
    margin_method_1 = 0.20 * underlying_price * 100 - otm_amount * 100
    margin_method_2 = 0.10 * strike * 100
    estimated_margin_per_contract = max(margin_method_1, margin_method_2) + premium * 100
    fallback_margin = strike * 100 * margin_rate_per_contract
    return max(estimated_margin_per_contract, fallback_margin)


class EventTracer:
    """Collect parity event trace and portfolio snapshots."""

    def __init__(self):
        self.event_trace: List[Dict[str, Any]] = []
        self.portfolio_snapshots: List[Dict[str, Any]] = []
        self._seq = 0

    def record(self, date_str: str, event_type: str, **payload: Any) -> Dict[str, Any]:
        self._seq += 1
        event = {"seq": self._seq, "date": date_str, "event_type": event_type}
        event.update(payload)
        self.event_trace.append(event)
        return event

    def snapshot(self, date_str: str, phase: str, **payload: Any) -> Dict[str, Any]:
        snap = {"date": date_str, "phase": phase}
        snap.update(payload)
        self.portfolio_snapshots.append(snap)
        return snap

    def build_parity_report(
        self,
        expected_trace: Optional[Sequence[Dict[str, Any]]] = None,
        expected_snapshots: Optional[Sequence[Dict[str, Any]]] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        thresholds = thresholds or {
            "daily_portfolio_value_pct": 1.0,
            "final_total_return_pct": 2.0,
            "max_drawdown_pct": 2.0,
            "total_trades": 1.0,
        }
        report: Dict[str, Any] = {
            "status": "no_baseline",
            "thresholds": thresholds,
            "event_count": len(self.event_trace),
            "snapshot_count": len(self.portfolio_snapshots),
            "first_mismatch": None,
        }
        if not expected_trace:
            return report

        def snapshots_for_date(
            snapshots: Sequence[Dict[str, Any]] | None,
            date_str: Optional[str],
        ) -> List[Dict[str, Any]]:
            if not snapshots or not date_str:
                return []
            return [dict(item) for item in snapshots if item.get("date") == date_str]

        def build_mismatch(
            *,
            index: int,
            reason: Optional[str] = None,
            field: Optional[str] = None,
            expected: Any = None,
            actual: Any = None,
            expected_event: Optional[Dict[str, Any]] = None,
            actual_event: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            mismatch_date = None
            if actual_event and actual_event.get("date"):
                mismatch_date = actual_event.get("date")
            elif expected_event and expected_event.get("date"):
                mismatch_date = expected_event.get("date")
            payload: Dict[str, Any] = {
                "index": index,
                "actual_event": dict(actual_event) if actual_event else None,
                "expected_event": dict(expected_event) if expected_event else None,
                "actual_snapshots": snapshots_for_date(self.portfolio_snapshots, mismatch_date),
                "expected_snapshots": snapshots_for_date(expected_snapshots, mismatch_date),
            }
            if reason is not None:
                payload["reason"] = reason
            if field is not None:
                payload["field"] = field
                payload["expected"] = expected
                payload["actual"] = actual
            return payload

        report["status"] = "matched"
        compare_keys = (
            "date",
            "event_type",
            "symbol",
            "action",
            "right",
            "qty",
            "strike",
            "expiry",
            "exit_reason",
            "assignment",
        )
        for index, actual in enumerate(self.event_trace):
            if index >= len(expected_trace):
                report["status"] = "length_mismatch"
                report["first_mismatch"] = build_mismatch(
                    index=index,
                    reason="actual_trace_longer",
                    actual_event=actual,
                )
                return report
            expected = expected_trace[index]
            for key in compare_keys:
                if expected.get(key) != actual.get(key):
                    report["status"] = "mismatch"
                    report["first_mismatch"] = build_mismatch(
                        index=index,
                        field=key,
                        expected=expected.get(key),
                        actual=actual.get(key),
                        expected_event=expected,
                        actual_event=actual,
                    )
                    return report
        if len(expected_trace) > len(self.event_trace):
            report["status"] = "length_mismatch"
            missing_index = len(self.event_trace)
            report["first_mismatch"] = build_mismatch(
                index=missing_index,
                reason="expected_trace_longer",
                expected_event=expected_trace[missing_index],
            )
            return report

        if expected_snapshots:
            report["expected_snapshot_count"] = len(expected_snapshots)
            for index, actual in enumerate(self.portfolio_snapshots):
                if index >= len(expected_snapshots):
                    report["status"] = "length_mismatch"
                    report["first_mismatch"] = {
                        "index": index,
                        "reason": "actual_snapshots_longer",
                        "actual_snapshot": dict(actual),
                        "expected_snapshot": None,
                    }
                    return report
                expected = expected_snapshots[index]
                if expected.get("date") != actual.get("date"):
                    report["status"] = "mismatch"
                    report["first_mismatch"] = {
                        "index": index,
                        "field": "snapshot_date",
                        "expected": expected.get("date"),
                        "actual": actual.get("date"),
                        "actual_snapshot": dict(actual),
                        "expected_snapshot": dict(expected),
                    }
                    return report
                if expected.get("phase") != actual.get("phase"):
                    report["status"] = "mismatch"
                    report["first_mismatch"] = {
                        "index": index,
                        "field": "snapshot_phase",
                        "expected": expected.get("phase"),
                        "actual": actual.get("phase"),
                        "actual_snapshot": dict(actual),
                        "expected_snapshot": dict(expected),
                    }
                    return report
                actual_value = actual.get("portfolio_value")
                expected_value = expected.get("portfolio_value")
                if actual_value is not None and expected_value not in (None, 0):
                    pct_diff = abs(actual_value - expected_value) / abs(expected_value) * 100
                    if pct_diff > thresholds["daily_portfolio_value_pct"]:
                        report["status"] = "mismatch"
                        report["first_mismatch"] = {
                            "index": index,
                            "field": "portfolio_value",
                            "expected": expected_value,
                            "actual": actual_value,
                            "pct_diff": round(pct_diff, 4),
                            "actual_snapshot": dict(actual),
                            "expected_snapshot": dict(expected),
                        }
                        return report
            if len(expected_snapshots) > len(self.portfolio_snapshots):
                missing_index = len(self.portfolio_snapshots)
                report["status"] = "length_mismatch"
                report["first_mismatch"] = {
                    "index": missing_index,
                    "reason": "expected_snapshots_longer",
                    "actual_snapshot": None,
                    "expected_snapshot": dict(expected_snapshots[missing_index]),
                }
        return report


def select_best_signal_with_memory(
    signals: List[Any],
    last_selected_stock: Optional[str],
    selection_count: int,
    min_hold_cycles: int,
    last_stock_scores: Dict[str, float],
) -> tuple[Any, str, int, Dict[str, float]]:
    """QC-style stock selection memory for SP candidates."""
    if not signals:
        return None, last_selected_stock, selection_count, last_stock_scores

    updated_scores = dict(last_stock_scores)
    for signal in signals:
        signal.confidence += getattr(signal, "ml_score_adjustment", 0.0) * 0.5
        updated_scores[signal.symbol] = signal.confidence

    signals_sorted = sorted(signals, key=lambda item: item.confidence, reverse=True)
    best_signal = signals_sorted[0]
    best_symbol = best_signal.symbol
    new_count = selection_count
    new_last = last_selected_stock

    if last_selected_stock is not None:
        new_count += 1
        if last_selected_stock != best_symbol:
            prev_score = last_stock_scores.get(last_selected_stock, 0.0)
            new_score = best_signal.confidence
            score_improvement = ((new_score - prev_score) / prev_score * 100) if prev_score > 0 else 100.0
            if new_count < min_hold_cycles and score_improvement < 10:
                for signal in signals_sorted:
                    if signal.symbol == last_selected_stock:
                        return signal, last_selected_stock, new_count, updated_scores
            new_count = 0
            new_last = best_symbol
    else:
        new_last = best_symbol
        new_count = 0

    return best_signal, new_last, new_count, updated_scores
