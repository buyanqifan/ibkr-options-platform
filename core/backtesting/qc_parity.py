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
    "target_margin_utilization": 0.65,
    "symbol_assignment_base_cap": 0.35,
    "max_assignment_risk_per_trade": 0.20,
    "roll_threshold_pct": 80.0,
    "min_dte_for_roll": 7,
    "roll_target_dte_min": 21,
    "roll_target_dte_max": 45,
    "sp_primary_delta_tolerance": 0.12,
    "sp_relaxed_delta_tolerance": 0.22,
    "sp_min_option_premium": 0.05,
    "cc_below_cost_enabled": True,
    "cc_target_delta": 0.25,
    "cc_target_dte_min": 10,
    "cc_target_dte_max": 28,
    "cc_max_discount_to_cost": 0.03,
    "assigned_stock_fail_safe_enabled": True,
    "assigned_stock_min_days_held": 5,
    "assigned_stock_drawdown_pct": 0.12,
    "assigned_stock_force_exit_pct": 1.0,
    "max_new_puts_per_day": 3,
    "ml_enabled": True,
    "ml_min_confidence": 0.45,
    "ml_adoption_rate": 0.5,
    "ml_exploration_rate": 0.1,
    "ml_learning_rate": 0.01,
    "profit_target_pct": 50.0,
    "stop_loss_pct": 999999.0,
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
    "target_margin_utilization": float(QC_PARAMETER_DEFAULTS["target_margin_utilization"]),
    "max_assignment_risk_per_trade": float(QC_PARAMETER_DEFAULTS["max_assignment_risk_per_trade"]),
    "symbol_assignment_base_cap": float(QC_PARAMETER_DEFAULTS["symbol_assignment_base_cap"]),
    "roll_threshold_pct": float(QC_PARAMETER_DEFAULTS["roll_threshold_pct"]),
    "min_dte_for_roll": int(QC_PARAMETER_DEFAULTS["min_dte_for_roll"]),
    "roll_target_dte_min": int(QC_PARAMETER_DEFAULTS["roll_target_dte_min"]),
    "roll_target_dte_max": int(QC_PARAMETER_DEFAULTS["roll_target_dte_max"]),
    "cc_below_cost_enabled": bool(QC_PARAMETER_DEFAULTS["cc_below_cost_enabled"]),
    "cc_target_delta": float(QC_PARAMETER_DEFAULTS["cc_target_delta"]),
    "cc_target_dte_min": int(QC_PARAMETER_DEFAULTS["cc_target_dte_min"]),
    "cc_target_dte_max": int(QC_PARAMETER_DEFAULTS["cc_target_dte_max"]),
    "cc_max_discount_to_cost": float(QC_PARAMETER_DEFAULTS["cc_max_discount_to_cost"]),
    "assigned_stock_fail_safe_enabled": bool(QC_PARAMETER_DEFAULTS["assigned_stock_fail_safe_enabled"]),
    "assigned_stock_min_days_held": int(QC_PARAMETER_DEFAULTS["assigned_stock_min_days_held"]),
    "assigned_stock_drawdown_pct": float(QC_PARAMETER_DEFAULTS["assigned_stock_drawdown_pct"]),
    "assigned_stock_force_exit_pct": float(QC_PARAMETER_DEFAULTS["assigned_stock_force_exit_pct"]),
    "max_new_puts_per_day": int(QC_PARAMETER_DEFAULTS["max_new_puts_per_day"]),
    "sp_primary_delta_tolerance": float(QC_PARAMETER_DEFAULTS["sp_primary_delta_tolerance"]),
    "sp_relaxed_delta_tolerance": float(QC_PARAMETER_DEFAULTS["sp_relaxed_delta_tolerance"]),
    "sp_min_option_premium": float(QC_PARAMETER_DEFAULTS["sp_min_option_premium"]),
    "ml_enabled": bool(QC_PARAMETER_DEFAULTS["ml_enabled"]),
    "ml_min_confidence": float(QC_PARAMETER_DEFAULTS["ml_min_confidence"]),
    "ml_adoption_rate": float(QC_PARAMETER_DEFAULTS["ml_adoption_rate"]),
    "ml_exploration_rate": float(QC_PARAMETER_DEFAULTS["ml_exploration_rate"]),
    "ml_learning_rate": float(QC_PARAMETER_DEFAULTS["ml_learning_rate"]),
    "profit_target_pct": float(QC_PARAMETER_DEFAULTS.get("profit_target_pct", 50.0)),
    "stop_loss_pct": float(QC_PARAMETER_DEFAULTS.get("stop_loss_pct", 999999.0)),
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
    ml_confidence_gate: float = QC_BINBIN_DEFAULTS["ml_min_confidence"]
    initial_capital: float = QC_BINBIN_DEFAULTS["initial_capital"]
    max_positions_ceiling: int = QC_BINBIN_DEFAULTS["max_positions_ceiling"]
    target_margin_utilization: float = QC_BINBIN_DEFAULTS["target_margin_utilization"]
    max_assignment_risk_per_trade: float = QC_BINBIN_DEFAULTS["max_assignment_risk_per_trade"]
    symbol_assignment_base_cap: float = QC_BINBIN_DEFAULTS["symbol_assignment_base_cap"]
    roll_threshold_pct: float = QC_BINBIN_DEFAULTS["roll_threshold_pct"]
    min_dte_for_roll: int = QC_BINBIN_DEFAULTS["min_dte_for_roll"]
    roll_target_dte_min: int = QC_BINBIN_DEFAULTS["roll_target_dte_min"]
    roll_target_dte_max: int = QC_BINBIN_DEFAULTS["roll_target_dte_max"]
    cc_below_cost_enabled: bool = bool(QC_BINBIN_DEFAULTS["cc_below_cost_enabled"])
    cc_target_delta: float = QC_BINBIN_DEFAULTS["cc_target_delta"]
    cc_target_dte_min: int = QC_BINBIN_DEFAULTS["cc_target_dte_min"]
    cc_target_dte_max: int = QC_BINBIN_DEFAULTS["cc_target_dte_max"]
    cc_max_discount_to_cost: float = QC_BINBIN_DEFAULTS["cc_max_discount_to_cost"]
    assigned_stock_fail_safe_enabled: bool = bool(QC_BINBIN_DEFAULTS["assigned_stock_fail_safe_enabled"])
    assigned_stock_min_days_held: int = QC_BINBIN_DEFAULTS["assigned_stock_min_days_held"]
    assigned_stock_drawdown_pct: float = QC_BINBIN_DEFAULTS["assigned_stock_drawdown_pct"]
    assigned_stock_force_exit_pct: float = QC_BINBIN_DEFAULTS["assigned_stock_force_exit_pct"]
    max_new_puts_per_day: int = QC_BINBIN_DEFAULTS["max_new_puts_per_day"]
    sp_primary_delta_tolerance: float = QC_BINBIN_DEFAULTS["sp_primary_delta_tolerance"]
    sp_relaxed_delta_tolerance: float = QC_BINBIN_DEFAULTS["sp_relaxed_delta_tolerance"]
    sp_min_option_premium: float = QC_BINBIN_DEFAULTS["sp_min_option_premium"]
    ml_enabled: bool = bool(QC_BINBIN_DEFAULTS["ml_enabled"])
    ml_min_confidence: float = QC_BINBIN_DEFAULTS["ml_min_confidence"]
    ml_adoption_rate: float = QC_BINBIN_DEFAULTS["ml_adoption_rate"]
    ml_exploration_rate: float = QC_BINBIN_DEFAULTS["ml_exploration_rate"]
    ml_learning_rate: float = QC_BINBIN_DEFAULTS["ml_learning_rate"]

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
        max_positions_ceiling = _to_int(
            merged.get("max_positions_ceiling", merged.get("max_positions", QC_BINBIN_DEFAULTS["max_positions_ceiling"])),
            QC_BINBIN_DEFAULTS["max_positions_ceiling"],
        )
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
            target_margin_utilization=_to_float(
                merged.get("target_margin_utilization", QC_BINBIN_DEFAULTS["target_margin_utilization"]),
                QC_BINBIN_DEFAULTS["target_margin_utilization"],
            ),
            max_assignment_risk_per_trade=_to_float(merged.get("max_assignment_risk_per_trade", QC_BINBIN_DEFAULTS["max_assignment_risk_per_trade"]), QC_BINBIN_DEFAULTS["max_assignment_risk_per_trade"]),
            symbol_assignment_base_cap=_clamp(
                _to_float(merged.get("symbol_assignment_base_cap", QC_BINBIN_DEFAULTS["symbol_assignment_base_cap"]), QC_BINBIN_DEFAULTS["symbol_assignment_base_cap"]),
                0.05,
                1.5,
            ),
            roll_threshold_pct=_to_float(merged.get("roll_threshold_pct", QC_BINBIN_DEFAULTS["roll_threshold_pct"]), QC_BINBIN_DEFAULTS["roll_threshold_pct"]),
            min_dte_for_roll=_to_int(merged.get("min_dte_for_roll", QC_BINBIN_DEFAULTS["min_dte_for_roll"]), QC_BINBIN_DEFAULTS["min_dte_for_roll"]),
            roll_target_dte_min=_to_int(merged.get("roll_target_dte_min", QC_BINBIN_DEFAULTS["roll_target_dte_min"]), QC_BINBIN_DEFAULTS["roll_target_dte_min"]),
            roll_target_dte_max=_to_int(merged.get("roll_target_dte_max", QC_BINBIN_DEFAULTS["roll_target_dte_max"]), QC_BINBIN_DEFAULTS["roll_target_dte_max"]),
            cc_below_cost_enabled=bool(merged.get("cc_below_cost_enabled", QC_BINBIN_DEFAULTS["cc_below_cost_enabled"])),
            cc_target_delta=_clamp(_to_float(merged.get("cc_target_delta", QC_BINBIN_DEFAULTS["cc_target_delta"]), QC_BINBIN_DEFAULTS["cc_target_delta"]), 0.10, 0.60),
            cc_target_dte_min=_to_int(merged.get("cc_target_dte_min", QC_BINBIN_DEFAULTS["cc_target_dte_min"]), QC_BINBIN_DEFAULTS["cc_target_dte_min"]),
            cc_target_dte_max=_to_int(merged.get("cc_target_dte_max", QC_BINBIN_DEFAULTS["cc_target_dte_max"]), QC_BINBIN_DEFAULTS["cc_target_dte_max"]),
            cc_max_discount_to_cost=_clamp(_to_float(merged.get("cc_max_discount_to_cost", QC_BINBIN_DEFAULTS["cc_max_discount_to_cost"]), QC_BINBIN_DEFAULTS["cc_max_discount_to_cost"]), 0.0, 0.30),
            assigned_stock_fail_safe_enabled=bool(merged.get("assigned_stock_fail_safe_enabled", QC_BINBIN_DEFAULTS["assigned_stock_fail_safe_enabled"])),
            assigned_stock_min_days_held=_to_int(merged.get("assigned_stock_min_days_held", QC_BINBIN_DEFAULTS["assigned_stock_min_days_held"]), QC_BINBIN_DEFAULTS["assigned_stock_min_days_held"]),
            assigned_stock_drawdown_pct=_to_float(merged.get("assigned_stock_drawdown_pct", QC_BINBIN_DEFAULTS["assigned_stock_drawdown_pct"]), QC_BINBIN_DEFAULTS["assigned_stock_drawdown_pct"]),
            assigned_stock_force_exit_pct=_clamp(_to_float(merged.get("assigned_stock_force_exit_pct", QC_BINBIN_DEFAULTS["assigned_stock_force_exit_pct"]), QC_BINBIN_DEFAULTS["assigned_stock_force_exit_pct"]), 0.0, 1.0),
            max_new_puts_per_day=max(1, _to_int(merged.get("max_new_puts_per_day", QC_BINBIN_DEFAULTS["max_new_puts_per_day"]), QC_BINBIN_DEFAULTS["max_new_puts_per_day"])),
            ml_enabled=bool(merged.get("ml_enabled", QC_BINBIN_DEFAULTS["ml_enabled"])),
            ml_min_confidence=ml_min_confidence,
            ml_adoption_rate=_to_float(merged.get("ml_adoption_rate", QC_BINBIN_DEFAULTS["ml_adoption_rate"]), QC_BINBIN_DEFAULTS["ml_adoption_rate"]),
            ml_exploration_rate=_to_float(merged.get("ml_exploration_rate", QC_BINBIN_DEFAULTS["ml_exploration_rate"]), QC_BINBIN_DEFAULTS["ml_exploration_rate"]),
            ml_learning_rate=_to_float(merged.get("ml_learning_rate", QC_BINBIN_DEFAULTS["ml_learning_rate"]), QC_BINBIN_DEFAULTS["ml_learning_rate"]),
            sp_primary_delta_tolerance=_clamp(_to_float(merged.get("sp_primary_delta_tolerance", QC_BINBIN_DEFAULTS["sp_primary_delta_tolerance"]), QC_BINBIN_DEFAULTS["sp_primary_delta_tolerance"]), 0.04, 0.40),
            sp_relaxed_delta_tolerance=_clamp(_to_float(merged.get("sp_relaxed_delta_tolerance", QC_BINBIN_DEFAULTS["sp_relaxed_delta_tolerance"]), QC_BINBIN_DEFAULTS["sp_relaxed_delta_tolerance"]), _clamp(_to_float(merged.get("sp_primary_delta_tolerance", QC_BINBIN_DEFAULTS["sp_primary_delta_tolerance"]), QC_BINBIN_DEFAULTS["sp_primary_delta_tolerance"]), 0.04, 0.40), 0.45),
            sp_min_option_premium=_clamp(_to_float(merged.get("sp_min_option_premium", QC_BINBIN_DEFAULTS["sp_min_option_premium"]), QC_BINBIN_DEFAULTS["sp_min_option_premium"]), 0.01, 1.0),
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
                "target_margin_utilization": self.target_margin_utilization,
                "max_assignment_risk_per_trade": self.max_assignment_risk_per_trade,
                "symbol_assignment_base_cap": self.symbol_assignment_base_cap,
                "roll_threshold_pct": self.roll_threshold_pct,
                "min_dte_for_roll": self.min_dte_for_roll,
                "roll_target_dte_min": self.roll_target_dte_min,
                "roll_target_dte_max": self.roll_target_dte_max,
                "cc_below_cost_enabled": self.cc_below_cost_enabled,
                "cc_target_delta": self.cc_target_delta,
                "cc_target_dte_min": self.cc_target_dte_min,
                "cc_target_dte_max": self.cc_target_dte_max,
                "cc_max_discount_to_cost": self.cc_max_discount_to_cost,
                "assigned_stock_fail_safe_enabled": self.assigned_stock_fail_safe_enabled,
                "assigned_stock_min_days_held": self.assigned_stock_min_days_held,
                "assigned_stock_drawdown_pct": self.assigned_stock_drawdown_pct,
                "assigned_stock_force_exit_pct": self.assigned_stock_force_exit_pct,
                "max_new_puts_per_day": self.max_new_puts_per_day,
                "ml_enabled": self.ml_enabled,
                "ml_min_confidence": self.ml_min_confidence,
                "ml_adoption_rate": self.ml_adoption_rate,
                "ml_exploration_rate": self.ml_exploration_rate,
                "ml_learning_rate": self.ml_learning_rate,
                "sp_primary_delta_tolerance": self.sp_primary_delta_tolerance,
                "sp_relaxed_delta_tolerance": self.sp_relaxed_delta_tolerance,
                "sp_min_option_premium": self.sp_min_option_premium,
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
    selection_tier: str = "primary"

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
            "selection_tier": self.selection_tier,
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


def build_cc_selection_tiers_qc(
    *,
    config: BinbinGodParityConfig,
    underlying_price: float,
    cost_basis: float,
    primary_dte_min: int,
    primary_dte_max: int,
    primary_delta_tolerance: float,
    primary_min_strike: float | None,
) -> List[Dict[str, float]]:
    return [
        {
            "label": "primary",
            "delta_tolerance": primary_delta_tolerance,
            "dte_min": primary_dte_min,
            "dte_max": primary_dte_max,
            "min_strike": primary_min_strike,
        },
        {
            "label": "delta_relaxed",
            "delta_tolerance": 0.16,
            "dte_min": primary_dte_min,
            "dte_max": primary_dte_max,
            "min_strike": primary_min_strike,
        },
    ]


def build_sp_selection_tiers_qc(
    *,
    config: BinbinGodParityConfig,
    primary_dte_min: int,
    primary_dte_max: int,
) -> List[Dict[str, float]]:
    return [
        {
            "label": "primary",
            "delta_tolerance": config.sp_primary_delta_tolerance,
            "dte_min": primary_dte_min,
            "dte_max": primary_dte_max,
        },
        {
            "label": "delta_relaxed",
            "delta_tolerance": config.sp_relaxed_delta_tolerance,
            "dte_min": primary_dte_min,
            "dte_max": primary_dte_max,
        },
    ]


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
            min_premium_threshold = 0.05 if right == "P" else 0.10
            if premium <= min_premium_threshold:
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
    selection_tiers: Optional[List[Dict[str, Any]]] = None,
) -> Optional[LatticeContract]:
    tiers = selection_tiers or [
        {
            "label": "primary",
            "delta_tolerance": delta_tolerance,
            "dte_min": dte_min,
            "dte_max": dte_max,
            "min_strike": min_strike,
        }
    ]
    for tier in tiers:
        contracts = build_contract_lattice(
            symbol=symbol,
            current_date=current_date,
            underlying_price=underlying_price,
            iv=iv,
            target_right=target_right,
            target_delta=target_delta,
            dte_min=int(tier.get("dte_min", dte_min)),
            dte_max=int(tier.get("dte_max", dte_max)),
            delta_tolerance=float(tier.get("delta_tolerance", delta_tolerance)),
            min_strike=tier.get("min_strike", min_strike),
        )
        if contracts:
            contracts[0].selection_tier = str(tier.get("label", "primary"))
            return contracts[0]
    return None


def calculate_dynamic_max_positions_from_prices(
    prices: Sequence[float],
    config: BinbinGodParityConfig,
    portfolio_value: float | None = None,
) -> int:
    valid_prices = [price for price in prices if price and price > 0]
    if not valid_prices:
        return config.max_positions_ceiling
    avg_price = sum(valid_prices) / len(valid_prices)
    capital_base = portfolio_value if portfolio_value is not None else config.initial_capital
    margin_budget = capital_base * config.target_margin_utilization
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
    del symbol, symbol_history_bars, pool_history_bars
    return max(1, int(config.max_positions_ceiling))


def calculate_symbol_state_risk_multiplier_qc(
    config: BinbinGodParityConfig,
    symbol_history_bars: Sequence[Dict[str, Any]] | None,
    pool_history_bars: Dict[str, Sequence[Dict[str, Any]]] | None,
    underlying_price: float,
    symbol_put_notional: float,
    symbol_stock_notional: float,
    portfolio_value: float,
) -> tuple[float, Dict[str, float]]:
    drawdown = 0.0
    symbol_bars = [float(bar.get("close", 0.0)) for bar in list(symbol_history_bars or []) if bar.get("close", 0)]
    if symbol_bars:
        peak_price = max(symbol_bars[-60:] or [underlying_price])
        if peak_price > 0:
            drawdown = max(0.0, (peak_price - underlying_price) / peak_price)
    exposure_ratio = (symbol_put_notional + symbol_stock_notional) / portfolio_value if portfolio_value > 0 else 0.0
    multiplier = 1.0
    return multiplier, {
        "vol_ratio": 1.0,
        "drawdown": drawdown,
        "momentum_20d": 0.0,
        "exposure_ratio": exposure_ratio,
    }


def calculate_stock_inventory_cap_qc(
    config: BinbinGodParityConfig,
    portfolio_value: float,
    symbol_state_multiplier: float,
) -> float:
    del config, symbol_state_multiplier
    return portfolio_value


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
    estimated_margin_per_contract = estimate_put_margin_qc(
        strike=strike,
        premium=premium,
        underlying_price=underlying_price,
        margin_rate_per_contract=0.25,
    )

    total_put_notional = 0.0
    symbol_put_notional = 0.0
    for pos in open_option_positions:
        if getattr(pos, "right", "") != "P":
            continue
        contracts = abs(int(getattr(pos, "quantity", 0)))
        strike_h = float(getattr(pos, "strike", 0.0))
        notional = contracts * strike_h * 100
        total_put_notional += notional
        if getattr(pos, "symbol", "") == symbol:
            symbol_put_notional += notional

    symbol_stock_notional = shares_held * max(underlying_price, 0.0)
    symbol_state_multiplier, symbol_state = calculate_symbol_state_risk_multiplier_qc(
        config,
        symbol_history_bars,
        pool_history_bars,
        underlying_price,
        symbol_put_notional,
        symbol_stock_notional,
        portfolio_value,
    )
    margin_budget = portfolio_value * config.target_margin_utilization
    remaining_budget = max(0.0, margin_budget - total_margin_used)
    candidate_notional = strike * 100
    remaining_symbol_notional = max(
        0.0,
        portfolio_value * config.symbol_assignment_base_cap * symbol_state_multiplier - (symbol_put_notional + symbol_stock_notional),
    )
    assignment_trade_cap = portfolio_value * config.max_assignment_risk_per_trade
    diagnostics: Dict[str, float] = {
        "portfolio_margin_capacity": int(remaining_budget / estimated_margin_per_contract) if estimated_margin_per_contract > 0 else 0,
        "symbol_assignment_capacity": int(remaining_symbol_notional / candidate_notional) if candidate_notional > 0 else 0,
        "trade_assignment_capacity": int(assignment_trade_cap / candidate_notional) if candidate_notional > 0 else 0,
        "position_slot_capacity": max(0, dynamic_max_positions - current_positions),
        "state_multiplier": round(symbol_state_multiplier, 4),
        "vol_ratio": round(symbol_state["vol_ratio"], 4),
        "drawdown": round(symbol_state["drawdown"], 4),
        "momentum_20d": round(symbol_state["momentum_20d"], 4),
        "exposure_ratio": round(symbol_state["exposure_ratio"], 4),
    }
    block_sequence = (
        ("portfolio_margin", "portfolio_margin_capacity"),
        ("symbol_assignment", "symbol_assignment_capacity"),
        ("trade_assignment", "trade_assignment_capacity"),
        ("position_slots", "position_slot_capacity"),
    )
    for reason, key in block_sequence:
        if diagnostics[key] <= 0:
            diagnostics["block_reason"] = reason
            return 0, diagnostics

    diagnostics["block_reason"] = ""
    quantity = min(int(diagnostics[key]) for _, key in block_sequence)
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
    fallback_margin = strike * 100 * 0.20
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
