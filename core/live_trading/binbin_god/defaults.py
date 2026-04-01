"""Shared defaults for Binbin God live trading without heavy backtest imports."""

from __future__ import annotations

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_QC_CONFIG_PATH = _REPO_ROOT / "quantconnect" / "config.json"

QC_PARAMETER_FALLBACKS = {
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 100000.0,
    "stock_pool": "MSFT,AAPL,NVDA,GOOGL,AMZN,META,TSLA",
    "max_positions_ceiling": 20,
    "profit_target_pct": 50.0,
    "stop_loss_pct": 999999.0,
    "margin_buffer_pct": 0.40,
    "margin_rate_per_contract": 0.25,
    "target_margin_utilization": 0.50,
    "position_aggressiveness": 1.2,
    "max_risk_per_trade": 0.03,
    "symbol_assignment_base_cap": 0.28,
    "stock_inventory_base_cap": 0.17,
    "stock_inventory_block_threshold": 0.85,
    "defensive_put_roll_loss_pct": 85.0,
    "defensive_put_roll_itm_buffer_pct": 0.04,
    "ml_enabled": True,
    "dte_min": 21,
    "dte_max": 60,
    "put_delta": 0.30,
    "call_delta": 0.30,
}


def parse_default_stock_pool(raw_value) -> list[str]:
    if isinstance(raw_value, str):
        symbols = [symbol.strip().upper() for symbol in raw_value.split(",") if symbol.strip()]
        if symbols:
            return symbols
    if isinstance(raw_value, (list, tuple)):
        symbols = [str(symbol).strip().upper() for symbol in raw_value if str(symbol).strip()]
        if symbols:
            return symbols
    return ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]


def _load_qc_parameter_defaults() -> dict:
    try:
        import json

        raw = json.loads(_QC_CONFIG_PATH.read_text(encoding="utf-8"))
        parameters = raw.get("parameters", {})
        if isinstance(parameters, dict):
            merged = dict(QC_PARAMETER_FALLBACKS)
            merged.update(parameters)
            return merged
    except Exception:
        pass
    return dict(QC_PARAMETER_FALLBACKS)


QC_PARAMETER_DEFAULTS = _load_qc_parameter_defaults()
DEFAULT_STOCK_POOL = parse_default_stock_pool(QC_PARAMETER_DEFAULTS.get("stock_pool"))


def build_live_defaults() -> dict:
    return {
        "start_date": str(QC_PARAMETER_DEFAULTS.get("start_date", "2024-01-01")),
        "end_date": str(QC_PARAMETER_DEFAULTS.get("end_date", "2024-12-31")),
        "initial_capital": float(QC_PARAMETER_DEFAULTS.get("initial_capital", 100000.0)),
        "stock_pool": DEFAULT_STOCK_POOL.copy(),
        "stock_pool_text": ",".join(DEFAULT_STOCK_POOL),
        "max_positions_ceiling": int(QC_PARAMETER_DEFAULTS.get("max_positions_ceiling", 20)),
        "target_margin_utilization": float(QC_PARAMETER_DEFAULTS.get("target_margin_utilization", 0.50)),
        "position_aggressiveness": float(QC_PARAMETER_DEFAULTS.get("position_aggressiveness", 1.2)),
        "profit_target_pct": float(QC_PARAMETER_DEFAULTS.get("profit_target_pct", 50.0)),
        "stop_loss_pct": 999999.0,
        "margin_buffer_pct": float(QC_PARAMETER_DEFAULTS.get("margin_buffer_pct", 0.40)),
        "margin_rate_per_contract": float(QC_PARAMETER_DEFAULTS.get("margin_rate_per_contract", 0.25)),
        "max_risk_per_trade": float(QC_PARAMETER_DEFAULTS.get("max_risk_per_trade", 0.03)),
        "dte_min": int(QC_PARAMETER_DEFAULTS.get("dte_min", 21)),
        "dte_max": int(QC_PARAMETER_DEFAULTS.get("dte_max", 60)),
        "put_delta": float(QC_PARAMETER_DEFAULTS.get("put_delta", 0.30)),
        "call_delta": float(QC_PARAMETER_DEFAULTS.get("call_delta", 0.30)),
        "ml_enabled": bool(QC_PARAMETER_DEFAULTS.get("ml_enabled", True)),
        "defensive_put_roll_enabled": True,
        "defensive_put_roll_loss_pct": float(QC_PARAMETER_DEFAULTS.get("defensive_put_roll_loss_pct", 85.0)),
        "defensive_put_roll_itm_buffer_pct": float(QC_PARAMETER_DEFAULTS.get("defensive_put_roll_itm_buffer_pct", 0.04)),
        "assignment_cooldown_days": 20,
        "large_loss_cooldown_days": 15,
        "large_loss_cooldown_pct": 100,
        "dynamic_symbol_risk_enabled": True,
        "symbol_assignment_base_cap": float(QC_PARAMETER_DEFAULTS.get("symbol_assignment_base_cap", 0.28)),
        "stock_inventory_cap_enabled": True,
        "stock_inventory_base_cap": float(QC_PARAMETER_DEFAULTS.get("stock_inventory_base_cap", 0.17)),
        "stock_inventory_block_threshold": float(QC_PARAMETER_DEFAULTS.get("stock_inventory_block_threshold", 0.85)),
        "poll_interval_seconds": 60,
        "allow_new_entries": True,
        "max_parallel_open_orders": 2,
        "enable_emergency_controls": True,
    }
