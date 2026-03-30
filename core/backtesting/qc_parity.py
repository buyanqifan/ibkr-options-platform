"""QC parity helpers for BinbinGod backtests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

from core.backtesting.pricing import OptionsPricer


QC_BINBIN_DEFAULTS = {
    "initial_capital": 100000.0,
    "max_positions_ceiling": 15,
    "profit_target_pct": 50.0,
    "stop_loss_pct": 999999.0,
    "margin_buffer_pct": 0.50,
    "margin_rate_per_contract": 0.25,
    "target_margin_utilization": 0.60,
    "position_aggressiveness": 1.0,
    "max_leverage": 1.0,
    "ml_enabled": True,
    "ml_min_confidence": 0.40,
    "dte_min": 21,
    "dte_max": 60,
    "put_delta": 0.30,
    "call_delta": 0.30,
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

    parity_mode: str = "none"
    contract_universe_mode: str = "legacy_theoretical"
    ml_confidence_gate: float = 0.40
    initial_capital: float = QC_BINBIN_DEFAULTS["initial_capital"]
    max_positions_ceiling: int = QC_BINBIN_DEFAULTS["max_positions_ceiling"]
    profit_target_pct: float = QC_BINBIN_DEFAULTS["profit_target_pct"]
    stop_loss_pct: float = QC_BINBIN_DEFAULTS["stop_loss_pct"]
    margin_buffer_pct: float = QC_BINBIN_DEFAULTS["margin_buffer_pct"]
    margin_rate_per_contract: float = QC_BINBIN_DEFAULTS["margin_rate_per_contract"]
    target_margin_utilization: float = QC_BINBIN_DEFAULTS["target_margin_utilization"]
    position_aggressiveness: float = QC_BINBIN_DEFAULTS["position_aggressiveness"]
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

    @property
    def enabled(self) -> bool:
        return self.parity_mode == "qc"

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "BinbinGodParityConfig":
        parity_mode = str(params.get("parity_mode", "none") or "none").lower()
        contract_universe_mode = str(
            params.get("contract_universe_mode", "legacy_theoretical") or "legacy_theoretical"
        ).lower()
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
                "max_leverage": self.max_leverage,
                "ml_enabled": self.ml_enabled,
                "ml_min_confidence": self.ml_min_confidence,
                "dte_min": self.dte_min,
                "dte_max": self.dte_max,
                "put_delta": self.put_delta,
                "call_delta": self.call_delta,
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
) -> tuple[int, Dict[str, int]]:
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
    max_by_symbol_contracts = max(0, config.max_put_contracts_per_symbol - symbol_put_contracts)
    max_by_total_contracts = max(0, config.max_put_contracts_total - total_put_contracts)
    max_by_trade_cap = max(0, config.max_contracts_per_trade)
    max_by_margin = int(usable_margin / estimated_margin_per_contract) if estimated_margin_per_contract > 0 else 0
    max_by_limit = max(0, adjusted_max_positions - current_positions)

    margin_budget = portfolio_value * config.target_margin_utilization
    remaining_budget = max(0.0, margin_budget - total_margin_used)
    max_by_budget = int(remaining_budget / estimated_margin_per_contract) if estimated_margin_per_contract > 0 else 0

    leverage_budget = portfolio_value * config.max_leverage
    remaining_leverage_budget = max(0.0, leverage_budget - total_margin_used)
    max_by_leverage = int(remaining_leverage_budget / estimated_margin_per_contract) if estimated_margin_per_contract > 0 else 0

    aggr = config.position_aggressiveness
    per_symbol_notional_cap = portfolio_value * (0.25 + 0.35 * aggr)
    total_notional_cap = portfolio_value * (0.70 + 0.90 * aggr)
    candidate_notional = strike * 100
    remaining_symbol_notional = max(0.0, per_symbol_notional_cap - (symbol_put_notional + symbol_stock_notional))
    remaining_total_notional = max(0.0, total_notional_cap - (total_put_notional + stock_holdings_value))
    max_by_symbol_notional = int(remaining_symbol_notional / candidate_notional) if candidate_notional > 0 else 0
    max_by_total_notional = int(remaining_total_notional / candidate_notional) if candidate_notional > 0 else 0

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
    }
    quantity = min(diagnostics.values()) if diagnostics else 0
    return max(0, quantity), diagnostics


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

        report["status"] = "matched"
        compare_keys = ("symbol", "action", "right", "qty", "exit_reason", "assignment")
        for index, actual in enumerate(self.event_trace):
            if index >= len(expected_trace):
                report["status"] = "length_mismatch"
                report["first_mismatch"] = {"index": index, "reason": "actual_trace_longer"}
                return report
            expected = expected_trace[index]
            for key in compare_keys:
                if expected.get(key) != actual.get(key):
                    report["status"] = "mismatch"
                    report["first_mismatch"] = {
                        "index": index,
                        "field": key,
                        "expected": expected.get(key),
                        "actual": actual.get(key),
                    }
                    return report
        if len(expected_trace) > len(self.event_trace):
            report["status"] = "length_mismatch"
            report["first_mismatch"] = {"index": len(self.event_trace), "reason": "expected_trace_longer"}
        if expected_snapshots:
            report["expected_snapshot_count"] = len(expected_snapshots)
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
