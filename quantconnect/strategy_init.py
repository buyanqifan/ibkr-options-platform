"""Initialization functions for BinbinGod Strategy."""
from datetime import datetime, timedelta
from AlgorithmImports import Resolution, DataNormalizationMode
from helpers import StockScore
from qc_portfolio import init_position_tracking
from ml_integration import MLOptimizationConfig

MAG7_STOCKS = ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]


def _get_param(algo, name, default=None):
    """Read QC parameter with default fallback for empty string."""
    value = algo.GetParameter(name)
    if value is None or value == "":
        return default
    return value


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    return default


def _as_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def _as_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _clamp(value, low, high):
    return max(low, min(high, value))


def init_dates(algo):
    start_date_param = algo.GetParameter("start_date")
    end_date_param = algo.GetParameter("end_date")
    if start_date_param:
        algo.SetStartDate(datetime.strptime(start_date_param, "%Y-%m-%d"))
    else:
        algo.SetStartDate(2024, 1, 1)
    if end_date_param:
        algo.SetEndDate(datetime.strptime(end_date_param, "%Y-%m-%d"))
    else:
        algo.SetEndDate(datetime.now() - timedelta(days=1))


def init_parameters(algo):
    algo.initial_capital = _as_float(_get_param(algo, "initial_capital", 100000), 100000.0)
    algo.SetCash(algo.initial_capital)
    # Backward compatible: support both max_positions and max_positions_ceiling
    max_pos_raw = _get_param(algo, "max_positions", None)
    if max_pos_raw is None:
        max_pos_raw = _get_param(algo, "max_positions_ceiling", 20)
    algo.max_positions_ceiling = _as_int(max_pos_raw, 20)
    algo.max_positions = algo.max_positions_ceiling  # Will be recalculated dynamically
    algo.profit_target_pct = _as_float(_get_param(algo, "profit_target_pct", 50), 50.0)
    algo.stop_loss_pct = _as_float(_get_param(algo, "stop_loss_pct", 999999), 999999.0)
    algo.max_risk_per_trade = _as_float(_get_param(algo, "max_risk_per_trade", 0.03), 0.03)
    algo.max_assignment_risk_per_trade = _as_float(_get_param(algo, "max_assignment_risk_per_trade", 0.20), 0.20)
    algo.max_leverage = _as_float(_get_param(algo, "max_leverage", 1.0), 1.0)
    algo._profit_target_disabled = algo.profit_target_pct >= 999999
    algo._stop_loss_disabled = algo.stop_loss_pct >= 999999
    algo.margin_buffer_pct = _as_float(_get_param(algo, "margin_buffer_pct", 0.40), 0.40)
    algo.margin_rate_per_contract = _as_float(_get_param(algo, "margin_rate_per_contract", 0.25), 0.25)
    # Position aggressiveness (0.3 conservative -> 1.0 baseline -> 2.0 aggressive)
    algo.position_aggressiveness = _clamp(_as_float(_get_param(algo, "position_aggressiveness", 1.2), 1.2), 0.3, 2.0)

    # Dynamic caps derived from aggressiveness (no hard-coded contract counts)
    symbol_cap_factor = 0.30 + 0.30 * algo.position_aggressiveness
    total_cap_factor = 1.20 + 0.80 * algo.position_aggressiveness
    trade_cap_factor = 0.60 + 0.40 * algo.position_aggressiveness
    algo.max_put_contracts_per_symbol = max(1, int(algo.max_positions_ceiling * symbol_cap_factor))
    algo.max_put_contracts_total = max(2, int(algo.max_positions_ceiling * total_cap_factor))
    algo.max_contracts_per_trade = max(1, int(algo.max_put_contracts_per_symbol * trade_cap_factor))
    algo.cc_optimization_enabled = _as_bool(_get_param(algo, "cc_optimization_enabled", True), True)
    algo.cc_min_delta_cost = _as_float(_get_param(algo, "cc_min_delta_cost", 0.15), 0.15)
    algo.cc_cost_basis_threshold = _as_float(_get_param(algo, "cc_cost_basis_threshold", 0.05), 0.05)
    algo.cc_min_strike_premium = _as_float(_get_param(algo, "cc_min_strike_premium", 0.02), 0.02)
    algo.repair_call_threshold_pct = _as_float(_get_param(algo, "repair_call_threshold_pct", 0.08), 0.08)
    algo.repair_call_delta = _clamp(_as_float(_get_param(algo, "repair_call_delta", 0.35), 0.35), 0.20, 0.60)
    algo.repair_call_dte_min = _as_int(_get_param(algo, "repair_call_dte_min", 7), 7)
    algo.repair_call_dte_max = _as_int(_get_param(algo, "repair_call_dte_max", 21), 21)
    algo.repair_call_max_discount_pct = _as_float(_get_param(algo, "repair_call_max_discount_pct", 0.08), 0.08)
    algo.defensive_put_roll_enabled = _as_bool(_get_param(algo, "defensive_put_roll_enabled", True), True)
    algo.defensive_put_roll_loss_pct = _as_float(_get_param(algo, "defensive_put_roll_loss_pct", 70), 70.0)
    algo.defensive_put_roll_itm_buffer_pct = _as_float(_get_param(algo, "defensive_put_roll_itm_buffer_pct", 0.03), 0.03)
    algo.defensive_put_roll_min_dte = _as_int(_get_param(algo, "defensive_put_roll_min_dte", 7), 7)
    algo.defensive_put_roll_max_dte = _as_int(_get_param(algo, "defensive_put_roll_max_dte", 21), 21)
    algo.defensive_put_roll_dte_min = _as_int(_get_param(algo, "defensive_put_roll_dte_min", 21), 21)
    algo.defensive_put_roll_dte_max = _as_int(_get_param(algo, "defensive_put_roll_dte_max", 60), 60)
    algo.defensive_put_roll_delta = _clamp(_as_float(_get_param(algo, "defensive_put_roll_delta", 0.20), 0.20), 0.10, 0.40)
    algo.assignment_cooldown_days = _as_int(_get_param(algo, "assignment_cooldown_days", 20), 20)
    algo.large_loss_cooldown_days = _as_int(_get_param(algo, "large_loss_cooldown_days", 15), 15)
    algo.large_loss_cooldown_pct = _as_float(_get_param(algo, "large_loss_cooldown_pct", 100), 100.0)
    algo.volatility_cap_floor = _clamp(_as_float(_get_param(algo, "volatility_cap_floor", 0.35), 0.35), 0.10, 1.0)
    algo.volatility_cap_ceiling = _clamp(_as_float(_get_param(algo, "volatility_cap_ceiling", 1.0), 1.0), 1.0, 3.0)
    algo.volatility_lookback = _as_int(_get_param(algo, "volatility_lookback", 20), 20)
    algo.dynamic_symbol_risk_enabled = _as_bool(_get_param(algo, "dynamic_symbol_risk_enabled", True), True)
    algo.symbol_state_cap_floor = _clamp(_as_float(_get_param(algo, "symbol_state_cap_floor", 0.20), 0.20), 0.05, 1.0)
    algo.symbol_state_cap_ceiling = _clamp(_as_float(_get_param(algo, "symbol_state_cap_ceiling", 1.0), 1.0), 0.50, 1.0)
    algo.symbol_drawdown_lookback = _as_int(_get_param(algo, "symbol_drawdown_lookback", 60), 60)
    algo.symbol_drawdown_sensitivity = _as_float(_get_param(algo, "symbol_drawdown_sensitivity", 1.20), 1.20)
    algo.symbol_downtrend_sensitivity = _as_float(_get_param(algo, "symbol_downtrend_sensitivity", 1.50), 1.50)
    algo.symbol_volatility_sensitivity = _as_float(_get_param(algo, "symbol_volatility_sensitivity", 0.75), 0.75)
    algo.symbol_exposure_sensitivity = _as_float(_get_param(algo, "symbol_exposure_sensitivity", 1.25), 1.25)
    default_symbol_assignment_cap = 0.20
    algo.symbol_assignment_base_cap = _clamp(
        _as_float(_get_param(algo, "symbol_assignment_base_cap", default_symbol_assignment_cap), default_symbol_assignment_cap),
        0.05,
        1.5,
    )
    algo.stock_inventory_cap_enabled = _as_bool(_get_param(algo, "stock_inventory_cap_enabled", True), True)
    algo.stock_inventory_base_cap = _clamp(_as_float(_get_param(algo, "stock_inventory_base_cap", 0.12), 0.12), 0.05, 1.0)
    algo.stock_inventory_cap_floor = _clamp(_as_float(_get_param(algo, "stock_inventory_cap_floor", 0.50), 0.50), 0.10, 1.0)
    algo.stock_inventory_block_threshold = _clamp(_as_float(_get_param(algo, "stock_inventory_block_threshold", 0.75), 0.75), 0.50, 1.20)
    algo.max_new_puts_per_day = max(1, _as_int(_get_param(algo, "max_new_puts_per_day", 3), 3))
    algo._last_selected_stock, algo._selection_count, algo._min_hold_cycles, algo._last_stock_scores = None, 0, 3, {}
    algo.ml_enabled = _as_bool(_get_param(algo, "ml_enabled", True), True)
    algo.ml_exploration_rate = _as_float(_get_param(algo, "ml_exploration_rate", 0.1), 0.1)
    algo.ml_learning_rate = _as_float(_get_param(algo, "ml_learning_rate", 0.01), 0.01)
    algo.ml_adoption_rate = _as_float(_get_param(algo, "ml_adoption_rate", 0.5), 0.5)
    algo.ml_min_confidence = _as_float(_get_param(algo, "ml_min_confidence", 0.4), 0.4)
    algo.stock_pool = str(_get_param(algo, "stock_pool", ",".join(MAG7_STOCKS))).split(",")
    algo.weights = {"iv_rank": 0.35, "technical": 0.25, "momentum": 0.20, "pe_score": 0.20}
    # Target margin utilization for position sizing (60% of capital)
    algo.target_margin_utilization = _as_float(_get_param(algo, "target_margin_utilization", 0.45), 0.45)


def init_ml(algo):
    from ml_integration import BinbinGodMLIntegration, AdaptiveDeltaStrategy
    algo.ml_integration = BinbinGodMLIntegration(MLOptimizationConfig(
        ml_delta_enabled=algo.ml_enabled, ml_dte_enabled=algo.ml_enabled,
        ml_roll_enabled=algo.ml_enabled, ml_position_enabled=algo.ml_enabled,
        dte_min=30, dte_max=45, exploration_rate=algo.ml_exploration_rate,
        learning_rate=algo.ml_learning_rate))
    algo.adaptive_strategy = AdaptiveDeltaStrategy(algo.ml_integration, algo.ml_adoption_rate, algo.ml_min_confidence)
    algo._ml_pretrained = False


def init_securities(algo):
    algo.equities, algo.options, algo.price_history = {}, {}, {}
    for symbol in algo.stock_pool:
        e = algo.AddEquity(symbol, Resolution.Daily)
        e.SetDataNormalizationMode(DataNormalizationMode.Raw)
        algo.equities[symbol], algo.price_history[symbol] = e, []
        o = algo.AddOption(symbol, Resolution.Daily)
        o.SetFilter(-30, 30, timedelta(days=20), timedelta(days=60))
        algo.options[symbol] = o
    algo.vix = algo.AddEquity("VIXY", Resolution.Daily)
    algo._current_vix, algo._vix_history = 20.0, []


def init_state(algo):
    # No phase concept - strategy is holdings-driven
    # Use QC Portfolio for position tracking (no manual dict needed)
    # position_metadata tracks entry Greeks (QC doesn't have this)
    init_position_tracking(algo)
    algo.pending_order_metadata = {}
    algo.symbol_cooldowns = {}
    algo.trade_history, algo.ml_signals_history = [], []
    algo.total_trades, algo.winning_trades, algo.total_pnl = 0, 0, 0.0
    algo.SetWarmUp(60)


def log_effective_parameters(algo):
    """Log effective runtime parameters for QC backtest verification."""
    algo.Log(
        "EFFECTIVE_PARAMS: "
        f"initial_capital={algo.initial_capital}, "
        f"max_positions_ceiling={algo.max_positions_ceiling}, "
        f"target_margin_utilization={algo.target_margin_utilization}, "
        f"margin_buffer_pct={algo.margin_buffer_pct}, "
        f"max_leverage={algo.max_leverage}, "
        f"max_assignment_risk_per_trade={algo.max_assignment_risk_per_trade}, "
        f"max_put_contracts_per_symbol={algo.max_put_contracts_per_symbol}, "
        f"max_put_contracts_total={algo.max_put_contracts_total}, "
        f"max_contracts_per_trade={algo.max_contracts_per_trade}, "
        f"position_aggressiveness={algo.position_aggressiveness}, "
        f"defensive_put_roll_enabled={algo.defensive_put_roll_enabled}, "
        f"defensive_put_roll_loss_pct={algo.defensive_put_roll_loss_pct}, "
        f"defensive_put_roll_itm_buffer_pct={algo.defensive_put_roll_itm_buffer_pct}, "
        f"defensive_put_roll_max_dte={algo.defensive_put_roll_max_dte}, "
        f"assignment_cooldown_days={algo.assignment_cooldown_days}, "
        f"large_loss_cooldown_days={algo.large_loss_cooldown_days}, "
        f"dynamic_symbol_risk_enabled={algo.dynamic_symbol_risk_enabled}, "
        f"symbol_state_cap_floor={algo.symbol_state_cap_floor}, "
        f"symbol_assignment_base_cap={algo.symbol_assignment_base_cap}, "
        f"stock_inventory_cap_enabled={algo.stock_inventory_cap_enabled}, "
        f"stock_inventory_base_cap={algo.stock_inventory_base_cap}, "
        f"stock_inventory_block_threshold={algo.stock_inventory_block_threshold}, "
        f"max_new_puts_per_day={algo.max_new_puts_per_day}, "
        f"repair_call_delta={algo.repair_call_delta}, "
        f"stop_loss_pct={algo.stop_loss_pct}, "
        f"ml_enabled={algo.ml_enabled}, "
        f"ml_min_confidence={algo.ml_min_confidence}, "
        f"stock_pool={algo.stock_pool}"
    )


def schedule_events(algo):
    algo.Schedule.On(algo.DateRules.EveryDay(), algo.TimeRules.AfterMarketOpen("SPY", 30), algo.Rebalance)
    algo.Schedule.On(algo.DateRules.EveryDay(), algo.TimeRules.AfterMarketClose("SPY", -10), algo.CheckExpiredOptions)
    algo.Schedule.On(algo.DateRules.MonthEnd(), algo.TimeRules.AfterMarketClose("SPY", -30), algo.UpdateMLModels)
