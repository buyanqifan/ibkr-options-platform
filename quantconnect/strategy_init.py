"""Initialization functions for BinbinGod Strategy."""

from datetime import datetime, timedelta

from AlgorithmImports import Resolution, DataNormalizationMode

from qc_portfolio import init_position_tracking
from ml_integration import MLOptimizationConfig

MAG7_STOCKS = ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]


def _get_param(algo, name, default=None):
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

    max_pos_raw = _get_param(algo, "max_positions", None)
    if max_pos_raw is None:
        max_pos_raw = _get_param(algo, "max_positions_ceiling", 20)
    algo.max_positions_ceiling = _as_int(max_pos_raw, 20)
    algo.max_positions = algo.max_positions_ceiling

    algo.target_margin_utilization = _as_float(_get_param(algo, "target_margin_utilization", 0.65), 0.65)
    algo.symbol_assignment_base_cap = _clamp(
        _as_float(_get_param(algo, "symbol_assignment_base_cap", 0.45), 0.45),
        0.05,
        1.5,
    )
    algo.max_assignment_risk_per_trade = _as_float(_get_param(algo, "max_assignment_risk_per_trade", 0.20), 0.20)

    algo.roll_threshold_pct = _as_float(_get_param(algo, "roll_threshold_pct", 78), 78.0)
    algo.min_dte_for_roll = _as_int(_get_param(algo, "min_dte_for_roll", 7), 7)
    algo.roll_target_dte_min = _as_int(_get_param(algo, "roll_target_dte_min", 21), 21)
    algo.roll_target_dte_max = _as_int(_get_param(algo, "roll_target_dte_max", 45), 45)
    algo.sp_close_early_dte = max(0, _as_int(_get_param(algo, "sp_close_early_dte", 3), 3))
    algo.sp_close_early_loss_limit_pct = _as_float(
        _get_param(algo, "sp_close_early_loss_limit_pct", 999999),
        999999.0,
    )
    algo.sp_primary_delta_tolerance = _clamp(
        _as_float(_get_param(algo, "sp_primary_delta_tolerance", 0.12), 0.12),
        0.04,
        0.40,
    )
    algo.sp_relaxed_delta_tolerance = _clamp(
        _as_float(_get_param(algo, "sp_relaxed_delta_tolerance", 0.22), 0.22),
        algo.sp_primary_delta_tolerance,
        0.45,
    )
    algo.sp_min_option_premium = _clamp(
        _as_float(_get_param(algo, "sp_min_option_premium", 0.05), 0.05),
        0.01,
        1.0,
    )
    algo.sp_assignment_cooldown_days = max(0, _as_int(_get_param(algo, "sp_assignment_cooldown_days", 7), 7))
    algo.sp_weak_filter_enabled = _as_bool(_get_param(algo, "sp_weak_filter_enabled", True), True)
    algo.sp_weak_filter_ma20_break_pct = _clamp(
        _as_float(_get_param(algo, "sp_weak_filter_ma20_break_pct", 0.03), 0.03),
        0.0,
        0.20,
    )
    algo.sp_weak_filter_vol_spike_ratio = _clamp(
        _as_float(_get_param(algo, "sp_weak_filter_vol_spike_ratio", 1.25), 1.25),
        1.0,
        3.0,
    )

    algo.cc_below_cost_enabled = _as_bool(_get_param(algo, "cc_below_cost_enabled", True), True)
    algo.cc_target_delta = _clamp(_as_float(_get_param(algo, "cc_target_delta", 0.25), 0.25), 0.10, 0.60)
    algo.cc_target_dte_min = _as_int(_get_param(algo, "cc_target_dte_min", 10), 10)
    algo.cc_target_dte_max = _as_int(_get_param(algo, "cc_target_dte_max", 28), 28)
    algo.cc_max_discount_to_cost = _clamp(
        _as_float(_get_param(algo, "cc_max_discount_to_cost", 0.03), 0.03),
        0.0,
        0.30,
    )
    algo.cc_fallback_delta_tolerance_1 = _clamp(
        _as_float(_get_param(algo, "cc_fallback_delta_tolerance_1", 0.12), 0.12),
        0.08,
        0.30,
    )
    algo.cc_fallback_delta_tolerance_2 = _clamp(
        _as_float(_get_param(algo, "cc_fallback_delta_tolerance_2", 0.15), 0.15),
        algo.cc_fallback_delta_tolerance_1,
        0.40,
    )
    algo.cc_fallback_dte_min = _as_int(_get_param(algo, "cc_fallback_dte_min", 14), 14)
    algo.cc_fallback_dte_max = _as_int(_get_param(algo, "cc_fallback_dte_max", 30), 30)
    algo.cc_fallback_min_cost_basis_ratio = _clamp(
        _as_float(_get_param(algo, "cc_fallback_min_cost_basis_ratio", 0.85), 0.85),
        0.50,
        1.0,
    )
    algo.assigned_stock_cc_repair_delta = _clamp(
        _as_float(_get_param(algo, "assigned_stock_cc_repair_delta", 0.35), 0.35),
        0.10,
        0.60,
    )
    algo.assigned_stock_cc_repair_dte_min = _as_int(
        _get_param(algo, "assigned_stock_cc_repair_dte_min", 7),
        7,
    )
    algo.assigned_stock_cc_repair_dte_max = _as_int(
        _get_param(algo, "assigned_stock_cc_repair_dte_max", 21),
        21,
    )
    algo.assigned_stock_cc_max_discount_to_cost = _clamp(
        _as_float(_get_param(algo, "assigned_stock_cc_max_discount_to_cost", 0.10), 0.10),
        0.0,
        0.30,
    )

    algo.assigned_stock_fail_safe_enabled = _as_bool(_get_param(algo, "assigned_stock_fail_safe_enabled", True), True)
    algo.assigned_stock_inventory_cap_pct = _clamp(
        _as_float(_get_param(algo, "assigned_stock_inventory_cap_pct", 0.30), 0.30),
        0.0,
        1.0,
    )
    algo.assigned_stock_max_repair_days = max(
        1,
        _as_int(_get_param(algo, "assigned_stock_max_repair_days", 7), 7),
    )
    algo.assigned_stock_cc_miss_limit = max(
        1,
        _as_int(_get_param(algo, "assigned_stock_cc_miss_limit", 3), 3),
    )
    algo.assigned_stock_exit_fraction = _clamp(
        _as_float(_get_param(algo, "assigned_stock_exit_fraction", 1.0), 1.0),
        0.0,
        1.0,
    )
    algo.assigned_stock_min_days_held = _as_int(_get_param(algo, "assigned_stock_min_days_held", 5), 5)
    algo.assigned_stock_drawdown_pct = _as_float(_get_param(algo, "assigned_stock_drawdown_pct", 0.12), 0.12)
    algo.assigned_stock_force_exit_pct = _clamp(
        _as_float(_get_param(algo, "assigned_stock_force_exit_pct", 1.0), 1.0),
        0.0,
        1.0,
    )

    algo.max_new_puts_per_day = max(1, _as_int(_get_param(algo, "max_new_puts_per_day", 3), 3))
    algo._last_selected_stock, algo._selection_count, algo._min_hold_cycles, algo._last_stock_scores = None, 0, 3, {}

    algo.ml_enabled = _as_bool(_get_param(algo, "ml_enabled", True), True)
    algo.ml_exploration_rate = _as_float(_get_param(algo, "ml_exploration_rate", 0.1), 0.1)
    algo.ml_learning_rate = _as_float(_get_param(algo, "ml_learning_rate", 0.01), 0.01)
    algo.ml_adoption_rate = _as_float(_get_param(algo, "ml_adoption_rate", 0.5), 0.5)
    algo.ml_min_confidence = _as_float(_get_param(algo, "ml_min_confidence", 0.45), 0.45)

    algo.stock_pool = str(_get_param(algo, "stock_pool", ",".join(MAG7_STOCKS))).split(",")
    algo.weights = {"iv_rank": 0.25, "technical": 0.30, "momentum": 0.25, "pe_score": 0.20}


def init_ml(algo):
    from ml_integration import BinbinGodMLIntegration, AdaptiveDeltaStrategy

    algo.ml_integration = BinbinGodMLIntegration(
        MLOptimizationConfig(
            ml_delta_enabled=algo.ml_enabled,
            ml_dte_enabled=algo.ml_enabled,
            ml_roll_enabled=False,
            ml_position_enabled=algo.ml_enabled,
            dte_min=14,
            dte_max=28,
            exploration_rate=algo.ml_exploration_rate,
            learning_rate=algo.ml_learning_rate,
        )
    )
    algo.adaptive_strategy = AdaptiveDeltaStrategy(algo.ml_integration, algo.ml_adoption_rate, algo.ml_min_confidence)
    algo._ml_pretrained = False


def init_securities(algo):
    algo.equities, algo.options, algo.price_history = {}, {}, {}
    for symbol in algo.stock_pool:
        equity = algo.AddEquity(symbol, Resolution.Daily)
        equity.SetDataNormalizationMode(DataNormalizationMode.Raw)
        algo.equities[symbol], algo.price_history[symbol] = equity, []
        option = algo.AddOption(symbol, Resolution.Daily)
        option.SetFilter(-30, 30, timedelta(days=20), timedelta(days=60))
        algo.options[symbol] = option
    algo.vix = algo.AddEquity("VIXY", Resolution.Daily)
    algo._current_vix, algo._vix_history = 20.0, []


def init_state(algo):
    from debug_counters import DEFAULT_DEBUG_COUNTERS

    init_position_tracking(algo)
    algo.debug_counters = dict(DEFAULT_DEBUG_COUNTERS)
    algo.pending_order_metadata = {}
    algo.pending_open_orders = {}
    algo.pending_close_orders = {}
    algo.pending_roll_orders = {}
    algo.processed_assignment_keys = set()
    algo.symbol_cooldowns = {}
    algo.assigned_stock_state = {}
    algo.trade_history, algo.ml_signals_history = [], []
    algo.total_trades, algo.winning_trades, algo.total_pnl = 0, 0, 0.0
    algo.SetWarmUp(60)


def log_effective_parameters(algo):
    algo.Log(
        "EFFECTIVE_PARAMS: "
        f"initial_capital={algo.initial_capital}, "
        f"max_positions_ceiling={algo.max_positions_ceiling}, "
        f"target_margin_utilization={algo.target_margin_utilization}, "
        f"symbol_assignment_base_cap={algo.symbol_assignment_base_cap}, "
        f"max_assignment_risk_per_trade={algo.max_assignment_risk_per_trade}, "
        f"roll_threshold_pct={algo.roll_threshold_pct}, "
        f"min_dte_for_roll={algo.min_dte_for_roll}, "
        f"roll_target_dte={algo.roll_target_dte_min}-{algo.roll_target_dte_max}, "
        f"sp_close_early_dte={algo.sp_close_early_dte}, "
        f"sp_close_early_loss_limit_pct={algo.sp_close_early_loss_limit_pct}, "
        f"sp_delta_tolerance={algo.sp_primary_delta_tolerance}/{algo.sp_relaxed_delta_tolerance}, "
        f"sp_min_option_premium={algo.sp_min_option_premium}, "
        f"sp_assignment_cooldown_days={algo.sp_assignment_cooldown_days}, "
        f"sp_weak_filter_enabled={algo.sp_weak_filter_enabled}, "
        f"sp_weak_filter_ma20_break_pct={algo.sp_weak_filter_ma20_break_pct}, "
        f"sp_weak_filter_vol_spike_ratio={algo.sp_weak_filter_vol_spike_ratio}, "
        f"cc_target_delta={algo.cc_target_delta}, "
        f"cc_target_dte={algo.cc_target_dte_min}-{algo.cc_target_dte_max}, "
        f"cc_max_discount_to_cost={algo.cc_max_discount_to_cost}, "
        f"cc_fallback_delta_tolerance={algo.cc_fallback_delta_tolerance_1}/{algo.cc_fallback_delta_tolerance_2}, "
        f"cc_fallback_dte={algo.cc_fallback_dte_min}-{algo.cc_fallback_dte_max}, "
        f"cc_fallback_min_cost_basis_ratio={algo.cc_fallback_min_cost_basis_ratio}, "
        f"assigned_stock_cc_repair_delta={algo.assigned_stock_cc_repair_delta}, "
        f"assigned_stock_cc_repair_dte={algo.assigned_stock_cc_repair_dte_min}-{algo.assigned_stock_cc_repair_dte_max}, "
        f"assigned_stock_cc_max_discount_to_cost={algo.assigned_stock_cc_max_discount_to_cost}, "
        f"assigned_stock_inventory_cap_pct={algo.assigned_stock_inventory_cap_pct}, "
        f"assigned_stock_max_repair_days={algo.assigned_stock_max_repair_days}, "
        f"assigned_stock_cc_miss_limit={algo.assigned_stock_cc_miss_limit}, "
        f"assigned_stock_exit_fraction={algo.assigned_stock_exit_fraction}, "
        f"assigned_stock_min_days_held={algo.assigned_stock_min_days_held}, "
        f"assigned_stock_drawdown_pct={algo.assigned_stock_drawdown_pct}, "
        f"assigned_stock_force_exit_pct={algo.assigned_stock_force_exit_pct}, "
        f"max_new_puts_per_day={algo.max_new_puts_per_day}, "
        f"ml_enabled={algo.ml_enabled}, "
        f"ml_min_confidence={algo.ml_min_confidence}, "
        f"stock_pool={algo.stock_pool}"
    )


def schedule_events(algo):
    algo.Schedule.On(algo.DateRules.EveryDay(), algo.TimeRules.AfterMarketOpen("SPY", 30), algo.Rebalance)
    algo.Schedule.On(algo.DateRules.EveryDay(), algo.TimeRules.AfterMarketClose("SPY", -10), algo.CheckExpiredOptions)
    algo.Schedule.On(algo.DateRules.MonthEnd(), algo.TimeRules.AfterMarketClose("SPY", -30), algo.UpdateMLModels)
