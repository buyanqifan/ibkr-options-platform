"""Initialization functions for BinbinGod Strategy."""
from datetime import datetime, timedelta
from AlgorithmImports import Resolution, DataNormalizationMode
from helpers import StockScore
from qc_portfolio import init_position_tracking
from ml_integration import MLOptimizationConfig

MAG7_STOCKS = ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]


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
    algo.initial_capital = float(algo.GetParameter("initial_capital", 100000))
    algo.SetCash(algo.initial_capital)
    algo.max_positions = int(algo.GetParameter("max_positions", 10))
    algo.profit_target_pct = float(algo.GetParameter("profit_target_pct", 50))
    algo.stop_loss_pct = float(algo.GetParameter("stop_loss_pct", 999999))
    algo.max_risk_per_trade = float(algo.GetParameter("max_risk_per_trade", 0.02))
    algo.max_leverage = float(algo.GetParameter("max_leverage", 1.0))
    algo._profit_target_disabled = algo.profit_target_pct >= 999999
    algo._stop_loss_disabled = algo.stop_loss_pct >= 999999
    algo.margin_buffer_pct = float(algo.GetParameter("margin_buffer_pct", 0.50))
    algo.margin_rate_per_contract = float(algo.GetParameter("margin_rate_per_contract", 0.25))
    algo.allow_sp_in_cc_phase = bool(algo.GetParameter("allow_sp_in_cc_phase", True))
    algo.sp_in_cc_margin_threshold = float(algo.GetParameter("sp_in_cc_margin_threshold", 0.5))
    algo.sp_in_cc_max_positions = int(algo.GetParameter("sp_in_cc_max_positions", 3))
    algo.cc_optimization_enabled = bool(algo.GetParameter("cc_optimization_enabled", True))
    algo.cc_min_delta_cost = float(algo.GetParameter("cc_min_delta_cost", 0.15))
    algo.cc_cost_basis_threshold = float(algo.GetParameter("cc_cost_basis_threshold", 0.05))
    algo.cc_min_strike_premium = float(algo.GetParameter("cc_min_strike_premium", 0.02))
    # CC profit protection: when stock price rises significantly above cost
    algo.cc_profit_protection_enabled = bool(algo.GetParameter("cc_profit_protection_enabled", True))
    algo.cc_profit_threshold = float(algo.GetParameter("cc_profit_threshold", 0.20))  # 20% above cost
    algo.cc_profit_delta = float(algo.GetParameter("cc_profit_delta", 0.20))  # Lower delta when in profit
    algo._last_selected_stock, algo._selection_count, algo._min_hold_cycles, algo._last_stock_scores = None, 0, 3, {}
    algo.ml_enabled = bool(algo.GetParameter("ml_enabled", True))
    algo.ml_exploration_rate = float(algo.GetParameter("ml_exploration_rate", 0.1))
    algo.ml_learning_rate = float(algo.GetParameter("ml_learning_rate", 0.01))
    algo.ml_adoption_rate = float(algo.GetParameter("ml_adoption_rate", 0.5))
    algo.ml_min_confidence = float(algo.GetParameter("ml_min_confidence", 0.4))
    algo.stock_pool = algo.GetParameter("stock_pool", ",".join(MAG7_STOCKS)).split(",")
    algo.weights = {"iv_rank": 0.35, "technical": 0.25, "momentum": 0.20, "pe_score": 0.20}


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
    algo.trade_history, algo.ml_signals_history = [], []
    algo.total_trades, algo.winning_trades, algo.total_pnl = 0, 0, 0.0
    algo.SetWarmUp(60)


def schedule_events(algo):
    algo.Schedule.On(algo.DateRules.EveryDay(), algo.TimeRules.AfterMarketOpen("SPY", 30), algo.Rebalance)
    algo.Schedule.On(algo.DateRules.EveryDay(), algo.TimeRules.AfterMarketClose("SPY", -10), algo.CheckExpiredOptions)
    algo.Schedule.On(algo.DateRules.MonthEnd(), algo.TimeRules.AfterMarketClose("SPY", -30), algo.UpdateMLModels)