"""Strategy Functions for BinbinGod - All strategy methods as standalone functions."""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from AlgorithmImports import OptionRight, OrderStatus, Resolution, DataNormalizationMode
from helpers import StockScore, StockHolding
from scoring import score_single_stock, calculate_historical_vol
from ml_integration import MLOptimizationConfig, StrategySignal
from option_utils import filter_option_by_itm_protection, estimate_delta_from_moneyness, get_premium_from_security, should_roll_position, calculate_dte, build_option_result
from signals import select_best_signal_with_memory, get_cc_optimization_params, build_position_data, calculate_pnl_metrics
from option_pricing import BlackScholes

MAG7_STOCKS = ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]

# Risk-free rate for option pricing
RISK_FREE_RATE = 0.05

def bs_put_price(S, K, T, sigma):
    """Wrapper for BlackScholes put price."""
    return BlackScholes.put_price(S, K, T, RISK_FREE_RATE, sigma)

def bs_call_price(S, K, T, sigma):
    """Wrapper for BlackScholes call price."""
    return BlackScholes.call_price(S, K, T, RISK_FREE_RATE, sigma)


def make_signal(symbol, action, delta=0, dte_min=30, dte_max=45, num_contracts=1, confidence=0.5, reasoning=""):
    return StrategySignal(symbol=symbol, action=action, delta=delta, dte_min=dte_min, dte_max=dte_max,
        num_contracts=num_contracts, confidence=confidence, reasoning=reasoning,
        expected_premium=0.0, expected_return=0.0, expected_risk=0.0, assignment_probability=0.0)


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
    algo.allow_sp_in_cc_phase = bool(algo.GetParameter("allow_sp_in_cc_phase", True))
    algo.sp_in_cc_margin_threshold = float(algo.GetParameter("sp_in_cc_margin_threshold", 0.5))
    algo.sp_in_cc_max_positions = int(algo.GetParameter("sp_in_cc_max_positions", 3))
    algo.cc_optimization_enabled = bool(algo.GetParameter("cc_optimization_enabled", True))
    algo.cc_min_delta_cost = float(algo.GetParameter("cc_min_delta_cost", 0.15))
    algo.cc_cost_basis_threshold = float(algo.GetParameter("cc_cost_basis_threshold", 0.05))
    algo.cc_min_strike_premium = float(algo.GetParameter("cc_min_strike_premium", 0.02))
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
        # Wide filter to ensure all potential contracts are subscribed
        # -30 to +30 strikes, 20 to 60 days to expiry
        o.SetFilter(-30, 30, timedelta(days=20), timedelta(days=60))
        algo.options[symbol] = o
    algo.vix = algo.AddEquity("VIXY", Resolution.Daily)
    algo._current_vix, algo._vix_history = 20.0, []


def init_state(algo):
    algo.phase, algo.stock_holding = "SP", StockHolding()
    algo.open_option_positions, algo.trade_history, algo.ml_signals_history = {}, [], []
    algo.total_trades, algo.winning_trades, algo.total_pnl = 0, 0, 0.0
    algo.SetWarmUp(60)


def schedule_events(algo):
    algo.Schedule.On(algo.DateRules.EveryDay(), algo.TimeRules.AfterMarketOpen("SPY", 30), algo.Rebalance)
    algo.Schedule.On(algo.DateRules.EveryDay(), algo.TimeRules.AfterMarketClose("SPY", -10), algo.CheckExpiredOptions)
    algo.Schedule.On(algo.DateRules.MonthEnd(), algo.TimeRules.AfterMarketClose("SPY", -30), algo.UpdateMLModels)


def rebalance(algo):
    if algo.IsWarmingUp:
        return
    check_position_management(algo)
    open_count = len(algo.open_option_positions)
    algo.Log(f"Rebalance: open_positions={open_count}, max_positions={algo.max_positions}")
    if open_count >= algo.max_positions:
        algo.Log("Rebalance: max positions reached, skipping")
        return
    signals = generate_ml_signals(algo)
    algo.Log(f"Rebalance: generated {len(signals)} signals")
    if not signals:
        algo.Log("Rebalance: no signals generated")
        return
    best_signal, algo._last_selected_stock, algo._selection_count, algo._last_stock_scores = \
        select_best_signal_with_memory(signals, algo._last_selected_stock, algo._selection_count, algo._min_hold_cycles, algo._last_stock_scores)
    if best_signal:
        algo.Log(f"Rebalance: best_signal={best_signal.symbol} confidence={best_signal.confidence:.2f}")
        if best_signal.confidence >= algo.ml_min_confidence:
            execute_signal(algo, best_signal)
        else:
            algo.Log(f"Rebalance: confidence too low (min={algo.ml_min_confidence})")
    else:
        algo.Log("Rebalance: no best_signal selected")


def check_position_management(algo):
    for position_id, pos_info in list(algo.open_option_positions.items()):
        if position_id not in algo.open_option_positions:
            continue
        option_symbol = pos_info['option_symbol']
        security = algo.Securities.get(option_symbol)
        if not security:
            continue
        current_price, entry_price = security.Price, pos_info['entry_price']
        if current_price <= 0 or entry_price <= 0:
            continue
        pnl, pnl_pct = calculate_pnl_metrics(entry_price, current_price, pos_info['quantity'])
        premium_captured_pct = ((entry_price - current_price) / entry_price * 100) if entry_price > 0 else 0
        dte = calculate_dte(pos_info['expiry'], algo.Time)
        algo.Log(f"Position check: {pos_info['symbol']} {pos_info['right']} captured={premium_captured_pct:.1f}% DTE={dte} PnL={pnl_pct:.1f}%")
        if algo.ml_enabled and hasattr(algo.ml_integration, 'roll_optimizer'):
            position_data = build_position_data(pos_info, current_price, pnl_pct, dte)
            equity = algo.equities.get(pos_info['symbol'])
            underlying_price = algo.Securities[equity.Symbol].Price if equity else 0
            market_data = {
                'price': underlying_price,
                'iv': 0.25,
                'vix': algo._current_vix,
                'option_price': current_price
            }
            try:
                should_roll, roll_rec = algo.ml_integration.roll_optimizer.should_roll(
                    position=position_data, market_data=market_data,
                    current_date=algo.Time.strftime('%Y-%m-%d'), min_confidence=algo.ml_min_confidence)
                algo.Log(f"ML Roll: action={roll_rec.action} confidence={roll_rec.confidence:.2f} improvement={roll_rec.expected_pnl_improvement:.2f}")
                if should_roll and roll_rec.action in ["ROLL_FORWARD", "ROLL_OUT", "CLOSE_EARLY"]:
                    algo.Log(f"ML triggered action: {roll_rec.action} - {roll_rec.reasoning}")
                    handle_roll_action(algo, roll_rec, pos_info, position_id)
                    continue
            except Exception as e:
                algo.Log(f"ML Roll error: {e}")
        action, reasoning = should_roll_position(premium_captured_pct, dte, pnl_pct,
            algo.profit_target_pct, algo._profit_target_disabled, algo.stop_loss_pct, algo._stop_loss_disabled)
        algo.Log(f"Rule-based: action={action} reasoning={reasoning}")
        if action == "ROLL":
            signal = make_signal(pos_info['symbol'], "ROLL", delta=abs(pos_info['delta_at_entry']),
                dte_min=30, dte_max=45, num_contracts=abs(pos_info['quantity']), confidence=0.8, reasoning=reasoning)
            execute_roll(algo, signal)
        elif action.startswith("CLOSE"):
            algo.Log(f"Closing position: {reasoning}")
            signal = make_signal(pos_info['symbol'], "CLOSE", dte_min=0, dte_max=0, num_contracts=0, confidence=0.9, reasoning=reasoning)
            execute_close(algo, signal)


def handle_roll_action(algo, roll_rec, pos_info, position_id):
    if roll_rec.action == "ROLL_FORWARD":
        signal = make_signal(pos_info['symbol'], "ROLL",
            delta=roll_rec.optimal_delta or abs(pos_info['delta_at_entry']),
            dte_min=roll_rec.optimal_dte or 30, dte_max=roll_rec.optimal_dte or 45,
            num_contracts=abs(pos_info['quantity']), confidence=roll_rec.confidence, reasoning=roll_rec.reasoning)
        execute_roll(algo, signal)
    elif roll_rec.action == "ROLL_OUT":
        signal = make_signal(pos_info['symbol'], "ROLL",
            delta=roll_rec.optimal_delta or abs(pos_info['delta_at_entry']),
            dte_min=roll_rec.optimal_dte or 30, dte_max=roll_rec.optimal_dte or 45,
            num_contracts=abs(pos_info['quantity']), confidence=roll_rec.confidence, reasoning=roll_rec.reasoning)
        execute_roll(algo, signal)
    elif roll_rec.action == "CLOSE_EARLY":
        signal = make_signal(pos_info['symbol'], "CLOSE", dte_min=0, dte_max=0, num_contracts=0, confidence=roll_rec.confidence, reasoning=roll_rec.reasoning)
        execute_close(algo, signal)


def generate_ml_signals(algo) -> List[StrategySignal]:
    signals, portfolio_state = [], get_portfolio_state(algo)
    algo.Log(f"generate_ml_signals: phase={algo.phase}, stock_pool={algo.stock_pool}")
    if algo.phase == "SP":
        symbols_with_put = {p.get('symbol') for p in algo.open_option_positions.values() if p.get('right') == 'P'}
        algo.Log(f"generate_ml_signals: symbols_with_put={symbols_with_put}")
        for symbol in algo.stock_pool:
            if symbol not in symbols_with_put:
                sig = generate_signal_for_symbol(algo, symbol, "SP", portfolio_state)
                algo.Log(f"generate_ml_signals: symbol={symbol}, sig={'None' if sig is None else 'generated'}")
                if sig: signals.append(sig)
    elif algo.phase == "CC":
        for symbol in algo.stock_holding.get_symbols():
            sig = generate_signal_for_symbol(algo, symbol, "CC", portfolio_state)
            if sig: signals.append(sig)
        if algo.allow_sp_in_cc_phase:
            signals.extend(generate_cc_sp_signals(algo, portfolio_state))
    return signals


def generate_signal_for_symbol(algo, symbol: str, strategy_phase: str, portfolio_state: Dict) -> Optional[StrategySignal]:
    equity = algo.equities.get(symbol)
    if not equity:
        algo.Log(f"generate_signal_for_symbol: {symbol} - no equity")
        return None
    underlying_price = algo.Securities[equity.Symbol].Price
    if underlying_price <= 0:
        algo.Log(f"generate_signal_for_symbol: {symbol} - price <= 0")
        return None
    bars = algo.price_history.get(symbol, [])
    if len(bars) < 20:
        algo.Log(f"generate_signal_for_symbol: {symbol} - bars={len(bars)} < 20")
        return None
    cost_basis = algo.stock_holding.holdings.get(symbol, {}).get("cost_basis", 0)
    current_position = get_current_position(algo, symbol)
    traditional_delta, cc_min_strike = 0.30, None
    if strategy_phase == "CC" and algo.cc_optimization_enabled and cost_basis > 0:
        adj_delta, cc_min_strike, log_msg = get_cc_optimization_params(cost_basis, underlying_price,
            algo.cc_optimization_enabled, algo.cc_min_delta_cost, algo.cc_cost_basis_threshold, algo.cc_min_strike_premium)
        if log_msg: algo.Log(log_msg); traditional_delta = adj_delta
    signal = algo.ml_integration.generate_signal(symbol=symbol, current_price=underlying_price, cost_basis=cost_basis,
        bars=bars, strategy_phase=strategy_phase, portfolio_state=portfolio_state, current_position=current_position)
    if signal and algo.ml_enabled:
        right = "P" if strategy_phase in ("SP", "CC+SP") else "C"
        if right == "P":
            adaptive_delta, _ = algo.adaptive_strategy.select_put_delta(traditional_delta, signal.delta, signal.delta_confidence, signal.reasoning)
        else:
            adaptive_delta, _ = algo.adaptive_strategy.select_call_delta(traditional_delta, signal.delta, signal.delta_confidence, signal.reasoning)
        signal.delta = adaptive_delta
    if signal:
        score = score_single_stock(symbol, bars, underlying_price, algo.weights)
        signal.ml_score_adjustment = (score.total_score - 50) / 100
        if cc_min_strike is not None: signal.min_strike = cc_min_strike
    return signal


def generate_cc_sp_signals(algo, portfolio_state: Dict) -> List[StrategySignal]:
    signals = []
    margin_util = algo.Portfolio.TotalMarginUsed / algo.Portfolio.TotalPortfolioValue
    if margin_util > algo.sp_in_cc_margin_threshold: return signals
    sp_positions = sum(1 for p in algo.open_option_positions.values() if p.get('right') == 'P')
    if sp_positions >= algo.sp_in_cc_max_positions: return signals
    if len(algo.open_option_positions) >= algo.max_positions: return signals
    symbols_with_put = {p.get('symbol') for p in algo.open_option_positions.values() if p.get('right') == 'P'}
    held = algo.stock_holding.get_symbols()
    available = [s for s in algo.stock_pool if s not in held] or algo.stock_pool
    available = [s for s in available if s not in symbols_with_put]
    best_signal = None
    for symbol in available:
        signal = generate_signal_for_symbol(algo, symbol, "CC+SP", portfolio_state)
        if signal and signal.confidence > 0.6:
            if best_signal is None or signal.confidence > best_signal.confidence: best_signal = signal
    if best_signal: signals.append(best_signal)
    return signals


def get_portfolio_state(algo) -> Dict:
    positions = [{'symbol': str(h.Symbol), 'quantity': h.Quantity, 'market_value': h.HoldingsValue}
                 for h in algo.Portfolio.Values if h.Invested]
    return {'total_capital': algo.Portfolio.TotalPortfolioValue, 'available_margin': algo.Portfolio.Cash,
            'margin_used': algo.Portfolio.TotalMarginUsed, 'drawdown': calculate_drawdown(algo),
            'positions': positions, 'cost_basis': algo.stock_holding.cost_basis}


def calculate_drawdown(algo) -> float:
    current = algo.Portfolio.TotalPortfolioValue
    if current < algo.initial_capital: return (algo.initial_capital - current) / algo.initial_capital * 100
    return 0.0


def get_current_position(algo, symbol: str) -> Optional[Dict]:
    for pos_info in algo.open_option_positions.values():
        if pos_info.get('symbol') == symbol: return pos_info
    return None


def execute_signal(algo, signal: StrategySignal):
    if not signal or signal.action == "HOLD": return
    if signal.action == "ROLL": execute_roll(algo, signal); return
    if signal.action == "CLOSE": execute_close(algo, signal); return
    equity = algo.equities.get(signal.symbol)
    if not equity: return
    underlying_price = algo.Securities[equity.Symbol].Price
    target_right = OptionRight.Put if signal.action == "SELL_PUT" else OptionRight.Call
    target_delta = -signal.delta if target_right == OptionRight.Put else signal.delta
    min_strike = getattr(signal, 'min_strike', 0.0)
    selected = find_option_by_greeks(algo, symbol=signal.symbol, equity_symbol=equity.Symbol,
        target_right=target_right, target_delta=target_delta, dte_min=signal.dte_min, dte_max=signal.dte_max,
        delta_tolerance=0.05, min_strike=min_strike if min_strike > 0 else None)
    if not selected:
        algo.Log(f"No suitable options for {signal.symbol} delta ~{target_delta:.2f}")
        return
    current_positions = len(algo.open_option_positions)
    if target_right == OptionRight.Put:
        # Calculate quantity based on actual margin available
        # QC margin requirement for short put is roughly strike * 100 * margin_rate (typically 15-20%)
        estimated_margin_per_contract = selected['strike'] * 100 * 0.20
        available_margin = algo.Portfolio.MarginRemaining
        max_by_margin = max(1, int(available_margin / estimated_margin_per_contract)) if estimated_margin_per_contract > 0 else 1
        max_by_limit = algo.max_positions - current_positions
        quantity = min(max_by_margin, max_by_limit)
        algo.Log(f"Position sizing: available_margin=${available_margin:.0f}, margin_per_contract=${estimated_margin_per_contract:.0f}, max_by_margin={max_by_margin}, quantity={quantity}")
    else:
        shares_held = algo.stock_holding.get_shares(signal.symbol)
        existing_call_contracts = sum(abs(p.get('quantity', 0)) for p in algo.open_option_positions.values()
            if p.get('symbol') == signal.symbol and p.get('right') == 'C')
        shares_covered = existing_call_contracts * 100
        shares_available = shares_held - shares_covered
        quantity = min(max(0, shares_available // 100), algo.max_positions)
        if quantity <= 0:
            algo.Log(f"No available shares for {signal.symbol} call: held={shares_held}, covered={shares_covered}")
            return
    if quantity <= 0: return
    quantity = -quantity
    algo.Log(f"Selling {abs(quantity)} {signal.symbol} {target_right} @ ${selected['premium']:.2f}")
    
    # Subscribe to the option contract if not already subscribed
    option_symbol = selected['option_symbol']
    if not algo.Securities.ContainsKey(option_symbol):
        algo.Log(f"Subscribing to option contract: {option_symbol}")
        algo.AddOptionContract(option_symbol, Resolution.Daily)
    
    ticket = algo.MarketOrder(option_symbol, quantity)
    if ticket.Status == OrderStatus.Filled:
        fill_price = ticket.AverageFillPrice or selected['premium']
        right_str = 'P' if target_right == OptionRight.Put else 'C'
        position_id = f"{signal.symbol}_{algo.Time.strftime('%Y%m%d')}_{selected['strike']:.0f}_{right_str}"
        algo.open_option_positions[position_id] = {
            'symbol': signal.symbol, 'option_symbol': selected['option_symbol'], 'right': right_str,
            'strike': selected['strike'], 'expiry': selected['expiry'], 'entry_date': algo.Time.strftime('%Y-%m-%d'),
            'entry_price': fill_price, 'quantity': quantity, 'delta_at_entry': selected['delta'],
            'iv_at_entry': selected['iv'], 'strategy_phase': algo.phase, 'ml_signal': signal}
        algo.Log(f"Executed: {signal.action} {signal.num_contracts} {signal.symbol} @ ${fill_price:.2f}")


def execute_roll(algo, signal: StrategySignal):
    existing = None
    for pos_id, pos_info in algo.open_option_positions.items():
        if pos_info.get('symbol') == signal.symbol: existing = (pos_id, pos_info); break
    if not existing: return
    pos_id, pos_info = existing
    close_ticket = algo.MarketOrder(pos_info['option_symbol'], -pos_info['quantity'])
    if close_ticket.Status != OrderStatus.Filled: return
    pnl, _ = calculate_pnl_metrics(pos_info['entry_price'], close_ticket.AverageFillPrice, pos_info['quantity'])
    record_trade(algo, signal.symbol, pos_info['right'], pnl, "ROLL")
    del algo.open_option_positions[pos_id]
    equity = algo.equities.get(signal.symbol)
    if not equity: return
    target_right = OptionRight.Put if pos_info['right'] == 'P' else OptionRight.Call
    target_delta = -signal.delta if target_right == OptionRight.Put else signal.delta
    new_selected = find_option_by_greeks(algo, symbol=signal.symbol, equity_symbol=equity.Symbol,
        target_right=target_right, target_delta=target_delta, dte_min=signal.dte_min, dte_max=signal.dte_max)
    if new_selected:
        new_qty = pos_info['quantity']
        # Subscribe to the new option contract if not already subscribed
        new_option_symbol = new_selected['option_symbol']
        if not algo.Securities.ContainsKey(new_option_symbol):
            algo.Log(f"Subscribing to option contract: {new_option_symbol}")
            algo.AddOptionContract(new_option_symbol, Resolution.Daily)
        new_ticket = algo.MarketOrder(new_option_symbol, new_qty)
        if new_ticket.Status == OrderStatus.Filled:
            right_str = 'P' if target_right == OptionRight.Put else 'C'
            algo.open_option_positions[f"{signal.symbol}_{algo.Time.strftime('%Y%m%d')}_{new_selected['strike']:.0f}_{right_str}"] = {
                'symbol': signal.symbol, 'option_symbol': new_selected['option_symbol'], 'right': right_str,
                'strike': new_selected['strike'], 'expiry': new_selected['expiry'],
                'entry_date': algo.Time.strftime('%Y-%m-%d'), 'entry_price': new_ticket.AverageFillPrice,
                'quantity': new_qty, 'delta_at_entry': new_selected['delta'],
                'iv_at_entry': new_selected['iv'], 'strategy_phase': algo.phase, 'ml_signal': signal}


def execute_close(algo, signal: StrategySignal):
    for pos_id, pos_info in list(algo.open_option_positions.items()):
        if pos_info.get('symbol') != signal.symbol: continue
        close_ticket = algo.MarketOrder(pos_info['option_symbol'], -pos_info['quantity'])
        if close_ticket.Status == OrderStatus.Filled:
            pnl, _ = calculate_pnl_metrics(pos_info['entry_price'], close_ticket.AverageFillPrice, pos_info['quantity'])
            record_trade(algo, signal.symbol, pos_info['right'], pnl, signal.reasoning or "SIGNAL_CLOSE")
            del algo.open_option_positions[pos_id]


def record_trade(algo, symbol: str, right: str, pnl: float, reason: str):
    algo.total_trades += 1
    algo.total_pnl += pnl
    if pnl > 0: algo.winning_trades += 1
    algo.trade_history.append({"date": algo.Time.strftime("%Y-%m-%d"), "symbol": symbol, "type": right, "pnl": pnl, "exit_reason": reason})


def check_expired_options(algo):
    for pos_id, pos_info in list(algo.open_option_positions.items()):
        security = algo.Securities.get(pos_info['option_symbol'])
        if not security or not (security.IsDelisted or security.Price == 0): continue
        symbol = pos_info['symbol']
        pnl = security.Holdings.UnrealizedProfit if hasattr(security, 'Holdings') else 0
        algo.ml_integration.update_performance({
            'symbol': symbol, 'delta': abs(pos_info['delta_at_entry']),
            'dte': calculate_dte(pos_info['expiry'], datetime.strptime(pos_info['entry_date'], '%Y-%m-%d')),
            'num_contracts': abs(pos_info['quantity']), 'pnl': pnl / 100, 'assigned': False,
            'bars': algo.price_history.get(symbol, []),
            'cost_basis': algo.stock_holding.holdings.get(symbol, {}).get('cost_basis', 0),
            'strategy_phase': pos_info['strategy_phase']})
        record_trade(algo, symbol, pos_info['right'], pnl, "EXPIRY")
        del algo.open_option_positions[pos_id]
        equity = algo.equities.get(symbol)
        if equity:
            last_price = algo.Securities[equity.Symbol].Price
            strike = pos_info['strike']
            if pos_info['right'] == "P" and last_price < strike:
                shares = abs(pos_info['quantity']) * 100
                algo.stock_holding.add_shares(symbol, shares, strike)
                algo.stock_holding.add_premium(symbol, pos_info['entry_price'] * 100)
                algo.phase = "CC"
                algo.Log(f"Put assigned: {shares} {symbol} @ ${strike:.2f}")
            elif pos_info['right'] == "C" and last_price > strike:
                shares = abs(pos_info['quantity']) * 100
                cost = algo.stock_holding.holdings.get(symbol, {}).get('cost_basis', strike)
                algo.total_pnl += (strike - cost) * shares
                algo.stock_holding.remove_shares(symbol, shares)
                algo.Log(f"Call assigned: {shares} {symbol} @ ${strike:.2f}")
                if algo.stock_holding.shares == 0: algo.phase = "SP"


def update_ml_models(algo):
    if algo.ml_integration.should_retrain(): algo.Log("Retraining ML models...")


def find_option_by_greeks(algo, symbol: str, equity_symbol, target_right, target_delta: float,
                            dte_min: int, dte_max: int, delta_tolerance: float = 0.10, min_strike: float = None) -> Optional[Dict]:
    underlying_price = algo.Securities[equity_symbol].Price
    
    # Prefer cached option chain from OnData (has real price data)
    option_contracts = None
    if hasattr(algo, '_cached_option_chains') and symbol in algo._cached_option_chains:
        cached_chain = algo._cached_option_chains[symbol]
        option_contracts = list(cached_chain.Values()) if cached_chain else None
        if option_contracts:
            algo.Log(f"find_option: using cached chain for {symbol}, size={len(option_contracts)}")
    
    # Fallback to OptionChainProvider
    if not option_contracts:
        option_symbols = algo.OptionChainProvider.GetOptionContractList(equity_symbol, algo.Time)
        if not option_symbols:
            algo.Log(f"find_option: no option_chain for {symbol}")
            return None
        algo.Log(f"find_option: using provider for {symbol}, size={len(option_symbols)}")
        # Convert symbols to a common format for processing
        option_contracts = [type('obj', (object,), {
            'Symbol': s, 'Right': s.ID.OptionRight, 'Strike': s.ID.StrikePrice,
            'Expiry': s.ID.Date, 'BidPrice': 0, 'AskPrice': 0, 'LastPrice': 0
        })() for s in option_symbols]
    
    algo.Log(f"find_option: {symbol} chain_size={len(option_contracts)} target_delta={target_delta:.2f} dte=[{dte_min},{dte_max}] underlying={underlying_price:.2f}")
    suitable = []
    stats = {'right': 0, 'dte': 0, 'min_strike': 0, 'itm': 0, 'tolerance': 0}
    for contract in option_contracts:
        # Handle both cached OptionChain contracts and provider symbols
        if hasattr(contract, 'Symbol'):
            option_symbol = contract.Symbol
            right = contract.Right
            strike = contract.Strike
            expiry = contract.Expiry
        else:
            option_symbol = contract
            right = contract.ID.OptionRight
            strike = contract.ID.StrikePrice
            expiry = contract.ID.Date
        
        if right != target_right: 
            continue
        stats['right'] += 1
        dte = (expiry - algo.Time).days
        if not (dte_min <= dte <= dte_max): 
            continue
        stats['dte'] += 1
        if min_strike and strike < min_strike: 
            stats['min_strike'] += 1
            continue
        if not filter_option_by_itm_protection(strike, underlying_price, target_right): 
            stats['itm'] += 1
            continue
        # Estimate delta using moneyness
        delta = estimate_delta_from_moneyness(strike, underlying_price, target_right)
        iv = calculate_historical_vol(algo.price_history.get(symbol, []))
        if delta is None: 
            continue
        if abs(delta - target_delta) > delta_tolerance: 
            stats['tolerance'] += 1
            continue
        # Estimate premium using theoretical pricing
        T = dte / 365.0
        if target_right == OptionRight.Put:
            premium = bs_put_price(underlying_price, strike, T, iv)
        else:
            premium = bs_call_price(underlying_price, strike, T, iv)
        if premium <= 0.10:
            continue
        suitable.append(build_option_result(option_symbol, strike, expiry, dte,
            delta, iv, premium, abs(delta - target_delta), premium * 0.99, premium * 1.01))
    algo.Log(f"find_option stats: right={stats['right']} dte={stats['dte']} itm={stats['itm']} tolerance={stats['tolerance']} suitable={len(suitable)}")
    if not suitable:
        algo.Log(f"find_option: no suitable options found for {symbol}")
        return None
    algo.Log(f"find_option: found {len(suitable)} suitable options for {symbol}")
    suitable.sort(key=lambda x: x['delta_diff'])
    return suitable[0]


def on_end_of_algorithm(algo):
    wr = (algo.winning_trades / algo.total_trades * 100) if algo.total_trades > 0 else 0
    algo.Log(f"Trades: {algo.total_trades}, WinRate: {wr:.1f}%, PnL: ${algo.total_pnl:.2f}")
    algo.Log(f"Portfolio: ${algo.Portfolio.TotalPortfolioValue:.2f}, Phase: {algo.phase}")
    algo.Log(algo.ml_integration.get_status_report())