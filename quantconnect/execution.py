"""Trade execution functions for BinbinGod Strategy.
# Updated: 2026-03-25 - Fix import cache issue
"""
from typing import Dict, Optional, Tuple
from AlgorithmImports import OptionRight, OrderStatus, Resolution, SecurityType
from ml_integration import StrategySignal
from option_utils import calculate_dte, calculate_historical_vol
from signals import calculate_pnl_metrics, calculate_position_risk
from option_pricing import BlackScholes
from debug_counters import increment_debug_counter
from qc_portfolio import (
    get_option_position_count, get_shares_held, get_call_position_contracts,
    get_position_for_symbol, save_position_metadata, remove_position_metadata,
    get_total_stock_holdings_value, get_stock_holding_count
)

RISK_FREE_RATE = 0.05
MAX_DEFERRED_OPEN_ATTEMPTS = 3


def calculate_dynamic_max_positions(algo) -> int:
    """Dynamically calculate max positions based on capital and stock prices.
    
    Formula:
    - Available margin budget = initial_capital * target_margin_utilization
    - Average margin per position = avg_stock_price * 100 * margin_rate
    - Max positions = budget / margin_per_position
    - Capped by max_positions_ceiling from config
    
    Returns:
        Dynamically calculated max positions
    """
    # Get average stock price from the pool
    total_price = 0.0
    count = 0
    for symbol in algo.stock_pool:
        equity = algo.equities.get(symbol)
        if equity and algo.Securities.ContainsKey(equity.Symbol):
            price = algo.Securities[equity.Symbol].Price
            if price > 0:
                total_price += price
                count += 1
    
    if count == 0:
        return algo.max_positions_ceiling  # Fallback to config value
    
    avg_price = total_price / count
    
    # Calculate margin budget from current portfolio value so slot capacity compounds.
    portfolio_value = max(getattr(algo.Portfolio, "TotalPortfolioValue", getattr(algo, "initial_capital", 0.0)), 0.0)
    margin_budget = portfolio_value * algo.target_margin_utilization
    
    # Estimate margin per contract using standard formula
    # max(20% * price, 10% * price) * 100 ≈ 20% * price * 100
    margin_per_contract = avg_price * 100 * 0.20
    
    # Calculate max positions
    if margin_per_contract > 0:
        dynamic_max = int(margin_budget / margin_per_contract)
    else:
        dynamic_max = algo.max_positions_ceiling
    
    # Cap by ceiling from config and ensure at least 1
    result = max(1, min(dynamic_max, algo.max_positions_ceiling))
    
    return result


def bs_put_price(S, K, T, sigma):
    return BlackScholes.put_price(S, K, T, RISK_FREE_RATE, sigma)


def bs_call_price(S, K, T, sigma):
    return BlackScholes.call_price(S, K, T, RISK_FREE_RATE, sigma)


def calculate_volatility_weighted_symbol_cap(algo, symbol: str) -> int:
    """Adjust per-symbol put cap so high-vol names consume fewer slots.

    This cap is one-way: it can reduce exposure to volatile names, but it
    should never increase a symbol above the base configured cap.
    """
    base_cap = max(1, int(getattr(algo, "max_put_contracts_per_symbol", 1)))
    lookback = max(5, int(getattr(algo, "volatility_lookback", 20)))

    symbol_bars = algo.price_history.get(symbol, [])
    symbol_vol = calculate_historical_vol(symbol_bars, window=lookback)
    if symbol_vol <= 0:
        return base_cap

    pool_vols = []
    for pool_symbol in algo.stock_pool:
        bars = algo.price_history.get(pool_symbol, [])
        if len(bars) >= lookback + 1:
            vol = calculate_historical_vol(bars, window=lookback)
            if vol > 0:
                pool_vols.append(vol)

    if not pool_vols:
        return base_cap

    avg_pool_vol = sum(pool_vols) / len(pool_vols)
    raw_multiplier = avg_pool_vol / symbol_vol if symbol_vol > 0 else 1.0
    multiplier = max(
        getattr(algo, "volatility_cap_floor", 0.35),
        min(raw_multiplier, min(1.0, getattr(algo, "volatility_cap_ceiling", 1.0))),
    )
    return max(1, int(round(base_cap * multiplier)))


def _calculate_symbol_state_risk_multiplier(
    algo,
    symbol: str,
    underlying_price: float,
    symbol_put_notional: float,
    symbol_stock_notional: float,
    portfolio_value: float,
) -> Tuple[float, Dict[str, float]]:
    """Compute a one-way risk discount from symbol state and existing exposure."""
    if not getattr(algo, "dynamic_symbol_risk_enabled", True):
        return 1.0, {
            "vol_ratio": 1.0,
            "drawdown": 0.0,
            "momentum_20d": 0.0,
            "exposure_ratio": 0.0,
        }

    lookback = max(5, int(getattr(algo, "volatility_lookback", 20)))
    bars = algo.price_history.get(symbol, [])

    symbol_vol = calculate_historical_vol(bars, window=lookback) if len(bars) >= lookback + 1 else 0.25
    pool_vols = []
    for pool_symbol in algo.stock_pool:
        pool_bars = algo.price_history.get(pool_symbol, [])
        if len(pool_bars) >= lookback + 1:
            pool_vols.append(calculate_historical_vol(pool_bars, window=lookback))
    avg_pool_vol = (sum(pool_vols) / len(pool_vols)) if pool_vols else max(symbol_vol, 0.25)
    vol_ratio = symbol_vol / avg_pool_vol if avg_pool_vol > 0 else 1.0

    closes = [float(b.get("close", 0)) for b in bars if b.get("close", 0)]
    momentum_20d = 0.0
    if len(closes) >= 20 and closes[-20] > 0:
        momentum_20d = underlying_price / closes[-20] - 1.0

    dd_lookback = max(20, int(getattr(algo, "symbol_drawdown_lookback", 60)))
    recent_closes = closes[-dd_lookback:] if closes else []
    peak_price = max(recent_closes) if recent_closes else max(underlying_price, 1.0)
    drawdown = max(0.0, (peak_price - underlying_price) / peak_price) if peak_price > 0 else 0.0

    assignment_exposure = symbol_put_notional + symbol_stock_notional
    exposure_ratio = assignment_exposure / portfolio_value if portfolio_value > 0 else 0.0

    volatility_penalty = max(
        0.35,
        1.0 - max(0.0, vol_ratio - 1.0) * getattr(algo, "symbol_volatility_sensitivity", 0.75),
    )
    downtrend_penalty = max(
        0.35,
        1.0 - max(0.0, -momentum_20d) * getattr(algo, "symbol_downtrend_sensitivity", 1.50),
    )
    drawdown_penalty = max(
        0.25,
        1.0 - drawdown * getattr(algo, "symbol_drawdown_sensitivity", 1.20),
    )
    exposure_penalty = max(
        0.20,
        1.0 - exposure_ratio * getattr(algo, "symbol_exposure_sensitivity", 1.25),
    )

    raw_multiplier = volatility_penalty * downtrend_penalty * drawdown_penalty * exposure_penalty
    multiplier = max(
        getattr(algo, "symbol_state_cap_floor", 0.20),
        min(raw_multiplier, getattr(algo, "symbol_state_cap_ceiling", 1.0)),
    )
    return multiplier, {
        "vol_ratio": vol_ratio,
        "drawdown": drawdown,
        "momentum_20d": momentum_20d,
        "exposure_ratio": exposure_ratio,
    }


def _calculate_stock_inventory_cap(algo, portfolio_value: float, symbol_state_multiplier: float) -> float:
    """Dynamic cap for stock inventory held after assignments."""
    if not getattr(algo, "stock_inventory_cap_enabled", True):
        return portfolio_value
    base_cap = portfolio_value * getattr(algo, "stock_inventory_base_cap", 0.17)
    dynamic_multiplier = max(
        getattr(algo, "stock_inventory_cap_floor", 0.50),
        min(1.0, symbol_state_multiplier),
    )
    return max(0.0, base_cap * dynamic_multiplier)


def _format_pending_expiry(expiry) -> str:
    if hasattr(expiry, "strftime"):
        return expiry.strftime("%Y%m%d")
    return str(expiry).replace("-", "")


def _format_pending_right(target_right) -> str:
    return "C" if target_right in (OptionRight.Call, "Call", "C") else "P"


def _build_pending_open_key(signal, selected, target_right, quantity):
    return (
        f"{signal.symbol}_{_format_pending_expiry(selected['expiry'])}_"
        f"{float(selected['strike']):.0f}_{_format_pending_right(target_right)}_{quantity}"
    )


def _enqueue_pending_open_order(algo, queue_key, payload):
    if not hasattr(algo, "pending_open_orders") or algo.pending_open_orders is None:
        algo.pending_open_orders = {}
    algo.pending_open_orders[queue_key] = payload


def safe_execute_option_order(algo, option_symbol, quantity, theoretical_price, deferred_context=None):
    """Safely execute option order with data readiness check.
    
    QC dynamically subscribed options may not have price data immediately.
    This function checks if data is ready and uses LimitOrder as fallback.
    
    Args:
        algo: QCAlgorithm instance
        option_symbol: Option Symbol object
        quantity: Number of contracts (negative for sell)
        theoretical_price: BS theoretical price for limit order fallback
    
    Returns:
        OrderTicket or None
    """
    # Subscribe if not already in Securities
    if not algo.Securities.ContainsKey(option_symbol):
        algo.Log(f"Subscribing to option contract: {option_symbol}")
        # Use Minute so contract receives intraday bars before we trade.
        algo.AddOptionContract(option_symbol, Resolution.Minute)
    
    security = algo.Securities[option_symbol]
    
    # Check if price data is available
    if security.HasData and security.Price > 0:
        # Data ready - use MarketOrder
        
        return algo.MarketOrder(option_symbol, quantity)
    else:
        # Data not ready yet. Defer order to next rebalance cycle.
        algo.Log(f"ORDER_DEFERRED: {option_symbol} waiting for first bar")
        if deferred_context:
            _enqueue_pending_open_order(
                algo,
                deferred_context["queue_key"],
                {
                    **deferred_context,
                    "option_symbol": option_symbol,
                    "quantity": quantity,
                    "theoretical_price": theoretical_price,
                    "attempt_count": deferred_context.get("attempt_count", 0),
                },
            )
        return None


def make_signal(symbol, action, delta=0, dte_min=30, dte_max=45, num_contracts=1, confidence=0.5, reasoning=""):
    return StrategySignal(symbol=symbol, action=action, delta=delta, dte_min=dte_min, dte_max=dte_max,
        num_contracts=num_contracts, confidence=confidence, reasoning=reasoning,
        expected_premium=0.0, expected_return=0.0, expected_risk=0.0, assignment_probability=0.0)


def _enqueue_open_order_metadata(algo, ticket, signal: StrategySignal, selected: Dict, target_right):
    """Queue open-order metadata and persist after OrderEvent fill."""
    if not ticket or ticket.OrderId is None:
        return
    if not hasattr(algo, "pending_order_metadata"):
        algo.pending_order_metadata = {}
    strategy_phase = "SP" if target_right == OptionRight.Put else "CC"
    algo.pending_order_metadata[ticket.OrderId] = {
        "symbol": signal.symbol,
        "right": "P" if target_right == OptionRight.Put else "C",
        "strike": float(selected["strike"]),
        "expiry": selected["expiry"].strftime("%Y%m%d"),
        "delta_at_entry": selected.get("delta", 0),
        "iv_at_entry": selected.get("iv", 0.25),
        "strategy_phase": strategy_phase,
        "entry_date": algo.Time.strftime("%Y-%m-%d"),
        "ml_signal": signal,
    }


def _build_position_key(symbol: str, expiry, strike: float, right: str) -> str:
    return f"{symbol}_{expiry.strftime('%Y%m%d')}_{strike:.0f}_{right}"


def retry_pending_open_orders(algo, _find_option_func=None):
    """Retry deferred opening orders once contracts have price data."""
    pending = getattr(algo, "pending_open_orders", None)
    if not pending:
        return []

    completed = []
    items = sorted(
        pending.items(),
        key=lambda kv: 0 if kv[1]["signal"].action == "SELL_CALL" else 1,
    )
    for queue_key, item in items:
        attempts = int(item.get("attempt_count", 0))
        if attempts >= MAX_DEFERRED_OPEN_ATTEMPTS:
            algo.Log(f"ORDER_DEFERRED_EXPIRED: {queue_key}")
            completed.append(queue_key)
            continue

        security = algo.Securities[item["option_symbol"]] if algo.Securities.ContainsKey(item["option_symbol"]) else None
        if security is None or not security.HasData or security.Price <= 0:
            item["attempt_count"] = attempts + 1
            continue

        algo.Log(f"RETRY_DEFERRED_OPEN: {queue_key}")
        ticket = algo.MarketOrder(item["option_symbol"], item["quantity"])
        _enqueue_open_order_metadata(algo, ticket, item["signal"], item["selected"], item["target_right"])
        completed.append(queue_key)

    for queue_key in completed:
        pending.pop(queue_key, None)
    return completed


def handle_order_event(algo, order_event):
    """Persist option metadata on fill for asynchronously filled orders."""
    if order_event.Status != OrderStatus.Filled:
        return
    symbol = order_event.Symbol
    if not hasattr(symbol, "SecurityType") or symbol.SecurityType != SecurityType.Option:
        return
    # Opening short option position -> negative fill quantity
    if order_event.FillQuantity >= 0:
        return
    pending = getattr(algo, "pending_order_metadata", {}).pop(order_event.OrderId, None)
    if not pending:
        return
    pos_id = f"{pending['symbol']}_{pending['expiry']}_{pending['strike']:.0f}_{pending['right']}"
    save_position_metadata(algo, pos_id, {
        "delta_at_entry": pending["delta_at_entry"],
        "iv_at_entry": pending["iv_at_entry"],
        "strategy_phase": pending["strategy_phase"],
        "entry_date": pending["entry_date"],
        "ml_signal": pending["ml_signal"],
    })


def execute_signal(algo, signal: StrategySignal, find_option_func):
    if not signal or signal.action == "HOLD": return
    if signal.action == "ROLL": execute_roll(algo, signal, find_option_func); return
    if signal.action == "CLOSE": execute_close(algo, signal); return
    equity = algo.equities.get(signal.symbol)
    if not equity: return
    underlying_price = algo.Securities[equity.Symbol].Price
    target_right = OptionRight.Put if signal.action == "SELL_PUT" else OptionRight.Call
    target_delta = -signal.delta if target_right == OptionRight.Put else signal.delta
    min_strike = getattr(signal, 'min_strike', 0.0)
    selected = find_option_func(algo, symbol=signal.symbol, equity_symbol=equity.Symbol,
        target_right=target_right, target_delta=target_delta, dte_min=signal.dte_min, dte_max=signal.dte_max,
        delta_tolerance=0.08, min_strike=min_strike if min_strike > 0 else None,
        selection_tiers=getattr(signal, "selection_tiers", None))
    if not selected:
        algo.Log(f"No suitable options for {signal.symbol} delta ~{target_delta:.2f}")
        increment_debug_counter(algo, "no_suitable_options")
        return
    if signal.action == "SELL_CALL" and selected.get("selection_tier") and selected["selection_tier"] != "primary":
        algo.Log(f"CC_SELECTION_TIER:{signal.symbol}:{selected['selection_tier']}")
    current_positions = get_option_position_count(algo)
    if target_right == OptionRight.Put:
        quantity = calculate_put_quantity(algo, selected, current_positions, underlying_price, signal.symbol)
    else:
        shares_held = get_shares_held(algo, signal.symbol)
        existing_call_contracts = get_call_position_contracts(algo, signal.symbol)
        shares_covered = existing_call_contracts * 100  # Each contract covers 100 shares
        shares_available = shares_held - shares_covered
        quantity = min(max(0, shares_available // 100), algo.max_positions)
        if quantity <= 0:
            algo.Log(f"No available shares for {signal.symbol} call: held={shares_held}, covered={shares_covered}")
            return
    if quantity <= 0: return
    quantity = -quantity
    option_symbol = selected['option_symbol']
    queue_key = _build_pending_open_key(signal, selected, target_right, quantity)
    deferred_context = {
        "queue_key": queue_key,
        "signal": signal,
        "selected": selected,
        "target_right": target_right,
        "attempt_count": getattr(algo, "pending_open_orders", {}).get(queue_key, {}).get("attempt_count", 0),
    }
    # Use safe execution to handle data readiness
    ticket = safe_execute_option_order(
        algo,
        option_symbol,
        quantity,
        selected['premium'],
        deferred_context=deferred_context,
    )
    _enqueue_open_order_metadata(algo, ticket, signal, selected, target_right)


def calculate_put_quantity(algo, selected: Dict, current_positions: int, underlying_price: float, symbol: str) -> int:
    """Calculate number of Put contracts to sell with improved margin management.
    
    Improvements:
    1. Use more accurate margin estimation (QC standard formula)
    2. Reduce positions when holding stock (stock ties up capital)
    3. Apply global margin budget constraint
    4. Use dynamically calculated max_positions
    """
    strike = selected['strike']
    premium = selected.get('premium', 0)
    
    # === More accurate margin estimation ===
    # QC uses: max(20% * underlying - OTM amount, 10% * strike) + premium
    # For naked short put, OTM amount = max(0, underlying - strike)
    otm_amount = max(0, underlying_price - strike)
    margin_method_1 = 0.20 * underlying_price * 100 - otm_amount * 100
    margin_method_2 = 0.10 * strike * 100
    estimated_margin_per_contract = max(margin_method_1, margin_method_2) + premium * 100
    
    # Ensure minimum margin estimate
    fallback_margin = strike * 100 * algo.margin_rate_per_contract
    estimated_margin_per_contract = max(estimated_margin_per_contract, fallback_margin)
    
    # === Get available margin ===
    available_margin = max(0.0, algo.Portfolio.MarginRemaining)
    usable_margin = max(0.0, available_margin * (1 - algo.margin_buffer_pct))
    
    # === Reduce positions when holding stock ===
    # Stock holdings tie up capital, reduce max positions accordingly
    stock_value = get_total_stock_holdings_value(algo, algo.stock_pool)
    stock_holding_count = get_stock_holding_count(algo, algo.stock_pool)
    
    # Each stock holding reduces available Put slots
    # If we have 2 stocks, reduce max_positions by 2-3 for new Puts
    adjusted_max_positions = algo.max_positions - stock_holding_count
    
    # Also reduce based on stock value relative to initial capital
    # If stock value > 30% of capital, further reduce positions
    stock_value_ratio = stock_value / algo.initial_capital if algo.initial_capital > 0 else 0
    if stock_value_ratio > 0.30:
        # Reduce positions proportionally (max 50% reduction)
        reduction_factor = min(0.5, stock_value_ratio)
        adjusted_max_positions = max(1, int(adjusted_max_positions * (1 - reduction_factor)))
    
    # === Existing put exposure limits ===
    total_put_contracts = 0
    symbol_put_contracts = 0
    total_put_notional = 0.0
    symbol_put_notional = 0.0
    for holding in algo.Portfolio.Values:
        if not holding.Invested:
            continue
        hs = holding.Symbol
        if not (hasattr(hs, "SecurityType") and hs.SecurityType == SecurityType.Option):
            continue
        if hs.ID.OptionRight != OptionRight.Put:
            continue
        contracts = abs(holding.Quantity)
        strike_h = float(hs.ID.StrikePrice)
        put_notional = contracts * strike_h * 100
        total_put_contracts += contracts
        total_put_notional += put_notional
        if hasattr(hs, "Underlying") and hs.Underlying and hs.Underlying.Value == symbol:
            symbol_put_contracts += contracts
            symbol_put_notional += put_notional

    # Include current stock exposure in notional-based caps
    symbol_stock_notional = get_shares_held(algo, symbol) * max(underlying_price, 0)

    symbol_cap = calculate_volatility_weighted_symbol_cap(algo, symbol)
    portfolio_value = max(algo.Portfolio.TotalPortfolioValue, 0.0)
    symbol_state_multiplier, symbol_state = _calculate_symbol_state_risk_multiplier(
        algo,
        symbol,
        underlying_price,
        symbol_put_notional,
        symbol_stock_notional,
        portfolio_value,
    )
    symbol_cap = max(1, int(symbol_cap * symbol_state_multiplier))
    max_by_symbol_contracts = max(0, symbol_cap - symbol_put_contracts)
    max_by_total_contracts = max(0, algo.max_put_contracts_total - total_put_contracts)
    max_by_trade_cap = max(0, min(algo.max_contracts_per_trade, symbol_cap))

    # === Calculate quantity limits ===
    max_by_margin = int(usable_margin / estimated_margin_per_contract) if estimated_margin_per_contract > 0 else 0
    max_by_limit = max(0, adjusted_max_positions - current_positions)
    
    # === Global margin budget: don't use more than target utilization ===
    total_margin_used = algo.Portfolio.TotalMarginUsed
    margin_budget = portfolio_value * algo.target_margin_utilization
    remaining_budget = max(0, margin_budget - total_margin_used)
    max_by_budget = max(0, int(remaining_budget / estimated_margin_per_contract)) if estimated_margin_per_contract > 0 else 0
    
    # === Hard leverage budget ===
    leverage_budget = portfolio_value * algo.max_leverage
    remaining_leverage_budget = max(0, leverage_budget - total_margin_used)
    max_by_leverage = max(0, int(remaining_leverage_budget / estimated_margin_per_contract)) if estimated_margin_per_contract > 0 else 0

    # === Notional assignment caps (dynamic by strike and aggressiveness) ===
    # These caps keep worst-case assignment exposure proportional to account size.
    aggr = getattr(algo, "position_aggressiveness", 1.0)
    base_symbol_notional_cap = portfolio_value * getattr(algo, "symbol_assignment_base_cap", 0.28)
    per_symbol_notional_cap = base_symbol_notional_cap * symbol_state_multiplier
    total_notional_cap = portfolio_value * (0.70 + 0.90 * aggr)        # 0.97x ~ 2.50x PV
    candidate_notional = strike * 100
    stock_inventory_cap = _calculate_stock_inventory_cap(algo, portfolio_value, symbol_state_multiplier)

    remaining_symbol_notional = max(0.0, per_symbol_notional_cap - (symbol_put_notional + symbol_stock_notional))
    remaining_total_notional = max(0.0, total_notional_cap - (total_put_notional + stock_value))
    remaining_stock_inventory = max(0.0, stock_inventory_cap - symbol_stock_notional)
    assignment_trade_cap = portfolio_value * getattr(algo, "max_assignment_risk_per_trade", 0.20)
    max_by_symbol_notional = int(remaining_symbol_notional / candidate_notional) if candidate_notional > 0 else 0
    max_by_total_notional = int(remaining_total_notional / candidate_notional) if candidate_notional > 0 else 0
    max_by_stock_inventory = int(remaining_stock_inventory / candidate_notional) if candidate_notional > 0 else 0
    max_by_assignment_trade = int(assignment_trade_cap / candidate_notional) if candidate_notional > 0 else 0
    
    quantity = min(
        max_by_margin,
        max_by_limit,
        max_by_budget,
        max_by_leverage,
        max_by_symbol_contracts,
        max_by_total_contracts,
        max_by_trade_cap,
        max_by_symbol_notional,
        max_by_total_notional,
        max_by_stock_inventory,
        max_by_assignment_trade,
    )
    max_by_risk = quantity
    risk_message = ""
    if quantity > 0 and premium > 0 and portfolio_value > 0:
        risk_adjusted_quantity, risk_message = calculate_position_risk(
            premium=premium,
            quantity=-quantity,
            portfolio_value=portfolio_value,
            max_risk_per_trade=getattr(algo, "max_risk_per_trade", 0.02),
            max_leverage=algo.max_leverage,
            current_margin_used=total_margin_used,
        )
        max_by_risk = abs(risk_adjusted_quantity) if risk_adjusted_quantity < 0 else 0
        quantity = min(quantity, max_by_risk)
    
    # Log if position is limited due to stock holdings or margin
    if stock_holding_count > 0 and quantity < (algo.max_positions - current_positions):
        algo.Log(f"PUT_LIMIT: stocks={stock_holding_count}, adj_max={adjusted_max_positions}, qty={quantity}")
    if quantity <= 0:
        algo.Log(
            f"PUT_BLOCK:{symbol} margin={max_by_margin} budget={max_by_budget} lev={max_by_leverage} "
            f"symcap={max_by_symbol_contracts}/{symbol_cap} totalcap={max_by_total_contracts} "
            f"symnot={max_by_symbol_notional} totalnot={max_by_total_notional} stockcap={max_by_stock_inventory} "
            f"assigntrade={max_by_assignment_trade} "
            f"risk={max_by_risk} riskmsg={risk_message or 'NA'} "
            f"slots={max_by_limit} "
            f"state={symbol_state_multiplier:.2f} vol={symbol_state['vol_ratio']:.2f} "
            f"dd={symbol_state['drawdown']:.2f} mom={symbol_state['momentum_20d']:.2f} "
            f"exp={symbol_state['exposure_ratio']:.2f}"
        )
    
    return max(0, quantity)


def execute_roll(algo, signal: StrategySignal, find_option_func, existing_position: Optional[Dict] = None):
    existing = existing_position or get_position_for_symbol(algo, signal.symbol)
    if not existing: return
    pos_info = existing
    pos_id = _build_position_key(signal.symbol, pos_info['expiry'], pos_info['strike'], pos_info['right'])
    # Use safe execution for closing the existing position
    close_ticket = safe_execute_option_order(
        algo, pos_info['option_symbol'], -pos_info['quantity'], pos_info['entry_price'])
    if not close_ticket or close_ticket.Status != OrderStatus.Filled:
        if not hasattr(algo, "pending_roll_orders"):
            algo.pending_roll_orders = {}
        algo.pending_roll_orders[pos_id] = {
            "symbol": signal.symbol,
            "existing_position": pos_info,
            "signal": signal,
            "queued_at": getattr(algo, "Time", None),
            "close_order_id": getattr(close_ticket, "OrderId", None) if close_ticket else None,
        }
        return
    pnl, _ = calculate_pnl_metrics(pos_info['entry_price'], close_ticket.AverageFillPrice, pos_info['quantity'])
    record_trade(algo, signal.symbol, pos_info['right'], pnl, "ROLL")
    remove_position_metadata(algo, pos_id)
    equity = algo.equities.get(signal.symbol)
    if not equity: return
    target_right = OptionRight.Put if pos_info['right'] == 'P' else OptionRight.Call
    target_delta = -signal.delta if target_right == OptionRight.Put else signal.delta
    new_selected = find_option_func(algo, symbol=signal.symbol, equity_symbol=equity.Symbol,
        target_right=target_right, target_delta=target_delta, dte_min=signal.dte_min, dte_max=signal.dte_max)
    if new_selected:
        new_qty = pos_info['quantity']
        new_option_symbol = new_selected['option_symbol']
        # Use safe execution to handle data readiness
        new_ticket = safe_execute_option_order(algo, new_option_symbol, new_qty, new_selected['premium'])
        _enqueue_open_order_metadata(algo, new_ticket, signal, new_selected, target_right)


def execute_close(algo, signal: StrategySignal, existing_position: Optional[Dict] = None):
    pos_info = existing_position or get_position_for_symbol(algo, signal.symbol)
    if not pos_info: return
    pos_id = _build_position_key(signal.symbol, pos_info['expiry'], pos_info['strike'], pos_info['right'])
    # For closing, use entry price as reference for limit order fallback
    # (positions should have data, but use safe execution for consistency)
    close_ticket = safe_execute_option_order(
        algo, pos_info['option_symbol'], -pos_info['quantity'], pos_info['entry_price'])
    if not close_ticket or close_ticket.Status != OrderStatus.Filled:
        if not hasattr(algo, "pending_close_orders"):
            algo.pending_close_orders = {}
        algo.pending_close_orders[pos_id] = {
            "symbol": signal.symbol,
            "existing_position": pos_info,
            "reason": signal.reasoning or "SIGNAL_CLOSE",
            "queued_at": getattr(algo, "Time", None),
            "close_order_id": getattr(close_ticket, "OrderId", None) if close_ticket else None,
        }
        return
    pnl, _ = calculate_pnl_metrics(pos_info['entry_price'], close_ticket.AverageFillPrice, pos_info['quantity'])
    record_trade(algo, signal.symbol, pos_info['right'], pnl, signal.reasoning or "SIGNAL_CLOSE")
    remove_position_metadata(algo, pos_id)


def record_trade(algo, symbol: str, right: str, pnl: float, reason: str):
    algo.total_trades += 1
    algo.total_pnl += pnl
    if pnl > 0: algo.winning_trades += 1
    algo.trade_history.append({"date": algo.Time.strftime("%Y-%m-%d"), "symbol": symbol, "type": right, "pnl": pnl, "exit_reason": reason})
