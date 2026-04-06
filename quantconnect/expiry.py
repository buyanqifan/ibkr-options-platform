"""Option expiry and assignment handling for BinbinGod Strategy.

No phase concept - QC handles assignment automatically, we just log and record.
"""
from datetime import datetime, timedelta
from AlgorithmImports import OptionRight, SecurityType
from option_utils import calculate_dte
from execution import record_trade, execute_signal
from signals import calculate_pnl_metrics
from debug_counters import increment_debug_counter
from qc_portfolio import (
    get_option_positions, get_shares_held, get_cost_basis,
    get_position_metadata, remove_position_metadata, get_symbols_with_holdings
)
from signal_generation import generate_signal_for_symbol, get_portfolio_state
from option_selector import find_option_by_greeks


def _build_assignment_key(symbol: str, expiry, strike: float, right: str) -> str:
    return f"{symbol}_{expiry.strftime('%Y%m%d')}_{strike:.0f}_{right}"


def _extract_assignment_context(order_event):
    symbol = getattr(order_event, "Symbol", None)
    if not symbol or not hasattr(symbol, "SecurityType") or symbol.SecurityType != SecurityType.Option:
        return None

    underlying = None
    if hasattr(symbol, "Underlying") and symbol.Underlying:
        underlying = getattr(symbol.Underlying, "Value", None)
    if not underlying and hasattr(symbol, "ID") and hasattr(symbol.ID, "Underlying"):
        underlying = getattr(symbol.ID.Underlying, "Value", None)
    if not underlying:
        return None

    right = "P" if symbol.ID.OptionRight == OptionRight.Put else "C"
    strike = float(symbol.ID.StrikePrice)
    expiry = symbol.ID.Date
    quantity = abs(int(getattr(order_event, "FillQuantity", 0) or 0))
    return {
        "symbol": underlying,
        "right": right,
        "strike": strike,
        "expiry": expiry,
        "quantity": max(1, quantity),
        "option_symbol": symbol,
    }


def _track_assigned_stock(algo, symbol: str, assignment_cost_basis: float):
    """Track stock inventory created by put assignment for later fail-safe checks."""
    if not hasattr(algo, "assigned_stock_state"):
        algo.assigned_stock_state = {}
    repair_deadline = algo.Time + timedelta(days=max(1, getattr(algo, "assigned_stock_max_repair_days", 7)))
    algo.assigned_stock_state[symbol] = {
        "assignment_date": algo.Time,
        "assignment_cost_basis": assignment_cost_basis,
        "sp_reentry_block_until": algo.Time + timedelta(days=max(0, getattr(algo, "sp_assignment_cooldown_days", 7))),
        "inventory_mode": "repair",
        "repair_deadline": repair_deadline,
        "cc_miss_count": 0,
        "has_covered_call": False,
        "force_exit_triggered": False,
    }
    increment_debug_counter(algo, "assigned_stock_track")
    algo.Log(
        f"ASSIGNED_STOCK_TRACK:{symbol}:cost_basis={assignment_cost_basis:.2f}:"
        f"sp_block_until={algo.assigned_stock_state[symbol]['sp_reentry_block_until'].strftime('%Y-%m-%d')}:"
        f"repair_deadline={repair_deadline.strftime('%Y-%m-%d')}:mode=repair"
    )


def try_sell_cc_immediately(algo, symbol):
    """Try to sell CC immediately after Put assignment to protect stock from margin call.
    
    This is critical because:
    1. Put assignment happens at market close
    2. Margin check happens before next rebalance
    3. If we don't sell CC immediately, stock may be liquidated
    """
    increment_debug_counter(algo, "immediate_cc")
    algo.Log(f"IMMEDIATE_CC: Attempting to sell CC for {symbol} after Put assignment")
    
    portfolio_state = get_portfolio_state(algo)
    signal = generate_signal_for_symbol(algo, symbol, "CC", portfolio_state)
    
    if signal:
        algo.Log(f"IMMEDIATE_CC: Generated signal for {symbol}, delta={signal.delta:.2f}")
        if signal.confidence >= algo.ml_min_confidence:
            execute_signal(algo, signal, find_option_by_greeks)
            algo.Log(f"IMMEDIATE_CC: Executed CC for {symbol}")
        else:
            increment_debug_counter(algo, "cc_confidence_block")
            algo.Log(f"IMMEDIATE_CC: Confidence {signal.confidence:.2f} < min {algo.ml_min_confidence}")
    else:
        increment_debug_counter(algo, "cc_signal_missing")
        algo.Log(f"IMMEDIATE_CC: No valid CC signal for {symbol}")


def handle_assignment_order_event(algo, order_event):
    """Process QC assignment events immediately instead of relying on expiry scans."""
    if getattr(order_event, "Status", None) != "Filled":
        return
    if not getattr(order_event, "IsAssignment", False):
        return

    context = _extract_assignment_context(order_event)
    if not context:
        return

    symbol = context["symbol"]
    strike = context["strike"]
    right = context["right"]
    quantity = context["quantity"]
    pos_id = _build_assignment_key(symbol, context["expiry"], strike, right)
    processed = getattr(algo, "processed_assignment_keys", None)
    if processed is not None and pos_id in processed:
        return
    if processed is not None:
        processed.add(pos_id)
    metadata = get_position_metadata(algo, pos_id)
    cost_basis = get_cost_basis(algo, symbol) or strike

    if right == "P":
        algo.Log(f"Put assigned (event): +{quantity * 100} {symbol} @ ${strike:.2f}")
        _track_assigned_stock(algo, symbol, cost_basis)
        try_sell_cc_immediately(algo, symbol)
    else:
        algo.Log(f"Call assigned (event): {symbol} @ ${strike:.2f}")

    if hasattr(algo, "ml_integration") and hasattr(algo.ml_integration, "update_performance"):
        entry_date = metadata.get("entry_date", algo.Time.strftime("%Y-%m-%d"))
        algo.ml_integration.update_performance({
            "symbol": symbol,
            "delta": abs(metadata.get("delta_at_entry", 0)),
            "dte": calculate_dte(context["expiry"], datetime.strptime(entry_date, "%Y-%m-%d")),
            "num_contracts": quantity,
            "pnl": 0.0,
            "assigned": True,
            "bars": algo.price_history.get(symbol, []),
            "cost_basis": cost_basis,
            "strategy_phase": metadata.get("strategy_phase", "SP"),
        })

    remove_position_metadata(algo, pos_id)


def check_expired_options(algo):
    """Check for expired/assigned options and record ML performance data.
    
    QC automatically handles option expiration/assignment and stock delivery.
    No phase management needed - strategy is holdings-driven.
    """
    positions = get_option_positions(algo)
    for pos_id, pos_info in positions.items():
        processed = getattr(algo, "processed_assignment_keys", None)
        assignment_key = _build_assignment_key(pos_info["symbol"], pos_info["expiry"], pos_info["strike"], pos_info["right"])
        if processed is not None and assignment_key in processed:
            continue
        security = algo.Securities.get(pos_info['option_symbol'])
        if not security or not (security.IsDelisted or security.Price == 0):
            continue
        
        symbol = pos_info['symbol']
        strike = pos_info['strike']
        right = pos_info['right']
        quantity = abs(pos_info['quantity'])
        
        # Get current shares from QC Portfolio
        equity = algo.equities.get(symbol)
        qc_shares = 0
        if equity and algo.Portfolio.ContainsKey(equity.Symbol):
            qc_shares = algo.Portfolio[equity.Symbol].Quantity
        
        # Detect assignment by checking QC Portfolio state
        was_assigned = False
        exit_reason = "EXPIRY"
        
        if right == "P":
            # Put assignment: QC bought shares for us at strike price
            if qc_shares > 0:
                shares_acquired = qc_shares
                expected_shares = quantity * 100
                if shares_acquired > expected_shares * 1.1:
                    algo.Log(f"WARNING: Put assignment shares {shares_acquired} > expected {expected_shares}")
                was_assigned = True
                exit_reason = "ASSIGNMENT"
                if processed is not None:
                    processed.add(assignment_key)
                algo.Log(f"Put assigned: +{shares_acquired} {symbol} @ ${strike:.2f}")
                assignment_cost_basis = get_cost_basis(algo, symbol) or strike
                _track_assigned_stock(algo, symbol, assignment_cost_basis)
                
                # CRITICAL: Immediately try to sell CC to protect stock from margin call
                try_sell_cc_immediately(algo, symbol)
        
        elif right == "C":
            # Call assignment: QC sold shares for us at strike price
            if qc_shares == 0:
                was_assigned = True
                exit_reason = "ASSIGNMENT"
                if processed is not None:
                    processed.add(assignment_key)
                cost_basis = get_cost_basis(algo, symbol)
                shares_sold = quantity * 100
                stock_pnl = (strike - cost_basis) * shares_sold if cost_basis > 0 else 0
                algo.Log(f"Call assigned: -{shares_sold} {symbol} @ ${strike:.2f}, Stock P&L: ${stock_pnl:+.2f}")
        
        # Prefer option-leg P&L derived from entry and settlement proxy price.
        # This is more stable than UnrealizedProfit on delisted contracts.
        settlement_price = security.Price if security.Price and security.Price > 0 else 0
        pnl, _ = calculate_pnl_metrics(pos_info['entry_price'], settlement_price, pos_info['quantity'])
        
        # Get metadata for ML learning
        metadata = get_position_metadata(algo, pos_id)
        
        # Update ML with correct assignment flag
        algo.ml_integration.update_performance({
            'symbol': symbol,
            'delta': abs(metadata.get('delta_at_entry', 0)),
            'dte': calculate_dte(pos_info['expiry'], datetime.strptime(metadata.get('entry_date', algo.Time.strftime('%Y-%m-%d')), '%Y-%m-%d')),
            'num_contracts': quantity,
            'pnl': pnl / 100,
            'assigned': was_assigned,
            'bars': algo.price_history.get(symbol, []),
            'cost_basis': get_cost_basis(algo, symbol),
            'strategy_phase': metadata.get('strategy_phase', 'SP'),
        })
        
        # Record trade (P&L from QC, no manual calculation)
        record_trade(algo, symbol, right, pnl, exit_reason)
        
        # Clean up metadata
        remove_position_metadata(algo, pos_id)


def update_ml_models(algo):
    if algo.ml_integration.should_retrain():
        algo.Log("Retraining ML models...")
