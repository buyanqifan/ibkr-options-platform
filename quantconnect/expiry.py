"""Option expiry and assignment handling for BinbinGod Strategy.

No phase concept - QC handles assignment automatically, we just log and record.
"""
from datetime import datetime
from option_utils import calculate_dte
from execution import record_trade, execute_signal
from signals import calculate_pnl_metrics
from qc_portfolio import (
    get_option_positions, get_shares_held, get_cost_basis,
    get_position_metadata, remove_position_metadata, get_symbols_with_holdings
)
from signal_generation import generate_signal_for_symbol, get_portfolio_state
from option_selector import find_option_by_greeks
from helpers import set_symbol_cooldown


def try_sell_cc_immediately(algo, symbol):
    """Try to sell CC immediately after Put assignment to protect stock from margin call.
    
    This is critical because:
    1. Put assignment happens at market close
    2. Margin check happens before next rebalance
    3. If we don't sell CC immediately, stock may be liquidated
    """
    algo.Log(f"IMMEDIATE_CC: Attempting to sell CC for {symbol} after Put assignment")
    
    portfolio_state = get_portfolio_state(algo)
    signal = generate_signal_for_symbol(algo, symbol, "CC", portfolio_state)
    
    if signal:
        algo.Log(f"IMMEDIATE_CC: Generated signal for {symbol}, delta={signal.delta:.2f}")
        if signal.confidence >= algo.ml_min_confidence:
            execute_signal(algo, signal, find_option_by_greeks)
            algo.Log(f"IMMEDIATE_CC: Executed CC for {symbol}")
        else:
            algo.Log(f"IMMEDIATE_CC: Confidence {signal.confidence:.2f} < min {algo.ml_min_confidence}")
    else:
        algo.Log(f"IMMEDIATE_CC: No valid CC signal for {symbol}")


def check_expired_options(algo):
    """Check for expired/assigned options and record ML performance data.
    
    QC automatically handles option expiration/assignment and stock delivery.
    No phase management needed - strategy is holdings-driven.
    """
    positions = get_option_positions(algo)
    for pos_id, pos_info in positions.items():
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
                algo.Log(f"Put assigned: +{shares_acquired} {symbol} @ ${strike:.2f}")
                set_symbol_cooldown(algo, symbol, algo.assignment_cooldown_days, "put_assignment")
                
                # CRITICAL: Immediately try to sell CC to protect stock from margin call
                try_sell_cc_immediately(algo, symbol)
        
        elif right == "C":
            # Call assignment: QC sold shares for us at strike price
            if qc_shares == 0:
                was_assigned = True
                exit_reason = "ASSIGNMENT"
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
