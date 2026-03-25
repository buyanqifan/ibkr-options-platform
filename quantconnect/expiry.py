"""Option expiry and assignment handling for BinbinGod Strategy.

No phase concept - QC handles assignment automatically, we just log and record.
"""
from datetime import datetime
from option_utils import calculate_dte
from execution import record_trade
from qc_portfolio import (
    get_option_positions, get_shares_held, get_cost_basis,
    get_position_metadata, remove_position_metadata, get_symbols_with_holdings
)


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
        
        elif right == "C":
            # Call assignment: QC sold shares for us at strike price
            if qc_shares == 0:
                was_assigned = True
                exit_reason = "ASSIGNMENT"
                cost_basis = get_cost_basis(algo, symbol)
                shares_sold = quantity * 100
                stock_pnl = (strike - cost_basis) * shares_sold if cost_basis > 0 else 0
                algo.Log(f"Call assigned: -{shares_sold} {symbol} @ ${strike:.2f}, Stock P&L: ${stock_pnl:+.2f}")
        
        # Get P&L from QC (already calculated by platform)
        pnl = security.Holdings.UnrealizedProfit if hasattr(security, 'Holdings') else 0
        
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
    if algo.ml_integration.should_retrain(): algo.Log("Retraining ML models...")