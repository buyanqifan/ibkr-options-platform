"""Option expiry and assignment handling for BinbinGod Strategy."""
from datetime import datetime
from option_utils import calculate_dte
from execution import record_trade


def check_expired_options(algo):
    """Check for expired/assigned options and sync strategy state with QC Portfolio."""
    for pos_id, pos_info in list(algo.open_option_positions.items()):
        security = algo.Securities.get(pos_info['option_symbol'])
        if not security or not (security.IsDelisted or security.Price == 0):
            continue
        symbol = pos_info['symbol']
        strike = pos_info['strike']
        right = pos_info['right']
        quantity = abs(pos_info['quantity'])
        equity = algo.equities.get(symbol)
        qc_shares = 0
        if equity and algo.Portfolio.ContainsKey(equity.Symbol):
            qc_shares = algo.Portfolio[equity.Symbol].Quantity
        tracked_shares = algo.stock_holding.get_shares(symbol)
        was_assigned = False
        exit_reason = "EXPIRY"
        if right == "P":
            if qc_shares > tracked_shares:
                shares_acquired = qc_shares - tracked_shares
                expected_shares = quantity * 100
                if shares_acquired > expected_shares * 1.1:
                    algo.Log(f"WARNING: Put assignment shares {shares_acquired} > expected {expected_shares}")
                    shares_acquired = min(shares_acquired, expected_shares)
                algo.stock_holding.add_shares(symbol, shares_acquired, strike)
                algo.stock_holding.add_premium(symbol, pos_info['entry_price'] * 100)
                algo.phase = "CC"
                was_assigned = True
                exit_reason = "ASSIGNMENT"
                algo.Log(f"Put assigned: +{shares_acquired} {symbol} @ ${strike:.2f} (QC has {qc_shares} shares)")
        elif right == "C":
            if qc_shares < tracked_shares:
                shares_sold = tracked_shares - qc_shares
                expected_shares = quantity * 100
                if shares_sold > expected_shares * 1.1:
                    algo.Log(f"WARNING: Call assignment shares {shares_sold} > expected {expected_shares}")
                    shares_sold = min(shares_sold, expected_shares)
                algo.stock_holding.remove_shares(symbol, shares_sold)
                was_assigned = True
                exit_reason = "ASSIGNMENT"
                cost_basis = algo.stock_holding.holdings.get(symbol, {}).get('cost_basis', strike)
                stock_pnl = (strike - cost_basis) * shares_sold if cost_basis > 0 else 0
                algo.Log(f"Call assigned: -{shares_sold} {symbol} @ ${strike:.2f}, Stock P&L: ${stock_pnl:+.2f}")
                if qc_shares == 0 or algo.stock_holding.shares == 0:
                    algo.phase = "SP"
                    algo.Log(f"All shares sold, returning to SP phase")
        pnl = security.Holdings.UnrealizedProfit if hasattr(security, 'Holdings') else 0
        algo.ml_integration.update_performance({
            'symbol': symbol,
            'delta': abs(pos_info['delta_at_entry']),
            'dte': calculate_dte(pos_info['expiry'], datetime.strptime(pos_info['entry_date'], '%Y-%m-%d')),
            'num_contracts': quantity,
            'pnl': pnl / 100,
            'assigned': was_assigned,
            'bars': algo.price_history.get(symbol, []),
            'cost_basis': algo.stock_holding.holdings.get(symbol, {}).get('cost_basis', 0),
            'strategy_phase': pos_info['strategy_phase'],
        })
        record_trade(algo, symbol, right, pnl, exit_reason)
        del algo.open_option_positions[pos_id]


def update_ml_models(algo):
    if algo.ml_integration.should_retrain(): algo.Log("Retraining ML models...")