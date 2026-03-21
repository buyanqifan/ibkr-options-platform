"""
Analyze CC phase trading issue from backtest result
"""
import json
from pathlib import Path
from datetime import datetime

# Read backtest result
backtest_file = Path(__file__).parent / "backtest_binbin_god_MAG7_AUTO_20260321_124030.json"
with open(backtest_file, 'r') as f:
    data = json.load(f)

print("=" * 80)
print("CC Phase Trading Issue Analysis")
print("=" * 80)

trades = data['trades']
strategy_perf = data.get('strategy_performance', {})

# Count trade types
put_trades = [t for t in trades if t['trade_type'] == 'BINBIN_PUT']
call_trades = [t for t in trades if t['trade_type'] == 'BINBIN_CALL']

print(f"\nTotal trades: {len(trades)}")
print(f"BINBIN_PUT trades: {len(put_trades)}")
print(f"BINBIN_CALL trades: {len(call_trades)}")

# Find last assignment trades
assignments = [t for t in trades if t['exit_reason'] == 'ASSIGNMENT']
print(f"\nTotal assignments: {len(assignments)}")
if assignments:
    print("\nLast 3 assignments:")
    for trade in assignments[-3:]:
        print(f"  {trade['entry_date']} {trade['symbol']:6} {trade['trade_type']:12} "
              f"qty={trade['quantity']:3} strike=${trade['strike']:7.2f} "
              f"exit_reason={trade['exit_reason']}")

# Check strategy state
print(f"\nStrategy final state:")
print(f"  Phase: {strategy_perf['phase']}")
print(f"  Shares held: {strategy_perf['shares_held']}")
print(f"  Stock holdings:")
for symbol, holding in strategy_perf['stock_holdings'].items():
    print(f"    {symbol}: {holding['shares']} shares @ ${holding['cost_basis']:.2f}")

# Check last trade date vs end date
last_trade_date = trades[-1]['exit_date']
end_date = data['parameters']['end_date']
last_trade_dt = datetime.strptime(last_trade_date, '%Y-%m-%d')
end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
days_between = (end_date_dt - last_trade_dt).days

print(f"\nTimeline analysis:")
print(f"  Last trade date: {last_trade_date}")
print(f"  Backtest end date: {end_date}")
print(f"  Days between: {days_between} days")

# Check if max_positions could be the issue
max_positions = data['parameters']['max_positions']
print(f"\nParameters:")
print(f"  max_positions: {max_positions}")
print(f"  cc_optimization_enabled: {data['parameters']['cc_optimization_enabled']}")
print(f"  cc_min_delta_cost: {data['parameters']['cc_min_delta_cost']}")
print(f"  cc_cost_basis_threshold: {data['parameters']['cc_cost_basis_threshold']}")

# Analyze potential root causes
print(f"\n" + "=" * 80)
print("Root Cause Analysis:")
print("=" * 80)

print(f"\n1. No Call trades generated despite holding {strategy_perf['shares_held']} shares")
print(f"   Expected: At least {strategy_perf['shares_held'] // 100} Call contracts should be sold")

print(f"\n2. Strategy stuck in CC phase from {last_trade_date} to {end_date} ({days_between} days)")

print(f"\n3. Possible causes:")
print(f"   a) pool_data not available in generate_signals()")
print(f"   b) Stock price retrieval failed for held symbols")
print(f"   c) CC optimization constraints too strict (min_strike > current_price)")
print(f"   d) Premium calculation returned < 0.01")
print(f"   e) max_positions limit reached (but should have capacity)")

print(f"\n4. Key parameters that may prevent Call signals:")
print(f"   - cc_min_delta_cost: {data['parameters']['cc_min_delta_cost']} (very low, should allow OTM calls)")
print(f"   - cc_cost_basis_threshold: {data['parameters']['cc_cost_basis_threshold']} (5% below cost triggers optimization)")
print(f"   - cc_min_strike_premium: {data['parameters']['cc_min_strike_premium']} (2% below cost basis)")

print(f"\n5. Check stock holdings vs current prices:")
for symbol, holding in strategy_perf['stock_holdings'].items():
    cost_basis = holding['cost_basis']
    min_strike = cost_basis * (1 - data['parameters']['cc_min_strike_premium'])
    print(f"   {symbol}: cost=${cost_basis:.2f}, min_strike=${min_strike:.2f}")
    print(f"      If current_price < ${min_strike:.2f}, CC optimization may fail to find valid strike")

print(f"\nRecommendation:")
print(f"  1. Check logs for 'CC phase: pool_data keys=' to verify data availability")
print(f"  2. Check logs for strike selection and premium calculation")
print(f"  3. Verify that current stock prices allow valid strike selection")
print(f"  4. Consider relaxing cc_min_strike_premium if stock prices dropped significantly")
