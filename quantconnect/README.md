# BinbinGod Strategy for QuantConnect

A simplified, rules-first Wheel strategy for MAG7 stocks.

## Files

| File | Description |
|------|-------------|
| `main.py` | Main strategy file |
| `ml_integration.py` | ML integration module |
| `ml_delta_optimizer.py` | Delta optimization using Q-learning |
| `ml_dte_optimizer.py` | DTE optimization using Q-learning |
| `ml_position_optimizer.py` | Position sizing with Kelly criterion |
| `ml_volatility_model.py` | Volatility prediction model |
| `option_pricing.py` | Black-Scholes option pricing |
| `config.json` | Strategy configuration |

## Usage

1. Upload all `.py` files to QuantConnect project
2. Upload `config.json` for parameters
3. Run backtest

## Configuration

```json
{
    "initial_capital": 300000,
    "stock_pool": "MSFT,AAPL,NVDA,GOOGL,AMZN,META,TSLA",
    "max_positions_ceiling": 20,
    "target_margin_utilization": 0.65,
    "symbol_assignment_base_cap": 0.35,
    "max_assignment_risk_per_trade": 0.20,
    "roll_threshold_pct": 80,
    "min_dte_for_roll": 7,
    "cc_target_delta": 0.25,
    "cc_target_dte_min": 10,
    "cc_target_dte_max": 28,
    "cc_max_discount_to_cost": 0.03,
    "assigned_stock_min_days_held": 5,
    "assigned_stock_drawdown_pct": 0.12,
    "assigned_stock_force_exit_pct": 1.0,
    "ml_enabled": true,
    "ml_min_confidence": 0.45
}
```

## Strategy Logic

### Sell Put (SP)
- Open short puts using one signal layer and one sizing function
- Only three active opening caps remain: portfolio margin, per-symbol assignment, per-trade assignment
- Open short puts are managed by one rule engine:
  - `ROLL` when premium captured >= threshold and DTE is still above the minimum
  - `EXPIRY` when DTE <= 0
  - otherwise `HOLD`

### Covered Call (CC)
- Assignment immediately records stock state and attempts a covered call
- When stock is below cost, CC strike selection still prefers the cost line:
  - `strike >= cost_basis * (1 - cc_max_discount_to_cost)`
- If no call fits that floor, the selector only relaxes delta once instead of walking a fallback ladder

### Assigned Stock Fail-Safe
- If assigned stock has no covered call for long enough and drawdown exceeds the configured threshold, a single emergency exit sells the configured fraction of shares
