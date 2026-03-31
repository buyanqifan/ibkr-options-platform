# BinbinGod Strategy for QuantConnect

A dynamic stock selection Wheel strategy for MAG7 stocks with ML optimization.

## Files

| File | Description |
|------|-------------|
| `main.py` | Main strategy file |
| `ml_integration.py` | ML integration module |
| `ml_delta_optimizer.py` | Delta optimization using Q-learning |
| `ml_dte_optimizer.py` | DTE optimization using Q-learning |
| `ml_roll_optimizer.py` | Roll decision optimization |
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
    "target_margin_utilization": 0.35,
    "margin_buffer_pct": 0.50,
    "position_aggressiveness": 1.0,
    "ml_enabled": true,
    "stop_loss_pct": 999999,
    "max_risk_per_trade": 0.02
}
```

## Strategy Logic

### Phase 1: Sell Put (SP)
- Sell OTM puts on best-scoring stock
- If expires worthless: keep premium, continue
- If assigned: buy shares, switch to CC phase

### Phase 2: Covered Call (CC)
- Sell OTM calls against held shares
- If expires worthless: keep premium and shares
- If assigned: sell shares, return to SP phase
