"""Backtest engine: orchestrates strategy execution over historical data."""

import numpy as np
from datetime import datetime, date
from core.backtesting.pricing import OptionsPricer
from core.backtesting.simulator import TradeSimulator, OptionPosition
from core.backtesting.metrics import PerformanceMetrics
from core.backtesting.strategies.base import BaseStrategy
from core.backtesting.strategies.sell_put import SellPutStrategy
from core.backtesting.strategies.covered_call import CoveredCallStrategy
from core.backtesting.strategies.iron_condor import IronCondorStrategy
from core.backtesting.strategies.spreads import BullPutSpreadStrategy, BearCallSpreadStrategy
from core.backtesting.strategies.straddle import StraddleStrategy, StrangleStrategy
from core.backtesting.strategies.wheel import WheelStrategy
from utils.logger import setup_logger

logger = setup_logger("backtest_engine")

STRATEGY_MAP = {
    "sell_put": SellPutStrategy,
    "covered_call": CoveredCallStrategy,
    "iron_condor": IronCondorStrategy,
    "bull_put_spread": BullPutSpreadStrategy,
    "bear_call_spread": BearCallSpreadStrategy,
    "straddle": StraddleStrategy,
    "strangle": StrangleStrategy,
    "wheel": WheelStrategy,
}


class BacktestEngine:
    """Run options strategy backtests using historical stock data + BS model pricing."""

    def __init__(self, data_client=None):
        self._client = data_client

    def run(self, params: dict) -> dict:
        """Execute a backtest and return results.

        params keys: strategy, symbol, start_date, end_date,
                     initial_capital, dte_min, dte_max, delta_target,
                     profit_target_pct, stop_loss_pct
        """
        strategy_name = params["strategy"]
        symbol = params["symbol"]
        start_date = params["start_date"]
        end_date = params["end_date"]
        initial_capital = params.get("initial_capital", 100000)

        # Create strategy instance
        strategy_cls = STRATEGY_MAP.get(strategy_name)
        if not strategy_cls:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        strategy = strategy_cls(params)

        # Fetch historical price data
        bars = self._get_historical_data(symbol, start_date, end_date)
        if not bars:
            raise ValueError(f"No historical data for {symbol}")

        # Estimate historical volatility for IV proxy
        prices = [b["close"] for b in bars]
        hv = self._rolling_hv(prices, window=20)

        # Run simulation
        simulator = TradeSimulator()
        daily_pnl = []
        cumulative_pnl = 0.0
        last_entry_idx = -999  # Track cooldown between entries
        
        # Initialize margin interest tracking
        margin_interest_rate_daily = 0.05 / 365  # 5% annual rate divided by 365 days
        borrowed_funds = 0.0  # Track borrowed funds for leverage

        for i, bar in enumerate(bars):
            bar_date = bar["date"][:10]  # YYYY-MM-DD
            underlying_price = bar["close"]
            iv = hv[i] if i < len(hv) else 0.3

            if iv <= 0.01:
                iv = 0.3  # fallback

            # Check exits
            closed = simulator.check_exits(
                bar_date,
                underlying_price,
                iv,
                strategy.profit_target_pct,
                strategy.stop_loss_pct,
                min_dte=0,
            )
            for trade in closed:
                cumulative_pnl += trade.pnl
                # Notify strategy of closed trade (for stateful strategies like Wheel)
                if hasattr(strategy, 'on_trade_closed'):
                    strategy.on_trade_closed(trade.to_dict())

            # Generate new signals (with cooldown)
            cooldown = int(strategy.dte_min * 0.8)
            if i - last_entry_idx >= cooldown:
                signals = strategy.generate_signals(
                    bar_date, underlying_price, iv, simulator.open_positions
                )
                for sig in signals:
                    pos = OptionPosition(
                        symbol=sig.symbol,
                        entry_date=bar_date,
                        expiry=sig.expiry,
                        strike=sig.strike,
                        right=sig.right,
                        trade_type=sig.trade_type,
                        quantity=sig.quantity,
                        entry_price=sig.premium,
                        underlying_entry=underlying_price,
                        iv_at_entry=sig.iv,
                        delta_at_entry=sig.delta,
                    )
                    simulator.open_position(pos)
                    last_entry_idx = i

            # Apply daily margin interest if leverage is used
            if borrowed_funds > 0:
                daily_interest = borrowed_funds * margin_interest_rate_daily
                cumulative_pnl -= daily_interest  # Subtract interest from P&L
            
            # Daily mark-to-market
            open_pnl = simulator.get_total_open_pnl()
            portfolio_value = initial_capital + cumulative_pnl + open_pnl
            
            # Update strategy daily stats if it supports monitoring
            if hasattr(strategy, 'update_daily_stats'):
                strategy.update_daily_stats(bar_date, portfolio_value, open_pnl)
            
            daily_pnl.append({
                "date": bar_date,
                "cumulative_pnl": cumulative_pnl + open_pnl,
                "closed_pnl": cumulative_pnl,
                "open_pnl": open_pnl,
                "portfolio_value": portfolio_value,
                "margin_interest": daily_interest if borrowed_funds > 0 else 0,
            })

        # Force close remaining positions at end
        if bars:
            last_bar = bars[-1]
            last_date = last_bar["date"][:10]
            last_price = last_bar["close"]
            last_iv = hv[-1] if hv else 0.3
            # Close all remaining positions at the end of backtest
            while simulator.open_positions:
                # Process one position at a time with immediate expiration
                temp_simulator = TradeSimulator()
                pos = simulator.open_positions.pop(0)
                temp_simulator.open_position(pos)
                
                # Force close with min_dte=0 to trigger expiration logic
                closed_trades = temp_simulator.check_exits(
                    last_date, last_price, last_iv,
                    profit_target_pct=9999,
                    stop_loss_pct=9999,
                    min_dte=0,
                )
                
                # Add the closed trades to main simulator
                simulator.closed_trades.extend(closed_trades)
            
            cumulative_pnl = sum(t.pnl for t in simulator.closed_trades)

        # Calculate metrics
        trades = [t.to_dict() for t in simulator.closed_trades]
        metrics = PerformanceMetrics.calculate(trades, daily_pnl, initial_capital)
        
        # Add strategy-specific performance data if available
        strategy_performance = {}
        if hasattr(strategy, 'get_performance_report'):
            strategy_performance = strategy.get_performance_report()

        return {
            "metrics": metrics,
            "trades": trades,
            "daily_pnl": daily_pnl,
            "params": params,
            "strategy_performance": strategy_performance,
        }

    def _get_historical_data(self, symbol: str, start_date: str, end_date: str) -> list[dict]:
        """Fetch historical bars from IBKR or generate synthetic data for testing."""
        if self._client:
            try:
                # Calculate duration
                sd = datetime.strptime(start_date, "%Y-%m-%d")
                ed = datetime.strptime(end_date, "%Y-%m-%d")
                days = (ed - sd).days
                if days <= 365:
                    duration = "1 Y"
                elif days <= 730:
                    duration = "2 Y"
                else:
                    duration = f"{min(days // 365 + 1, 5)} Y"

                bars = self._client.get_historical_bars(symbol, duration, "1 day")
                # Filter to date range
                filtered = [
                    b for b in bars
                    if start_date <= b["date"][:10] <= end_date
                ]
                if filtered:
                    return filtered
            except Exception as e:
                logger.warning("Failed to fetch live data, using synthetic: %s", e)

        # Generate synthetic data for testing when IBKR is not connected
        return self._generate_synthetic_data(symbol, start_date, end_date)

    def _generate_synthetic_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> list[dict]:
        """Generate synthetic price data using geometric Brownian motion."""
        from utils.date_utils import get_trading_days
        sd = datetime.strptime(start_date, "%Y-%m-%d").date()
        ed = datetime.strptime(end_date, "%Y-%m-%d").date()
        trading_days = get_trading_days(sd, ed)

        if not trading_days:
            return []

        np.random.seed(42)
        S0 = 150.0  # starting price
        mu = 0.08 / 252  # daily drift
        sigma = 0.25 / np.sqrt(252)  # daily vol

        prices = [S0]
        for _ in range(len(trading_days) - 1):
            ret = np.random.normal(mu, sigma)
            prices.append(prices[-1] * (1 + ret))

        bars = []
        for i, d in enumerate(trading_days):
            p = prices[i]
            daily_range = p * 0.02 * np.random.random()
            bars.append({
                "date": d.isoformat(),
                "open": round(p - daily_range / 2, 2),
                "high": round(p + daily_range, 2),
                "low": round(p - daily_range, 2),
                "close": round(p, 2),
                "volume": int(np.random.uniform(1e6, 5e6)),
                "average": round(p, 2),
                "barCount": 1000,
            })
        return bars

    def _rolling_hv(self, prices: list[float], window: int = 20) -> list[float]:
        """Calculate rolling historical volatility (annualized)."""
        if len(prices) < 2:
            return [0.3] * len(prices)

        returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                returns.append(np.log(prices[i] / prices[i - 1]))
            else:
                returns.append(0)

        hvs = [0.3]  # default for first bar
        for i in range(len(returns)):
            start = max(0, i - window + 1)
            window_returns = returns[start:i + 1]
            if len(window_returns) >= 5:
                std = np.std(window_returns, ddof=1)
                hvs.append(std * np.sqrt(252))
            else:
                hvs.append(0.3)

        return hvs
