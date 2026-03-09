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
from core.backtesting.position_manager import PositionManager
from core.backtesting.cost_model import TradingCostModel  # New: Trading cost model
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

        # Initialize position manager for capital allocation and margin tracking
        position_mgr = PositionManager(
            initial_capital=initial_capital,
            max_leverage=params.get("max_leverage", 1.0),
            position_percentage=params.get("position_percentage", 0.10),
            margin_interest_rate=0.05,
        )
        
        # Initialize cost model for trading costs
        cost_model = TradingCostModel(
            commission_per_contract=params.get("commission_per_contract", 0.65),
            commission_min=params.get("commission_min", 1.00),
            slippage_per_contract=params.get("slippage_per_contract", 0.05),
        )
        
        # Run simulation
        simulator = TradeSimulator()
        daily_pnl = []
        last_entry_idx = -999  # Track cooldown between entries
        total_commission = 0.0
        total_slippage = 0.0

        for i, bar in enumerate(bars):
            bar_date = bar["date"][:10]  # YYYY-MM-DD
            underlying_price = bar["close"]
            iv = hv[i] if i < len(hv) else 0.3

            if iv <= 0.01:
                iv = 0.3  # fallback

            # Check exits and release margin
            # For Wheel strategy SP phase, skip profit target/stop loss - only check assignment at expiry
            if strategy.name == "wheel":
                # Check which phase the Wheel strategy is in
                wheel_phase = getattr(strategy, 'phase', 'SP')  # Default to SP if not found
                
                if wheel_phase == "SP":
                    # Sell Put phase: disable profit target/stop loss (want assignment)
                    closed = simulator.check_exits(
                        bar_date,
                        underlying_price,
                        iv,
                        profit_target_pct=999999,  # Disable profit target
                        stop_loss_pct=999999,      # Disable stop loss
                        min_dte=0,
                    )
                else:  # CC phase
                    # Covered Call phase: use normal profit target/stop loss (unless disabled by user)
                    # Check if user explicitly disabled profit target/stop loss
                    profit_target_to_use = 999999 if strategy._profit_target_disabled else strategy.profit_target_pct
                    stop_loss_to_use = 999999 if strategy._stop_loss_disabled else strategy.stop_loss_pct
                    
                    closed = simulator.check_exits(
                        bar_date,
                        underlying_price,
                        iv,
                        profit_target_to_use,
                        stop_loss_to_use,
                        min_dte=0,
                    )
            else:
                # Normal strategies use configured profit target and stop loss
                # Unless user explicitly disabled them
                profit_target_to_use = 999999 if strategy._profit_target_disabled else strategy.profit_target_pct
                stop_loss_to_use = 999999 if strategy._stop_loss_disabled else strategy.stop_loss_pct
                
                closed = simulator.check_exits(
                    bar_date,
                    underlying_price,
                    iv,
                    profit_target_to_use,
                    stop_loss_to_use,
                    min_dte=0,
                )
            for trade in closed:
                # Calculate exit trading costs
                exit_cost = cost_model.calculate_total_cost(-trade.quantity)  # Opposite sign for closing
                total_commission += cost_model.calculate_commission(-trade.quantity)
                total_slippage += cost_model.calculate_slippage(-trade.quantity)
                
                # Adjust P&L for round-trip costs (entry + exit already calculated)
                # The trade.pnl is already calculated by simulator, we just need to subtract exit costs
                adjusted_pnl = trade.pnl - exit_cost
                
                # Create position ID and release margin
                position_id = f"{trade.symbol}_{trade.entry_date}_{trade.strike}_{trade.right}"
                position_mgr.release_margin(position_id, adjusted_pnl)  # Use adjusted P&L
                
                # Update trade record with capital information at exit
                trade.capital_at_exit = position_mgr.net_capital  # Record total capital after closing
                
                # Notify strategy of closed trade (for stateful strategies like Wheel)
                if hasattr(strategy, 'on_trade_closed'):
                    strategy.on_trade_closed(trade.to_dict())

            # Generate new signals (no cooldown for strategies that should trade frequently)
            # Cooldown was causing delays between closing and reopening positions
            # For sell_put strategy, we want to deploy capital as soon as it's available
            signals = strategy.generate_signals(
                bar_date, underlying_price, iv, simulator.open_positions,
                position_mgr=position_mgr
            )
            for sig in signals:
                    # Use strategy-provided margin requirement if available
                    if sig.margin_requirement is not None and sig.margin_requirement > 0:
                        # Strategy has provided specific margin requirement (e.g., spreads, straddles)
                        margin_per_contract = sig.margin_requirement
                        logger.debug(
                            f"Using strategy-provided margin for {sig.trade_type}: "
                            f"${margin_per_contract:.2f}"
                        )
                    else:
                        # Fallback to legacy calculation for simple strategies
                        if "PUT" in sig.trade_type:
                            # Cash-secured put or naked put
                            margin_per_contract = sig.strike * 100
                        elif "CALL" in sig.trade_type and "COVERED" not in sig.trade_type:
                            # Naked call
                            margin_per_contract = underlying_price * 100
                        else:
                            # Covered calls or other: use premium as reference
                            margin_per_contract = sig.premium * 100 * 10
                    
                    total_margin = abs(sig.quantity) * margin_per_contract
                    
                    # Allocate margin before opening position
                    position_id = f"{sig.symbol}_{bar_date}_{sig.strike}_{sig.right}"
                    if position_mgr.allocate_margin(
                        position_id=position_id,
                        strategy=strategy.name,
                        symbol=sig.symbol,
                        entry_date=bar_date,
                        margin_amount=total_margin,
                    ):
                        # Margin allocated successfully, open position
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
                            capital_at_entry=position_mgr.net_capital,  # Record total capital at entry
                        )
                        simulator.open_position(pos)
                        
                        # Calculate and track trading costs (commission + slippage)
                        entry_cost = cost_model.calculate_total_cost(sig.quantity)
                        total_commission += cost_model.calculate_commission(sig.quantity)
                        total_slippage += cost_model.calculate_slippage(sig.quantity)
                        
                        logger.debug(
                            f"Opened {sig.symbol} {sig.trade_type}: "
                            f"premium={sig.premium:.2f}, cost={entry_cost:.2f}"
                        )
                        
                        last_entry_idx = i
                    else:
                        logger.warning(
                            f"Insufficient margin for {sig.symbol} {sig.trade_type}: "
                            f"required {total_margin:.2f}, available {position_mgr.available_margin:.2f}"
                        )

            # Apply daily margin interest on borrowed funds
            daily_interest = position_mgr.apply_daily_interest()
            
            # Daily mark-to-market
            open_pnl = simulator.get_total_open_pnl()
            portfolio_value = position_mgr.net_capital + open_pnl
            
            # Update strategy daily stats if it supports monitoring
            if hasattr(strategy, 'update_daily_stats'):
                strategy.update_daily_stats(bar_date, portfolio_value, open_pnl)
            
            daily_pnl.append({
                "date": bar_date,
                "cumulative_pnl": position_mgr.cumulative_pnl + open_pnl,
                "closed_pnl": position_mgr.cumulative_pnl,
                "open_pnl": open_pnl,
                "portfolio_value": portfolio_value,
                "margin_interest": daily_interest,
                "margin_used": position_mgr.total_margin_used,
                "available_margin": position_mgr.available_margin,
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

        # Get underlying price data for timeline chart
        underlying_prices = []
        if bars:
            underlying_prices = [
                {"date": bar["date"][:10], "close": bar["close"]}
                for bar in bars
            ]
        
        return {
            "metrics": metrics,
            "trades": trades,
            "daily_pnl": daily_pnl,
            "underlying_prices": underlying_prices,
            "params": params,
            "strategy_performance": strategy_performance,
            "trading_costs": {  # New: Trading cost breakdown
                "total_commission": round(total_commission, 2),
                "total_slippage": round(total_slippage, 2),
                "total_costs": round(total_commission + total_slippage, 2),
                "commission_rate": cost_model.commission_per_contract,
                "slippage_rate": cost_model.slippage_per_contract,
            },
        }

    def _get_historical_data(self, symbol: str, start_date: str, end_date: str) -> list[dict]:
        """Fetch historical bars from IBKR. Raises error if data not available."""
        if not self._client:
            raise ValueError(
                f"IBKR client not connected. Cannot fetch historical data for {symbol}. "
                "Please ensure IB Gateway/TWS is running and connected."
            )
        
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
            if not filtered:
                raise ValueError(
                    f"No historical data available for {symbol} in the range "
                    f"{start_date} to {end_date}. Please check:\n"
                    f"1. IB Gateway/TWS connection status\n"
                    f"2. Whether the symbol '{symbol}' exists\n"
                    f"3. The requested date range"
                )
            
            return filtered
            
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(
                f"Failed to fetch historical data for {symbol}: {str(e)}\n"
                "Real-time data connection required. Please check IB Gateway/TWS status."
            )

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
        
        # Use realistic starting prices for common symbols (as of 2026)
        # These are approximate real-world prices for better backtest realism
        SYMBOL_START_PRICES = {
            "NVDA": 800.0,   # ~2026 expected range: $700-900
            "AAPL": 220.0,   # ~2026 expected range: $200-250
            "MSFT": 500.0,   # ~2026 expected range: $450-550
            "TSLA": 350.0,   # ~2026 expected range: $300-400
            "GOOGL": 200.0,  # ~2026 expected range: $180-220
            "AMZN": 220.0,   # ~2026 expected range: $200-250
            "META": 600.0,   # ~2026 expected range: $550-650
        }
        
        S0 = SYMBOL_START_PRICES.get(symbol.upper(), 150.0)  # Default to $150 if unknown
        
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
