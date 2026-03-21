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
from core.backtesting.strategies.binbin_god import BinbinGodStrategy  # New: Binbin God strategy
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
    "binbin_god": BinbinGodStrategy,  # New: Binbin God strategy v0.1.0
}


class BacktestEngine:
    """Run options strategy backtests using historical stock data + BS model pricing."""

    def __init__(self, data_client=None, vol_predictor=None):
        self._client = data_client
        self._vol_predictor = vol_predictor  # ML volatility predictor

    def _get_price_for_symbol(self, symbol: str, date_str: str, mag7_data: dict) -> float | None:
        """Get the closing price for a specific symbol on a specific date.

        Args:
            symbol: Stock symbol
            date_str: Date string in YYYY-MM-DD format
            mag7_data: Dictionary of {symbol: list of bars}

        Returns:
            Closing price or None if not found
        """
        if not mag7_data or symbol not in mag7_data:
            return None

        bars = mag7_data.get(symbol, [])
        for bar in bars:
            bar_date = str(bar.get("date", ""))[:10]
            if bar_date == date_str:
                return bar.get("close")

        # If exact date not found, try to find the closest previous date
        closest_price = None
        for bar in bars:
            bar_date = str(bar.get("date", ""))[:10]
            if bar_date <= date_str:
                closest_price = bar.get("close")
            else:
                break

        return closest_price

    def run(self, params: dict) -> dict:
        """Execute a backtest and return results.

        params keys: strategy, symbol, start_date, end_date,
                     initial_capital, dte_min, dte_max, delta_target,
                     profit_target_pct, stop_loss_pct, use_synthetic_data
        """
        strategy_name = params["strategy"]
        symbol = params["symbol"]
        start_date = params["start_date"]
        end_date = params["end_date"]
        initial_capital = params.get("initial_capital", 100000)
        use_synthetic = params.get("use_synthetic_data", False)  # New parameter

        # Create strategy instance
        strategy_cls = STRATEGY_MAP.get(strategy_name)
        if not strategy_cls:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        strategy = strategy_cls(params)

        # Fetch historical price data (or generate synthetic data)
        # For BinbinGod strategy with auto stock selection, fetch data for all stocks in pool
        if strategy_name == "binbin_god" and "AUTO" in symbol:
            # Use stock_pool from params, or fallback to MAG7_STOCKS
            from core.backtesting.strategies.binbin_god import MAG7_STOCKS
            stock_pool = params.get("stock_pool", MAG7_STOCKS)
            
            # Fetch data for all stocks in pool
            pool_data = {}
            for pool_symbol in stock_pool:
                pool_data[pool_symbol] = self._get_historical_data(
                    pool_symbol, start_date, end_date, use_synthetic=use_synthetic
                )
            
            # Filter out stocks with no data
            valid_stocks = [s for s in stock_pool if pool_data.get(s)]
            if not valid_stocks:
                raise ValueError(f"No historical data for any stock in pool: {stock_pool}")
            
            # Use the first valid stock's data as primary (strategy will switch as needed)
            bars = pool_data[valid_stocks[0]]
            # Store all pool data in strategy for dynamic selection
            strategy.mag7_data = pool_data
            strategy.stock_pool = valid_stocks  # Update strategy's stock pool with validated symbols
            logger.info(f"BinbinGod: Using stock pool {valid_stocks} with {symbol} symbol")
        else:
            bars = self._get_historical_data(symbol, start_date, end_date, use_synthetic=use_synthetic)
        
        if not bars:
            raise ValueError(f"No historical data for {symbol}")

        # Pretrain ML model if enabled (use early portion of historical data)
        if params.get("ml_delta_optimization", False) and hasattr(strategy, 'pretrain_ml_model'):
            # Use first 30% of data for pretraining
            pretrain_size = int(len(bars) * 0.3)
            pretrain_bars = bars[:pretrain_size]
            
            if len(pretrain_bars) >= 60:
                # Estimate IV for pretraining
                pretrain_prices = [b["close"] for b in pretrain_bars]
                pretrain_hv = self._rolling_hv(pretrain_prices, window=20)
                avg_iv = np.mean([v for v in pretrain_hv if v > 0.01]) if pretrain_hv else 0.25
                avg_iv = max(0.15, min(0.50, avg_iv))  # Clamp to reasonable range
                
                pretrain_stats = strategy.pretrain_ml_model(pretrain_bars, iv_estimate=avg_iv)
                logger.info(f"ML Delta pretraining: {pretrain_stats}")

        # Estimate historical volatility for IV proxy
        prices = [b["close"] for b in bars]
        hv = self._rolling_hv(prices, window=20)
        
        # For multi-stock strategies, calculate HV for each stock
        stock_hv = {}  # symbol -> list of HV values
        if strategy_name == "binbin_god" and hasattr(strategy, 'mag7_data'):
            for sym, sym_bars in strategy.mag7_data.items():
                if sym_bars:
                    sym_prices = [b["close"] for b in sym_bars]
                    stock_hv[sym] = self._rolling_hv(sym_prices, window=20)
            logger.info(f"Calculated HV for {len(stock_hv)} stocks in pool")
            # Store stock_hv in strategy so it can access correct IV for each stock
            strategy.stock_hv = stock_hv

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
        
        # Initialize ML roll optimizer if enabled
        ml_roll_optimizer = None
        if params.get("ml_roll_optimization", False):
            from core.ml.roll_optimizer import MLRollOptimizer
            ml_roll_optimizer = MLRollOptimizer()
            logger.info("ML Roll Optimizer initialized")
        
        # Run simulation
        simulator = TradeSimulator()
        daily_pnl = []
        last_entry_idx = -999
        total_commission = 0.0
        total_slippage = 0.0
        
        # Initialize stock position for Covered Call strategy
        # Must own shares before selling calls
        if strategy.name == "covered_call" and bars:
            first_price = bars[0]["close"]
            if hasattr(strategy, 'initialize_stock_position'):
                strategy.initialize_stock_position(first_price)
                logger.info(f"Covered Call: Initialized with {strategy.stock_holding.shares} shares @ ${first_price:.2f}")

        for i, bar in enumerate(bars):
            # Ensure bar_date is string in YYYY-MM-DD format
            bar_date_raw = bar["date"]
            if hasattr(bar_date_raw, 'isoformat'):
                # It's a datetime/date object
                bar_date = bar_date_raw.isoformat()[:10]
            else:
                # It's already a string
                bar_date = str(bar_date_raw)[:10]
            underlying_price = bar["close"]
            
            # Use ML predictor for IV if available and ready
            if self._vol_predictor and self._vol_predictor.is_ready() and i >= 60:
                # Use ML predicted volatility
                recent_bars = bars[max(0, i-60):i+1]
                predicted_vol = self._vol_predictor.predict_from_bars(recent_bars)
                if predicted_vol is not None:
                    iv = predicted_vol / 100  # Convert % to decimal
                    logger.debug(f"Using ML predicted IV: {iv:.3f}")
                else:
                    iv = hv[i] if i < len(hv) else 0.3
            else:
                # Fallback to historical volatility
                iv = hv[i] if i < len(hv) else 0.3

            if iv <= 0.01:
                iv = 0.3  # fallback

            # Check exits and release margin
            # For BinbinGod strategy with multi-stock, check each position with its correct price
            mag7_data = getattr(strategy, 'mag7_data', None)
            is_multi_stock = strategy_name == "binbin_god" and mag7_data

            if is_multi_stock:
                # Multi-stock mode: check each position with its correct underlying price AND IV
                profit_target_to_use = 999999 if strategy._profit_target_disabled else strategy.profit_target_pct
                stop_loss_to_use = 999999 if strategy._stop_loss_disabled else strategy.stop_loss_pct

                closed = []
                remaining_positions = []
                for pos in simulator.open_positions:
                    # Get the correct price for this position's symbol
                    pos_price = self._get_price_for_symbol(pos.symbol, bar_date, mag7_data)
                    if pos_price is None:
                        pos_price = underlying_price  # Fallback to primary stock price
                    
                    # Get the correct IV for this position's symbol
                    pos_iv = iv  # Default to primary stock IV
                    if pos.symbol in stock_hv:
                        sym_hv = stock_hv[pos.symbol]
                        pos_iv = sym_hv[i] if i < len(sym_hv) and sym_hv[i] > 0.01 else 0.3

                    trade = simulator.check_exits_for_position(
                        pos,
                        bar_date,
                        pos_price,
                        pos_iv,
                        profit_target_to_use,
                        stop_loss_to_use,
                        min_dte=0,
                    )
                    if trade:
                        closed.append(trade)
                    else:
                        remaining_positions.append(pos)

                simulator.open_positions = remaining_positions
            # For Wheel strategy SP phase, skip profit target/stop loss - only check assignment at expiry
            elif strategy.name == "wheel":
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

                # Use ML roll optimization if enabled (for Wheel strategies)
                # Note: Roll optimization is handled in strategy.should_exit_position()
                # The simulator just uses standard profit/stop logic as fallback

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
                # Use position_id from trade record (must match the one used during allocation)
                position_id = trade.position_id if trade.position_id else f"{trade.symbol}_{trade.entry_date}_{trade.strike}_{trade.right}"
                position_mgr.release_margin(position_id, adjusted_pnl)  # Use adjusted P&L
                
                # Handle stock capital allocation for Wheel strategy
                # When Put is assigned, we buy shares - the margin should be converted to stock capital
                # This prevents double-counting of available capital
                if trade.exit_reason == "ASSIGNMENT" and trade.trade_type in ("WHEEL_PUT", "BINBIN_PUT"):
                    # Put assigned: allocate capital for stock position
                    shares_acquired = abs(trade.quantity) * 100
                    stock_cost = trade.strike * shares_acquired
                    stock_position_id = f"{trade.symbol}_{bar_date}_STOCK"
                    
                    # Allocate stock capital (this is the money used to buy shares)
                    position_mgr.allocate_margin(
                        position_id=stock_position_id,
                        strategy=strategy.name,
                        symbol=trade.symbol,
                        entry_date=bar_date,
                        margin_amount=stock_cost,
                    )
                    logger.debug(
                        f"Allocated stock capital: {stock_position_id} = ${stock_cost:.2f} "
                        f"({shares_acquired} shares @ ${trade.strike:.2f})"
                    )
                
                # When Call is assigned, we sell shares - release stock capital
                if trade.exit_reason == "ASSIGNMENT" and trade.trade_type in ("WHEEL_CALL", "BINBIN_CALL"):
                    # Call assigned: release stock capital
                    shares_sold = abs(trade.quantity) * 100
                    stock_cost_basis = getattr(strategy, 'stock_holding', None)
                    
                    if stock_cost_basis and hasattr(stock_cost_basis, 'cost_basis'):
                        # Calculate stock capital to release
                        stock_capital = stock_cost_basis.cost_basis * shares_sold
                        
                        # Find and release stock position (release first unreleased stock allocation)
                        for pid, alloc in list(position_mgr.allocations.items()):
                            if "_STOCK" in pid and not alloc.released:
                                position_mgr.release_margin(pid, 0)  # Release with 0 P&L (stock P&L handled separately)
                                logger.debug(f"Released stock capital: {pid}")
                                break
                
                # Update trade record with capital information at exit
                trade.capital_at_exit = position_mgr.net_capital  # Record total capital after closing
                
                # Notify strategy of closed trade (for stateful strategies like Wheel)
                if hasattr(strategy, 'on_trade_closed'):
                    additional_pnl = strategy.on_trade_closed(trade.to_dict())
                    # For Wheel/CoveredCall/BinbinGod strategies: add stock P&L from call assignment
                    if additional_pnl and additional_pnl != 0:
                        position_mgr.cumulative_pnl += additional_pnl
                        # net_capital is a property (initial_capital + cumulative_pnl), auto-updates

                # Update ML performance tracking for learning
                # This enables ML optimizers to learn from actual trade outcomes
                if hasattr(strategy, 'ml_integration') and strategy.ml_integration:
                    try:
                        ml_integration = strategy.ml_integration
                        if hasattr(ml_integration, 'update_performance'):
                            # Determine actual assignment status
                            actual_assignment = trade.exit_reason == "ASSIGNMENT"
                            
                            # Get bars for this symbol up to current date
                            symbol_bars = []
                            if is_multi_stock and mag7_data:
                                symbol_bars = mag7_data.get(trade.symbol, [])
                            
                            ml_integration.update_performance(
                                delta=abs(trade.delta_at_entry) if trade.delta_at_entry else 0.3,
                                symbol=trade.symbol,
                                current_price=trade.underlying_exit,
                                cost_basis=trade.underlying_entry,  # Use entry price as reference
                                bars=symbol_bars,
                                options_data=[],
                                actual_pnl=trade.pnl,
                                actual_assignment=actual_assignment
                            )
                            logger.info(f"Updated ML performance: {trade.symbol} delta={abs(trade.delta_at_entry) if trade.delta_at_entry else 0.3:.2f} pnl={trade.pnl:.2f}")
                    except Exception as e:
                        logger.warning(f"Failed to update ML performance: {e}")

                # Generate roll signal for Wheel-style strategies
                # When profit target is hit, immediately roll to new position
                roll_signal = None
                if hasattr(strategy, 'generate_roll_signal') and trade.exit_reason in ('PROFIT_TARGET', 'ROLL_FORWARD', 'ROLL_OUT'):
                    # Get underlying price for this symbol
                    roll_underlying_price = underlying_price
                    if is_multi_stock:
                        symbol_price = self._get_price_for_symbol(trade.symbol, bar_date, mag7_data)
                        if symbol_price is not None:
                            roll_underlying_price = symbol_price

                    roll_signal = strategy.generate_roll_signal(
                        closed_trade=trade.to_dict(),
                        current_date=bar_date,
                        underlying_price=roll_underlying_price,
                        iv=iv,
                    )

                    if roll_signal:
                        # Open roll position immediately (same day)
                        if "PUT" in roll_signal.trade_type:
                            margin_per_contract = roll_signal.strike * 100
                        elif "CALL" in roll_signal.trade_type:
                            # Covered calls need no margin if shares held
                            has_shares = hasattr(strategy, 'stock_holding') and strategy.stock_holding.shares > 0
                            margin_per_contract = 0 if has_shares else roll_signal.strike * 100
                        else:
                            margin_per_contract = roll_signal.margin_requirement or 0

                        total_margin = abs(roll_signal.quantity) * margin_per_contract
                        roll_position_id = f"{roll_signal.symbol}_{bar_date}_ROLL_{roll_signal.strike}_{roll_signal.right}"

                        if position_mgr.allocate_margin(
                            position_id=roll_position_id,
                            strategy=strategy.name,
                            symbol=roll_signal.symbol,
                            entry_date=bar_date,
                            margin_amount=total_margin,
                        ):
                            roll_pos = OptionPosition(
                                symbol=roll_signal.symbol,
                                entry_date=bar_date,
                                expiry=roll_signal.expiry,
                                strike=roll_signal.strike,
                                right=roll_signal.right,
                                trade_type=roll_signal.trade_type,
                                quantity=roll_signal.quantity,
                                entry_price=roll_signal.premium,
                                underlying_entry=roll_underlying_price,
                                iv_at_entry=roll_signal.iv,
                                delta_at_entry=roll_signal.delta,
                                position_id=roll_position_id,  # Use the same position_id for margin tracking
                                capital_at_entry=position_mgr.net_capital,
                            )
                            simulator.open_position(roll_pos)

                            # Track roll costs
                            roll_cost = cost_model.calculate_total_cost(roll_signal.quantity)
                            total_commission += cost_model.calculate_commission(roll_signal.quantity)
                            total_slippage += cost_model.calculate_slippage(roll_signal.quantity)

                            logger.info(
                                f"Rolled position: {trade.symbol} {trade.exit_reason} -> "
                                f"{roll_signal.symbol} {roll_signal.trade_type} strike={roll_signal.strike:.2f} "
                                f"premium={roll_signal.premium:.2f}"
                            )
                        else:
                            logger.warning(
                                f"Insufficient margin for roll: {roll_signal.symbol} {roll_signal.trade_type}"
                            )

            # Generate new signals (no cooldown for strategies that should trade frequently)
            # Cooldown was causing delays between closing and reopening positions
            # For sell_put strategy, we want to deploy capital as soon as it's available
            signals = strategy.generate_signals(
                bar_date, underlying_price, iv, simulator.open_positions,
                position_mgr=position_mgr
            )
            
            # DEBUG: Log number of signals generated
            if signals:
                logger.info(f"Generated {len(signals)} signals on {bar_date}")
            for sig in signals:
                    # DEBUG: Log each signal being processed
                    logger.info(f"Processing signal: {sig.symbol} {sig.trade_type} {sig.right} strike={sig.strike:.2f} qty={sig.quantity} margin_req={sig.margin_requirement}")
                    
                    # Use strategy-provided margin requirement if available
                    # Note: margin_requirement can be 0 for covered calls (shares already owned)
                    if sig.margin_requirement is not None:
                        # Strategy has provided specific margin requirement (e.g., spreads, straddles, covered calls)
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
                        elif "CALL" in sig.trade_type and "COVERED" not in sig.trade_type and "WHEEL" not in sig.trade_type:
                            # Naked call (not covered call or wheel call)
                            margin_per_contract = underlying_price * 100
                        else:
                            # Covered calls or other: use premium as reference
                            margin_per_contract = sig.premium * 100 * 10
                    
                    total_margin = abs(sig.quantity) * margin_per_contract
                    
                    # Allocate margin before opening position
                    position_id = f"{sig.symbol}_{bar_date}_{sig.strike}_{sig.right}"
                    logger.info(f"Attempting to allocate margin: position_id={position_id}, total_margin=${total_margin:.2f}")
                    if position_mgr.allocate_margin(
                        position_id=position_id,
                        strategy=strategy.name,
                        symbol=sig.symbol,
                        entry_date=bar_date,
                        margin_amount=total_margin,
                    ):
                        logger.info(f"Margin allocated successfully for {position_id}")
                        # Get correct underlying price for this symbol (important for multi-stock strategies)
                        entry_underlying_price = underlying_price  # Default: use primary stock
                        if is_multi_stock:
                            symbol_price = self._get_price_for_symbol(sig.symbol, bar_date, mag7_data)
                            if symbol_price is not None:
                                entry_underlying_price = symbol_price
                                logger.debug(f"Using {sig.symbol} price ${entry_underlying_price:.2f} for entry (vs primary ${underlying_price:.2f})")

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
                            underlying_entry=entry_underlying_price,
                            iv_at_entry=sig.iv,
                            delta_at_entry=sig.delta,
                            position_id=position_id,  # Use the same position_id for margin tracking
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
            
            # For Wheel strategy, also calculate unrealized P&L from stock holdings
            stock_unrealized_pnl = 0.0
            if hasattr(strategy, 'stock_holding') and strategy.stock_holding.shares > 0:
                # Support multi-stock holdings
                holdings = getattr(strategy.stock_holding, 'holdings', None)
                
                if holdings and is_multi_stock:
                    # Multi-stock: calculate each stock's unrealized P&L separately
                    for stock_symbol, holding_info in holdings.items():
                        shares = holding_info.get("shares", 0)
                        cost_basis = holding_info.get("cost_basis", 0)
                        
                        if shares <= 0 or cost_basis <= 0:
                            continue
                        
                        # Get the correct price for this stock
                        stock_price = self._get_price_for_symbol(stock_symbol, bar_date, mag7_data)
                        if stock_price is None:
                            stock_price = underlying_price  # Fallback
                        
                        stock_cost = shares * cost_basis
                        stock_market_value = shares * stock_price
                        stock_unrealized_pnl += stock_market_value - stock_cost
                else:
                    # Single stock (backward compatible)
                    stock_cost = strategy.stock_holding.shares * strategy.stock_holding.cost_basis
                    stock_market_value = strategy.stock_holding.shares * underlying_price
                    stock_unrealized_pnl = stock_market_value - stock_cost
            
            portfolio_value = position_mgr.net_capital + open_pnl + stock_unrealized_pnl
            total_open_pnl = open_pnl + stock_unrealized_pnl
            
            # Update strategy daily stats if it supports monitoring
            if hasattr(strategy, 'update_daily_stats'):
               strategy.update_daily_stats(bar_date, portfolio_value, total_open_pnl)
            
            daily_pnl.append({
                "date": bar_date,
                "cumulative_pnl": position_mgr.cumulative_pnl + total_open_pnl,
                "closed_pnl": position_mgr.cumulative_pnl,
                "open_pnl": total_open_pnl,
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
            
            # For multi-stock strategies, get price/IV for each position's symbol
            mag7_data = getattr(strategy, 'mag7_data', None)
            stock_hv = getattr(strategy, 'stock_hv', {})
            is_multi_stock = strategy_name == "binbin_god" and mag7_data
            
            # Close all remaining positions at the end of backtest
            while simulator.open_positions:
                # Process one position at a time with immediate expiration
                temp_simulator = TradeSimulator()
                pos = simulator.open_positions.pop(0)
                temp_simulator.open_position(pos)
                
                # Get the correct price and IV for this position's symbol
                pos_price = last_price
                pos_iv = last_iv
                if is_multi_stock:
                    # Find the price for this position's symbol
                    symbol_price = self._get_price_for_symbol(pos.symbol, last_date, mag7_data)
                    if symbol_price is not None:
                        pos_price = symbol_price
                    
                    # Get the correct IV for this position's symbol
                    if pos.symbol in stock_hv:
                        sym_hv = stock_hv[pos.symbol]
                        # Get the last available IV for this symbol
                        pos_iv = sym_hv[-1] if sym_hv and sym_hv[-1] > 0.01 else 0.3
                
                # Force close with min_dte=0 to trigger expiration logic
                closed_trades = temp_simulator.check_exits(
                    last_date, pos_price, pos_iv,
                    profit_target_pct=9999,
                    stop_loss_pct=9999,
                    min_dte=0,
                )
                
                # Add the closed trades to main simulator
                simulator.closed_trades.extend(closed_trades)
            
            cumulative_pnl = sum(t.pnl for t in simulator.closed_trades)
            
            # IMPORTANT: Get strategy performance report BEFORE liquidating stock holdings
            # This preserves the final state for UI display (Current Holdings, Current Strategy State)
            strategy_performance = {}
            if hasattr(strategy, 'get_performance_report'):
                strategy_performance = strategy.get_performance_report()
                logger.info(f"Engine: Retrieved strategy_performance - shares={strategy_performance.get('shares_held')}, cost={strategy_performance.get('cost_basis')}")
            
            # Handle remaining stock position for Wheel/BinbinGod strategies
            # At end of backtest, liquidate any remaining stock holdings
            if hasattr(strategy, 'stock_holding') and strategy.stock_holding.shares > 0:
                # Support multi-stock holdings
                holdings = getattr(strategy.stock_holding, 'holdings', {})
                
                if holdings:
                    # Multi-stock: liquidate each stock separately
                    for stock_symbol, holding_info in holdings.items():
                        shares = holding_info.get("shares", 0)
                        cost_basis = holding_info.get("cost_basis", 0)
                        
                        if shares <= 0:
                            continue
                        
                        # Get the correct price for this stock
                        liquidation_price = last_price  # Default
                        if is_multi_stock:
                            symbol_price = self._get_price_for_symbol(stock_symbol, last_date, mag7_data)
                            if symbol_price is not None:
                                liquidation_price = symbol_price
                        
                        # Calculate realized P&L from stock liquidation
                        stock_pnl = (liquidation_price - cost_basis) * shares
                        position_mgr.cumulative_pnl += stock_pnl
                        logger.info(
                            f"Backtest end: Liquidated {shares} shares of {stock_symbol} @ ${liquidation_price:.2f} "
                            f"(cost: ${cost_basis:.2f}), P&L: ${stock_pnl:+.2f}"
                        )
                else:
                    # Fallback to single-stock logic (backward compatibility)
                    shares = strategy.stock_holding.shares
                    cost_basis = strategy.stock_holding.cost_basis
                    stock_symbol = getattr(strategy.stock_holding, 'symbol', '')
                    
                    # Get the correct price for the stock we're holding
                    liquidation_price = last_price  # Default to main stock price
                    if is_multi_stock and stock_symbol:
                        # For multi-stock strategies, get the correct price for the held stock
                        symbol_price = self._get_price_for_symbol(stock_symbol, last_date, mag7_data)
                        if symbol_price is not None:
                            liquidation_price = symbol_price
                            logger.info(f"Using {stock_symbol} price ${liquidation_price:.2f} for stock liquidation (vs main ${last_price:.2f})")
                    
                    # Calculate realized P&L from stock liquidation
                    stock_pnl = (liquidation_price - cost_basis) * shares
                    position_mgr.cumulative_pnl += stock_pnl
                    logger.info(
                        f"Backtest end: Liquidated {shares} shares of {stock_symbol or 'stock'} @ ${liquidation_price:.2f} "
                        f"(cost: ${cost_basis:.2f}), P&L: ${stock_pnl:+.2f}"
                    )
                
                # Release all stock capital allocations
                released_count = 0
                for pid, alloc in list(position_mgr.allocations.items()):
                    if "_STOCK" in pid and not alloc.released:
                        position_mgr.release_margin(pid, 0)  # PnL already added above
                        logger.debug(f"Released stock capital at backtest end: {pid}")
                        released_count += 1
                
                # Reset stock holding
                strategy.stock_holding.shares = 0
                strategy.stock_holding.cost_basis = 0.0
                if hasattr(strategy.stock_holding, 'holdings'):
                    strategy.stock_holding.holdings = {}

        # Calculate metrics
        # Sort trades by exit_date (then entry_date for same exit date) for consistent ordering
        sorted_closed_trades = sorted(
            simulator.closed_trades,
            key=lambda t: (t.exit_date, t.entry_date)
        )
        trades = [t.to_dict() for t in sorted_closed_trades]
        metrics = PerformanceMetrics.calculate(trades, daily_pnl, initial_capital)

        # Train ML roll optimizer if enabled and we have trades
        if ml_roll_optimizer and trades:
            try:
                # For roll optimizer, we would need labeled data with optimal roll decisions
                # This is typically done offline with historical analysis
                logger.info(f"ML Roll Optimizer enabled with {len(trades)} trades")
                # Training would be done separately with labeled optimal roll decisions

            except Exception as e:
                logger.error(f"ML roll optimization error: {e}")
        
        # Merge engine-calculated metrics into strategy_performance
        # This ensures performance_metrics in UI shows real data, not placeholders
        if "performance_metrics" not in strategy_performance:
            strategy_performance["performance_metrics"] = {}
        
        # Update performance_metrics with actual calculated metrics
        strategy_performance["performance_metrics"].update({
            "total_trades": metrics.get("total_trades", 0),
            "winning_trades": metrics.get("winning_trades", 0),
            "losing_trades": metrics.get("losing_trades", 0),
            "total_pnl": metrics.get("total_pnl", 0),
            "win_rate": metrics.get("win_rate", 0),
            "avg_pnl_per_trade": metrics.get("avg_pnl_per_trade", 0),
            "max_drawdown": metrics.get("max_drawdown_pct", 0),
            "profit_factor": metrics.get("profit_factor", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "sortino_ratio": metrics.get("sortino_ratio", 0),
        })
        
        # Add open positions info for UI display (convert OptionPosition to dict)
        if simulator.open_positions:
            open_positions = []
            for pos in simulator.open_positions:
                open_positions.append({
                    "symbol": pos.symbol,
                    "trade_type": pos.trade_type,
                    "right": pos.right,
                    "strike": pos.strike,
                    "expiry": pos.expiry,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "market_value": pos.current_price * abs(pos.quantity) * 100,
                })
            strategy_performance["open_positions"] = open_positions

        # Get underlying price data for timeline chart
        underlying_prices = []
        multi_stock_prices = {}
        if bars:
            underlying_prices = [
                {"date": bar["date"][:10], "close": bar["close"]}
                for bar in bars
            ]
        
        # For BinbinGod strategy, collect prices for all stocks in the pool
        if strategy_name == "binbin_god" and hasattr(strategy, 'mag7_data'):
            stock_pool = getattr(strategy, 'stock_pool', [])
            for symbol in stock_pool:
                stock_bars = strategy.mag7_data.get(symbol, [])
                if stock_bars:
                    multi_stock_prices[symbol] = [
                        {"date": bar["date"][:10], "close": bar["close"]}
                        for bar in stock_bars
                    ]
        
        return {
            "metrics": metrics,
            "trades": trades,
            "daily_pnl": daily_pnl,
            "underlying_prices": underlying_prices,
            "multi_stock_prices": multi_stock_prices,
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

    def _get_historical_data(self, symbol: str, start_date: str, end_date: str, use_synthetic: bool = False) -> list[dict]:
        """Fetch historical bars from IBKR. If use_synthetic=True, generate random data instead.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_synthetic: If True, generate synthetic data instead of fetching from IBKR
            
        Returns:
            List of OHLCV bars
            
        Raises:
            ValueError: If IBKR not connected and use_synthetic=False
        """
        # If synthetic data requested, generate it directly
        if use_synthetic:
            logger.info(f"Generating synthetic data for {symbol} from {start_date} to {end_date}")
            return self._generate_synthetic_data(symbol, start_date, end_date)
        
        # Require real IBKR data
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
        """Generate synthetic price data using geometric Brownian motion.
        
        Uses realistic market parameters for better backtest realism.
        Different date ranges will produce different (but reproducible) price paths.
        """
        from utils.date_utils import get_trading_days
        sd = datetime.strptime(start_date, "%Y-%m-%d").date()
        ed = datetime.strptime(end_date, "%Y-%m-%d").date()
        trading_days = get_trading_days(sd, ed)

        if not trading_days:
            return []

        # Use date-based seed for reproducibility with variation across different periods
        # This ensures same date range always produces same results, but different ranges differ
        date_seed = int(start_date.replace("-", "")) + int(end_date.replace("-", ""))
        np.random.seed(date_seed)
        
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
        
        mu = 0.08 / 252  # daily drift (8% annual return)
        sigma = 0.25 / np.sqrt(252)  # daily vol (25% annual vol)

        # Generate daily close prices using GBM
        prices = [S0]
        for _ in range(len(trading_days) - 1):
            ret = np.random.normal(mu, sigma)
            prices.append(prices[-1] * (1 + ret))

        # Generate OHLCV bars with realistic intraday dynamics
        bars = []
        for i, d in enumerate(trading_days):
            prev_close = prices[i-1] if i > 0 else S0
            curr_close = prices[i]
            
            # Generate realistic open price (gap from previous close)
            gap = np.random.normal(0, sigma * 0.3)  # Small gap effect
            open_price = prev_close * (1 + gap)
            
            # Generate high/low with asymmetric range
            daily_vol = sigma * np.random.lognormal(0, 0.5)  # Volatility clustering
            high_price = max(open_price, curr_close) * (1 + abs(np.random.normal(0, daily_vol)))
            low_price = min(open_price, curr_close) * (1 - abs(np.random.normal(0, daily_vol)))
            
            # Ensure logical consistency: low <= open,close <= high
            low_price = min(low_price, open_price, curr_close)
            high_price = max(high_price, open_price, curr_close)
            
            # Generate volume based on symbol characteristics
            base_volume = self._get_base_volume_for_symbol(symbol)
            volume_shock = np.random.lognormal(0, 0.5)  # Volume varies day to day
            volume = int(base_volume * volume_shock)
            
            bars.append({
                "date": d.isoformat(),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(curr_close, 2),
                "volume": volume,
                "average": round((open_price + high_price + low_price + curr_close) / 4, 2),
                "barCount": int(np.random.uniform(800, 1200)),
            })
        return bars
    
    def _get_base_volume_for_symbol(self, symbol: str) -> int:
        """Get typical daily volume for a symbol."""
        VOLUME_MAP = {
            "NVDA": 50_000_000,   # Very high volume tech stock
            "AAPL": 60_000_000,   # Highest volume
            "MSFT": 30_000_000,
            "TSLA": 80_000_000,   # Extremely high volume
            "GOOGL": 25_000_000,
            "AMZN": 40_000_000,
            "META": 20_000_000,
        }
        return VOLUME_MAP.get(symbol.upper(), 10_000_000)  # Default 10M

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
