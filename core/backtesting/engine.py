"""Backtest engine: orchestrates strategy execution over historical data."""

import numpy as np
from datetime import datetime, date
from core.backtesting.pricing import OptionsPricer
from core.backtesting.simulator import TradeSimulator, OptionPosition, TradeRecord
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
from core.backtesting.qc_parity import (
    BinbinGodParityConfig,
    EventTracer,
    calculate_dynamic_max_positions_from_prices,
)
from core.backtesting.qc_trace_adapter import adapt_qc_trace
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

    def _build_parity_context(
        self,
        strategy,
        position_mgr: PositionManager,
        simulator: TradeSimulator,
        bar_date: str,
        pool_data: dict,
        fallback_price: float,
        dynamic_max_positions: int,
    ) -> dict:
        """Build a QC-style portfolio snapshot for parity sizing."""
        price_by_symbol = {}
        stock_holdings_value = 0.0
        for symbol in getattr(strategy, "stock_pool", []):
            price = self._get_price_for_symbol(symbol, bar_date, pool_data) or fallback_price
            price_by_symbol[symbol] = price
        for symbol, holding in getattr(strategy.stock_holding, "holdings", {}).items():
            price = price_by_symbol.get(symbol, fallback_price)
            stock_holdings_value += holding["shares"] * price

        open_option_pnl = simulator.get_total_open_pnl()
        portfolio_value = position_mgr.net_capital + stock_holdings_value + open_option_pnl
        margin_remaining = max(0.0, portfolio_value * position_mgr.max_leverage - position_mgr.total_margin_used)
        return {
            "portfolio_value": portfolio_value,
            "margin_remaining": margin_remaining,
            "total_margin_used": position_mgr.total_margin_used,
            "stock_holdings_value": stock_holdings_value,
            "stock_holding_count": len(getattr(strategy.stock_holding, "holdings", {})),
            "price_by_symbol": price_by_symbol,
            "dynamic_max_positions": dynamic_max_positions,
        }

    def _open_signal_position(
        self,
        strategy,
        signal,
        simulator: TradeSimulator,
        position_mgr: PositionManager,
        cost_model: TradingCostModel,
        tracer: EventTracer,
        bar_date: str,
        entry_underlying_price: float,
    ) -> bool:
        """Open a parity-mode option position."""
        if signal.confidence < strategy.ml_min_confidence:
            tracer.record(
                bar_date,
                "order_deferred",
                symbol=signal.symbol,
                action="SELL_PUT" if signal.right == "P" else "SELL_CALL",
                right=signal.right,
                qty=abs(signal.quantity),
                reason="below_confidence_gate",
                confidence=signal.confidence,
            )
            return False

        margin_per_contract = signal.margin_requirement if signal.margin_requirement is not None else signal.strike * 100
        total_margin = abs(signal.quantity) * margin_per_contract
        position_id = f"{signal.symbol}_{bar_date}_{signal.strike}_{signal.right}"
        if not position_mgr.allocate_margin(
            position_id=position_id,
            strategy=strategy.name,
            symbol=signal.symbol,
            entry_date=bar_date,
            margin_amount=total_margin,
        ):
            tracer.record(
                bar_date,
                "order_deferred",
                symbol=signal.symbol,
                action="SELL_PUT" if signal.right == "P" else "SELL_CALL",
                right=signal.right,
                qty=abs(signal.quantity),
                reason="insufficient_margin",
            )
            return False

        entry_commission = cost_model.calculate_commission(signal.quantity)
        entry_slippage = cost_model.calculate_slippage(signal.quantity)
        pos = OptionPosition(
            symbol=signal.symbol,
            entry_date=bar_date,
            expiry=signal.expiry,
            strike=signal.strike,
            right=signal.right,
            trade_type=signal.trade_type,
            quantity=signal.quantity,
            entry_price=signal.premium,
            underlying_entry=entry_underlying_price,
            iv_at_entry=signal.iv,
            delta_at_entry=signal.delta,
            capital_at_entry=position_mgr.net_capital,
            position_id=position_id,
            strategy_phase=getattr(signal, "strategy_phase", "SP"),
            entry_commission=entry_commission,
            entry_slippage=entry_slippage,
        )
        simulator.open_position(pos)
        position_mgr.cumulative_pnl -= (entry_commission + entry_slippage)
        tracer.record(
            bar_date,
            "contract_selected",
            symbol=signal.symbol,
            action="SELL_PUT" if signal.right == "P" else "SELL_CALL",
            right=signal.right,
            strike=signal.strike,
            expiry=signal.expiry,
            qty=abs(signal.quantity),
        )
        tracer.record(
            bar_date,
            "order_opened",
            symbol=signal.symbol,
            action="SELL_PUT" if signal.right == "P" else "SELL_CALL",
            right=signal.right,
            strike=signal.strike,
            expiry=signal.expiry,
            qty=abs(signal.quantity),
            confidence=signal.confidence,
        )
        return True

    def _process_closed_trades_parity(
        self,
        strategy,
        closed: list,
        simulator: TradeSimulator,
        position_mgr: PositionManager,
        cost_model: TradingCostModel,
        tracer: EventTracer,
        bar_date: str,
        underlying_price: float,
        iv: float,
        pool_data: dict,
        allow_roll: bool,
    ) -> list[str]:
        """Apply QC-style bookkeeping for closed trades."""
        assigned_put_symbols: list[str] = []
        for trade in closed:
            exit_cost = cost_model.calculate_total_cost(-trade.quantity)
            adjusted_pnl = trade.pnl - exit_cost
            position_id = trade.position_id or f"{trade.symbol}_{trade.entry_date}_{trade.strike}_{trade.right}"
            position_mgr.release_margin(position_id, adjusted_pnl)

            event_type = "expired"
            if trade.exit_reason == "ASSIGNMENT" and trade.right == "P":
                event_type = "assigned_put"
            elif trade.exit_reason == "ASSIGNMENT" and trade.right == "C":
                event_type = "assigned_call"
            elif trade.exit_reason in ("PROFIT_TARGET", "ROLL_FORWARD", "ROLL_OUT"):
                event_type = "rolled"

            tracer.record(
                bar_date,
                event_type,
                symbol=trade.symbol,
                right=trade.right,
                qty=abs(trade.quantity),
                strike=trade.strike,
                exit_reason=trade.exit_reason,
                assignment=trade.exit_reason == "ASSIGNMENT",
            )

            if trade.exit_reason == "ASSIGNMENT" and trade.trade_type == "BINBIN_PUT":
                additional_pnl = strategy.on_trade_closed(trade.to_dict()) if hasattr(strategy, "on_trade_closed") else 0.0
                if additional_pnl:
                    position_mgr.cumulative_pnl += additional_pnl
                    trade.pnl += additional_pnl
                shares_acquired = abs(trade.quantity) * 100
                stock_cost = trade.strike * shares_acquired
                stock_position_id = f"{trade.symbol}_{bar_date}_STOCK"
                position_mgr.allocate_margin(
                    position_id=stock_position_id,
                    strategy=strategy.name,
                    symbol=trade.symbol,
                    entry_date=bar_date,
                    margin_amount=stock_cost,
                )
                tracer.record(
                    bar_date,
                    "stock_opened",
                    symbol=trade.symbol,
                    qty=shares_acquired,
                    strike=trade.strike,
                    portfolio_pnl_effect=0.0,
                )
                assigned_put_symbols.append(trade.symbol)
            elif trade.exit_reason == "ASSIGNMENT" and trade.trade_type == "BINBIN_CALL":
                stock_pnl = strategy.on_trade_closed(trade.to_dict()) if hasattr(strategy, "on_trade_closed") else 0.0
                if stock_pnl:
                    position_mgr.cumulative_pnl += stock_pnl
                    trade.pnl += stock_pnl
                shares_sold = abs(trade.quantity) * 100
                for pid, alloc in list(position_mgr.allocations.items()):
                    if pid.startswith(f"{trade.symbol}_") and "_STOCK" in pid and not alloc.released:
                        position_mgr.release_margin(pid, 0)
                        break
                tracer.record(
                    bar_date,
                    "stock_closed",
                    symbol=trade.symbol,
                    qty=shares_sold,
                    strike=trade.strike,
                    portfolio_pnl_effect=round(stock_pnl, 2),
                )
            elif hasattr(strategy, "on_trade_closed"):
                additional_pnl = strategy.on_trade_closed(trade.to_dict())
                if additional_pnl:
                    position_mgr.cumulative_pnl += additional_pnl
                    trade.pnl += additional_pnl

            if allow_roll and hasattr(strategy, "generate_roll_signal") and trade.exit_reason in ("PROFIT_TARGET", "ROLL_FORWARD", "ROLL_OUT"):
                symbol_price = self._get_price_for_symbol(trade.symbol, bar_date, pool_data) or underlying_price
                roll_signal = strategy.generate_roll_signal(
                    closed_trade=trade.to_dict(),
                    current_date=bar_date,
                    underlying_price=symbol_price,
                    iv=iv,
                )
                if roll_signal:
                    self._open_signal_position(
                        strategy=strategy,
                        signal=roll_signal,
                        simulator=simulator,
                        position_mgr=position_mgr,
                        cost_model=cost_model,
                        tracer=tracer,
                        bar_date=bar_date,
                        entry_underlying_price=symbol_price,
                    )

        return assigned_put_symbols

    def _run_binbin_god_qc_parity(
        self,
        strategy,
        params: dict,
        bars: list[dict],
        hv: list[float],
        pool_data: dict,
    ) -> dict:
        """Run BinbinGod using QC-style parity timing and bookkeeping."""
        initial_capital = params.get("initial_capital", 100000)
        warmup_bars = 60
        warmup_pretrained = False
        position_mgr = PositionManager(
            initial_capital=initial_capital,
            max_leverage=params.get("max_leverage", 1.0),
            position_percentage=params.get("position_percentage", 0.10),
            margin_interest_rate=0.05,
        )
        cost_model = TradingCostModel(
            commission_per_contract=params.get("commission_per_contract", 0.65),
            commission_min=params.get("commission_min", 1.00),
            slippage_per_contract=params.get("slippage_per_contract", 0.05),
        )
        simulator = TradeSimulator()
        tracer = EventTracer()
        strategy.set_event_recorder(tracer)
        daily_pnl = []
        total_commission = 0.0
        total_slippage = 0.0

        for i, bar in enumerate(bars):
            bar_date = str(bar["date"])[:10]
            underlying_price = bar["close"]
            iv = hv[i] if i < len(hv) else 0.3
            if iv <= 0.01:
                iv = 0.3

            price_series = [
                self._get_price_for_symbol(symbol, bar_date, pool_data) or underlying_price
                for symbol in getattr(strategy, "stock_pool", [])
            ]
            dynamic_max_positions = calculate_dynamic_max_positions_from_prices(price_series, strategy.parity_config)
            strategy.max_positions = dynamic_max_positions
            parity_context = self._build_parity_context(
                strategy=strategy,
                position_mgr=position_mgr,
                simulator=simulator,
                bar_date=bar_date,
                pool_data=pool_data,
                fallback_price=underlying_price,
                dynamic_max_positions=dynamic_max_positions,
            )
            strategy.set_parity_context(parity_context)
            tracer.snapshot(
                bar_date,
                "rebalance_snapshot",
                portfolio_value=round(parity_context["portfolio_value"], 2),
                margin_used=round(parity_context["total_margin_used"], 2),
                margin_remaining=round(parity_context["margin_remaining"], 2),
                dynamic_max_positions=dynamic_max_positions,
            )

            if i < warmup_bars:
                tracer.snapshot(
                    bar_date,
                    "warmup_snapshot",
                    progress=i + 1,
                    required_bars=warmup_bars,
                )
                daily_pnl.append(
                    {
                        "date": bar_date,
                        "cumulative_pnl": 0.0,
                        "closed_pnl": 0.0,
                        "open_pnl": 0.0,
                        "portfolio_value": initial_capital,
                        "margin_interest": 0.0,
                        "margin_used": position_mgr.total_margin_used,
                        "available_margin": position_mgr.available_margin,
                    }
                )
                continue

            if (
                not warmup_pretrained
                and params.get("ml_delta_optimization", False)
                and hasattr(strategy, "pretrain_ml_model")
            ):
                warmup_history = bars[:warmup_bars]
                warmup_iv = [value for value in hv[:warmup_bars] if value and value > 0.01]
                avg_iv = float(np.mean(warmup_iv)) if warmup_iv else 0.25
                avg_iv = max(0.15, min(0.50, avg_iv))
                pretrain_stats = strategy.pretrain_ml_model(
                    warmup_history,
                    iv_estimate=avg_iv,
                )
                logger.info(f"ML Delta pretraining after warmup: {pretrain_stats}")
                warmup_pretrained = True

            managed, still_open = simulator.check_position_management(
                current_date=bar_date,
                price_lookup=parity_context["price_by_symbol"],
                iv=iv,
                profit_target_pct=strategy.profit_target_pct,
                stop_loss_pct=999999,
            )
            simulator.open_positions = still_open
            assigned_symbols = self._process_closed_trades_parity(
                strategy=strategy,
                closed=managed,
                simulator=simulator,
                position_mgr=position_mgr,
                cost_model=cost_model,
                tracer=tracer,
                bar_date=bar_date,
                underlying_price=underlying_price,
                iv=iv,
                pool_data=pool_data,
                allow_roll=True,
            )

            strategy.set_parity_context(
                self._build_parity_context(
                    strategy=strategy,
                    position_mgr=position_mgr,
                    simulator=simulator,
                    bar_date=bar_date,
                    pool_data=pool_data,
                    fallback_price=underlying_price,
                    dynamic_max_positions=dynamic_max_positions,
                )
            )
            signals = strategy.generate_signals(bar_date, underlying_price, iv, simulator.open_positions, position_mgr=position_mgr)
            for signal in signals:
                entry_price = self._get_price_for_symbol(signal.symbol, bar_date, pool_data) or underlying_price
                opened = self._open_signal_position(
                    strategy=strategy,
                    signal=signal,
                    simulator=simulator,
                    position_mgr=position_mgr,
                    cost_model=cost_model,
                    tracer=tracer,
                    bar_date=bar_date,
                    entry_underlying_price=entry_price,
                )
                if opened:
                    total_commission += cost_model.calculate_commission(signal.quantity)
                    total_slippage += cost_model.calculate_slippage(signal.quantity)

            expired, still_open = simulator.check_expiries(
                current_date=bar_date,
                price_lookup=self._build_parity_context(
                    strategy=strategy,
                    position_mgr=position_mgr,
                    simulator=simulator,
                    bar_date=bar_date,
                    pool_data=pool_data,
                    fallback_price=underlying_price,
                    dynamic_max_positions=dynamic_max_positions,
                )["price_by_symbol"],
                iv=iv,
            )
            simulator.open_positions = still_open
            assigned_symbols.extend(
                self._process_closed_trades_parity(
                    strategy=strategy,
                    closed=expired,
                    simulator=simulator,
                    position_mgr=position_mgr,
                    cost_model=cost_model,
                    tracer=tracer,
                    bar_date=bar_date,
                    underlying_price=underlying_price,
                    iv=iv,
                    pool_data=pool_data,
                    allow_roll=False,
                )
            )

            for symbol in assigned_symbols:
                symbol_price = self._get_price_for_symbol(symbol, bar_date, pool_data) or underlying_price
                immediate_signal = strategy.generate_immediate_cc_signal(
                    symbol=symbol,
                    current_date=bar_date,
                    underlying_price=symbol_price,
                    iv=iv,
                    open_positions=simulator.open_positions,
                    position_mgr=position_mgr,
                )
                if immediate_signal:
                    opened = self._open_signal_position(
                        strategy=strategy,
                        signal=immediate_signal,
                        simulator=simulator,
                        position_mgr=position_mgr,
                        cost_model=cost_model,
                        tracer=tracer,
                        bar_date=bar_date,
                        entry_underlying_price=symbol_price,
                    )
                    if opened:
                        total_commission += cost_model.calculate_commission(immediate_signal.quantity)
                        total_slippage += cost_model.calculate_slippage(immediate_signal.quantity)

            daily_interest = position_mgr.apply_daily_interest()
            stock_unrealized_pnl = 0.0
            for symbol, holding in getattr(strategy.stock_holding, "holdings", {}).items():
                symbol_price = self._get_price_for_symbol(symbol, bar_date, pool_data) or underlying_price
                stock_unrealized_pnl += holding["shares"] * (symbol_price - holding["cost_basis"])
            open_pnl = simulator.get_total_open_pnl() + stock_unrealized_pnl
            portfolio_value = position_mgr.net_capital + open_pnl
            tracer.snapshot(
                bar_date,
                "end_of_day_valuation",
                portfolio_value=round(portfolio_value, 2),
                open_pnl=round(open_pnl, 2),
                closed_pnl=round(position_mgr.cumulative_pnl, 2),
                margin_interest=round(daily_interest, 2),
            )
            daily_pnl.append(
                {
                    "date": bar_date,
                    "cumulative_pnl": position_mgr.cumulative_pnl + open_pnl,
                    "closed_pnl": position_mgr.cumulative_pnl,
                    "open_pnl": open_pnl,
                    "portfolio_value": portfolio_value,
                    "margin_interest": daily_interest,
                    "margin_used": position_mgr.total_margin_used,
                    "available_margin": position_mgr.available_margin,
                }
            )

        if bars:
            last_bar = bars[-1]
            last_date = str(last_bar["date"])[:10]
            last_price = last_bar["close"]
            last_iv = hv[-1] if hv else 0.3
            while simulator.open_positions:
                pos = simulator.open_positions.pop(0)
                temp = TradeSimulator()
                temp.open_position(pos)
                closed_trades = temp.check_exits(last_date, last_price, last_iv, 9999, 9999, min_dte=0)
                simulator.closed_trades.extend(closed_trades)

        trades = [t.to_dict() for t in simulator.closed_trades]
        metrics = PerformanceMetrics.calculate(trades, daily_pnl, initial_capital)
        strategy_performance = strategy.get_performance_report() if hasattr(strategy, "get_performance_report") else {}

        qc_trace_source = params.get("qc_trace") or params.get("qc_trace_path")
        qc_trace = adapt_qc_trace(qc_trace_source) if qc_trace_source else {"event_trace": [], "portfolio_snapshots": []}
        parity_report = tracer.build_parity_report(
            expected_trace=qc_trace.get("event_trace"),
            expected_snapshots=qc_trace.get("portfolio_snapshots"),
        )

        underlying_prices = [{"date": str(bar["date"])[:10], "close": bar["close"]} for bar in bars]
        multi_stock_prices = {}
        for symbol in getattr(strategy, "stock_pool", []):
            stock_bars = pool_data.get(symbol, [])
            if stock_bars:
                multi_stock_prices[symbol] = [{"date": str(bar["date"])[:10], "close": bar["close"]} for bar in stock_bars]

        return {
            "metrics": metrics,
            "trades": trades,
            "daily_pnl": daily_pnl,
            "underlying_prices": underlying_prices,
            "multi_stock_prices": multi_stock_prices,
            "params": params,
            "strategy_performance": strategy_performance,
            "trading_costs": {
                "total_commission": round(total_commission, 2),
                "total_slippage": round(total_slippage, 2),
                "total_costs": round(total_commission + total_slippage, 2),
                "commission_rate": cost_model.commission_per_contract,
                "slippage_rate": cost_model.slippage_per_contract,
            },
            "event_trace": tracer.event_trace,
            "portfolio_snapshots": tracer.portfolio_snapshots,
            "parity_report": parity_report,
            "qc_trace": qc_trace,
        }

    def run(self, params: dict) -> dict:
        """Execute a backtest and return results.

        params keys: strategy, symbol, start_date, end_date,
                     initial_capital, dte_min, dte_max, delta_target,
                     profit_target_pct, stop_loss_pct, use_synthetic_data
        """
        strategy_name = params["strategy"]
        parity_config = None
        if strategy_name == "binbin_god":
            parity_config = BinbinGodParityConfig.from_params(params)
            params = parity_config.apply_to_params(params)
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

        # Pretrain ML model if enabled (use early portion of historical data).
        # Binbin God QC replay performs this after its QC-style warmup instead.
        if (
            strategy_name != "binbin_god"
            and params.get("ml_delta_optimization", False)
            and hasattr(strategy, 'pretrain_ml_model')
        ):
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
        if strategy_name == "binbin_god" and "AUTO" in symbol:
            stock_hv = {}
            for pool_symbol in valid_stocks:
                pool_bars = pool_data.get(pool_symbol, [])
                if pool_bars:
                    pool_prices = [b["close"] for b in pool_bars]
                    stock_hv[pool_symbol] = self._rolling_hv(pool_prices, window=20)
            strategy.stock_hv = stock_hv
            logger.info(f"BinbinGod: Calculated HV for {list(stock_hv.keys())}")

        if strategy_name == "binbin_god" and parity_config:
            return self._run_binbin_god_qc_parity(
                strategy=strategy,
                params=params,
                bars=bars,
                hv=hv,
                pool_data=getattr(strategy, "mag7_data", {}),
            )

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
                # Multi-stock mode: check each position with its correct underlying price
                # For Wheel-like strategies (binbin_god):
                # - Disable stop loss: hold through drawdowns, let time work for us
                # - Enable profit target: roll to new position when profitable
                profit_target_to_use = strategy.profit_target_pct  # Enable profit target for rolling
                stop_loss_to_use = 999999      # Disable stop loss for Wheel-like strategies

                closed = []
                remaining_positions = []
                for pos in simulator.open_positions:
                    # Get the correct price for this position's symbol
                    pos_price = self._get_price_for_symbol(pos.symbol, bar_date, mag7_data)
                    if pos_price is None:
                        pos_price = underlying_price  # Fallback to primary stock price

                    trade = simulator.check_exits_for_position(
                        pos,
                        bar_date,
                        pos_price,
                        iv,
                        profit_target_to_use,
                        stop_loss_to_use,
                        min_dte=0,
                    )
                    if trade:
                        closed.append(trade)
                    else:
                        remaining_positions.append(pos)

                simulator.open_positions = remaining_positions
            # For Wheel strategy SP phase, skip stop loss - only check assignment at expiry
            # But enable profit target for rolling
            elif strategy.name == "wheel":
                # Check which phase the Wheel strategy is in
                wheel_phase = getattr(strategy, 'phase', 'SP')  # Default to SP if not found
                
                # For Wheel strategy:
                # - Disable stop loss: hold through drawdowns
                # - Enable profit target: roll when profitable
                closed = simulator.check_exits(
                    bar_date,
                    underlying_price,
                    iv,
                    profit_target_pct=strategy.profit_target_pct,  # Enable profit target for rolling
                    stop_loss_pct=999999,  # Disable stop loss
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
                # Use the position_id from trade record if available (more reliable)
                # For ROLL positions, position_id contains "_ROLL_" which is different from reconstructed ID
                if hasattr(trade, 'position_id') and trade.position_id:
                    position_id = trade.position_id
                else:
                    position_id = f"{trade.symbol}_{trade.entry_date}_{trade.strike}_{trade.right}"
                
                release_success = position_mgr.release_margin(position_id, adjusted_pnl)  # Use adjusted P&L
                if not release_success:
                    logger.error(f"Failed to release margin for {position_id}, trying fallback...")
                    # Fallback: search for matching allocation by symbol and strike
                    for pid, alloc in list(position_mgr.allocations.items()):
                        if trade.symbol in pid and str(trade.strike) in pid and not alloc.released:
                            position_mgr.release_margin(pid, adjusted_pnl)
                            logger.info(f"Released margin using fallback: {pid}")
                            break
                
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
                    
                    # Create stock buy trade record
                    stock_buy_trade = TradeRecord(
                        symbol=trade.symbol,
                        trade_type="STOCK_BUY",
                        entry_date=bar_date,
                        exit_date=bar_date,
                        expiry=bar_date,
                        strike=trade.strike,  # Purchase price = strike
                        right="S",  # S for Stock
                        entry_price=trade.strike,
                        exit_price=trade.strike,
                        quantity=shares_acquired,
                        pnl=0,  # No P&L at purchase
                        pnl_pct=0,
                        exit_reason="PUT_ASSIGNMENT",
                        underlying_entry=trade.strike,
                        underlying_exit=trade.strike,
                        iv_at_entry=trade.iv_at_entry,
                        delta_at_entry=0,
                        position_id=stock_position_id,
                        capital_at_entry=position_mgr.net_capital,
                        capital_at_exit=position_mgr.net_capital,
                        strategy_phase=trade.strategy_phase,
                    )
                    simulator.closed_trades.append(stock_buy_trade)
                
                # When Call is assigned, we sell shares - release stock capital
                if trade.exit_reason == "ASSIGNMENT" and trade.trade_type in ("WHEEL_CALL", "BINBIN_CALL"):
                    # Call assigned: release stock capital
                    shares_sold = abs(trade.quantity) * 100
                    stock_holding = getattr(strategy, 'stock_holding', None)
                    stock_sell_price = trade.strike  # Sell at strike price
                    stock_buy_price = 0
                    
                    # Get the cost basis for this specific stock
                    if stock_holding:
                        # For multi-stock mode, get cost basis from holdings dict
                        if hasattr(stock_holding, 'holdings') and stock_holding.holdings:
                            stock_buy_price = stock_holding.holdings.get(trade.symbol, {}).get("cost_basis", 0)
                        elif hasattr(stock_holding, 'cost_basis'):
                            # Single stock mode
                            stock_buy_price = stock_holding.cost_basis
                        
                        if stock_buy_price > 0:
                            # Calculate stock capital to release
                            stock_capital = stock_buy_price * shares_sold
                            
                            # Find and release stock position (release first unreleased stock allocation)
                            for pid, alloc in list(position_mgr.allocations.items()):
                                if "_STOCK" in pid and not alloc.released:
                                    position_mgr.release_margin(pid, 0)  # Release with 0 P&L (stock P&L handled separately)
                                    logger.debug(f"Released stock capital: {pid}")
                                    break
                    
                    # Get stock P&L from strategy.on_trade_closed (will be called below)
                    # We need to call it first to get the actual stock P&L
                    if hasattr(strategy, 'on_trade_closed'):
                        stock_pnl = strategy.on_trade_closed(trade.to_dict())
                        if stock_pnl and stock_pnl != 0:
                            position_mgr.cumulative_pnl += stock_pnl
                            trade.pnl += stock_pnl
                            
                            # Create stock sell trade record
                            stock_position_id = f"{trade.symbol}_{bar_date}_STOCK"
                            stock_sell_trade = TradeRecord(
                                symbol=trade.symbol,
                                trade_type="STOCK_SELL",
                                entry_date=bar_date,  # Same day as buy for simplicity
                                exit_date=bar_date,
                                expiry=bar_date,
                                strike=stock_sell_price,
                                right="S",  # S for Stock
                                entry_price=stock_buy_price,
                                exit_price=stock_sell_price,
                                quantity=shares_sold,
                                pnl=stock_pnl,
                                pnl_pct=(stock_pnl / (stock_buy_price * shares_sold) * 100) if stock_buy_price > 0 else 0,
                                exit_reason="CALL_ASSIGNMENT",
                                underlying_entry=stock_buy_price,
                                underlying_exit=stock_sell_price,
                                iv_at_entry=trade.iv_at_entry,
                                delta_at_entry=0,
                                position_id=stock_position_id,
                                capital_at_entry=position_mgr.net_capital - stock_pnl,
                                capital_at_exit=position_mgr.net_capital,
                                strategy_phase=trade.strategy_phase,
                            )
                            simulator.closed_trades.append(stock_sell_trade)
                else:
                    # Non-assignment trades: call on_trade_closed normally
                    if hasattr(strategy, 'on_trade_closed'):
                        additional_pnl = strategy.on_trade_closed(trade.to_dict())
                        if additional_pnl and additional_pnl != 0:
                            position_mgr.cumulative_pnl += additional_pnl
                            trade.pnl += additional_pnl
                
                # Update trade record with capital information at exit
                # After stock capital is allocated and strategy state is updated
                trade.capital_at_exit = position_mgr.net_capital

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
                            # Calculate entry costs for P&L breakdown
                            roll_entry_commission = cost_model.calculate_commission(roll_signal.quantity)
                            roll_entry_slippage = cost_model.calculate_slippage(roll_signal.quantity)
                            
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
                                capital_at_entry=position_mgr.net_capital,
                                position_id=roll_position_id,  # Store position_id for reliable margin tracking
                                strategy_phase=getattr(roll_signal, 'strategy_phase', 'SP'),  # Pass strategy phase from signal
                                entry_commission=roll_entry_commission,
                                entry_slippage=roll_entry_slippage,
                            )
                            simulator.open_position(roll_pos)

                            # Track roll costs
                            roll_cost = cost_model.calculate_total_cost(roll_signal.quantity)
                            total_commission += roll_entry_commission
                            total_slippage += roll_entry_slippage

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
            for sig in signals:
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
                    if position_mgr.allocate_margin(
                        position_id=position_id,
                        strategy=strategy.name,
                        symbol=sig.symbol,
                        entry_date=bar_date,
                        margin_amount=total_margin,
                    ):
                        # Get correct underlying price for this symbol (important for multi-stock strategies)
                        entry_underlying_price = underlying_price  # Default: use primary stock
                        if is_multi_stock:
                            symbol_price = self._get_price_for_symbol(sig.symbol, bar_date, mag7_data)
                            if symbol_price is not None:
                                entry_underlying_price = symbol_price
                                logger.debug(f"Using {sig.symbol} price ${entry_underlying_price:.2f} for entry (vs primary ${underlying_price:.2f})")

                        # Calculate entry costs for P&L breakdown
                        entry_commission = cost_model.calculate_commission(sig.quantity)
                        entry_slippage = cost_model.calculate_slippage(sig.quantity)

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
                            capital_at_entry=position_mgr.net_capital,  # Record total capital at entry
                            position_id=position_id,  # Store position_id for reliable margin tracking
                            strategy_phase=getattr(sig, 'strategy_phase', 'SP'),  # Pass strategy phase from signal
                            entry_commission=entry_commission,
                            entry_slippage=entry_slippage,
                        )
                        simulator.open_position(pos)
                        
                        # Calculate and track trading costs (commission + slippage)
                        entry_cost = entry_commission + entry_slippage
                        total_commission += entry_commission
                        total_slippage += entry_slippage

                        # Deduct entry cost from cumulative P&L (trading costs reduce realized gains)
                        position_mgr.cumulative_pnl -= entry_cost
                        
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
                # CRITICAL FIX: In multi-stock mode, use the correct price for each stock holding
                # The legacy approach used underlying_price which could be wrong for multi-stock
                if hasattr(strategy.stock_holding, 'holdings') and strategy.stock_holding.holdings:
                    # Multi-stock mode: calculate unrealized P&L for each stock separately
                    for sym, holding in strategy.stock_holding.holdings.items():
                        sym_price = underlying_price  # Default to primary stock
                        if is_multi_stock and mag7_data:
                            sym_price = self._get_price_for_symbol(sym, bar_date, mag7_data) or underlying_price
                        sym_market_value = holding["shares"] * sym_price
                        sym_cost = holding["shares"] * holding["cost_basis"]
                        stock_unrealized_pnl += sym_market_value - sym_cost
                else:
                    # Single-stock mode (legacy behavior)
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
            
            # IMPORTANT: Get strategy performance report BEFORE liquidating stock holdings
            # This preserves the final state for UI display (Current Holdings, Current Strategy State)
            strategy_performance = {}
            if hasattr(strategy, 'get_performance_report'):
                strategy_performance = strategy.get_performance_report()
                logger.info(f"Engine: Retrieved strategy_performance - shares={strategy_performance.get('shares_held')}, cost={strategy_performance.get('cost_basis')}")
            
            # Handle remaining stock position for Wheel/BinbinGod strategies
            # At end of backtest, liquidate any remaining stock holdings
            if hasattr(strategy, 'stock_holding') and strategy.stock_holding.shares > 0:
                shares = strategy.stock_holding.shares
                cost_basis = strategy.stock_holding.cost_basis
                
                # Calculate realized P&L from stock liquidation
                stock_pnl = (last_price - cost_basis) * shares
                position_mgr.cumulative_pnl += stock_pnl
                logger.info(
                    f"Backtest end: Liquidated {shares} shares @ ${last_price:.2f} "
                    f"(cost: ${cost_basis:.2f}), P&L: ${stock_pnl:+.2f}"
                )
                
                # Release all stock capital allocations (for BinbinGod, symbol may differ from params symbol)
                released_count = 0
                for pid, alloc in list(position_mgr.allocations.items()):
                    if "_STOCK" in pid and not alloc.released:
                        position_mgr.release_margin(pid, stock_pnl if released_count == 0 else 0)
                        logger.debug(f"Released stock capital at backtest end: {pid}")
                        released_count += 1
                
                # Reset stock holding AFTER getting performance report
                strategy.stock_holding.shares = 0
                strategy.stock_holding.cost_basis = 0.0

        # Calculate metrics
        trades = [t.to_dict() for t in simulator.closed_trades]
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
            "event_trace": [],
            "portfolio_snapshots": [],
            "parity_report": {
                "status": "disabled",
                "thresholds": {
                    "daily_portfolio_value_pct": 1.0,
                    "final_total_return_pct": 2.0,
                    "max_drawdown_pct": 2.0,
                    "total_trades": 1.0,
                },
                "event_count": 0,
                "snapshot_count": 0,
                "first_mismatch": None,
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
