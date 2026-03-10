"""Binbin God Strategy: Dynamic MAG7 stock selection with full Wheel strategy logic.

This strategy intelligently selects the best stock from MAG7 universe based on:
- P/E Ratio (20% weight) - Value stocks preferred
- Option IV (40% weight) - Higher premium income  
- Momentum (20% weight) - Positive trend
- Stability (20% weight) - Risk management

Then executes FULL Wheel strategy logic (both Sell Put AND Covered Call phases).

Phase 1 (SP): Sell OTM puts, collect premium
  - If expires worthless: keep premium, continue selling puts
  - If assigned: buy shares at strike, switch to Phase 2

Phase 2 (CC): Sell OTM calls against owned shares
  - If expires worthless: keep premium + shares, continue selling calls
  - If assigned: sell shares at strike, return to Phase 1
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any
from core.backtesting.strategies.base import BaseStrategy, Signal
from core.backtesting.pricing import OptionsPricer
import logging


logger = logging.getLogger("binbin_god")


# MAG7 Universe
MAG7_STOCKS = ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]


@dataclass
class StockScore:
    """Score for a single stock."""
    symbol: str
    pe_ratio: float
    iv_rank: float
    momentum: float
    stability: float
    total_score: float
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "pe_ratio": self.pe_ratio,
            "iv_rank": self.iv_rank,
            "momentum": self.momentum,
            "stability": self.stability,
            "total_score": round(self.total_score, 2),
        }


@dataclass
class StockHolding:
    """Tracks stock position from put assignment."""
    shares: int = 0
    cost_basis: float = 0.0  # average cost per share
    total_premium_collected: float = 0.0  # cumulative premium from both phases


class BinbinGodStrategy(BaseStrategy):
    """
    Binbin God Strategy - Intelligent stock selection + Full Wheel logic.
    
    Implements BOTH Sell Put and Covered Call phases.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config
        self.symbol = config.get("symbol", "MAG7_AUTO")
        self.dte_min = config.get("dte_min", 30)
        self.dte_max = config.get("dte_max", 45)
        self.delta_target = config.get("delta_target", 0.30)
        self.profit_target_pct = config.get("profit_target_pct", 50)
        self.stop_loss_pct = config.get("stop_loss_pct", 200)
        self.max_positions = config.get("max_positions", 10)
        self.use_synthetic_data = config.get("use_synthetic_data", False)
        
        # Wheel-specific parameters
        self.put_delta = config.get("put_delta", 0.30)
        self.call_delta = config.get("call_delta", 0.30)
        
        # Scoring weights
        self.weights = {
            "pe_ratio": 0.20,
            "iv_rank": 0.40,
            "momentum": 0.20,
            "stability": 0.20,
        }
        
        # Phase tracking
        self.phase = "SP"  # Start with Sell Put phase
        self.stock_holding = StockHolding()
        
        # Storage for MAG7 analysis
        self.mag7_analysis = {
            "ranked_stocks": [],
            "best_pick": None,
        }
        
        # Check if profit target/stop loss are disabled
        self._profit_target_disabled = self.profit_target_pct >= 999999
        self._stop_loss_disabled = self.stop_loss_pct >= 999999
    
    def _score_stocks(self, market_data: Dict[str, Any]) -> List[StockScore]:
        """Score all MAG7 stocks based on metrics."""
        scores = []
        
        for symbol in MAG7_STOCKS:
            # Get market data for this symbol
            if symbol not in market_data:
                continue
                
            data = market_data[symbol]
            
            # Extract metrics (use defaults if not available)
            pe_ratio = data.get("fundamentals", {}).get("pe_ratio", 25.0)
            iv_rank = data.get("options", {}).get("iv_rank", 50.0)
            momentum = data.get("technical", {}).get("momentum_score", 50.0)
            stability = data.get("technical", {}).get("stability_score", 50.0)
            
            # Normalize PE ratio (lower is better, invert the score)
            pe_score = max(0, min(100, 100 - pe_ratio))
            
            # Calculate weighted total score
            total_score = (
                pe_score * self.weights["pe_ratio"] +
                iv_rank * self.weights["iv_rank"] +
                momentum * self.weights["momentum"] +
                stability * self.weights["stability"]
            )
            
            scores.append(StockScore(
                symbol=symbol,
                pe_ratio=pe_ratio,
                iv_rank=iv_rank,
                momentum=momentum,
                stability=stability,
                total_score=total_score,
            ))
        
        # Sort by total score (descending)
        scores.sort(key=lambda x: x.total_score, reverse=True)
        return scores
    
    def _select_best_stock(self, market_data: Dict[str, Any]) -> str:
        """Select the best stock from MAG7 based on scoring."""
        scores = self._score_stocks(market_data)
        
        if not scores:
            logger.warning("No stocks scored, defaulting to NVDA")
            return "NVDA"
        
        # Store analysis results
        self.mag7_analysis["ranked_stocks"] = [s.to_dict() for s in scores]
        self.mag7_analysis["best_pick"] = scores[0].to_dict()
        
        best_symbol = scores[0].symbol
        logger.info(f"Selected {best_symbol} with score {scores[0].total_score:.1f}")
        
        return best_symbol
    
    def generate_signal(
        self,
        symbol: str,
        current_dt: datetime,
        bars: List[Dict],
        contracts: List[Dict],
        portfolio: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> Signal | None:
        """Generate trading signal for Binbin God strategy."""
        
        # If symbol is MAG7_AUTO, select the best stock
        if self.symbol == "MAG7_AUTO":
            actual_symbol = self._select_best_stock(market_data)
        else:
            actual_symbol = self.symbol
        
        # Check phase and generate appropriate signal
        if self.phase == "SP":
            return self._generate_put_signal(
                actual_symbol, current_dt, bars, contracts, portfolio, market_data
            )
        else:  # CC phase
            return self._generate_call_signal(
                actual_symbol, current_dt, bars, contracts, portfolio, market_data
            )
    
    def _generate_put_signal(
        self,
        symbol: str,
        current_dt: datetime,
        bars: List[Dict],
        contracts: List[Dict],
        portfolio: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> Signal | None:
        """Generate Sell Put signal (Phase 1)."""
        
        # Check if we already have max positions
        current_positions = len(portfolio.get("positions", []))
        if current_positions >= self.max_positions:
            logger.debug(f"Max positions ({self.max_positions}) reached")
            return None
        
        # Filter for puts with target DTE and delta
        suitable_contracts = []
        
        for contract in contracts:
            if contract.get("right", "") != "P":
                continue
            
            # Check DTE
            expiry = contract.get("expiry")
            if expiry:
                if isinstance(expiry, datetime):
                    dte = (expiry.date() - current_dt.date()).days
                else:
                    dte = 0
            else:
                dte = 0
                
            if not (self.dte_min <= dte <= self.dte_max):
                continue
            
            # Check delta (absolute value)
            delta = abs(contract.get("delta", 0))
            if 0.25 <= delta <= 0.35:  # Allow some flexibility around target
                suitable_contracts.append((contract, abs(delta - self.put_delta)))
        
        if not suitable_contracts:
            logger.debug(f"No suitable put contracts found for {symbol}")
            return None
        
        # Select contract closest to target delta
        suitable_contracts.sort(key=lambda x: x[1])
        selected_contract = suitable_contracts[0][0]
        
        # Calculate position size (1 contract per $10k capital as rough guide)
        capital = portfolio.get("cash", 100000)
        max_contracts_by_capital = int(capital / 10000)
        max_contracts_by_limit = self.max_positions - current_positions
        quantity = min(max_contracts_by_capital, max_contracts_by_limit, 10)
        
        if quantity <= 0:
            return None
        
        logger.info(f"Selling {quantity} put(s) on {symbol} @ ${selected_contract.get('bid', 0):.2f}")
        
        return Signal(
            action="SELL_PUT",
            symbol=symbol,
            contract=selected_contract,
            quantity=quantity,
            price=selected_contract.get("bid", 0),
        )
    
    def _generate_call_signal(
        self,
        symbol: str,
        current_dt: datetime,
        bars: List[Dict],
        contracts: List[Dict],
        portfolio: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> Signal | None:
        """Generate Covered Call signal (Phase 2)."""
        
        # Check if we have shares to sell calls against
        if self.stock_holding.shares <= 0:
            logger.warning("In CC phase but no shares held, switching back to SP")
            self.phase = "SP"
            return None
        
        # Calculate how many call contracts we can sell (1 contract per 100 shares)
        max_contracts = self.stock_holding.shares // 100
        if max_contracts <= 0:
            return None
        
        # Filter for calls with target DTE and delta
        suitable_contracts = []
        
        for contract in contracts:
            if contract.get("right", "") != "C":
                continue
            
            # Check DTE
            expiry = contract.get("expiry")
            if expiry:
                if isinstance(expiry, datetime):
                    dte = (expiry.date() - current_dt.date()).days
                else:
                    dte = 0
            else:
                dte = 0
                
            if not (self.dte_min <= dte <= self.dte_max):
                continue
            
            # Check delta (absolute value)
            delta = abs(contract.get("delta", 0))
            if 0.25 <= delta <= 0.35:  # Allow some flexibility around target
                suitable_contracts.append((contract, abs(delta - self.call_delta)))
        
        if not suitable_contracts:
            logger.debug(f"No suitable call contracts found for {symbol}")
            return None
        
        # Select contract closest to target delta
        suitable_contracts.sort(key=lambda x: x[1])
        selected_contract = suitable_contracts[0][0]
        
        # Sell calls against shares (limited by shares owned)
        quantity = min(max_contracts, 10)
        
        if quantity <= 0:
            return None
        
        logger.info(f"Selling {quantity} call(s) on {symbol} @ ${selected_contract.get('bid', 0):.2f}")
        
        return Signal(
            action="SELL_CALL",
            symbol=symbol,
            contract=selected_contract,
            quantity=quantity,
            price=selected_contract.get("bid", 0),
        )
    
    def should_exit_position(
        self,
        position: Dict[str, Any],
        current_price: float,
        entry_price: float,
        current_dt: datetime,
    ) -> tuple[bool, str]:
        """Check if position should be exited."""
        
        # Calculate P&L
        pnl = current_price - entry_price
        pnl_pct = (pnl / entry_price) * 100 if entry_price > 0 else 0
        
        # Profit target exit
        if not self._profit_target_disabled:
            profit_threshold = self.profit_target_pct / 100.0 * abs(entry_price)
            if abs(pnl) >= profit_threshold and pnl < 0:  # Premium decayed enough
                return True, "PROFIT_TARGET"
        
        # Stop loss exit
        if not self._stop_loss_disabled:
            loss_threshold = self.stop_loss_pct / 100.0 * abs(entry_price)
            if pnl >= loss_threshold:  # Loss exceeded threshold
                return True, "STOP_LOSS"
        
        # Expiry exit
        expiry = position.get("expiry")
        if expiry and current_dt >= expiry:
            return True, "EXPIRY"
        
        return False, ""
    
    def on_assignment(self, position: Dict[str, Any]):
        """Called when option is assigned/exercised."""
        right = position.get("right", "")
        quantity = abs(position.get("quantity", 0))
        strike = position.get("strike", 0)
        
        if right == "P":
            # Put assignment: we bought shares
            shares_acquired = quantity * 100
            self.stock_holding.shares += shares_acquired
            self.stock_holding.cost_basis = strike
            self.phase = "CC"  # Switch to Covered Call phase
            logger.info(f"Put assigned: Bought {shares_acquired} shares @ ${strike}, switching to CC phase")
        
        elif right == "C":
            # Call assignment: we sold shares
            shares_sold = quantity * 100
            if shares_sold <= self.stock_holding.shares:
                # Calculate realized stock P&L
                stock_cost_basis = self.stock_holding.cost_basis * shares_sold
                stock_proceeds = strike * shares_sold
                stock_pnl = stock_proceeds - stock_cost_basis
               
                # Log complete P&L breakdown
                option_pnl = position.get("pnl", 0)
                total_trade_pnl = option_pnl + stock_pnl
                logger.info(
                    f"Call assigned: Option P&L=${option_pnl:+.2f}, Stock P&L=${stock_pnl:+.2f}, "
                    f"Total=${total_trade_pnl:+.2f} (bought at ${self.stock_holding.cost_basis:.2f}, "
                    f"sold at ${strike:.2f}, {shares_sold} shares)"
                )
               
                self.stock_holding.shares -= shares_sold
                if self.stock_holding.shares == 0:
                    self.phase = "SP"  # Switch back to Sell Put phase
            else:
                logger.warning(f"Call assignment error: Trying to sell {shares_sold} shares but only have {self.stock_holding.shares}")
    
    def get_mag7_analysis(self) -> Dict[str, Any]:
        """Get MAG7 analysis results."""
        return self.mag7_analysis
