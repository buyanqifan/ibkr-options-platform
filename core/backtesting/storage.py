"""Backtest result storage service for saving and retrieving backtest records."""

import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from models.base import SessionLocal
from models.backtest import BacktestResult, BacktestTrade
from utils.logger import setup_logger

logger = setup_logger("backtest_storage")


class BacktestStorage:
    """Service for storing and retrieving backtest results from database."""

    def __init__(self):
        self.Session = SessionLocal

    def save_backtest(
        self,
        params: Dict[str, Any],
        result: Dict[str, Any],
        run_time_seconds: float = 0,
    ) -> int:
        """Save a backtest result to the database.
        
        Args:
            params: Backtest parameters (strategy, symbol, dates, etc.)
            result: Backtest result from engine.run()
            run_time_seconds: Time taken to run the backtest
            
        Returns:
            The ID of the saved backtest record
        """
        metrics = result.get("metrics", {})
        trades = result.get("trades", [])
        daily_pnl = result.get("daily_pnl", [])
        strategy_performance = result.get("strategy_performance", {})
        
        session = self.Session()
        try:
            # Calculate final capital
            initial_capital = params.get("initial_capital", 100000)
            final_capital = initial_capital + metrics.get("total_return", 0)
            if final_capital == initial_capital and daily_pnl:
                # Fallback: calculate from last daily P&L
                final_capital = daily_pnl[-1].get("portfolio_value", initial_capital)
            
            # Count assignments
            assignment_count = sum(
                1 for t in trades if t.get("exit_reason") == "ASSIGNMENT"
            )
            
            # Calculate average DTE at entry
            dte_values = [params.get("dte_min", 30), params.get("dte_max", 45)]
            avg_dte = sum(dte_values) / 2 if dte_values else 30
            
            # Create main backtest record
            backtest = BacktestResult(
                strategy_name=params.get("strategy", "unknown"),
                symbol=params.get("symbol", ""),
                start_date=params.get("start_date", ""),
                end_date=params.get("end_date", ""),
                params=params,
                initial_capital=initial_capital,
                final_capital=final_capital,
                total_return_pct=metrics.get("total_return_pct", 0),
                annualized_return_pct=metrics.get("annualized_return_pct", 0),
                max_drawdown_pct=metrics.get("max_drawdown_pct", 0),
                sharpe_ratio=metrics.get("sharpe_ratio", 0),
                sortino_ratio=metrics.get("sortino_ratio", 0),
                win_rate=metrics.get("win_rate", 0),
                total_trades=metrics.get("total_trades", 0),
                avg_profit=metrics.get("avg_profit", 0),
                avg_loss=metrics.get("avg_loss", 0),
                profit_factor=metrics.get("profit_factor", 0),
                avg_dte_at_entry=avg_dte,
                assignment_count=assignment_count,
                daily_pnl=json.dumps(daily_pnl),
                created_at=datetime.utcnow(),
            )
            
            session.add(backtest)
            session.flush()  # Get the ID
            
            backtest_id = backtest.id
            
            # Save individual trades
            for trade in trades:
                trade_record = BacktestTrade(
                    backtest_id=backtest_id,
                    symbol=trade.get("symbol", params.get("symbol", "")),
                    strategy_name=params.get("strategy", "unknown"),
                    trade_type=trade.get("trade_type", ""),
                    entry_date=trade.get("entry_date", ""),
                    exit_date=trade.get("exit_date", ""),
                    expiry=trade.get("expiry", ""),
                    strike=trade.get("strike", 0),
                    entry_price=trade.get("entry_price", 0),
                    exit_price=trade.get("exit_price", 0),
                    quantity=trade.get("quantity", 1),
                    pnl=trade.get("pnl", 0),
                    pnl_pct=trade.get("pnl_pct", 0),
                    exit_reason=trade.get("exit_reason", ""),
                    underlying_entry=trade.get("underlying_entry", 0),
                    underlying_exit=trade.get("underlying_exit", 0),
                    iv_at_entry=trade.get("iv_at_entry", 0),
                    delta_at_entry=trade.get("delta_at_entry", 0),
                    capital_at_entry=trade.get("capital_at_entry", 0),
                    capital_at_exit=trade.get("capital_at_exit", 0),
                )
                session.add(trade_record)
            
            session.commit()
            logger.info(f"Saved backtest #{backtest_id}: {params.get('strategy')} on {params.get('symbol')}")
            return backtest_id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save backtest: {e}")
            raise
        finally:
            session.close()

    def get_backtest(self, backtest_id: int) -> Optional[Dict[str, Any]]:
        """Get a single backtest result by ID.
        
        Args:
            backtest_id: The ID of the backtest to retrieve
            
        Returns:
            Dictionary containing backtest data and trades, or None if not found
        """
        session = self.Session()
        try:
            backtest = session.query(BacktestResult).filter(
                BacktestResult.id == backtest_id
            ).first()
            
            if not backtest:
                return None
            
            trades = session.query(BacktestTrade).filter(
                BacktestTrade.backtest_id == backtest_id
            ).order_by(BacktestTrade.entry_date).all()
            
            return {
                "id": backtest.id,
                "strategy_name": backtest.strategy_name,
                "symbol": backtest.symbol,
                "start_date": backtest.start_date,
                "end_date": backtest.end_date,
                "params": backtest.params or {},
                "initial_capital": backtest.initial_capital,
                "final_capital": backtest.final_capital,
                "total_return_pct": backtest.total_return_pct,
                "annualized_return_pct": backtest.annualized_return_pct,
                "max_drawdown_pct": backtest.max_drawdown_pct,
                "sharpe_ratio": backtest.sharpe_ratio,
                "sortino_ratio": backtest.sortino_ratio,
                "win_rate": backtest.win_rate,
                "total_trades": backtest.total_trades,
                "avg_profit": backtest.avg_profit,
                "avg_loss": backtest.avg_loss,
                "profit_factor": backtest.profit_factor,
                "avg_dte_at_entry": backtest.avg_dte_at_entry,
                "assignment_count": backtest.assignment_count,
                "daily_pnl": json.loads(backtest.daily_pnl) if backtest.daily_pnl else [],
                "created_at": backtest.created_at.isoformat() if backtest.created_at else None,
                "trades": [
                    {
                        "id": t.id,
                        "symbol": t.symbol,
                        "trade_type": t.trade_type,
                        "entry_date": t.entry_date,
                        "exit_date": t.exit_date,
                        "expiry": t.expiry,
                        "strike": t.strike,
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "quantity": t.quantity,
                        "pnl": t.pnl,
                        "pnl_pct": t.pnl_pct,
                        "exit_reason": t.exit_reason,
                        "underlying_entry": t.underlying_entry,
                        "underlying_exit": t.underlying_exit,
                        "iv_at_entry": t.iv_at_entry,
                        "delta_at_entry": t.delta_at_entry,
                    }
                    for t in trades
                ],
            }
        finally:
            session.close()

    def list_backtests(
        self,
        strategy: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List backtest results with optional filtering.
        
        Args:
            strategy: Filter by strategy name
            symbol: Filter by symbol
            limit: Maximum number of results to return
            offset: Number of results to skip
            
        Returns:
            List of backtest summaries
        """
        session = self.Session()
        try:
            query = session.query(BacktestResult)
            
            if strategy:
                query = query.filter(BacktestResult.strategy_name == strategy)
            if symbol:
                query = query.filter(BacktestResult.symbol == symbol.upper())
            
            query = query.order_by(BacktestResult.created_at.desc())
            query = query.limit(limit).offset(offset)
            
            results = query.all()
            
            return [
                {
                    "id": r.id,
                    "strategy_name": r.strategy_name,
                    "symbol": r.symbol,
                    "start_date": r.start_date,
                    "end_date": r.end_date,
                    "initial_capital": r.initial_capital,
                    "total_return_pct": r.total_return_pct,
                    "sharpe_ratio": r.sharpe_ratio,
                    "win_rate": r.win_rate,
                    "total_trades": r.total_trades,
                    "max_drawdown_pct": r.max_drawdown_pct,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in results
            ]
        finally:
            session.close()

    def delete_backtest(self, backtest_id: int) -> bool:
        """Delete a backtest and its associated trades.
        
        Args:
            backtest_id: The ID of the backtest to delete
            
        Returns:
            True if deleted, False if not found
        """
        session = self.Session()
        try:
            # Delete trades first
            session.query(BacktestTrade).filter(
                BacktestTrade.backtest_id == backtest_id
            ).delete()
            
            # Delete backtest
            deleted = session.query(BacktestResult).filter(
                BacktestResult.id == backtest_id
            ).delete()
            
            session.commit()
            return deleted > 0
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete backtest {backtest_id}: {e}")
            raise
        finally:
            session.close()

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all backtests.
        
        Returns:
            Dictionary with summary statistics
        """
        session = self.Session()
        try:
            from sqlalchemy import func
            
            total_count = session.query(func.count(BacktestResult.id)).scalar() or 0
            
            if total_count == 0:
                return {
                    "total_backtests": 0,
                    "by_strategy": {},
                    "by_symbol": {},
                }
            
            # Count by strategy
            strategy_counts = session.query(
                BacktestResult.strategy_name,
                func.count(BacktestResult.id)
            ).group_by(BacktestResult.strategy_name).all()
            
            # Count by symbol
            symbol_counts = session.query(
                BacktestResult.symbol,
                func.count(BacktestResult.id)
            ).group_by(BacktestResult.symbol).all()
            
            return {
                "total_backtests": total_count,
                "by_strategy": {s: c for s, c in strategy_counts},
                "by_symbol": {s: c for s, c in symbol_counts},
            }
        finally:
            session.close()


# Singleton instance
_storage_instance: Optional[BacktestStorage] = None


def get_backtest_storage() -> BacktestStorage:
    """Get the singleton BacktestStorage instance."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = BacktestStorage()
    return _storage_instance