"""Backtest result database models."""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Text, JSON
from models.base import Base


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    start_date = Column(String(10), nullable=False)
    end_date = Column(String(10), nullable=False)
    params = Column(JSON)                  # strategy parameters
    initial_capital = Column(Float, nullable=False)
    final_capital = Column(Float)
    total_return_pct = Column(Float)
    annualized_return_pct = Column(Float)
    max_drawdown_pct = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    win_rate = Column(Float)
    total_trades = Column(Integer)
    avg_profit = Column(Float)
    avg_loss = Column(Float)
    profit_factor = Column(Float)
    avg_dte_at_entry = Column(Float)
    assignment_count = Column(Integer, default=0)
    daily_pnl = Column(Text)               # JSON-encoded list of daily P&L
    created_at = Column(DateTime, default=datetime.utcnow)


class BacktestTrade(Base):
    __tablename__ = "backtest_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    backtest_id = Column(Integer, nullable=False)
    symbol = Column(String(20), nullable=False)
    strategy_name = Column(String(50))
    trade_type = Column(String(20))        # SELL_PUT, COVERED_CALL, etc.
    strategy_phase = Column(String(10))    # SP, CC, or CC+SP (simultaneous mode)
    entry_date = Column(String(10), nullable=False)
    exit_date = Column(String(10))
    expiry = Column(String(10))
    strike = Column(Float)
    entry_price = Column(Float)            # premium received/paid
    exit_price = Column(Float)
    quantity = Column(Integer, default=1)
    pnl = Column(Float)
    pnl_pct = Column(Float)
    exit_reason = Column(String(50))       # PROFIT_TARGET, STOP_LOSS, EXPIRY, ASSIGNMENT, ROLL
    underlying_entry = Column(Float)       # stock price at entry
    underlying_exit = Column(Float)        # stock price at exit
    iv_at_entry = Column(Float)
    delta_at_entry = Column(Float)
    capital_at_entry = Column(Float)       # total portfolio capital when opening position
    capital_at_exit = Column(Float)        # total portfolio capital when closing position
