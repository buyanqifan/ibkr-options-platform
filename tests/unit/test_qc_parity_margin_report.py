"""Focused QC parity tests for margin accounting and divergence reporting."""

from core.backtesting.cost_model import TradingCostModel
from core.backtesting.engine import BacktestEngine
from core.backtesting.position_manager import PositionManager
from core.backtesting.qc_parity import EventTracer
from core.backtesting.simulator import TradeRecord, TradeSimulator
from core.backtesting.strategies.base import Signal
from core.backtesting.strategies.binbin_god import BinbinGodStrategy


def _make_strategy() -> BinbinGodStrategy:
    return BinbinGodStrategy(
        {
            "strategy": "binbin_god",
            "symbol": "MAG7_AUTO",
            "stock_pool": ["NVDA"],
            "parity_mode": "qc",
            "contract_universe_mode": "qc_emulated_lattice",
            "ml_enabled": False,
        }
    )


def test_open_signal_position_uses_qc_estimated_put_margin():
    engine = BacktestEngine()
    strategy = _make_strategy()
    simulator = TradeSimulator()
    position_mgr = PositionManager(initial_capital=100000.0, max_leverage=1.0)
    tracer = EventTracer()
    cost_model = TradingCostModel()
    signal = Signal(
        symbol="NVDA",
        trade_type="BINBIN_PUT",
        right="P",
        strike=95.0,
        expiry="20240419",
        quantity=-2,
        iv=0.25,
        delta=-0.20,
        premium=1.50,
        underlying_price=100.0,
        margin_requirement=None,
        confidence=0.95,
    )

    opened = engine._open_signal_position(
        strategy=strategy,
        signal=signal,
        simulator=simulator,
        position_mgr=position_mgr,
        cost_model=cost_model,
        tracer=tracer,
        bar_date="2024-03-15",
        entry_underlying_price=100.0,
    )

    assert opened is True
    assert position_mgr.total_margin_used == 4750.0


def test_put_assignment_does_not_allocate_extra_stock_margin():
    engine = BacktestEngine()
    strategy = _make_strategy()
    simulator = TradeSimulator()
    position_mgr = PositionManager(initial_capital=100000.0, max_leverage=1.0)
    tracer = EventTracer()
    cost_model = TradingCostModel()

    option_position_id = "NVDA_2024-03-01_95.0_P"
    allocated = position_mgr.allocate_margin(
        position_id=option_position_id,
        strategy=strategy.name,
        symbol="NVDA",
        entry_date="2024-03-01",
        margin_amount=2375.0,
    )
    assert allocated is True

    assigned_trade = TradeRecord(
        symbol="NVDA",
        trade_type="BINBIN_PUT",
        entry_date="2024-03-01",
        exit_date="2024-04-19",
        expiry="20240419",
        strike=95.0,
        right="P",
        entry_price=1.5,
        exit_price=0.0,
        quantity=-1,
        pnl=150.0,
        pnl_pct=100.0,
        exit_reason="ASSIGNMENT",
        underlying_entry=100.0,
        underlying_exit=90.0,
        iv_at_entry=0.25,
        delta_at_entry=-0.20,
        position_id=option_position_id,
        capital_at_entry=100000.0,
        capital_at_exit=100000.0,
        strategy_phase="SP",
    )

    assigned_symbols = engine._process_closed_trades_parity(
        strategy=strategy,
        closed=[assigned_trade],
        simulator=simulator,
        position_mgr=position_mgr,
        cost_model=cost_model,
        tracer=tracer,
        bar_date="2024-04-19",
        underlying_price=90.0,
        iv=0.25,
        pool_data={"NVDA": [{"date": "2024-04-19", "close": 90.0}]},
        allow_roll=False,
    )

    assert assigned_symbols == ["NVDA"]
    assert position_mgr.total_margin_used == 0.0


def test_build_parity_report_includes_first_strike_mismatch_and_snapshots():
    tracer = EventTracer()
    tracer.record(
        "2024-03-15",
        "contract_selected",
        symbol="NVDA",
        action="SELL_PUT",
        right="P",
        qty=1,
        strike=95.0,
        expiry="20240419",
    )
    tracer.snapshot("2024-03-15", "rebalance_snapshot", portfolio_value=100000.0)

    report = tracer.build_parity_report(
        expected_trace=[
            {
                "date": "2024-03-15",
                "event_type": "contract_selected",
                "symbol": "NVDA",
                "action": "SELL_PUT",
                "right": "P",
                "qty": 1,
                "strike": 94.0,
                "expiry": "20240419",
            }
        ],
        expected_snapshots=[
            {
                "date": "2024-03-15",
                "phase": "rebalance_snapshot",
                "portfolio_value": 100500.0,
            }
        ],
    )

    assert report["status"] == "mismatch"
    assert report["first_mismatch"]["field"] == "strike"
    assert report["first_mismatch"]["actual_event"]["strike"] == 95.0
    assert report["first_mismatch"]["expected_event"]["strike"] == 94.0
    assert report["first_mismatch"]["actual_snapshots"][0]["phase"] == "rebalance_snapshot"
    assert report["first_mismatch"]["expected_snapshots"][0]["phase"] == "rebalance_snapshot"
