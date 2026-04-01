"""Tests for BinbinGod QC parity helpers and engine mode."""

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

pytest.importorskip("optionlab")

ROOT = Path(__file__).resolve().parents[2]
QC_DIR = ROOT / "quantconnect"
if str(QC_DIR) not in sys.path:
    sys.path.insert(0, str(QC_DIR))

import core.backtesting.strategies.binbin_god as binbin_god_module
from core.backtesting.engine import BacktestEngine
from core.backtesting.position_manager import PositionManager
from core.backtesting.qc_parity import (
    BinbinGodParityConfig,
    QC_PARAMETER_DEFAULTS,
    _extract_strategy_init_parameter_defaults,
    build_contract_lattice,
    calculate_symbol_state_risk_multiplier_qc,
    calculate_dynamic_max_positions_from_prices,
    calculate_put_quantity_qc,
)
from core.backtesting.simulator import OptionPosition, TradeSimulator
from core.backtesting.strategies.binbin_god import BinbinGodStrategy
from scoring import DEFAULT_WEIGHTS, score_single_stock


def _make_bars(start_price: float = 100.0, days: int = 80):
    bars = []
    current = start_price
    start = datetime(2024, 1, 1)
    for offset in range(days):
        current *= 1.002
        bars.append(
            {
                "date": (start + timedelta(days=offset)).strftime("%Y-%m-%d"),
                "open": current * 0.99,
                "high": current * 1.01,
                "low": current * 0.98,
                "close": current,
                "volume": 1_000_000 + offset * 1000,
            }
        )
    return bars


def test_qc_parity_config_uses_qc_defaults():
    config = BinbinGodParityConfig.from_params({"parity_mode": "qc"})
    assert config.enabled is True
    assert config.initial_capital == 300000.0
    assert config.max_positions_ceiling == 20
    assert config.dte_min == 21
    assert config.dte_max == 60
    assert config.position_aggressiveness == pytest.approx(1.2)
    assert config.profit_target_pct == pytest.approx(70.0)
    assert config.margin_buffer_pct == pytest.approx(0.40)
    assert config.target_margin_utilization == pytest.approx(0.45)
    assert config.max_risk_per_trade == pytest.approx(0.03)
    assert config.max_assignment_risk_per_trade == pytest.approx(0.20)
    assert config.ml_min_confidence == pytest.approx(0.40)
    assert config.defensive_put_roll_enabled is True
    assert config.assignment_cooldown_days == 20
    assert config.stock_inventory_cap_enabled is True
    assert config.stock_inventory_base_cap == pytest.approx(0.12)
    assert config.stock_inventory_block_threshold == pytest.approx(0.75)
    assert config.defensive_put_roll_loss_pct == pytest.approx(70.0)
    assert config.defensive_put_roll_itm_buffer_pct == pytest.approx(0.03)
    assert config.defensive_put_roll_max_dte == 21
    assert config.symbol_assignment_base_cap == pytest.approx(0.20)
    assert config.max_new_puts_per_day == 3


def test_extract_strategy_init_parameter_defaults_reads_qc_source_defaults():
    defaults = _extract_strategy_init_parameter_defaults()

    assert defaults["max_positions_ceiling"] == 20
    assert defaults["margin_buffer_pct"] == pytest.approx(0.40)
    assert defaults["target_margin_utilization"] == pytest.approx(0.45)
    assert defaults["max_risk_per_trade"] == pytest.approx(0.03)
    assert defaults["max_assignment_risk_per_trade"] == pytest.approx(0.20)
    assert defaults["defensive_put_roll_loss_pct"] == pytest.approx(70.0)
    assert defaults["defensive_put_roll_itm_buffer_pct"] == pytest.approx(0.03)
    assert defaults["defensive_put_roll_max_dte"] == 21
    assert defaults["symbol_assignment_base_cap"] == pytest.approx(0.20)
    assert defaults["stock_inventory_base_cap"] == pytest.approx(0.12)
    assert defaults["stock_inventory_block_threshold"] == pytest.approx(0.75)
    assert defaults["max_new_puts_per_day"] == 3


def test_qc_parameter_defaults_merge_config_and_strategy_init_sources():
    assert QC_PARAMETER_DEFAULTS["initial_capital"] == 300000.0
    assert QC_PARAMETER_DEFAULTS["profit_target_pct"] == pytest.approx(70.0)
    assert QC_PARAMETER_DEFAULTS["max_risk_per_trade"] == pytest.approx(0.03)
    assert QC_PARAMETER_DEFAULTS["max_assignment_risk_per_trade"] == pytest.approx(0.20)
    assert QC_PARAMETER_DEFAULTS["stock_inventory_base_cap"] == pytest.approx(0.12)
    assert QC_PARAMETER_DEFAULTS["symbol_assignment_base_cap"] == pytest.approx(0.20)
    assert QC_PARAMETER_DEFAULTS["defensive_put_roll_loss_pct"] == pytest.approx(70.0)
    assert QC_PARAMETER_DEFAULTS["max_new_puts_per_day"] == 3


def test_score_single_stock_penalizes_extreme_volatility():
    assert DEFAULT_WEIGHTS["iv_rank"] == pytest.approx(0.20)

    moderate_bars = []
    extreme_bars = []
    moderate_price = 100.0
    extreme_price = 100.0
    start = datetime(2024, 1, 1)

    for offset in range(40):
        moderate_price *= 1.006 if offset % 2 == 0 else 0.998
        extreme_price *= 1.08 if offset % 2 == 0 else 0.90
        moderate_bars.append(
            {
                "date": (start + timedelta(days=offset)).strftime("%Y-%m-%d"),
                "open": moderate_price * 0.99,
                "high": moderate_price * 1.01,
                "low": moderate_price * 0.98,
                "close": moderate_price,
                "volume": 1_000_000,
            }
        )
        extreme_bars.append(
            {
                "date": (start + timedelta(days=offset)).strftime("%Y-%m-%d"),
                "open": extreme_price * 0.98,
                "high": extreme_price * 1.03,
                "low": extreme_price * 0.93,
                "close": extreme_price,
                "volume": 1_000_000,
            }
        )

    moderate_score = score_single_stock("MSFT", moderate_bars, moderate_bars[-1]["close"]).total_score
    extreme_score = score_single_stock("TSLA", extreme_bars, extreme_bars[-1]["close"]).total_score

    assert extreme_score < moderate_score


def test_binbin_god_strategy_forces_qc_replay_defaults():
    config = BinbinGodParityConfig.from_params({"strategy": "binbin_god"})

    assert config.enabled is True
    assert config.parity_mode == "qc"
    assert config.contract_universe_mode == "qc_emulated_lattice"


def test_contract_lattice_respects_min_strike():
    contracts = build_contract_lattice(
        symbol="NVDA",
        current_date="2024-03-01",
        underlying_price=100.0,
        iv=0.25,
        target_right="C",
        target_delta=0.12,
        dte_min=21,
        dte_max=45,
        min_strike=105.0,
    )
    assert contracts
    assert all(contract.strike >= 105.0 for contract in contracts)


def test_dynamic_max_positions_matches_qc_style_formula():
    config = BinbinGodParityConfig.from_params({"parity_mode": "qc"})
    result = calculate_dynamic_max_positions_from_prices([100.0, 120.0, 140.0], config)
    assert result >= 1
    assert result <= config.max_positions_ceiling


def test_put_quantity_qc_applies_caps():
    config = BinbinGodParityConfig.from_params({"parity_mode": "qc"})
    contract = build_contract_lattice(
        symbol="NVDA",
        current_date="2024-03-01",
        underlying_price=100.0,
        iv=0.25,
        target_right="P",
        target_delta=-0.12,
        dte_min=21,
        dte_max=45,
    )[0]
    quantity, diagnostics = calculate_put_quantity_qc(
        config=config,
        selected_contract=contract,
        current_positions=0,
        underlying_price=100.0,
        symbol="NVDA",
        portfolio_value=100000.0,
        margin_remaining=100000.0,
        total_margin_used=0.0,
        stock_holdings_value=0.0,
        stock_holding_count=0,
        open_option_positions=[],
        shares_held=0,
        dynamic_max_positions=10,
    )
    assert quantity >= 0
    assert set(diagnostics) >= {"margin", "slots", "budget", "leverage"}


def test_put_quantity_qc_blocks_when_stock_inventory_is_full():
    config = BinbinGodParityConfig.from_params(
        {
            "parity_mode": "qc",
            "stock_inventory_cap_enabled": True,
            "stock_inventory_base_cap": 0.20,
            "stock_inventory_cap_floor": 0.50,
        }
    )
    contract = build_contract_lattice(
        symbol="NVDA",
        current_date="2024-03-01",
        underlying_price=100.0,
        iv=0.25,
        target_right="P",
        target_delta=-0.12,
        dte_min=21,
        dte_max=45,
    )[0]
    quantity, diagnostics = calculate_put_quantity_qc(
        config=config,
        selected_contract=contract,
        current_positions=0,
        underlying_price=100.0,
        symbol="NVDA",
        portfolio_value=100000.0,
        margin_remaining=100000.0,
        total_margin_used=0.0,
        stock_holdings_value=18000.0,
        stock_holding_count=1,
        open_option_positions=[],
        shares_held=180,
        dynamic_max_positions=10,
    )
    assert quantity == 0
    assert "stock_inventory" in diagnostics


def test_put_quantity_qc_applies_max_risk_per_trade_cap():
    config = BinbinGodParityConfig.from_params(
        {
            "parity_mode": "qc",
            "max_risk_per_trade": 0.005,
        }
    )
    contract = build_contract_lattice(
        symbol="NVDA",
        current_date="2024-03-01",
        underlying_price=100.0,
        iv=0.80,
        target_right="P",
        target_delta=-0.30,
        dte_min=21,
        dte_max=45,
    )[0]
    quantity, diagnostics = calculate_put_quantity_qc(
        config=config,
        selected_contract=contract,
        current_positions=0,
        underlying_price=100.0,
        symbol="NVDA",
        portfolio_value=100000.0,
        margin_remaining=100000.0,
        total_margin_used=0.0,
        stock_holdings_value=0.0,
        stock_holding_count=0,
        open_option_positions=[],
        shares_held=0,
        dynamic_max_positions=10,
    )
    assert diagnostics["risk"] == 1
    assert quantity == 1
    assert quantity < diagnostics["margin"]


def test_put_quantity_qc_caps_high_notional_single_trade_assignment_risk():
    config = BinbinGodParityConfig.from_params(
        {
            "parity_mode": "qc",
            "max_assignment_risk_per_trade": 0.20,
        }
    )
    contract = build_contract_lattice(
        symbol="NVDA",
        current_date="2024-03-01",
        underlying_price=900.0,
        iv=0.25,
        target_right="P",
        target_delta=-0.12,
        dte_min=21,
        dte_max=45,
    )[0]
    quantity, diagnostics = calculate_put_quantity_qc(
        config=config,
        selected_contract=contract,
        current_positions=0,
        underlying_price=900.0,
        symbol="NVDA",
        portfolio_value=300000.0,
        margin_remaining=300000.0,
        total_margin_used=0.0,
        stock_holdings_value=0.0,
        stock_holding_count=0,
        open_option_positions=[],
        shares_held=0,
        dynamic_max_positions=10,
    )
    assert diagnostics["assignment_trade"] == 0
    assert quantity == 0


def test_generate_immediate_cc_signal_in_parity_mode():
    strategy = BinbinGodStrategy(
        {
            "strategy": "binbin_god",
            "symbol": "MAG7_AUTO",
            "stock_pool": ["NVDA"],
            "parity_mode": "qc",
            "contract_universe_mode": "qc_emulated_lattice",
            "ml_confidence_gate": 0.2,
            "ml_enabled": False,
        }
    )
    bars = _make_bars(100.0, 90)
    strategy.mag7_data = {"NVDA": bars}
    strategy.stock_hv = {"NVDA": [0.25] * len(bars)}
    strategy.stock_holding.add_shares("NVDA", 100, 100.0)
    strategy.set_parity_context(
        {
            "portfolio_value": 100000.0,
            "margin_remaining": 100000.0,
            "total_margin_used": 0.0,
            "stock_holdings_value": 10000.0,
            "stock_holding_count": 1,
            "price_by_symbol": {"NVDA": bars[-1]["close"]},
            "dynamic_max_positions": 10,
        }
    )

    signal = strategy.generate_immediate_cc_signal(
        symbol="NVDA",
        current_date=bars[-1]["date"],
        underlying_price=bars[-1]["close"],
        iv=0.25,
        open_positions=[],
    )
    assert signal is not None
    assert signal.right == "C"
    assert signal.confidence > 0


def test_build_parity_context_includes_stock_market_value():
    engine = BacktestEngine()
    strategy = BinbinGodStrategy(
        {
            "strategy": "binbin_god",
            "symbol": "MAG7_AUTO",
            "stock_pool": ["NVDA"],
            "parity_mode": "qc",
            "contract_universe_mode": "qc_emulated_lattice",
            "ml_enabled": False,
        }
    )
    strategy.stock_holding.add_shares("NVDA", 100, 90.0)
    position_mgr = PositionManager(initial_capital=100000.0, max_leverage=1.0)
    simulator = TradeSimulator()
    simulator.open_position(
        OptionPosition(
            symbol="NVDA",
            entry_date="2024-03-01",
            expiry="20240419",
            strike=95.0,
            right="P",
            trade_type="BINBIN_PUT",
            quantity=-1,
            entry_price=1.5,
            underlying_entry=100.0,
            iv_at_entry=0.25,
            delta_at_entry=-0.25,
            current_pnl=250.0,
        )
    )
    pool_data = {"NVDA": [{"date": "2024-03-15", "close": 110.0}]}

    context = engine._build_parity_context(
        strategy=strategy,
        position_mgr=position_mgr,
        simulator=simulator,
        bar_date="2024-03-15",
        pool_data=pool_data,
        fallback_price=110.0,
        dynamic_max_positions=10,
    )

    assert context["stock_holdings_value"] == pytest.approx(11000.0)
    assert context["portfolio_value"] == pytest.approx(111250.0)


def test_parity_put_signal_confidence_follows_qc_ml_semantics(monkeypatch):
    strategy = BinbinGodStrategy(
        {
            "strategy": "binbin_god",
            "symbol": "MAG7_AUTO",
            "stock_pool": ["NVDA"],
            "parity_mode": "qc",
            "contract_universe_mode": "qc_emulated_lattice",
            "ml_enabled": True,
            "ml_delta_optimization": True,
            "ml_dte_optimization": True,
        }
    )
    bars = _make_bars(100.0, 90)
    strategy.mag7_data = {"NVDA": bars}
    strategy.set_parity_context(
        {
            "portfolio_value": 100000.0,
            "margin_remaining": 100000.0,
            "total_margin_used": 0.0,
            "stock_holdings_value": 0.0,
            "stock_holding_count": 0,
            "price_by_symbol": {"NVDA": bars[-1]["close"]},
            "dynamic_max_positions": 10,
        }
    )

    monkeypatch.setattr(strategy, "_score_stocks", lambda *_args, **_kwargs: [SimpleNamespace(total_score=10.0)])
    monkeypatch.setattr(
        strategy.ml_integration,
        "optimize_put_delta",
        lambda **_kwargs: SimpleNamespace(optimal_delta=0.28, confidence=0.9, reasoning="delta"),
    )
    monkeypatch.setattr(
        strategy.ml_integration,
        "optimize_put_dte",
        lambda **_kwargs: SimpleNamespace(optimal_dte_min=31, optimal_dte_max=37, confidence=0.8, reasoning="dte"),
    )

    signals = strategy._generate_backtest_put_signal(
        symbol="NVDA",
        current_date=bars[-1]["date"],
        underlying_price=bars[-1]["close"],
        iv=0.25,
        open_positions=[],
    )

    assert signals
    assert signals[0].confidence == pytest.approx(0.73)
    assert signals[0].ml_score_adjustment == pytest.approx(-0.40)


def test_parity_put_signal_uses_ml_dte_window_for_contract_selection(monkeypatch):
    strategy = BinbinGodStrategy(
        {
            "strategy": "binbin_god",
            "symbol": "MAG7_AUTO",
            "stock_pool": ["NVDA"],
            "parity_mode": "qc",
            "contract_universe_mode": "qc_emulated_lattice",
            "ml_enabled": True,
            "ml_delta_optimization": True,
            "ml_dte_optimization": True,
        }
    )
    bars = _make_bars(100.0, 90)
    strategy.mag7_data = {"NVDA": bars}
    strategy.set_parity_context(
        {
            "portfolio_value": 100000.0,
            "margin_remaining": 100000.0,
            "total_margin_used": 0.0,
            "stock_holdings_value": 0.0,
            "stock_holding_count": 0,
            "price_by_symbol": {"NVDA": bars[-1]["close"]},
            "dynamic_max_positions": 10,
        }
    )

    monkeypatch.setattr(strategy, "_score_stocks", lambda *_args, **_kwargs: [SimpleNamespace(total_score=60.0)])
    monkeypatch.setattr(
        strategy.ml_integration,
        "optimize_put_delta",
        lambda **_kwargs: SimpleNamespace(optimal_delta=0.28, confidence=0.9, reasoning="delta"),
    )
    monkeypatch.setattr(
        strategy.ml_integration,
        "optimize_put_dte",
        lambda **_kwargs: SimpleNamespace(optimal_dte_min=31, optimal_dte_max=37, confidence=0.8, reasoning="dte"),
    )

    captured_windows = []

    class FakeContract:
        def __init__(self):
            self.strike = 95.0
            self.expiry = datetime.strptime(bars[-1]["date"], "%Y-%m-%d") + timedelta(days=35)
            self.dte = 35
            self.premium = 1.5
            self.delta = -0.28

        def to_dict(self):
            return {
                "strike": self.strike,
                "expiry": self.expiry,
                "dte": self.dte,
                "premium": self.premium,
                "delta": self.delta,
            }

    def fake_select_contract_from_lattice(**kwargs):
        captured_windows.append((kwargs["dte_min"], kwargs["dte_max"]))
        return FakeContract()

    monkeypatch.setattr(binbin_god_module, "select_contract_from_lattice", fake_select_contract_from_lattice)

    signals = strategy._generate_backtest_put_signal(
        symbol="NVDA",
        current_date=bars[-1]["date"],
        underlying_price=bars[-1]["close"],
        iv=0.25,
        open_positions=[],
    )

    assert signals
    assert captured_windows == [(31, 37)]


def test_parity_put_signal_uses_symbol_specific_cost_basis(monkeypatch):
    strategy = BinbinGodStrategy(
        {
            "strategy": "binbin_god",
            "symbol": "MAG7_AUTO",
            "stock_pool": ["NVDA", "AAPL"],
            "parity_mode": "qc",
            "contract_universe_mode": "qc_emulated_lattice",
            "ml_enabled": True,
            "ml_delta_optimization": True,
            "ml_dte_optimization": True,
        }
    )
    bars = _make_bars(100.0, 90)
    strategy.mag7_data = {"NVDA": bars, "AAPL": bars}
    strategy.stock_holding.add_shares("NVDA", 100, 250.0)
    strategy.set_parity_context(
        {
            "portfolio_value": 100000.0,
            "margin_remaining": 100000.0,
            "total_margin_used": 0.0,
            "stock_holdings_value": 25000.0,
            "stock_holding_count": 1,
            "price_by_symbol": {"NVDA": bars[-1]["close"], "AAPL": bars[-1]["close"]},
            "dynamic_max_positions": 10,
        }
    )

    seen_cost_basis = []
    monkeypatch.setattr(strategy, "_score_stocks", lambda *_args, **_kwargs: [SimpleNamespace(total_score=60.0)])

    def fake_optimize_put_delta(**kwargs):
        seen_cost_basis.append(kwargs["cost_basis"])
        return SimpleNamespace(optimal_delta=0.28, confidence=0.9, reasoning="delta")

    monkeypatch.setattr(strategy.ml_integration, "optimize_put_delta", fake_optimize_put_delta)
    monkeypatch.setattr(
        strategy.ml_integration,
        "optimize_put_dte",
        lambda **kwargs: SimpleNamespace(optimal_dte_min=31, optimal_dte_max=37, confidence=0.8, reasoning=f"dte:{kwargs['cost_basis']}"),
    )

    signals = strategy._generate_backtest_put_signal(
        symbol="AAPL",
        current_date=bars[-1]["date"],
        underlying_price=bars[-1]["close"],
        iv=0.25,
        open_positions=[],
    )

    assert signals
    assert seen_cost_basis == [0.0]


def test_qc_parity_skips_sell_put_when_stock_inventory_exceeds_block_threshold(monkeypatch):
    strategy = BinbinGodStrategy(
        {
            "strategy": "binbin_god",
            "symbol": "MAG7_AUTO",
            "stock_pool": ["NVDA"],
            "parity_mode": "qc",
            "contract_universe_mode": "qc_emulated_lattice",
            "ml_enabled": False,
            "stock_inventory_base_cap": 0.20,
            "stock_inventory_block_threshold": 0.90,
        }
    )
    bars = _make_bars(100.0, 90)
    strategy.mag7_data = {"NVDA": bars}
    strategy.stock_holding.add_shares("NVDA", 200, 100.0)
    strategy.set_parity_context(
        {
            "portfolio_value": 100000.0,
            "margin_remaining": 100000.0,
            "total_margin_used": 0.0,
            "stock_holdings_value": 20000.0,
            "stock_holding_count": 1,
            "price_by_symbol": {"NVDA": 100.0},
            "dynamic_max_positions": 10,
        }
    )
    recorded = []
    monkeypatch.setattr(strategy, "_record_event", lambda date, event_type, **payload: recorded.append((date, event_type, payload)))
    monkeypatch.setattr(strategy, "_generate_backtest_call_signal", lambda *args, **kwargs: [])
    monkeypatch.setattr(strategy, "_generate_backtest_put_signal", lambda *args, **kwargs: [SimpleNamespace(symbol="NVDA")])

    signals = strategy._generate_qc_parity_signals(
        current_date=bars[-1]["date"],
        underlying_price=100.0,
        iv=0.25,
        open_positions=[],
        position_mgr=None,
    )

    assert signals == []
    assert recorded
    assert recorded[0][1] == "order_deferred"
    assert recorded[0][2]["reason"] == "sp_existing_stock"


def test_qc_parity_skips_sell_put_for_symbol_with_existing_stock_holdings(monkeypatch):
    strategy = BinbinGodStrategy(
        {
            "strategy": "binbin_god",
            "symbol": "MAG7_AUTO",
            "stock_pool": ["NVDA", "AAPL"],
            "parity_mode": "qc",
            "contract_universe_mode": "qc_emulated_lattice",
            "ml_enabled": False,
        }
    )
    bars = _make_bars(100.0, 90)
    strategy.mag7_data = {"NVDA": bars, "AAPL": bars}
    strategy.stock_holding.add_shares("NVDA", 100, 100.0)
    strategy.set_parity_context(
        {
            "portfolio_value": 100000.0,
            "margin_remaining": 100000.0,
            "total_margin_used": 0.0,
            "stock_holdings_value": 10000.0,
            "stock_holding_count": 1,
            "price_by_symbol": {"NVDA": 100.0, "AAPL": 100.0},
            "dynamic_max_positions": 10,
        }
    )

    recorded = []
    monkeypatch.setattr(strategy, "_record_event", lambda date, event_type, **payload: recorded.append((date, event_type, payload)))
    monkeypatch.setattr(strategy, "_generate_backtest_call_signal", lambda *args, **kwargs: [])

    def fake_put_signal(symbol, *args, **kwargs):
        return [SimpleNamespace(symbol=symbol, right="P", quantity=-1, confidence=0.9, strategy_phase="SP")]

    monkeypatch.setattr(strategy, "_generate_backtest_put_signal", fake_put_signal)

    signals = strategy._generate_qc_parity_signals(
        current_date=bars[-1]["date"],
        underlying_price=100.0,
        iv=0.25,
        open_positions=[],
        position_mgr=None,
    )

    assert [signal.symbol for signal in signals] == ["AAPL"]
    assert recorded
    assert recorded[0][1] == "order_deferred"
    assert recorded[0][2]["symbol"] == "NVDA"
    assert recorded[0][2]["reason"] == "sp_existing_stock"


def test_qc_parity_returns_multiple_sp_signals_when_slots_available(monkeypatch):
    strategy = BinbinGodStrategy(
        {
            "strategy": "binbin_god",
            "symbol": "MAG7_AUTO",
            "stock_pool": ["NVDA", "AAPL", "MSFT"],
            "parity_mode": "qc",
            "contract_universe_mode": "qc_emulated_lattice",
            "ml_enabled": False,
            "max_new_puts_per_day": 3,
        }
    )
    bars = _make_bars(100.0, 90)
    strategy.mag7_data = {"NVDA": bars, "AAPL": bars, "MSFT": bars}
    strategy.set_parity_context(
        {
            "portfolio_value": 100000.0,
            "margin_remaining": 100000.0,
            "total_margin_used": 0.0,
            "stock_holdings_value": 0.0,
            "stock_holding_count": 0,
            "price_by_symbol": {"NVDA": 100.0, "AAPL": 100.0, "MSFT": 100.0},
            "dynamic_max_positions": 10,
        }
    )

    monkeypatch.setattr(strategy, "_generate_backtest_call_signal", lambda *args, **kwargs: [])

    def fake_put_signal(symbol, *args, **kwargs):
        confidence_by_symbol = {"NVDA": 0.95, "AAPL": 0.90, "MSFT": 0.85}
        return [SimpleNamespace(symbol=symbol, right="P", quantity=-1, confidence=confidence_by_symbol[symbol], strategy_phase="SP")]

    monkeypatch.setattr(strategy, "_generate_backtest_put_signal", fake_put_signal)

    signals = strategy._generate_qc_parity_signals(
        current_date=bars[-1]["date"],
        underlying_price=100.0,
        iv=0.25,
        open_positions=[],
        position_mgr=None,
    )

    assert [signal.symbol for signal in signals] == ["NVDA", "AAPL", "MSFT"]


def test_symbol_state_risk_multiplier_matches_qc_penalty_shape():
    bars = _make_bars(100.0, 90)
    stressed_bars = []
    start = datetime(2024, 1, 1)
    price = 120.0
    for offset in range(90):
        if offset < 60:
            price *= 0.995
        else:
            price *= 0.97
        stressed_bars.append(
            {
                "date": (start + timedelta(days=offset)).strftime("%Y-%m-%d"),
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": 1_000_000,
            }
        )

    config = BinbinGodParityConfig.from_params({"parity_mode": "qc"})
    multiplier, diagnostics = calculate_symbol_state_risk_multiplier_qc(
        config=config,
        symbol_history_bars=stressed_bars,
        pool_history_bars={"NVDA": stressed_bars, "AAPL": bars},
        underlying_price=stressed_bars[-1]["close"],
        symbol_put_notional=25000.0,
        symbol_stock_notional=25000.0,
        portfolio_value=100000.0,
    )

    assert multiplier < 1.0
    assert diagnostics["drawdown"] > 0
    assert diagnostics["exposure_ratio"] > 0


def test_engine_returns_parity_artifacts():
    engine = BacktestEngine()
    result = engine.run(
        {
            "strategy": "binbin_god",
            "symbol": "MAG7_AUTO",
            "stock_pool": ["NVDA", "AAPL"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
            "initial_capital": 100000,
            "max_leverage": 1.0,
            "use_synthetic_data": True,
            "parity_mode": "qc",
            "contract_universe_mode": "qc_emulated_lattice",
            "ml_confidence_gate": 0.2,
            "ml_enabled": False,
        }
    )
    assert "event_trace" in result
    assert "portfolio_snapshots" in result
    assert "parity_report" in result
    assert result["parity_report"]["status"] in {"no_baseline", "matched", "mismatch", "length_mismatch"}


def test_binbin_god_strategy_does_not_expose_legacy_cc_sp_entrypoints():
    strategy = BinbinGodStrategy({"symbol": "NVDA"})

    assert not hasattr(strategy, "allow_sp_in_cc_phase")
    assert not hasattr(strategy, "sp_in_cc_margin_threshold")
    assert not hasattr(strategy, "sp_in_cc_max_positions")
    assert not hasattr(strategy, "_generate_sp_in_cc_phase")


def test_repo_has_no_legacy_cc_sp_markers_in_runtime_paths():
    repo_root = Path(__file__).resolve().parents[2]
    runtime_files = [
        repo_root / "core" / "backtesting" / "strategies" / "binbin_god.py",
        repo_root / "core" / "backtesting" / "strategies" / "base.py",
        repo_root / "core" / "backtesting" / "simulator.py",
        repo_root / "core" / "ml" / "position_optimizer.py",
        repo_root / "core" / "ml" / "dte_optimizer.py",
        repo_root / "core" / "ml" / "roll_optimizer.py",
        repo_root / "core" / "ml" / "delta_strategy_integration.py",
        repo_root / "quantconnect" / "ml_position_optimizer.py",
        repo_root / "quantconnect" / "ml_dte_optimizer.py",
        repo_root / "quantconnect" / "ml_integration.py",
        repo_root / "quantconnect" / "README.md",
    ]
    banned_markers = ("CC+SP", "allow_sp_in_cc_phase", "sp_in_cc_")

    for path in runtime_files:
        text = path.read_text(encoding="utf-8")
        assert not any(marker in text for marker in banned_markers), path.as_posix()
