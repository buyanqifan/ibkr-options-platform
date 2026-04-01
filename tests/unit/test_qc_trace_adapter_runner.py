"""Tests for QC trace adaptation and parity runner baseline wiring."""

from core.backtesting.qc_parity_runner import run_qc_parity_backtest
from core.backtesting.qc_trace_adapter import adapt_qc_trace


def test_adapt_qc_trace_extracts_dates_and_deferred_events_from_log_text():
    log_text = """
2024-03-15 09:30:00 SP_SIGNAL: NVDA delta=-0.30
2024-03-15 09:31:00 ORDER_DEFERRED: NVDA 20240419P95000 waiting for first bar
2024-03-15 15:50:00 Put assigned: +100 NVDA @ $95.00
2024-03-15 16:00:00 Final Portfolio: $301,250.25
""".strip()

    trace = adapt_qc_trace(log_text)

    assert trace["event_trace"][0]["date"] == "2024-03-15"
    assert trace["event_trace"][0]["event_type"] == "signal_generated"
    assert trace["event_trace"][1]["event_type"] == "order_deferred"
    assert trace["event_trace"][1]["symbol"] == "NVDA"
    assert trace["event_trace"][1]["reason"] == "waiting_for_first_bar"
    assert trace["event_trace"][2]["event_type"] == "assigned_put"
    assert trace["portfolio_snapshots"][0]["date"] == "2024-03-15"
    assert trace["portfolio_snapshots"][0]["portfolio_value"] == 301250.25


def test_run_qc_parity_backtest_passes_qc_baseline_into_engine():
    received = {}

    class DummyEngine:
        def run(self, params):
            received["params"] = params
            return {
                "parity_report": {"status": "matched"},
                "qc_trace": params.get("qc_trace", {}),
            }

    result = run_qc_parity_backtest(
        DummyEngine(),
        {"strategy": "binbin_god", "symbol": "MAG7_AUTO"},
        qc_source={"event_trace": [{"type": "signal_generated", "symbol": "NVDA"}], "portfolio_snapshots": []},
    )

    assert received["params"]["parity_mode"] == "qc"
    assert "qc_trace" in received["params"]
    assert received["params"]["qc_trace"]["event_trace"][0]["event_type"] == "signal_generated"
    assert result["parity_report"]["status"] == "matched"
