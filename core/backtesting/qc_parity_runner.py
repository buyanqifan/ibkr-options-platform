"""Helpers to run local QC parity backtests and compare against QC traces."""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.backtesting.qc_trace_adapter import adapt_qc_trace


def run_qc_parity_backtest(engine, params: Dict[str, Any], qc_source: Optional[Any] = None) -> Dict[str, Any]:
    """Run BinbinGod in QC parity mode and attach an optional QC baseline."""
    parity_params = dict(params)
    parity_params["parity_mode"] = "qc"
    if qc_source is not None:
        parity_params["qc_trace"] = adapt_qc_trace(qc_source)
    result = engine.run(parity_params)
    if qc_source is not None:
        result["qc_trace"] = parity_params["qc_trace"]
    return result
