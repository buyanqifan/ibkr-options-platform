"""
Backtest Verification Module

This module provides tools for verifying backtest calculations and
comparing results with external platforms like QuantConnect.

Main components:
- BacktestVerifier: AI-powered verification engine
- QuantConnect templates: Reference implementations for comparison
"""

from .backtest_verifier import (
    BacktestVerifier,
    VerificationReport,
    VerificationIssue,
    verify_backtest_file,
)

__all__ = [
    "BacktestVerifier",
    "VerificationReport",
    "VerificationIssue",
    "verify_backtest_file",
]