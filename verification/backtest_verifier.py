"""
AI-Powered Backtest Verification Tool

This module provides automated verification of backtest results by:
1. Running local backtest with detailed P&L tracking
2. Comparing with QuantConnect results (if available)
3. Detecting discrepancies and anomalies
4. Generating detailed verification reports

Usage:
    from verification.backtest_verifier import BacktestVerifier
    
    verifier = BacktestVerifier()
    report = verifier.verify_backtest(backtest_results)
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class VerificationIssue:
    """Represents a single verification issue."""
    severity: str  # "critical", "warning", "info"
    category: str  # "pnl_calculation", "assignment", "commission", etc.
    trade_id: Optional[int]
    description: str
    expected: float
    actual: float
    difference_pct: float
    recommendation: str


@dataclass
class VerificationReport:
    """Complete verification report for a backtest."""
    timestamp: str
    backtest_id: str
    total_trades: int
    issues: List[Dict]
    summary: Dict
    passed: bool
    confidence_score: float  # 0-100
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "backtest_id": self.backtest_id,
            "total_trades": self.total_trades,
            "issues": self.issues,
            "summary": self.summary,
            "passed": self.passed,
            "confidence_score": self.confidence_score,
        }


class BacktestVerifier:
    """AI-powered backtest verification engine."""
    
    # Tolerance thresholds
    PNL_TOLERANCE_PCT = 0.01  # 1% tolerance for P&L
    COMMISSION_TOLERANCE_PCT = 0.05  # 5% tolerance for commissions
    ASSIGNMENT_PNL_MIN = 0.0  # Assignment P&L should be >= 0
    
    def __init__(self, reference_data: Optional[Dict] = None):
        """Initialize verifier with optional reference data.
        
        Args:
            reference_data: Optional reference backtest results for comparison
        """
        self.reference_data = reference_data
        self.issues: List[VerificationIssue] = []
        
    def verify_backtest(self, results: Dict) -> VerificationReport:
        """Perform comprehensive verification of backtest results.
        
        Args:
            results: Backtest results dictionary from engine.run()
            
        Returns:
            VerificationReport with detailed findings
        """
        self.issues = []
        backtest_id = self._generate_backtest_id(results)
        trades = results.get("trades", [])
        metrics = results.get("metrics", {})
        
        # Run all verification checks
        self._verify_assignment_pnl(trades)
        self._verify_pnl_consistency(trades)
        self._verify_commission_tracking(trades, results)
        self._verify_profit_target_pnl(trades)
        self._verify_stop_loss_pnl(trades)
        self._verify_capital_tracking(trades, results)
        self._verify_total_pnl(trades, metrics)
        self._verify_pnl_breakdown(trades)
        
        # Generate summary
        summary = self._generate_summary(trades, metrics)
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(trades)
        
        # Determine if passed
        critical_issues = [i for i in self.issues if i.severity == "critical"]
        passed = len(critical_issues) == 0
        
        return VerificationReport(
            timestamp=datetime.now().isoformat(),
            backtest_id=backtest_id,
            total_trades=len(trades),
            issues=[asdict(i) for i in self.issues],
            summary=summary,
            passed=passed,
            confidence_score=confidence,
        )
    
    def _generate_backtest_id(self, results: Dict) -> str:
        """Generate unique ID for the backtest."""
        params = results.get("params", {})
        return f"{params.get('strategy', 'unknown')}_{params.get('symbol', 'unknown')}_{params.get('start_date', 'unknown')}"
    
    def _verify_assignment_pnl(self, trades: List[Dict]):
        """Verify P&L calculation for assigned options.
        
        Rule: When an option is assigned, the seller keeps the entire premium.
        P&L = premium_received × 100 × contracts
        """
        for i, trade in enumerate(trades):
            if trade.get("exit_reason") != "ASSIGNMENT":
                continue
                
            entry_price = trade.get("entry_price", 0)
            quantity = abs(trade.get("quantity", 0))
            expected_pnl = entry_price * 100 * quantity
            
            actual_pnl = trade.get("pnl", 0)
            pnl_pct = trade.get("pnl_pct", 0)
            
            # Check if P&L is correct
            if actual_pnl < 0:
                self.issues.append(VerificationIssue(
                    severity="critical",
                    category="assignment",
                    trade_id=i,
                    description=f"Assignment P&L is negative: ${actual_pnl:.2f}. Should be positive (premium received).",
                    expected=expected_pnl,
                    actual=actual_pnl,
                    difference_pct=abs(actual_pnl - expected_pnl) / expected_pnl * 100 if expected_pnl > 0 else 0,
                    recommendation="Check simulator.py assignment handling. P&L should be premium received, not (premium - intrinsic)."
                ))
            
            # Check if pnl_pct is 100%
            if abs(pnl_pct - 100.0) > 0.1:
                self.issues.append(VerificationIssue(
                    severity="warning",
                    category="assignment",
                    trade_id=i,
                    description=f"Assignment pnl_pct is {pnl_pct:.1f}%, expected 100%.",
                    expected=100.0,
                    actual=pnl_pct,
                    difference_pct=abs(pnl_pct - 100.0),
                    recommendation="pnl_pct for assignment should be 100% (full premium retained)."
                ))
    
    def _verify_pnl_consistency(self, trades: List[Dict]):
        """Verify P&L is consistent with entry/exit prices.
        
        For short options:
        P&L = (entry_price - exit_price) × 100 × |quantity|
        """
        for i, trade in enumerate(trades):
            if trade.get("right") == "S":  # Skip stock trades
                continue
                
            entry_price = trade.get("entry_price", 0)
            exit_price = trade.get("exit_price", 0)
            quantity = abs(trade.get("quantity", 0))
            actual_pnl = trade.get("pnl", 0)
            
            # Skip assignments (handled separately)
            if trade.get("exit_reason") == "ASSIGNMENT":
                continue
            
            # Calculate expected P&L
            expected_pnl = (entry_price - exit_price) * 100 * quantity
            
            if abs(expected_pnl) > 0.01:  # Avoid division by zero
                diff_pct = abs(actual_pnl - expected_pnl) / abs(expected_pnl) * 100
                
                if diff_pct > 5.0:  # More than 5% difference
                    self.issues.append(VerificationIssue(
                        severity="warning",
                        category="pnl_calculation",
                        trade_id=i,
                        description=f"P&L calculation mismatch: expected ${expected_pnl:.2f}, got ${actual_pnl:.2f}",
                        expected=expected_pnl,
                        actual=actual_pnl,
                        difference_pct=diff_pct,
                        recommendation=f"Verify P&L calculation: (entry - exit) × 100 × quantity = ({entry_price} - {exit_price}) × 100 × {quantity}"
                    ))
    
    def _verify_commission_tracking(self, trades: List[Dict], results: Dict):
        """Verify commission and slippage are properly tracked."""
        trading_costs = results.get("trading_costs", {})
        
        if not trading_costs:
            self.issues.append(VerificationIssue(
                severity="info",
                category="commission",
                trade_id=None,
                description="No trading costs breakdown found in results.",
                expected=0,
                actual=0,
                difference_pct=0,
                recommendation="Enable TradingCostModel in engine to track commissions and slippage."
            ))
            return
        
        total_trades = len([t for t in trades if t.get("right") != "S"])
        total_commission = trading_costs.get("total_commission", 0)
        expected_commission = total_trades * 2 * trading_costs.get("commission_rate", 0.65)  # Entry + exit
        
        if total_commission < expected_commission * 0.9:  # Allow 10% tolerance
            self.issues.append(VerificationIssue(
                severity="warning",
                category="commission",
                trade_id=None,
                description=f"Commission tracking may be incomplete: ${total_commission:.2f} vs expected ~${expected_commission:.2f}",
                expected=expected_commission,
                actual=total_commission,
                difference_pct=abs(total_commission - expected_commission) / expected_commission * 100,
                recommendation="Check that all trades have commission costs recorded."
            ))
    
    def _verify_profit_target_pnl(self, trades: List[Dict]):
        """Verify profit target exits show positive P&L."""
        for i, trade in enumerate(trades):
            if trade.get("exit_reason") != "PROFIT_TARGET":
                continue
                
            pnl = trade.get("pnl", 0)
            pnl_pct = trade.get("pnl_pct", 0)
            
            if pnl < 0:
                self.issues.append(VerificationIssue(
                    severity="critical",
                    category="profit_target",
                    trade_id=i,
                    description=f"Profit target exit has negative P&L: ${pnl:.2f}",
                    expected=abs(pnl),
                    actual=pnl,
                    difference_pct=100,
                    recommendation="Profit target exits should always have positive P&L."
                ))
    
    def _verify_stop_loss_pnl(self, trades: List[Dict]):
        """Verify stop loss exits show negative P&L."""
        for i, trade in enumerate(trades):
            if trade.get("exit_reason") != "STOP_LOSS":
                continue
                
            pnl = trade.get("pnl", 0)
            
            if pnl > 0:
                self.issues.append(VerificationIssue(
                    severity="warning",
                    category="stop_loss",
                    trade_id=i,
                    description=f"Stop loss exit has positive P&L: ${pnl:.2f}. Check stop loss logic.",
                    expected=-abs(pnl),
                    actual=pnl,
                    difference_pct=100,
                    recommendation="Stop loss exits should typically have negative P&L."
                ))
    
    def _verify_capital_tracking(self, trades: List[Dict], results: Dict):
        """Verify capital tracking consistency."""
        for i, trade in enumerate(trades):
            capital_entry = trade.get("capital_at_entry", 0)
            capital_exit = trade.get("capital_at_exit", 0)
            
            if capital_entry <= 0 and trade.get("right") != "S":
                self.issues.append(VerificationIssue(
                    severity="info",
                    category="capital_tracking",
                    trade_id=i,
                    description="Capital at entry not recorded.",
                    expected=0,
                    actual=0,
                    difference_pct=0,
                    recommendation="Record capital_at_entry when opening positions for better tracking."
                ))
    
    def _verify_total_pnl(self, trades: List[Dict], metrics: Dict):
        """Verify total P&L matches sum of individual trades."""
        # Calculate sum from trades
        trade_pnl_sum = sum(t.get("pnl", 0) for t in trades if t.get("right") != "S")
        
        # Get reported total P&L
        reported_pnl = metrics.get("total_pnl", 0)
        
        if abs(trade_pnl_sum - reported_pnl) > 0.01:
            self.issues.append(VerificationIssue(
                severity="warning",
                category="pnl_calculation",
                trade_id=None,
                description=f"Total P&L mismatch: trades sum to ${trade_pnl_sum:.2f}, but reported is ${reported_pnl:.2f}",
                expected=trade_pnl_sum,
                actual=reported_pnl,
                difference_pct=abs(trade_pnl_sum - reported_pnl) / abs(trade_pnl_sum) * 100 if trade_pnl_sum != 0 else 0,
                recommendation="Verify that cumulative P&L is calculated correctly."
            ))
    
    def _verify_pnl_breakdown(self, trades: List[Dict]):
        """Verify P&L breakdown is present and consistent."""
        trades_with_breakdown = 0
        
        for i, trade in enumerate(trades):
            breakdown = trade.get("pnl_breakdown")
            
            if breakdown:
                trades_with_breakdown += 1
                
                # Verify breakdown totals match trade pnl
                breakdown_pnl = breakdown.get("net_pnl", 0)
                trade_pnl = trade.get("pnl", 0)
                
                if abs(breakdown_pnl - trade_pnl) > 0.01:
                    self.issues.append(VerificationIssue(
                        severity="info",
                        category="pnl_breakdown",
                        trade_id=i,
                        description=f"P&L breakdown (${breakdown_pnl:.2f}) doesn't match trade P&L (${trade_pnl:.2f})",
                        expected=trade_pnl,
                        actual=breakdown_pnl,
                        difference_pct=abs(breakdown_pnl - trade_pnl) / abs(trade_pnl) * 100 if trade_pnl != 0 else 0,
                        recommendation="Ensure P&L breakdown is calculated consistently."
                    ))
        
        if trades_with_breakdown == 0:
            self.issues.append(VerificationIssue(
                severity="info",
                category="pnl_breakdown",
                trade_id=None,
                description="No P&L breakdown found in any trade. Enable PnLBreakdown tracking for better transparency.",
                expected=0,
                actual=0,
                difference_pct=0,
                recommendation="Update TradeRecord to include pnl_breakdown field for detailed P&L analysis."
            ))
    
    def _generate_summary(self, trades: List[Dict], metrics: Dict) -> Dict:
        """Generate summary statistics for verification."""
        total_issues = len(self.issues)
        critical = len([i for i in self.issues if i.severity == "critical"])
        warnings = len([i for i in self.issues if i.severity == "warning"])
        
        return {
            "total_issues": total_issues,
            "critical_issues": critical,
            "warnings": warnings,
            "info_count": total_issues - critical - warnings,
            "total_pnl": metrics.get("total_pnl", 0),
            "win_rate": metrics.get("win_rate", 0),
            "total_trades": len(trades),
            "assignment_trades": len([t for t in trades if t.get("exit_reason") == "ASSIGNMENT"]),
            "profit_target_trades": len([t for t in trades if t.get("exit_reason") == "PROFIT_TARGET"]),
            "stop_loss_trades": len([t for t in trades if t.get("exit_reason") == "STOP_LOSS"]),
        }
    
    def _calculate_confidence_score(self, trades: List[Dict]) -> float:
        """Calculate confidence score based on issues found.
        
        Returns:
            Score from 0-100, where 100 is highest confidence
        """
        base_score = 100.0
        
        # Deduct for issues
        for issue in self.issues:
            if issue.severity == "critical":
                base_score -= 20
            elif issue.severity == "warning":
                base_score -= 5
            else:
                base_score -= 1
        
        return max(0.0, min(100.0, base_score))
    
    def compare_with_reference(self, results: Dict, reference: Dict) -> Dict:
        """Compare backtest results with reference (e.g., QuantConnect).
        
        Args:
            results: Local backtest results
            reference: Reference platform results
            
        Returns:
            Comparison report with discrepancies
        """
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "discrepancies": [],
            "correlation": None,
            "pnl_difference": None,
        }
        
        # Compare total P&L
        local_pnl = results.get("metrics", {}).get("total_pnl", 0)
        ref_pnl = reference.get("total_pnl", 0)
        
        if ref_pnl != 0:
            pnl_diff_pct = (local_pnl - ref_pnl) / abs(ref_pnl) * 100
            comparison["pnl_difference"] = {
                "local": local_pnl,
                "reference": ref_pnl,
                "difference_pct": pnl_diff_pct,
            }
            
            if abs(pnl_diff_pct) > 5:
                comparison["discrepancies"].append({
                    "type": "pnl_mismatch",
                    "severity": "critical" if abs(pnl_diff_pct) > 20 else "warning",
                    "description": f"P&L differs by {pnl_diff_pct:.1f}%",
                    "local": local_pnl,
                    "reference": ref_pnl,
                })
        
        # Compare win rate
        local_win_rate = results.get("metrics", {}).get("win_rate", 0)
        ref_win_rate = reference.get("win_rate", 0)
        
        if ref_win_rate != 0:
            win_rate_diff = abs(local_win_rate - ref_win_rate)
            if win_rate_diff > 5:
                comparison["discrepancies"].append({
                    "type": "win_rate_mismatch",
                    "severity": "warning",
                    "description": f"Win rate differs by {win_rate_diff:.1f}%",
                    "local": local_win_rate,
                    "reference": ref_win_rate,
                })
        
        return comparison
    
    def generate_report_markdown(self, report: VerificationReport) -> str:
        """Generate human-readable markdown report.
        
        Args:
            report: VerificationReport to format
            
        Returns:
            Markdown formatted string
        """
        md = f"""# Backtest Verification Report

**Generated:** {report.timestamp}  
**Backtest ID:** {report.backtest_id}  
**Status:** {"PASSED" if report.passed else "FAILED"}  
**Confidence Score:** {report.confidence_score:.1f}/100

## Summary

| Metric | Value |
|--------|-------|
| Total Trades | {report.total_trades} |
| Critical Issues | {report.summary.get("critical_issues", 0)} |
| Warnings | {report.summary.get("warnings", 0)} |
| Info Messages | {report.summary.get("info_count", 0)} |
| Total P&L | ${report.summary.get("total_pnl", 0):.2f} |
| Win Rate | {report.summary.get("win_rate", 0):.1f}% |

## Trade Breakdown

| Exit Type | Count |
|-----------|-------|
| Assignment | {report.summary.get("assignment_trades", 0)} |
| Profit Target | {report.summary.get("profit_target_trades", 0)} |
| Stop Loss | {report.summary.get("stop_loss_trades", 0)} |

## Issues Found

"""
        if report.issues:
            for issue in report.issues:
                severity_emoji = {"critical": "red_circle", "warning": "warning", "info": "information_source"}
                md += f"""### {severity_emoji.get(issue["severity"], "warning")} {issue["severity"].upper()}: {issue["category"]}

**Trade ID:** {issue.get("trade_id", "N/A")}  
**Description:** {issue["description"]}  
**Expected:** ${issue["expected"]:.2f}  
**Actual:** ${issue["actual"]:.2f}  
**Recommendation:** {issue["recommendation"]}

---
"""
        else:
            md += "No issues found. All checks passed.\n"
        
        return md


def verify_backtest_file(file_path: str) -> VerificationReport:
    """Convenience function to verify a backtest JSON file.
    
    Args:
        file_path: Path to backtest results JSON
        
    Returns:
        VerificationReport
    """
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    verifier = BacktestVerifier()
    return verifier.verify_backtest(results)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python backtest_verifier.py <backtest_json_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    report = verify_backtest_file(file_path)
    
    # Print markdown report
    verifier = BacktestVerifier()
    print(verifier.generate_report_markdown(report))