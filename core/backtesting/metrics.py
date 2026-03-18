"""Performance metrics calculator for backtest results."""

import math
import numpy as np
from collections import defaultdict


class PerformanceMetrics:
    """Calculate comprehensive performance metrics from backtest trades and daily P&L."""

    @staticmethod
    def calculate(
        trades: list[dict],
        daily_pnl: list[dict],
        initial_capital: float,
    ) -> dict:
        if not trades:
            return {
                "total_return_pct": 0,
                "annualized_return_pct": 0,
                "max_drawdown_pct": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "win_rate": 0,
                "total_trades": 0,
                "avg_profit": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "monthly_returns": {},
            }

        # Basic trade stats
        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # Calculate total P&L from daily_pnl (includes option P&L + stock P&L)
        # This ensures Wheel/BinbinGod strategies' stock P&L is properly counted
        if daily_pnl:
            final_cumulative_pnl = daily_pnl[-1]["cumulative_pnl"]
            total_pnl = final_cumulative_pnl  # Use final cumulative P&L (includes all realized + unrealized)
        else:
            total_pnl = sum(pnls)  # Fallback to trades only
        
        total_return_pct = (total_pnl / initial_capital) * 100

        # Annualized return
        if daily_pnl and len(daily_pnl) > 1:
            n_days = len(daily_pnl)
            n_years = n_days / 252
            if n_years > 0:
                final_val = initial_capital + total_pnl
                annualized = ((final_val / initial_capital) ** (1 / n_years) - 1) * 100
            else:
                annualized = 0
        else:
            annualized = 0

        # Daily returns for Sharpe/Sortino
        daily_returns = []
        if daily_pnl:
            prev = initial_capital
            for d in daily_pnl:
                curr = initial_capital + d["cumulative_pnl"]
                if prev > 0:
                    daily_returns.append((curr - prev) / prev)
                prev = curr

        # Sharpe ratio (annualized, assuming 252 trading days)
        sharpe = 0
        if daily_returns:
            mean_r = np.mean(daily_returns)
            std_r = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 0
            if std_r > 0:
                sharpe = (mean_r / std_r) * math.sqrt(252)

        # Sortino ratio
        sortino = 0
        if daily_returns:
            downside = [r for r in daily_returns if r < 0]
            downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0
            if downside_std > 0:
                sortino = (np.mean(daily_returns) / downside_std) * math.sqrt(252)

        # Max drawdown
        max_dd = 0
        if daily_pnl:
            equity = [initial_capital + d["cumulative_pnl"] for d in daily_pnl]
            peak = equity[0]
            for val in equity:
                peak = max(peak, val)
                dd = (peak - val) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)

        # Monthly returns
        monthly_returns = defaultdict(float)
        if daily_pnl:
            prev_cum = 0
            for d in daily_pnl:
                date_str = d["date"]
                year_month = date_str[:7]  # YYYY-MM
                day_pnl = d["cumulative_pnl"] - prev_cum
                monthly_returns[year_month] += day_pnl
                prev_cum = d["cumulative_pnl"]

        # Convert monthly to % of initial capital
        monthly_pct = {k: (v / initial_capital) * 100 for k, v in monthly_returns.items()}

        return {
            "total_return_pct": round(total_return_pct, 2),
            "annualized_return_pct": round(annualized, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
            "total_trades": len(trades),
            "avg_profit": round(np.mean(wins), 2) if wins else 0,
            "avg_loss": round(np.mean(losses), 2) if losses else 0,
            "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) != 0 else 999,
            "monthly_returns": monthly_pct,
        }
