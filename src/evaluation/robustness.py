"""
Robustness & Sensitivity Analysis Engine (Model Attack Suite).
Evaluates strategy stability across estimation windows, transaction cost tiers, and risk parameters.
"""

from typing import Dict, List, Any
import pandas as pd
from src.backtest.engine import WalkForwardBacktester
from src.evaluation.performance import PerformanceEvaluator


class RobustnessAnalyzer:
    """Stress tests model parameters to prevent overfitting and cherry-picking."""

    def __init__(self, df_factors: pd.DataFrame, df_yield_spread: pd.Series = None):
        self.df_factors = df_factors
        self.df_yield_spread = df_yield_spread
        self.evaluator = PerformanceEvaluator()

    def run_cost_sensitivity(self, cost_bps_list: list = None) -> pd.DataFrame:
        """Evaluate net Sharpe ratios across varying transaction cost assumptions."""
        if cost_bps_list is None:
            cost_bps_list = [0.0, 10.0, 25.0, 50.0]

        records = []
        for cost in cost_bps_list:
            bt = WalkForwardBacktester(estimation_window=252, rebalance_freq="monthly", cost_bps=cost)
            results = bt.run_backtest(self.df_factors, self.df_yield_spread)
            
            for strat_name, df_ret in results.items():
                perf = self.evaluator.evaluate_strategy(df_ret, strat_name)
                records.append({
                    "Cost (bps)": cost,
                    "Strategy": strat_name,
                    "CAGR": perf["CAGR"],
                    "Sharpe Ratio": perf["Sharpe Ratio"],
                    "Max Drawdown": perf["Max Drawdown"]
                })

        return pd.DataFrame(records)

    def run_window_sensitivity(self, windows: list = None) -> pd.DataFrame:
        """Evaluate strategy performance across estimation window lengths (60, 126, 252 days)."""
        if windows is None:
            windows = [60, 126, 252]

        records = []
        for w in windows:
            bt = WalkForwardBacktester(estimation_window=w, rebalance_freq="monthly", cost_bps=10.0)
            results = bt.run_backtest(self.df_factors, self.df_yield_spread)
            
            for strat_name, df_ret in results.items():
                perf = self.evaluator.evaluate_strategy(df_ret, strat_name)
                records.append({
                    "Estimation Window": w,
                    "Strategy": strat_name,
                    "CAGR": perf["CAGR"],
                    "Sharpe Ratio": perf["Sharpe Ratio"],
                    "Max Drawdown": perf["Max Drawdown"]
                })

        return pd.DataFrame(records)
