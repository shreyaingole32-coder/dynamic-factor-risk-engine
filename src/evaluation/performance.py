"""
Performance & Risk Evaluation Metrics Engine.
Computes CAGR, Sharpe, Sortino, Information Ratio, VaR, ES, Max Drawdown, and Turnover statistics.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from src.risk.volatility import calculate_annualized_volatility, calculate_downside_deviation
from src.risk.var import calculate_historical_var
from src.risk.expected_shortfall import calculate_historical_expected_shortfall
from src.risk.drawdown import DrawdownEngine


class PerformanceEvaluator:
    """Computes comprehensive quantitative performance and risk metrics."""

    def __init__(self, ann_factor: int = 252, rf_daily: float = 0.0):
        self.ann_factor = ann_factor
        self.rf_daily = rf_daily

    def evaluate_strategy(self, returns_df: pd.DataFrame, strategy_name: str) -> Dict[str, Any]:
        """
        Compute performance summary dictionary for a strategy DataFrame.
        Expects columns: net_return, gross_return, transaction_cost, turnover
        """
        net_ret = returns_df["net_return"].dropna()
        if len(net_ret) == 0:
            return {}

        total_days = len(net_ret)
        cum_ret = float((1.0 + net_ret).prod() - 1.0)
        cagr = float((1.0 + cum_ret) ** (self.ann_factor / total_days) - 1.0) if total_days > 0 else 0.0

        ann_vol = calculate_annualized_volatility(net_ret, ann_factor=self.ann_factor)
        downside_vol = calculate_downside_deviation(net_ret, target=self.rf_daily, ann_factor=self.ann_factor)

        # Excess return Sharpe & Sortino
        excess_mean_ann = (net_ret.mean() - self.rf_daily) * self.ann_factor
        sharpe = (excess_mean_ann / ann_vol) if ann_vol > 0 else 0.0
        sortino = (excess_mean_ann / downside_vol) if downside_vol > 0 else 0.0

        # Risk metrics
        var_95 = calculate_historical_var(net_ret, confidence_level=0.95)
        es_975 = calculate_historical_expected_shortfall(net_ret, confidence_level=0.975)
        dd_details = DrawdownEngine.calculate_drawdown_details(net_ret)

        avg_turnover = float(returns_df["turnover"].mean()) if "turnover" in returns_df.columns else 0.0
        total_costs = float(returns_df["transaction_cost"].sum()) if "transaction_cost" in returns_df.columns else 0.0

        return {
            "Strategy": strategy_name,
            "Cumulative Return": cum_ret,
            "CAGR": cagr,
            "Annualized Vol": ann_vol,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Historical VaR (95%)": var_95,
            "Expected Shortfall (97.5%)": es_975,
            "Max Drawdown": dd_details["max_drawdown"],
            "Avg Rebalance Turnover": avg_turnover,
            "Total Transaction Cost": total_costs
        }
