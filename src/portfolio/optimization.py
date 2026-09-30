"""
Portfolio Optimization Engine.
Implements Equal Weight, Minimum Variance, Expected Shortfall (CVaR), and Regime-Aware ES Allocations.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.risk.covariance import CovarianceEstimator
from src.risk.expected_shortfall import calculate_historical_expected_shortfall
from src.portfolio.constraints import PortfolioConstraints


class PortfolioOptimizer:
    """Portfolio optimization algorithms for quantitative allocation."""

    def __init__(self, constraints: Optional[PortfolioConstraints] = None):
        self.constraints = constraints or PortfolioConstraints()

    def optimize_equal_weight(self, asset_names: list) -> pd.Series:
        """Equal weight allocation (1/N)."""
        n = len(asset_names)
        if n == 0:
            return pd.Series()
        w = np.full(n, 1.0 / n)
        return pd.Series(w, index=asset_names)

    def optimize_minimum_variance(self, returns_df: pd.DataFrame, cov_matrix: Optional[pd.DataFrame] = None) -> pd.Series:
        """Minimum Variance portfolio optimization."""
        asset_names = returns_df.columns.tolist()
        n = len(asset_names)
        if n == 0:
            return pd.Series()

        if cov_matrix is None:
            cov_matrix = CovarianceEstimator.estimate_covariance(returns_df, method="shrinkage")

        cov_values = cov_matrix.values
        w0 = np.full(n, 1.0 / n)

        def objective(w):
            return float(w.T @ cov_values @ w)

        bounds = self.constraints.get_bounds(n)
        budget_constraint = self.constraints.get_budget_constraint()

        res = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=[budget_constraint],
            options={"ftol": 1e-9, "maxiter": 500}
        )

        if not res.success:
            print(f"[Opt Warning] MinVar optimization failed: {res.message}. Falling back to EW.")
            return self.optimize_equal_weight(asset_names)

        weights = res.x
        weights = np.maximum(0.0, weights)
        weights = weights / np.sum(weights) # Re-normalize
        return pd.Series(weights, index=asset_names)

    def optimize_expected_shortfall(self, returns_df: pd.DataFrame, alpha: float = 0.975) -> pd.Series:
        """Expected Shortfall (CVaR) portfolio optimization."""
        asset_names = returns_df.columns.tolist()
        n = len(asset_names)
        if n == 0:
            return pd.Series()

        clean_returns = returns_df.dropna().values
        if len(clean_returns) < 10:
            return self.optimize_equal_weight(asset_names)

        w0 = np.full(n, 1.0 / n)

        def objective(w):
            port_returns = pd.Series(clean_returns @ w)
            es = calculate_historical_expected_shortfall(port_returns, confidence_level=alpha)
            return float(es)

        bounds = self.constraints.get_bounds(n)
        budget_constraint = self.constraints.get_budget_constraint()

        res = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=[budget_constraint],
            options={"ftol": 1e-4, "maxiter": 50}
        )

        if not res.success:
            return self.optimize_minimum_variance(returns_df)

        weights = np.maximum(0.0, res.x)
        weights = weights / np.sum(weights)
        return pd.Series(weights, index=asset_names)
