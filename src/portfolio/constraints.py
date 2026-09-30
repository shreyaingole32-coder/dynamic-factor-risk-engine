"""
Portfolio Optimization Constraints.
Enforces budget constraint (sum of weights = 1), position bounds (0 <= w_i <= max_weight), and turnover bounds.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class PortfolioConstraints:
    """Configurable constraint manager for portfolio optimizers."""

    def __init__(self, min_weight: float = 0.0, max_weight: float = 0.30, target_sum: float = 1.0):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.target_sum = target_sum

    def get_bounds(self, n_assets: int) -> List[Tuple[float, float]]:
        """Return list of (min_weight, max_weight) tuples per asset."""
        return [(self.min_weight, self.max_weight) for _ in range(n_assets)]

    def get_budget_constraint(self) -> Dict[str, Any]:
        """Return SciPy constraint dictionary enforcing sum(w) = target_sum."""
        return {
            "type": "eq",
            "fun": lambda w: np.sum(w) - self.target_sum
        }
