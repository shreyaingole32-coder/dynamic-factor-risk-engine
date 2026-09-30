"""
Expected Shortfall (CVaR) Calculation Engine.
Calculates tail risk loss conditional on exceeding VaR threshold.
"""

import numpy as np
import pandas as pd
from src.risk.var import calculate_historical_var


def calculate_historical_expected_shortfall(returns: pd.Series, confidence_level: float = 0.975) -> float:
    """
    Calculate Historical Expected Shortfall (Conditional Value at Risk).
    ES_alpha = - E[ R | R <= -VaR_alpha ]
    Returns a positive number representing expected tail loss magnitude.
    """
    clean_r = returns.dropna()
    if len(clean_r) == 0:
        return 0.0
    
    alpha = 1.0 - confidence_level
    cutoff_quantile = np.percentile(clean_r, alpha * 100.0)
    
    tail_losses = clean_r[clean_r <= cutoff_quantile]
    if len(tail_losses) == 0:
        return float(-cutoff_quantile)
    
    expected_shortfall = -tail_losses.mean()
    return float(expected_shortfall)
