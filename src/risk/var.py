"""
Value at Risk (VaR) Calculation Engine.
Implements Historical VaR and Parametric (Gaussian & Cornish-Fisher) VaR.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def calculate_historical_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Historical Value at Risk.
    Returns positive value representing loss magnitude.
    """
    clean_r = returns.dropna()
    if len(clean_r) == 0:
        return 0.0
    alpha = 1.0 - confidence_level
    quantile_val = np.percentile(clean_r, alpha * 100.0)
    return float(-quantile_val)


def calculate_parametric_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate Parametric (Gaussian) Value at Risk."""
    clean_r = returns.dropna()
    if len(clean_r) < 2:
        return 0.0
    mu = clean_r.mean()
    sigma = clean_r.std(ddof=1)
    z_score = norm.ppf(confidence_level)
    var_val = -(mu - z_score * sigma)
    return float(var_val)
