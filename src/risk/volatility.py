"""
Volatility and Downside Risk Metrics.
Calculates annualized volatility and downside deviation (semi-volatility below target).
"""

import numpy as np
import pandas as pd


def calculate_annualized_volatility(returns: pd.Series, ann_factor: int = 252) -> float:
    """Calculate annualized volatility from daily or monthly returns."""
    clean_r = returns.dropna()
    if len(clean_r) < 2:
        return 0.0
    return float(clean_r.std(ddof=1) * np.sqrt(ann_factor))


def calculate_downside_deviation(returns: pd.Series, target: float = 0.0, ann_factor: int = 252) -> float:
    """
    Calculate downside deviation (semi-deviation) below a target rate (e.g. 0.0 or risk-free).
    $$\text{DD} = \sqrt{\frac{1}{T} \sum \min(0, R_t - \text{target})^2} \times \sqrt{\text{ann\_factor}}$$
    """
    clean_r = returns.dropna()
    if len(clean_r) == 0:
        return 0.0
    downside_diff = np.minimum(0.0, clean_r - target)
    downside_var = np.mean(downside_diff ** 2)
    return float(np.sqrt(downside_var) * np.sqrt(ann_factor))
