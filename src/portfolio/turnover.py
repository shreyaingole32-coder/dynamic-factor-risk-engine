"""
Portfolio Turnover and Weight Drift Engine.
Tracks weight drift during holding periods and calculates portfolio rebalance turnover.
"""

import numpy as np
import pandas as pd


def calculate_portfolio_turnover(w_new: pd.Series, w_old: pd.Series) -> float:
    """
    Calculate two-sided portfolio turnover:
    $$\text{Turnover} = \sum_{i} |w_{new, i} - w_{old, i}|$$
    """
    all_assets = list(set(w_new.index).union(set(w_old.index)))
    s_new = w_new.reindex(all_assets, fill_value=0.0)
    s_old = w_old.reindex(all_assets, fill_value=0.0)
    
    turnover = np.sum(np.abs(s_new - s_old))
    return float(turnover)


def calculate_drifted_weights(w_old: pd.Series, asset_returns: pd.Series) -> pd.Series:
    """Calculate asset weight drift over holding period prior to rebalance."""
    growth = (1.0 + asset_returns.reindex(w_old.index, fill_value=0.0))
    wealth_unnormalized = w_old * growth
    total_wealth = wealth_unnormalized.sum()
    if total_wealth <= 0:
        return w_old
    return wealth_unnormalized / total_wealth
