"""
Covariance Estimation Engine.
Supports Sample Covariance, Rolling Covariance, and Ledoit-Wolf Shrinkage Covariance.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


class CovarianceEstimator:
    """Estimates asset/factor covariance matrices using robust statistical techniques."""

    @staticmethod
    def estimate_sample_covariance(returns_df: pd.DataFrame) -> pd.DataFrame:
        """Standard sample covariance matrix."""
        clean_df = returns_df.dropna()
        return clean_df.cov()

    @staticmethod
    def estimate_ledoit_wolf_covariance(returns_df: pd.DataFrame) -> pd.DataFrame:
        """Ledoit-Wolf shrinkage covariance matrix."""
        clean_df = returns_df.dropna()
        if len(clean_df) < 5:
            return clean_df.cov()
        
        lw = LedoitWolf()
        lw.fit(clean_df.values)
        cov_matrix = pd.DataFrame(lw.covariance_, index=clean_df.columns, columns=clean_df.columns)
        return cov_matrix

    @classmethod
    def estimate_covariance(cls, returns_df: pd.DataFrame, method: str = "sample") -> pd.DataFrame:
        """Factory method to estimate covariance by named technique."""
        if method == "shrinkage" or method == "ledoit_wolf":
            return cls.estimate_ledoit_wolf_covariance(returns_df)
        return cls.estimate_sample_covariance(returns_df)
