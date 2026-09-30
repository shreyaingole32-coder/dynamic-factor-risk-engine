"""
Factor Statistics Engine.
Calculates annualized returns, volatility, Sharpe ratio, skewness, kurtosis, and rolling metrics.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


class FactorStatisticsEngine:
    """Calculates statistical summary properties and rolling factor dynamics."""

    def __init__(self, ann_factor: int = 252):
        self.ann_factor = ann_factor

    def compute_summary_table(self, df_factors: pd.DataFrame, factor_cols: list = None) -> pd.DataFrame:
        """
        Compute summary table for factor returns.
        Columns: Factor, Mean (Ann), Volatility (Ann), Sharpe Ratio, Skewness, Kurtosis, Max Drawdown
        """
        if factor_cols is None:
            factor_cols = [c for c in df_factors.columns if c not in ["date", "rf"]]

        results = []
        for col in factor_cols:
            s = df_factors[col].dropna()
            if len(s) == 0:
                continue

            mean_daily = s.mean()
            vol_daily = s.std(ddof=1)
            ann_return = mean_daily * self.ann_factor
            ann_vol = vol_daily * np.sqrt(self.ann_factor)
            sharpe = (ann_return / ann_vol) if ann_vol > 0 else np.nan

            skew = s.skew()
            kurt = s.kurtosis() # Excess kurtosis

            # Max Drawdown of cumulative factor wealth
            cum_wealth = (1 + s).cumprod()
            peak = cum_wealth.cummax()
            dd = (cum_wealth - peak) / peak
            max_dd = dd.min()

            results.append({
                "Factor": col.upper(),
                "Mean (Ann)": ann_return,
                "Vol (Ann)": ann_vol,
                "Sharpe": sharpe,
                "Skewness": skew,
                "Kurtosis": kurt,
                "Max Drawdown": max_dd
            })

        return pd.DataFrame(results)

    def compute_rolling_metrics(self, df_factors: pd.DataFrame, window: int = 252, factor_cols: list = None) -> Dict[str, pd.DataFrame]:
        """Compute rolling volatility, rolling Sharpe ratio, and rolling factor correlations."""
        if factor_cols is None:
            factor_cols = [c for c in df_factors.columns if c not in ["date", "rf"]]

        df_calc = df_factors.set_index("date")[factor_cols]

        rolling_vol = df_calc.rolling(window).std() * np.sqrt(self.ann_factor)
        rolling_mean = df_calc.rolling(window).mean() * self.ann_factor
        rolling_sharpe = rolling_mean / rolling_vol

        # Rolling correlations between MKT and SMB/HML/MOM etc.
        rolling_corrs = df_calc.rolling(window).corr()

        return {
            "rolling_vol": rolling_vol,
            "rolling_sharpe": rolling_sharpe,
            "rolling_corrs": rolling_corrs
        }
