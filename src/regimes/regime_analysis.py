"""
Regime-Conditioned Factor Analysis.
Calculates factor mean, volatility, Sharpe, skewness, and cross-factor correlation matrices across macro regimes.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from src.factors.factor_statistics import FactorStatisticsEngine


class RegimeAnalysisEngine:
    """Analyzes factor behavior conditioned on macro regime states (H1 and H2 testing)."""

    def __init__(self, ann_factor: int = 252):
        self.stats_engine = FactorStatisticsEngine(ann_factor=ann_factor)

    def compute_regime_conditioned_stats(self, df_factors: pd.DataFrame, df_regimes: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute factor metrics and correlation matrices grouped by regime.
        """
        # Align index/dates
        df_combined = pd.merge(df_factors, df_regimes[["regime"]], left_on="date", right_index=True, how="inner")
        
        regime_stats = {}
        regime_corrs = {}

        factor_cols = [c for c in df_factors.columns if c not in ["date", "rf"]]

        for regime_name, group in df_combined.groupby("regime"):
            if len(group) < 20: # Skip sparse regimes
                continue
            
            # Summary statistics per factor in this regime
            stats_df = self.stats_engine.compute_summary_table(group, factor_cols=factor_cols)
            regime_stats[regime_name] = stats_df

            # Cross-factor correlation matrix in this regime
            corr_matrix = group[factor_cols].corr()
            regime_corrs[regime_name] = corr_matrix

        return {
            "regime_stats": regime_stats,
            "regime_corrs": regime_corrs
        }
