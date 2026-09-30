"""
Performance Attribution Engine.
Decomposes portfolio returns into factor contributions, allocation changes, and residual components.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


class PerformanceAttributionEngine:
    """Decomposes realized strategy returns into factor exposures and active allocation decisions."""

    @staticmethod
    def compute_factor_attribution(weights_df: pd.DataFrame, factor_returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate contribution of each factor to total strategy return over time:
        $$\text{Contribution}_{k, t} = w_{k, t} \times F_{k, t}$$
        """
        # Align weights and returns by date
        merged = pd.merge(weights_df, factor_returns_df, on="date", how="inner")
        
        contrib_records = []
        for date_val, group in merged.groupby("date"):
            record = {"date": date_val}
            total_contrib = 0.0
            for idx, row in group.iterrows():
                asset = row["asset"]
                weight = row["weight"]
                f_ret = row.get(asset, 0.0)
                contrib = weight * f_ret
                record[asset] = contrib
                total_contrib += contrib
            record["total_factor_return"] = total_contrib
            contrib_records.append(record)

        df_attrib = pd.DataFrame(contrib_records).sort_values("date").reset_index(drop=True)
        return df_attrib
