"""
Macro Regime Classification Engine.
Implements rule-based, interpretable regime identification based on volatility and yield curve dynamics.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class MacroRegimeEngine:
    """Classifies market environment into macro/volatility regimes."""

    def __init__(self, vol_window: int = 252, vol_percentile: float = 70.0, yield_invert_threshold: float = 0.0):
        self.vol_window = vol_window
        self.vol_percentile = vol_percentile
        self.yield_invert_threshold = yield_invert_threshold

    def classify_regimes(self, df_market_returns: pd.Series, df_yield_spread: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Classify each date into regime states:
        - Volatility State: 'LowVol' vs 'HighVol'
        - Yield State: 'NormalYield' vs 'InvertedYield'
        - Combined Regime: 'LowVol_NormalYield', 'HighVol_Stressed', etc.
        """
        df_reg = pd.DataFrame(index=df_market_returns.index)
        
        # 1. Rolling Volatility state
        rolling_vol = df_market_returns.rolling(self.vol_window, min_periods=60).std() * np.sqrt(252)
        vol_threshold = rolling_vol.expanding(min_periods=252).quantile(self.vol_percentile / 100.0)
        
        # Fallback for early periods
        vol_threshold = vol_threshold.fillna(0.20)
        
        df_reg["rolling_vol"] = rolling_vol
        df_reg["vol_threshold"] = vol_threshold
        df_reg["vol_regime"] = np.where(df_reg["rolling_vol"] > df_reg["vol_threshold"], "HighVol", "LowVol")

        # 2. Yield Curve State
        if df_yield_spread is not None:
            aligned_spread = df_yield_spread.reindex(df_reg.index).ffill()
            df_reg["yield_spread"] = aligned_spread
            df_reg["yield_regime"] = np.where(aligned_spread <= self.yield_invert_threshold, "InvertedYield", "NormalYield")
        else:
            df_reg["yield_regime"] = "NormalYield"

        # 3. Combined Regime State
        df_reg["regime"] = df_reg["vol_regime"] + "_" + df_reg["yield_regime"]
        return df_reg
