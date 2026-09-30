"""
Drawdown Engine.
Calculates high-water mark, drawdown series, maximum drawdown, peak/trough/recovery dates and durations.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


class DrawdownEngine:
    """Computes drawdown dynamics and peak-to-trough recovery analytics."""

    @staticmethod
    def calculate_drawdown_series(returns: pd.Series) -> tuple[pd.Series, pd.Series]:
        """
        Compute cumulative wealth index and drawdown time-series.
        wealth_t = cumprod(1 + R_t)
        peak_t = cummax(wealth_t)
        drawdown_t = (wealth_t / peak_t) - 1.0
        """
        clean_r = returns.fillna(0.0)
        wealth = (1.0 + clean_r).cumprod()
        peak = wealth.cummax()
        drawdown = (wealth / peak) - 1.0
        return wealth, drawdown

    @classmethod
    def calculate_drawdown_details(cls, returns: pd.Series) -> Dict[str, Any]:
        """
        Calculate comprehensive drawdown statistics:
        Max Drawdown, Current Drawdown, Peak Date, Trough Date, Recovery Date, Recovery Duration.
        """
        if len(returns) == 0:
            return {
                "max_drawdown": 0.0,
                "current_drawdown": 0.0,
                "peak_date": None,
                "trough_date": None,
                "recovery_date": None,
                "recovery_duration_days": None,
                "is_recovered": True
            }

        wealth, drawdown = cls.calculate_drawdown_series(returns)
        max_dd = float(drawdown.min())
        current_dd = float(drawdown.iloc[-1])

        if max_dd == 0.0:
            return {
                "max_drawdown": 0.0,
                "current_drawdown": 0.0,
                "peak_date": returns.index[0] if isinstance(returns.index, pd.DatetimeIndex) else None,
                "trough_date": returns.index[0] if isinstance(returns.index, pd.DatetimeIndex) else None,
                "recovery_date": None,
                "recovery_duration_days": 0,
                "is_recovered": True
            }

        trough_idx = drawdown.idxmin()
        trough_loc = drawdown.index.get_loc(trough_idx)

        # Peak date prior to trough
        wealth_before_trough = wealth.iloc[:trough_loc + 1]
        peak_idx = wealth_before_trough.idxmax()

        # Recovery date post trough (when wealth >= peak wealth)
        peak_val = wealth.loc[peak_idx]
        wealth_after_trough = wealth.iloc[trough_loc:]
        recovered_slice = wealth_after_trough[wealth_after_trough >= (peak_val - 1e-9)]

        if len(recovered_slice) > 0:
            recovery_idx = recovered_slice.index[0]
            is_recovered = True
            if isinstance(returns.index, pd.DatetimeIndex):
                duration = (recovery_idx - peak_idx).days
            else:
                duration = int(returns.index.get_loc(recovery_idx) - returns.index.get_loc(peak_idx))
        else:
            recovery_idx = None
            is_recovered = False
            if isinstance(returns.index, pd.DatetimeIndex):
                duration = (returns.index[-1] - peak_idx).days
            else:
                duration = int(len(returns) - returns.index.get_loc(peak_idx))

        return {
            "max_drawdown": max_dd,
            "current_drawdown": current_dd,
            "peak_date": str(peak_idx),
            "trough_date": str(trough_idx),
            "recovery_date": str(recovery_idx) if recovery_idx is not None else None,
            "recovery_duration_days": duration,
            "is_recovered": is_recovered
        }
