"""
Factor Regression Engine.
Executes multi-factor OLS regression (FF5 + Momentum) with Newey-West robust standard errors.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


class FactorRegressionEngine:
    """Estimates asset factor exposures (betas) and risk-adjusted excess returns (alpha)."""

    def __init__(self, use_hac: bool = True, max_lags: int = 5):
        self.use_hac = use_hac
        self.max_lags = max_lags

    def run_regression(self, y: pd.Series, X_factors: pd.DataFrame, rf: pd.Series = None) -> Dict[str, Any]:
        """
        Run multi-factor OLS regression: (R_i - R_f) = alpha + sum(beta_k * F_k) + eps
        """
        # Align indexes
        if rf is not None:
            y_excess = y - rf
        else:
            y_excess = y

        data = pd.concat([y_excess.rename("y"), X_factors], axis=1).dropna()
        y_clean = data["y"]
        X_clean = data.drop(columns=["y"])
        X_design = sm.add_constant(X_clean)

        model = sm.OLS(y_clean, X_design)
        if self.use_hac:
            results = model.fit(cov_type="HAC", cov_kwds={"maxlags": self.max_lags})
        else:
            results = model.fit()

        betas = results.params.to_dict()
        std_errors = results.bse.to_dict()
        t_stats = results.tvalues.to_dict()
        p_values = results.pvalues.to_dict()
        conf_int = results.conf_int().to_dict(orient="index")

        return {
            "alpha": betas.get("const", 0.0),
            "alpha_ann": betas.get("const", 0.0) * 252, # Annualized alpha
            "betas": {k: v for k, v in betas.items() if k != "const"},
            "std_errors": std_errors,
            "t_stats": t_stats,
            "p_values": p_values,
            "conf_intervals": conf_int,
            "r_squared": results.rsquared,
            "r_squared_adj": results.rsquared_adj,
            "residual_vol_ann": np.sqrt(results.mse_resid) * np.sqrt(252),
            "n_observations": int(results.nobs),
            "summary_object": results
        }
