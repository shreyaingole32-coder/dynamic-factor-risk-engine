"""
Walk-Forward Event-Aware Backtesting Engine.
Executes out-of-sample portfolio rebalancing without future data leakage.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from src.portfolio.optimization import PortfolioOptimizer
from src.portfolio.turnover import calculate_portfolio_turnover, calculate_drifted_weights
from src.backtest.transaction_costs import TransactionCostModel
from src.regimes.regime_definition import MacroRegimeEngine
from src.risk.volatility import calculate_annualized_volatility
from src.risk.var import calculate_historical_var
from src.risk.expected_shortfall import calculate_historical_expected_shortfall
from src.risk.drawdown import DrawdownEngine


class WalkForwardBacktester:
    """Rigorous walk-forward backtest loop for factor allocation strategies."""

    def __init__(
        self,
        estimation_window: int = 252,
        rebalance_freq: str = "monthly",
        cost_bps: float = 10.0,
        es_alpha: float = 0.975,
        max_weight: float = 0.30
    ):
        self.estimation_window = estimation_window
        self.rebalance_freq = rebalance_freq
        self.cost_model = TransactionCostModel(cost_bps=cost_bps)
        self.es_alpha = es_alpha
        self.max_weight = max_weight
        self.optimizer = PortfolioOptimizer()
        self.regime_engine = MacroRegimeEngine(vol_window=252)

    def run_backtest(
        self,
        df_factors: pd.DataFrame,
        df_yield_spread: pd.Series = None,
        asset_cols: list = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Execute walk-forward backtest across four allocation models:
        1. EqualWeight
        2. MinVariance
        3. ExpectedShortfall
        4. RegimeAware_ES
        """
        if asset_cols is None:
            asset_cols = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
        
        # Ensure data is sorted
        df_factors = df_factors.sort_values("date").reset_index(drop=True)
        dates = df_factors["date"].values
        df_returns_indexed = df_factors.set_index("date")[asset_cols]
        market_series = df_factors.set_index("date")["mkt_rf"]

        # Pre-classify regimes for reference lookup at time t
        df_regimes = self.regime_engine.classify_regimes(market_series, df_yield_spread)

        # Identify rebalance dates
        if self.rebalance_freq == "monthly":
            rebalance_dates = df_factors.groupby(df_factors["date"].dt.to_period("M"))["date"].max().values
        elif self.rebalance_freq == "quarterly":
            rebalance_dates = df_factors.groupby(df_factors["date"].dt.to_period("Q"))["date"].max().values
        else: # Daily
            rebalance_dates = dates

        strategies = ["EqualWeight", "MinVariance", "ExpectedShortfall", "RegimeAware_ES"]
        results = {strat: {"returns": [], "weights": [], "turnover": []} for strat in strategies}

        # Weight tracking
        current_weights = {strat: pd.Series(1.0 / len(asset_cols), index=asset_cols) for strat in strategies}

        # Backtest loop over daily observations
        for i in range(self.estimation_window, len(df_factors)):
            t_date = dates[i]
            prev_date = dates[i - 1]
            daily_asset_ret = df_returns_indexed.loc[t_date]

            # Check if t_date is a rebalance date
            is_rebalance = (t_date in rebalance_dates)

            if is_rebalance:
                # STRICT LOOK-AHEAD CONTROL: Slice data ONLY UP TO i-1
                hist_window = df_returns_indexed.iloc[i - self.estimation_window : i]
                current_regime = df_regimes.loc[t_date, "regime"] if t_date in df_regimes.index else "LowVol_NormalYield"

                # 1. Equal Weight
                new_w_ew = self.optimizer.optimize_equal_weight(asset_cols)

                # 2. Minimum Variance
                new_w_mv = self.optimizer.optimize_minimum_variance(hist_window)

                # 3. Expected Shortfall
                new_w_es = self.optimizer.optimize_expected_shortfall(hist_window, alpha=self.es_alpha)

                # 4. Regime-Aware ES
                # Filter historical window for matching regime observations if available
                matching_dates = df_regimes.iloc[i - self.estimation_window : i]
                matching_indices = matching_dates[matching_dates["regime"] == current_regime].index
                regime_hist = hist_window.reindex(matching_indices).dropna()

                if len(regime_hist) >= 30:
                    new_w_reg_es = self.optimizer.optimize_expected_shortfall(regime_hist, alpha=self.es_alpha)
                else:
                    new_w_reg_es = new_w_es # Fallback to unconditional ES

                new_weights_dict = {
                    "EqualWeight": new_w_ew,
                    "MinVariance": new_w_mv,
                    "ExpectedShortfall": new_w_es,
                    "RegimeAware_ES": new_w_reg_es
                }

                # Calculate turnover & update current_weights
                for strat in strategies:
                    old_w = current_weights[strat]
                    target_w = new_weights_dict[strat]
                    turnover = calculate_portfolio_turnover(target_w, old_w)
                    current_weights[strat] = target_w
                    results[strat]["turnover"].append({"date": t_date, "turnover": turnover})

                    # Record weight
                    for a, w_val in target_w.items():
                        results[strat]["weights"].append({"date": t_date, "asset": a, "weight": float(w_val)})

            # Calculate daily strategy performance
            for strat in strategies:
                w = current_weights[strat]
                gross_ret = float(np.sum(w * daily_asset_ret))
                
                # Deduct transaction cost on rebalance day
                if is_rebalance:
                    turnover_val = results[strat]["turnover"][-1]["turnover"]
                    net_ret, cost = self.cost_model.calculate_net_return(gross_ret, turnover_val)
                else:
                    net_ret, cost = gross_ret, 0.0

                results[strat]["returns"].append({
                    "date": t_date,
                    "gross_return": gross_ret,
                    "transaction_cost": cost,
                    "net_return": net_ret
                })

                # Drift weights for non-rebalance day
                current_weights[strat] = calculate_drifted_weights(w, daily_asset_ret)

        # Structure DataFrame output per strategy
        output_dfs = {}
        for strat in strategies:
            df_ret = pd.DataFrame(results[strat]["returns"])
            df_ret["date"] = pd.to_datetime(df_ret["date"])
            df_ret = df_ret.set_index("date")
            output_dfs[strat] = df_ret

        return output_dfs
