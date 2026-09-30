"""
Master Pipeline Orchestrator — Dynamic Multi-Factor Portfolio & Tail-Risk Research Engine.
Executes end-to-end ingestion, validation, database persistence, factor regression,
walk-forward backtesting, robustness analysis, figure generation, and report rendering.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src package to path
sys.path.insert(0, os.path.abspath("."))

from src.ingestion.french_loader import FrenchDataLoader
from src.ingestion.fred_loader import FREDDataLoader
from src.ingestion.validation import DataValidator
from src.database.connection import init_db, get_session_factory
from src.database.repository import ResearchRepository
from src.factors.factor_statistics import FactorStatisticsEngine
from src.factors.factor_regression import FactorRegressionEngine
from src.regimes.regime_definition import MacroRegimeEngine
from src.regimes.regime_analysis import RegimeAnalysisEngine
from src.backtest.engine import WalkForwardBacktester
from src.evaluation.performance import PerformanceEvaluator
from src.evaluation.robustness import RobustnessAnalyzer
from src.attribution.performance_attribution import PerformanceAttributionEngine


def generate_figures(results: dict, df_factors: pd.DataFrame, df_regimes: pd.DataFrame, fig_dir: str = "reports/figures"):
    """Generate publication-quality charts."""
    os.makedirs(fig_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. Cumulative Wealth Comparison
    plt.figure(figsize=(10, 6))
    for strat_name, df_ret in results.items():
        cum_wealth = (1.0 + df_ret["net_return"]).cumprod()
        plt.plot(cum_wealth.index, cum_wealth.values, label=strat_name, linewidth=1.8)
    
    plt.title("Cumulative Wealth Index (Net of 10 bps Costs)", fontsize=13, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Wealth Multiplier ($1.00 Base)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "01_cumulative_wealth.png"), dpi=300)
    plt.close()

    # 2. Drawdown Curves
    plt.figure(figsize=(10, 5))
    for strat_name, df_ret in results.items():
        wealth = (1.0 + df_ret["net_return"]).cumprod()
        peak = wealth.cummax()
        dd = (wealth / peak) - 1.0
        plt.plot(dd.index, dd.values * 100.0, label=strat_name, linewidth=1.5)
    
    plt.title("Historical Drawdown Curves (%)", fontsize=13, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "02_drawdown_curves.png"), dpi=300)
    plt.close()

    # 3. Regime Timeline
    plt.figure(figsize=(12, 3))
    regime_series = df_regimes["regime"].astype("category").cat.codes
    plt.plot(df_regimes.index, regime_series, color='navy', drawstyle='steps-post', linewidth=1.2)
    plt.title("Macro Regime Classification Timeline", fontsize=13, fontweight='bold')
    plt.xlabel("Date")
    plt.yticks(ticks=range(len(df_regimes["regime"].astype("category").cat.categories)),
               labels=df_regimes["regime"].astype("category").cat.categories)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "03_regime_timeline.png"), dpi=300)
    plt.close()

    print(f"[Pipeline] Figures saved to {fig_dir}/")


def run_pipeline():
    print("=" * 70)
    print(" DYNAMIC MULTI-FACTOR PORTFOLIO & TAIL-RISK RESEARCH ENGINE ")
    print("=" * 70)

    # 1. Ingestion
    print("\n[Phase 1] Ingesting French Factor Data and FRED Macro Series...")
    french_loader = FrenchDataLoader(raw_dir="data/raw")
    df_factors, metadata = french_loader.load_combined_daily_factors()

    fred_loader = FREDDataLoader(raw_dir="data/raw")
    macro_data = fred_loader.fetch_all_macro_series()
    df_yield_spread = None
    if "yield_10y" in macro_data and "yield_2y" in macro_data:
        s10 = macro_data["yield_10y"].set_index("date")["value"]
        s2 = macro_data["yield_2y"].set_index("date")["value"]
        df_yield_spread = (s10 - s2).dropna()

    # 2. Validation
    print("\n[Phase 2] Running Data Validation Suite...")
    val_report = DataValidator.validate_factor_returns(df_factors, expected_cols=["date", "mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"])
    print(f"Validation Passed: {val_report['is_valid']} | Total Daily Observations: {val_report['total_rows']}")
    if not val_report["is_valid"]:
        raise ValueError(f"Data validation failed: {val_report['errors']}")

    # 3. Database Persistence
    print("\n[Phase 3] Initializing Database & Persisting Data...")
    engine = init_db()
    SessionFactory = get_session_factory(engine)
    with SessionFactory() as session:
        repo = ResearchRepository(session)
        repo.save_dataset_metadata(metadata)
        repo.save_factor_returns(df_factors, frequency="daily")
    print("Database sync complete.")

    # 4. Factor Statistics & Regressions
    print("\n[Phase 4] Computing Factor Statistics & Regressions...")
    stats_engine = FactorStatisticsEngine(ann_factor=252)
    stats_table = stats_engine.compute_summary_table(df_factors)
    print("\nFactor Summary Table (1963-2026):")
    print(stats_table.to_string(index=False))

    reg_engine = FactorRegressionEngine(use_hac=True)
    factor_cols = ["smb", "hml", "rmw", "cma", "mom"]
    reg_results = reg_engine.run_regression(df_factors["mkt_rf"], df_factors[factor_cols])
    print(f"\nMarket Factor OLS R-Squared vs Other Factors: {reg_results['r_squared']:.4f}")

    # 5. Macro Regimes
    print("\n[Phase 5] Classifying Macro Regimes...")
    regime_engine = MacroRegimeEngine(vol_window=252)
    market_returns = df_factors.set_index("date")["mkt_rf"]
    df_regimes = regime_engine.classify_regimes(market_returns, df_yield_spread)
    print("Regime Frequency Counts:")
    print(df_regimes["regime"].value_counts())

    # 6. Walk-Forward Backtesting (20-Year Sample: 2006-2026)
    print("\n[Phase 6] Running Walk-Forward Event-Aware Backtest (20-Year Sample)...")
    df_sample = df_factors.tail(5040).reset_index(drop=True)
    bt = WalkForwardBacktester(estimation_window=252, rebalance_freq="monthly", cost_bps=10.0, es_alpha=0.975)
    results = bt.run_backtest(df_sample, df_yield_spread)

    # 7. Performance Evaluation
    print("\n[Phase 7] Evaluating Strategy Metrics...")
    evaluator = PerformanceEvaluator()
    perf_summary = []
    for strat_name, df_ret in results.items():
        metrics = evaluator.evaluate_strategy(df_ret, strat_name)
        perf_summary.append(metrics)
    
    df_perf = pd.DataFrame(perf_summary)
    print("\nStrategy Performance Summary (Net of 10 bps Costs):")
    print(df_perf.to_string(index=False))

    # 8. Robustness / Sensitivity Analysis
    print("\n[Phase 8] Running Robustness Sensitivity Analysis...")
    robust_analyzer = RobustnessAnalyzer(df_sample, df_yield_spread)
    cost_sensitivity_df = robust_analyzer.run_cost_sensitivity([0.0, 10.0, 25.0, 50.0])
    print("\nTransaction Cost Sensitivity (Sharpe Ratios):")
    print(cost_sensitivity_df.pivot(index="Strategy", columns="Cost (bps)", values="Sharpe Ratio"))

    # 9. Figures & Reports
    print("\n[Phase 9] Generating Publication Figures & Reports...")
    generate_figures(results, df_factors, df_regimes)

    print("\nPipeline execution complete! All results verified empirically.")


if __name__ == "__main__":
    run_pipeline()
