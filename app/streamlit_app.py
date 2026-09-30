"""
Streamlit Quantitative Research Terminal.
Interactive web GUI for exploring factor statistics, macro regimes, portfolio optimization, and backtests.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

# Add root directory to path
sys.path.insert(0, os.path.abspath("."))

from src.ingestion.french_loader import FrenchDataLoader
from src.factors.factor_statistics import FactorStatisticsEngine
from src.regimes.regime_definition import MacroRegimeEngine
from src.backtest.engine import WalkForwardBacktester
from src.evaluation.performance import PerformanceEvaluator

st.set_page_config(
    page_title="Factor & Tail-Risk Research Terminal",
    page_icon="📈",
    layout="wide"
)

st.title("Dynamic Multi-Factor Portfolio & Tail-Risk Research Engine")
st.caption("Empirical Analysis of Factor Performance, Macro Regimes, and Risk-Aware Portfolio Construction")

@st.cache_data
def load_data():
    french_loader = FrenchDataLoader(raw_dir="data/raw")
    df_factors, metadata = french_loader.load_combined_daily_factors()
    return df_factors, metadata

df_factors, metadata = load_data()

# Navigation tabs
tabs = st.tabs([
    "Overview", 
    "Factor Research", 
    "Regime Analysis", 
    "Portfolio Lab", 
    "Backtest", 
    "Risk", 
    "Attribution", 
    "Research Notes"
])

# 1. Overview
with tabs[0]:
    st.header("Project & Research Overview")
    st.markdown("""
    **Core Research Question**:  
    *Does incorporating changing factor behavior and tail-risk conditions into portfolio construction produce materially different risk-adjusted outcomes than static or naive portfolio allocation?*
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset Observations", f"{len(df_factors):,}")
    col2.metric("Start Date", str(df_factors['date'].min().date()))
    col3.metric("End Date", str(df_factors['date'].max().date()))
    col4.metric("Factors Tracked", "FF5 + Momentum")

    st.subheader("Data Source Metadata")
    st.json(metadata)

# 2. Factor Research
with tabs[1]:
    st.header("Factor Summary Statistics")
    stats_engine = FactorStatisticsEngine()
    stats_df = stats_engine.compute_summary_table(df_factors)
    st.dataframe(stats_df, use_container_width=True)

    st.subheader("Factor Correlation Matrix")
    factor_cols = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
    corr_df = df_factors[factor_cols].corr()
    st.dataframe(corr_df.style.background_gradient(cmap="coolwarm", axis=None), use_container_width=True)

# 3. Regime Analysis
with tabs[2]:
    st.header("Macro Regime Analysis")
    regime_engine = MacroRegimeEngine(vol_window=252)
    market_series = df_factors.set_index("date")["mkt_rf"]
    df_regimes = regime_engine.classify_regimes(market_series)

    st.subheader("Regime Distribution")
    st.bar_chart(df_regimes["regime"].value_counts())

# 4. Portfolio Lab
with tabs[3]:
    st.header("Interactive Portfolio Optimizer")
    max_w = st.slider("Max Position Constraint (w_max)", 0.10, 1.00, 0.30, 0.05)
    st.info(f"Configured Maximum Weight per Factor: {max_w:.2f}")

# 5. Backtest
with tabs[4]:
    st.header("Walk-Forward Backtest Results")
    st.markdown("Run `python run_pipeline.py` to regenerate complete walk-forward simulation.")
    
    if st.button("Run Quick Simulation"):
        with st.spinner("Running walk-forward backtest..."):
            bt = WalkForwardBacktester(estimation_window=252, rebalance_freq="monthly")
            results = bt.run_backtest(df_factors.tail(2520)) # Last ~10 years
            
            chart_df = pd.DataFrame()
            for strat_name, df_ret in results.items():
                chart_df[strat_name] = (1.0 + df_ret["net_return"]).cumprod()
            
            st.line_chart(chart_df)

# 6. Risk
with tabs[5]:
    st.header("Tail-Risk & Drawdown Engine")
    st.markdown("Detailed breakdown of Value-at-Risk (95%) and Expected Shortfall (97.5%).")

# 7. Attribution
with tabs[6]:
    st.header("Factor Performance Attribution")
    st.markdown("Factor return contribution vs active allocation decisions.")

# 8. Research Notes
with tabs[7]:
    st.header("Interview Guide & Technical Defense Notes")
    if os.path.exists("docs/interview_notes.md"):
        with open("docs/interview_notes.md", "r") as f:
            st.markdown(f.read())
    else:
        st.write("Interview notes available in docs/interview_notes.md")
