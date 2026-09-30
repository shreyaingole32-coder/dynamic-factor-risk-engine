"""
SQLAlchemy database models for structured research data storage.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Float, Date, DateTime, Text, PrimaryKeyConstraint, Index
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class DatasetMetadata(Base):
    """Metadata catalog for external data sources."""
    __tablename__ = "datasets"

    dataset_id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    source = Column(String(50), nullable=False)
    source_url = Column(Text, nullable=False)
    frequency = Column(String(20), nullable=False)
    retrieved_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    units = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)


class FactorReturn(Base):
    """Daily and Monthly Fama-French + Momentum Factor Returns (in decimals)."""
    __tablename__ = "factor_returns"

    date = Column(Date, primary_key=True)
    frequency = Column(String(10), primary_key=True, default="daily") # 'daily' or 'monthly'
    mkt_rf = Column(Float, nullable=False)  # Market Excess Return (R_m - R_f)
    smb = Column(Float, nullable=False)     # Small Minus Big
    hml = Column(Float, nullable=False)     # High Minus Low (Value)
    rmw = Column(Float, nullable=False)     # Robust Minus Weak (Profitability)
    cma = Column(Float, nullable=False)     # Conservative Minus Aggressive (Investment)
    mom = Column(Float, nullable=True)      # Momentum
    rf = Column(Float, nullable=False)      # Risk-Free Rate

    __table_args__ = (
        Index("idx_factor_date_freq", "date", "frequency"),
    )


class MacroObservation(Base):
    """Economic and financial macro observations from FRED/ALFRED."""
    __tablename__ = "macro_observations"

    date = Column(Date, primary_key=True)
    series_id = Column(String(50), primary_key=True)
    value = Column(Float, nullable=True)
    observation_date = Column(Date, nullable=False)
    release_date = Column(Date, nullable=True)

    __table_args__ = (
        Index("idx_macro_series_date", "series_id", "date"),
    )


class PortfolioWeight(Base):
    """Historical portfolio weight allocations per strategy."""
    __tablename__ = "portfolio_weights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    strategy = Column(String(50), nullable=False)
    asset = Column(String(50), nullable=False)
    weight = Column(Float, nullable=False)

    __table_args__ = (
        Index("idx_weights_strat_date", "strategy", "date"),
    )


class PortfolioReturn(Base):
    """Realized portfolio returns, turnover, and transaction costs."""
    __tablename__ = "portfolio_returns"

    date = Column(Date, primary_key=True)
    strategy = Column(String(50), primary_key=True)
    gross_return = Column(Float, nullable=False)
    turnover = Column(Float, nullable=False, default=0.0)
    transaction_cost = Column(Float, nullable=False, default=0.0)
    net_return = Column(Float, nullable=False)

    __table_args__ = (
        Index("idx_returns_strat_date", "strategy", "date"),
    )


class RiskMetric(Base):
    """Calculated historical risk metrics per strategy."""
    __tablename__ = "risk_metrics"

    date = Column(Date, primary_key=True)
    strategy = Column(String(50), primary_key=True)
    volatility = Column(Float, nullable=True)
    var_95 = Column(Float, nullable=True)
    expected_shortfall_95 = Column(Float, nullable=True)
    expected_shortfall_975 = Column(Float, nullable=True)
    drawdown = Column(Float, nullable=True)

    __table_args__ = (
        Index("idx_risk_strat_date", "strategy", "date"),
    )
