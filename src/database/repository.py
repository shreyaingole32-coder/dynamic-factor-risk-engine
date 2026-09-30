"""
Database repository for saving and loading research metrics and observations.
"""

from typing import List, Optional
import pandas as pd
from sqlalchemy.orm import Session
from src.database.models import (
    DatasetMetadata, FactorReturn, MacroObservation, PortfolioWeight, PortfolioReturn, RiskMetric
)


class ResearchRepository:
    """Repository handling CRUD operations for research datasets."""

    def __init__(self, session: Session):
        self.session = session

    def save_dataset_metadata(self, metadata_dict: dict) -> DatasetMetadata:
        """Save or update dataset metadata catalog entry."""
        existing = self.session.query(DatasetMetadata).filter_by(dataset_id=metadata_dict["dataset_id"]).first()
        if existing:
            for k, v in metadata_dict.items():
                setattr(existing, k, v)
            record = existing
        else:
            record = DatasetMetadata(**metadata_dict)
            self.session.add(record)
        self.session.commit()
        return record

    def save_factor_returns(self, df: pd.DataFrame, frequency: str = "daily"):
        """
        Save Factor Returns DataFrame to database.
        Expects columns: date, mkt_rf, smb, hml, rmw, cma, mom, rf
        """
        records = []
        for idx, row in df.iterrows():
            date_val = pd.to_datetime(row["date"]).date()
            record = FactorReturn(
                date=date_val,
                frequency=frequency,
                mkt_rf=float(row["mkt_rf"]),
                smb=float(row["smb"]),
                hml=float(row["hml"]),
                rmw=float(row["rmw"]),
                cma=float(row["cma"]),
                mom=float(row["mom"]) if pd.notnull(row.get("mom")) else None,
                rf=float(row["rf"])
            )
            records.append(record)
        
        # Merge / insert
        for rec in records:
            self.session.merge(rec)
        self.session.commit()

    def get_factor_returns(self, frequency: str = "daily", start_date=None, end_date=None) -> pd.DataFrame:
        """Load factor returns into Pandas DataFrame."""
        query = self.session.query(FactorReturn).filter(FactorReturn.frequency == frequency)
        if start_date:
            query = query.filter(FactorReturn.date >= pd.to_datetime(start_date).date())
        if end_date:
            query = query.filter(FactorReturn.date <= pd.to_datetime(end_date).date())
        
        df = pd.read_sql(query.statement, self.session.bind)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        return df

    def save_macro_observations(self, df: pd.DataFrame, series_id: str):
        """Save macro observations DataFrame (date, value, observation_date, release_date)."""
        for idx, row in df.iterrows():
            obs_date = pd.to_datetime(row["date"]).date()
            rec = MacroObservation(
                date=obs_date,
                series_id=series_id,
                value=float(row["value"]) if pd.notnull(row["value"]) else None,
                observation_date=pd.to_datetime(row.get("observation_date", obs_date)).date(),
                release_date=pd.to_datetime(row.get("release_date", obs_date)).date()
            )
            self.session.merge(rec)
        self.session.commit()
