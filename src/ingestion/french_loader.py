"""
Kenneth French Data Library Ingestion & Parser.
Downloads official ZIP/CSV factor data, parses headers, standardizes dates and units.
"""

import os
import io
import zipfile
from datetime import datetime
import requests
import pandas as pd
import yaml


class FrenchDataLoader:
    """Downloader and parser for Kenneth French Data Library datasets."""

    def __init__(self, raw_dir: str = "data/raw", config_path: str = "configs/default.yaml"):
        self.raw_dir = raw_dir
        os.makedirs(self.raw_dir, exist_ok=True)
        self.base_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
        
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
                french_cfg = cfg.get("data_sources", {}).get("french", {})
                self.base_url = french_cfg.get("base_url", self.base_url)

    def fetch_dataset(self, filename: str, force_download: bool = False) -> str:
        """Download raw ZIP file from French library if not already cached."""
        target_path = os.path.join(self.raw_dir, filename)
        if not os.path.exists(target_path) or force_download:
            url = f"{self.base_url}/{filename}"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code != 200:
                raise ValueError(f"Failed to download {url}. Status code: {response.status_code}")
            with open(target_path, "wb") as f:
                f.write(response.content)
            print(f"[FrenchLoader] Downloaded: {filename}")
        else:
            print(f"[FrenchLoader] Using cached file: {target_path}")
        return target_path

    def parse_ff5_daily(self, filepath: str) -> pd.DataFrame:
        """Parse Daily Fama-French 5 Factors CSV from ZIP archive."""
        with zipfile.ZipFile(filepath, "r") as z:
            csv_name = [name for name in z.namelist() if name.endswith(".CSV") or name.endswith(".csv")][0]
            with z.open(csv_name) as f:
                content = f.read().decode("utf-8")

        lines = content.splitlines()
        header_idx = -1
        for i, line in enumerate(lines[:30]):
            if "Mkt-RF" in line or "MKT-RF" in line:
                header_idx = i
                break
        
        if header_idx == -1:
            raise ValueError("Could not locate header row in FF5 daily file.")

        # Read data from header row
        data_lines = []
        for line in lines[header_idx:]:
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) >= 6:
                # Check if first item is date (YYYYMMDD)
                if parts[0].isdigit() and len(parts[0]) == 8:
                    data_lines.append(line)

        df = pd.read_csv(io.StringIO("\n".join(data_lines)), header=None)
        df = df.iloc[:, :7]
        df.columns = ["date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
        
        # Parse date and convert percentages to decimals
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
        factor_cols = ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
        for col in factor_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df

    def parse_momentum_daily(self, filepath: str) -> pd.DataFrame:
        """Parse Daily Momentum Factor CSV from ZIP archive."""
        with zipfile.ZipFile(filepath, "r") as z:
            csv_name = [name for name in z.namelist() if name.endswith(".CSV") or name.endswith(".csv")][0]
            with z.open(csv_name) as f:
                content = f.read().decode("utf-8")

        lines = content.splitlines()
        data_lines = []
        for line in lines:
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 8:
                data_lines.append(f"{parts[0]},{parts[1]}")

        df = pd.read_csv(io.StringIO("\n".join(data_lines)), header=None)
        df.columns = ["date", "mom"]
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
        df["mom"] = pd.to_numeric(df["mom"], errors="coerce") / 100.0
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df

    def load_combined_daily_factors(self) -> tuple[pd.DataFrame, dict]:
        """Fetch, parse, and merge FF5 Daily and Momentum Daily datasets."""
        ff5_path = self.fetch_dataset("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
        mom_path = self.fetch_dataset("F-F_Momentum_Factor_daily_CSV.zip")

        df_ff5 = self.parse_ff5_daily(ff5_path)
        df_mom = self.parse_momentum_daily(mom_path)

        df_merged = pd.merge(df_ff5, df_mom, on="date", how="left")
        
        metadata = {
            "dataset_id": "ff5_mom_daily",
            "name": "Fama-French 5 Factors + Momentum Daily",
            "source": "Kenneth R. French Data Library",
            "source_url": f"{self.base_url}/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
            "frequency": "daily",
            "retrieved_at": datetime.utcnow(),
            "start_date": df_merged["date"].min().date(),
            "end_date": df_merged["date"].max().date(),
            "units": "decimal",
            "description": "Daily returns for Mkt-RF, SMB, HML, RMW, CMA, MOM, and Risk-Free rate."
        }
        return df_merged, metadata
