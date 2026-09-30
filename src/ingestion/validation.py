"""
Data Validation Engine for Quantitative Data Integrity.
Executes quality gates on factor returns and macroeconomic time series.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


class DataValidator:
    """Rigorous validator for financial factor data and macroeconomic variables."""

    @staticmethod
    def validate_factor_returns(df: pd.DataFrame, expected_cols: List[str] = None) -> Dict[str, Any]:
        """
        Validate factor returns dataset.
        Checks for: missing dates, duplicates, NaNs, decimal vs percentage formatting, extrema.
        """
        if expected_cols is None:
            expected_cols = ["date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"]

        report = {
            "is_valid": True,
            "total_rows": len(df),
            "start_date": None,
            "end_date": None,
            "missing_columns": [],
            "duplicate_dates": 0,
            "nan_counts": {},
            "percentage_warning": False,
            "extreme_value_warnings": [],
            "errors": []
        }

        # 1. Column presence
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            report["missing_columns"] = missing_cols
            report["is_valid"] = False
            report["errors"].append(f"Missing required columns: {missing_cols}")
            return report

        # 2. Date checks
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            report["errors"].append("Column 'date' is not datetime type.")
            report["is_valid"] = False
            return report

        report["start_date"] = df["date"].min().strftime("%Y-%m-%d")
        report["end_date"] = df["date"].max().strftime("%Y-%m-%d")

        # Duplicate dates
        dups = df["date"].duplicated().sum()
        report["duplicate_dates"] = int(dups)
        if dups > 0:
            report["is_valid"] = False
            report["errors"].append(f"Found {dups} duplicate dates in dataset.")

        # 3. NaNs and null values
        factor_cols = [c for c in expected_cols if c != "date"]
        for col in factor_cols:
            nans = int(df[col].isna().sum())
            report["nan_counts"][col] = nans
            if nans > 0:
                report["errors"].append(f"Column '{col}' contains {nans} missing (NaN) values.")

        # 4. Decimal vs Percentage sanity check
        # Returns in decimals should almost never exceed 1.0 (100% daily return)
        max_abs_val = df[factor_cols].abs().max().max()
        if max_abs_val > 1.0:
            report["percentage_warning"] = True
            report["errors"].append(f"Maximum absolute return value is {max_abs_val:.4f} > 1.0. Appears to be in percentage format instead of decimal.")
            report["is_valid"] = False

        # 5. Extreme value sanity check (e.g. daily return > 50%)
        for col in factor_cols:
            extreme_rows = df[df[col].abs() > 0.5]
            if len(extreme_rows) > 0:
                report["extreme_value_warnings"].append(f"Column '{col}' has {len(extreme_rows)} observations exceeding 50% absolute return.")

        return report

    @staticmethod
    def validate_macro_series(df: pd.DataFrame, series_id: str) -> Dict[str, Any]:
        """Validate macro series DataFrame (date, value)."""
        report = {
            "series_id": series_id,
            "is_valid": True,
            "total_rows": len(df),
            "start_date": None,
            "end_date": None,
            "duplicate_dates": 0,
            "null_value_count": 0,
            "errors": []
        }

        if "date" not in df.columns or "value" not in df.columns:
            report["is_valid"] = False
            report["errors"].append("Dataset missing 'date' or 'value' columns.")
            return report

        report["start_date"] = df["date"].min().strftime("%Y-%m-%d")
        report["end_date"] = df["date"].max().strftime("%Y-%m-%d")

        dups = df["date"].duplicated().sum()
        report["duplicate_dates"] = int(dups)
        if dups > 0:
            report["is_valid"] = False
            report["errors"].append(f"Found {dups} duplicate dates for macro series {series_id}.")

        nulls = int(df["value"].isna().sum())
        report["null_value_count"] = nulls

        return report
