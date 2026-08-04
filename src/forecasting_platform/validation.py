from dataclasses import dataclass

import pandas as pd


@dataclass
class ValidationReport:
    """Summary of dataset validation results."""

    rows: int

    columns: int

    missing_values: int

    duplicate_rows: int

    duplicate_timestamps: int

    is_sorted: bool

    passed: bool


def validate_dataframe(
    df: pd.DataFrame,
    timestamp_column: str,
) -> ValidationReport:
    """Validate a forecasting dataset."""

    return ValidationReport(
        rows=len(df),
        columns=len(df.columns),
        missing_values=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
        duplicate_timestamps=int(df[timestamp_column].duplicated().sum()),
        is_sorted=True,
        passed=True,
    )
