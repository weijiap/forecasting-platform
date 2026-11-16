from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ValidationReport:
    """Summary of dataset validation results."""

    rows: int

    columns: int

    missing_values: int

    duplicate_rows: int

    duplicate_timestamps: int

    is_sorted: bool

    passed: bool

    issues: tuple[str, ...] = ()


def validate_dataframe(
    df: pd.DataFrame,
    timestamp_column: str,
    target_column: str | None = None,
    min_rows: int = 0,
) -> ValidationReport:
    """Validate a forecasting dataset.

    Unsorted timestamps are reported but do not fail validation, because the
    pipeline sorts the series. Problems that would silently corrupt lag
    features (unparseable or duplicate timestamps, missing or non-numeric
    target values, too few rows) fail validation and are listed in ``issues``.
    """

    issues: list[str] = []
    rows = len(df)
    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    if timestamp_column not in df.columns:
        issues.append(f"Timestamp column '{timestamp_column}' was not found.")

        return ValidationReport(
            rows=rows,
            columns=len(df.columns),
            missing_values=missing_values,
            duplicate_rows=duplicate_rows,
            duplicate_timestamps=0,
            is_sorted=False,
            passed=False,
            issues=tuple(issues),
        )

    timestamps = pd.to_datetime(df[timestamp_column], errors="coerce")
    valid_timestamps = timestamps.dropna()

    invalid_timestamps = int(timestamps.isna().sum())
    if invalid_timestamps:
        issues.append(f"{invalid_timestamps} timestamp value(s) could not be parsed.")

    duplicate_timestamps = int(valid_timestamps.duplicated().sum())
    if duplicate_timestamps:
        issues.append(f"{duplicate_timestamps} duplicate timestamp(s) were found.")

    if target_column is not None:
        if target_column not in df.columns:
            issues.append(f"Target column '{target_column}' was not found.")
        else:
            values = pd.to_numeric(df[target_column], errors="coerce")
            invalid_values = int(values.isna().sum())

            if invalid_values:
                issues.append(
                    f"{invalid_values} target value(s) are missing or non-numeric."
                )

    if rows < min_rows:
        issues.append(f"At least {min_rows} rows are required; received {rows}.")

    return ValidationReport(
        rows=rows,
        columns=len(df.columns),
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
        duplicate_timestamps=duplicate_timestamps,
        is_sorted=bool(valid_timestamps.is_monotonic_increasing),
        passed=not issues,
        issues=tuple(issues),
    )
