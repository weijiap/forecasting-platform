import pandas as pd


def prepare_series(
    df: pd.DataFrame,
    timestamp_column: str,
    target_column: str,
) -> pd.DataFrame:
    """Return a clean, chronologically sorted two-column series.

    Lag and rolling features assume rows are in time order, so the series is
    sorted before any feature is built.
    """

    for column in (timestamp_column, target_column):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' was not found.")

    series = pd.DataFrame(
        {
            timestamp_column: pd.to_datetime(df[timestamp_column]),
            target_column: pd.to_numeric(df[target_column]).astype(float),
        }
    )

    return series.sort_values(timestamp_column, kind="stable").reset_index(drop=True)
