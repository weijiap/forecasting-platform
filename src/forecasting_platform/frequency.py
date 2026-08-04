import pandas as pd


def detect_frequency(
    df: pd.DataFrame,
    timestamp_column: str,
) -> str | None:
    """Detect the frequency of a time-series dataset."""

    timestamps = pd.to_datetime(df[timestamp_column])

    return pd.infer_freq(timestamps)
