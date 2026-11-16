import pandas as pd


def detect_frequency(
    df: pd.DataFrame,
    timestamp_column: str,
) -> str | None:
    """Detect the frequency of a time-series dataset."""

    timestamps = pd.to_datetime(df[timestamp_column])

    return pd.infer_freq(timestamps)


def infer_step(
    df: pd.DataFrame,
    timestamp_column: str,
) -> str | pd.Timedelta:
    """Return the spacing used to generate future timestamps.

    Uses the pandas frequency when one can be inferred (for example ``"D"`` or
    ``"MS"``), and otherwise falls back to the median gap between observations.
    """

    timestamps = pd.to_datetime(df[timestamp_column])

    if len(timestamps) >= 3:
        frequency = pd.infer_freq(timestamps)

        if frequency is not None:
            return frequency

    gaps = timestamps.diff().dropna()

    if gaps.empty:
        raise ValueError("At least two timestamps are required to infer a step.")

    return pd.Timedelta(gaps.median())
