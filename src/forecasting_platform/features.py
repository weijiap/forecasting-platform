import pandas as pd

LAGS: tuple[int, ...] = (1, 7)
ROLLING_WINDOWS: tuple[int, ...] = (7, 14)

# Rows lost at the start of the series before every feature is defined.
FEATURE_WARMUP_ROWS = max(*LAGS, *ROLLING_WINDOWS)


def lag_column_name(target_column: str, lag: int) -> str:
    """Return the name of the lag feature for a target column."""

    return f"{target_column}_lag_{lag}"


def create_features(
    df: pd.DataFrame,
    target_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    """Create all forecasting features."""

    result = create_lag_features(
        df,
        target_column=target_column,
        lags=list(LAGS),
    )

    result = create_rolling_features(
        result,
        target_column=target_column,
        windows=list(ROLLING_WINDOWS),
    )

    result = create_calendar_features(
        result,
        timestamp_column=timestamp_column,
    )

    return result


def create_lag_features(
    df: pd.DataFrame,
    target_column: str,
    lags: list[int],
) -> pd.DataFrame:
    """Create lagged features for a target column."""

    result = df.copy()

    for lag in lags:
        result[lag_column_name(target_column, lag)] = result[target_column].shift(lag)

    return result


def create_rolling_features(
    df: pd.DataFrame,
    target_column: str,
    windows: list[int],
) -> pd.DataFrame:
    """Create rolling statistical features for a target column."""

    result = df.copy()

    for window in windows:
        shifted_target = result[target_column].shift(1)

        result[f"{target_column}_rolling_mean_{window}"] = shifted_target.rolling(
            window=window
        ).mean()

        result[f"{target_column}_rolling_std_{window}"] = shifted_target.rolling(
            window=window
        ).std()

    return result


def create_calendar_features(
    df: pd.DataFrame,
    timestamp_column: str,
) -> pd.DataFrame:
    """Create calendar-based features from a timestamp column."""

    result = df.copy()
    timestamps = pd.to_datetime(result[timestamp_column])

    result["year"] = timestamps.dt.year
    result["month"] = timestamps.dt.month
    result["quarter"] = timestamps.dt.quarter
    result["day_of_week"] = timestamps.dt.dayofweek
    result["day_of_month"] = timestamps.dt.day
    result["week_of_year"] = timestamps.dt.isocalendar().week.astype(int)

    return result
