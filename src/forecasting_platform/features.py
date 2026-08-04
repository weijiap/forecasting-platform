import pandas as pd


def create_features(
    df: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    """Create all forecasting features."""

    result = df.copy()

    result = create_lag_features(
        result,
        target_column=target_column,
        lags=[1, 7],
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
        result[f"{target_column}_lag_{lag}"] = result[target_column].shift(lag)

    return result
