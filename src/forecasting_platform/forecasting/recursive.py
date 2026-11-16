import pandas as pd

from forecasting_platform.features import create_features
from forecasting_platform.training.models import Forecaster


def recursive_forecast(
    forecaster: Forecaster,
    history: pd.DataFrame,
    target_column: str,
    timestamp_column: str,
    feature_columns: list[str],
    horizon: int,
    step: str | pd.Timedelta,
) -> pd.DataFrame:
    """Forecast beyond the last observation, one step at a time.

    Future lag and rolling features are unknown, so each prediction is appended
    to the history and the features for the next step are rebuilt from it
    (the recursive multi-step strategy). Errors can compound with the horizon.
    """

    if horizon <= 0:
        raise ValueError("horizon must be greater than zero.")

    working = history[[timestamp_column, target_column]].reset_index(drop=True)
    last_timestamp = working[timestamp_column].iloc[-1]
    future_timestamps = pd.date_range(
        start=last_timestamp,
        periods=horizon + 1,
        freq=step,
    )[1:]

    forecasts: list[float] = []

    for timestamp in future_timestamps:
        next_row = pd.DataFrame(
            {
                timestamp_column: [timestamp],
                target_column: [float("nan")],
            }
        )
        extended = pd.concat([working, next_row], ignore_index=True)
        features = create_features(
            extended,
            target_column=target_column,
            timestamp_column=timestamp_column,
        )

        value = float(forecaster.predict(features.iloc[[-1]][feature_columns]).iloc[0])
        extended.loc[extended.index[-1], target_column] = value

        working = extended
        forecasts.append(value)

    return pd.DataFrame(
        {
            timestamp_column: future_timestamps,
            "forecast": forecasts,
        }
    )
