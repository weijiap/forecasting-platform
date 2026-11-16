import pandas as pd
import pytest

from forecasting_platform.features import create_features
from forecasting_platform.forecasting import recursive_forecast
from forecasting_platform.training.models import LagForecaster


def make_history(periods: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=periods, freq="D"),
            "sales": [float(i % 7) for i in range(periods)],
        }
    )


def feature_columns() -> list[str]:
    features = create_features(make_history(), "sales", "date")

    return [c for c in features.columns if c not in {"sales", "date"}]


def test_recursive_forecast_dates_and_length() -> None:
    """Forecasts should start one step after the last observation."""

    history = make_history()

    result = recursive_forecast(
        LagForecaster(name="naive", lag_column="sales_lag_1"),
        history=history,
        target_column="sales",
        timestamp_column="date",
        feature_columns=feature_columns(),
        horizon=5,
        step="D",
    )

    assert len(result) == 5
    assert result["date"].iloc[0] == pd.Timestamp("2026-01-31")
    assert list(result.columns) == ["date", "forecast"]


def test_recursive_forecast_feeds_predictions_back() -> None:
    """A naive forecast repeats the last value at every step."""

    history = make_history()
    last_value = history["sales"].iloc[-1]

    result = recursive_forecast(
        LagForecaster(name="naive", lag_column="sales_lag_1"),
        history=history,
        target_column="sales",
        timestamp_column="date",
        feature_columns=feature_columns(),
        horizon=4,
        step="D",
    )

    assert result["forecast"].tolist() == [last_value] * 4


def test_recursive_forecast_seasonal_naive_repeats_the_week() -> None:
    """A seasonal-naive forecast should replay the last seven observations."""

    history = make_history()

    result = recursive_forecast(
        LagForecaster(name="seasonal_naive", lag_column="sales_lag_7"),
        history=history,
        target_column="sales",
        timestamp_column="date",
        feature_columns=feature_columns(),
        horizon=7,
        step="D",
    )

    assert result["forecast"].tolist() == history["sales"].iloc[-7:].tolist()


def test_recursive_forecast_requires_positive_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        recursive_forecast(
            LagForecaster(name="naive", lag_column="sales_lag_1"),
            history=make_history(),
            target_column="sales",
            timestamp_column="date",
            feature_columns=feature_columns(),
            horizon=0,
            step="D",
        )
