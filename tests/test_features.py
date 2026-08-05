import pandas as pd

from forecasting_platform.features import (
    create_calendar_features,
    create_features,
    create_lag_features,
    create_rolling_features,
)


def test_create_lag_features() -> None:
    """Lag columns should contain shifted target values."""

    df = pd.DataFrame(
        {
            "sales": [100, 110, 120, 130],
        }
    )

    result = create_lag_features(
        df,
        target_column="sales",
        lags=[1, 2],
    )

    assert result["sales_lag_1"].tolist()[1:] == [100.0, 110.0, 120.0]
    assert result["sales_lag_2"].tolist()[2:] == [100.0, 110.0]


def test_create_rolling_features_uses_only_past_values() -> None:
    """Rolling features should exclude the current target value."""

    df = pd.DataFrame(
        {
            "sales": [10, 20, 30, 40, 50],
        }
    )

    result = create_rolling_features(
        df,
        target_column="sales",
        windows=[2],
    )

    assert pd.isna(result.loc[0, "sales_rolling_mean_2"])
    assert pd.isna(result.loc[1, "sales_rolling_mean_2"])
    assert result.loc[2, "sales_rolling_mean_2"] == 15.0
    assert result.loc[3, "sales_rolling_mean_2"] == 25.0
    assert result.loc[4, "sales_rolling_mean_2"] == 35.0


def test_create_features_combines_feature_groups() -> None:
    """The feature pipeline should create lag, rolling, and calendar features."""

    df = pd.DataFrame(
        {
            "date": pd.date_range(
                start="2026-01-01",
                periods=20,
                freq="D",
            ),
            "sales": list(range(1, 21)),
        }
    )

    result = create_features(
        df,
        target_column="sales",
        timestamp_column="date",
    )

    expected_columns = {
        "sales_lag_1",
        "sales_lag_7",
        "sales_rolling_mean_7",
        "sales_rolling_std_7",
        "sales_rolling_mean_14",
        "sales_rolling_std_14",
        "year",
        "month",
        "quarter",
        "day_of_week",
        "day_of_month",
        "week_of_year",
    }

    assert expected_columns.issubset(result.columns)


def test_create_calendar_features() -> None:
    """Calendar features should be extracted correctly."""

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-02-15",
                ]
            )
        }
    )

    result = create_calendar_features(
        df,
        timestamp_column="date",
    )

    assert result["year"].tolist() == [2026, 2026]
    assert result["month"].tolist() == [1, 2]
    assert result["quarter"].tolist() == [1, 1]
