import pandas as pd

from forecasting_platform.features import create_lag_features


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
