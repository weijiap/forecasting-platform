import pandas as pd

from forecasting_platform.training import prepare_training_data


def test_prepare_training_data() -> None:
    """Training data should exclude timestamp and target columns."""

    df = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=5,
            ),
            "sales": [1, 2, 3, 4, 5],
            "sales_lag_1": [None, 1, 2, 3, 4],
            "month": [1, 1, 1, 1, 1],
        }
    )

    X, y = prepare_training_data(
        df,
        target_column="sales",
        timestamp_column="date",
    )

    assert list(X.columns) == [
        "sales_lag_1",
        "month",
    ]

    assert y.name == "sales"

    assert len(X) == len(y)
