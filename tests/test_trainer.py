import pandas as pd

from forecasting_platform.training import (
    TrainerConfig,
    train_random_forest,
)


def test_train_random_forest() -> None:
    """Random Forest trainer should return a TrainingResult."""

    X = pd.DataFrame(
        {
            "lag1": [1, 2, 3, 4, 5],
            "month": [1, 1, 1, 1, 1],
        }
    )

    y = pd.Series(
        [2, 3, 4, 5, 6],
        name="sales",
    )

    result = train_random_forest(
        X,
        y,
        TrainerConfig(),
    )

    assert result.train_rows == 5
    assert result.target_column == "sales"
    assert len(result.feature_columns) == 2
