import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from forecasting_platform.training.benchmark import (
    ModelEvaluation,
    evaluate_model,
    select_best_model,
)


def test_evaluate_model() -> None:
    """Model evaluation should return non-negative metrics."""

    X_train = pd.DataFrame(
        {
            "lag1": [1, 2, 3, 4, 5],
        }
    )
    y_train = pd.Series(
        [2, 3, 4, 5, 6],
        name="sales",
    )

    model = RandomForestRegressor(
        n_estimators=10,
        random_state=42,
    )
    model.fit(X_train, y_train)

    evaluation = evaluate_model(
        model=model,
        model_name="random_forest",
        X_test=X_train,
        y_test=y_train,
    )

    assert evaluation.model_name == "random_forest"
    assert evaluation.rmse >= 0
    assert evaluation.mae >= 0


def test_select_best_model() -> None:
    """The model with the lowest RMSE should be selected."""

    evaluations = [
        ModelEvaluation(
            model_name="random_forest",
            rmse=2.5,
            mae=2.0,
        ),
        ModelEvaluation(
            model_name="xgboost",
            rmse=1.8,
            mae=1.5,
        ),
        ModelEvaluation(
            model_name="lightgbm",
            rmse=2.1,
            mae=1.7,
        ),
    ]

    best_model = select_best_model(evaluations)

    assert best_model.model_name == "xgboost"
    assert best_model.rmse == 1.8


def test_select_best_model_requires_evaluations() -> None:
    """Selecting from an empty evaluation list should fail."""

    with pytest.raises(
        ValueError,
        match="At least one model evaluation is required.",
    ):
        select_best_model([])
