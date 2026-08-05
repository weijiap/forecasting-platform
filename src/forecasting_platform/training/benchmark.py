from dataclasses import dataclass

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass(frozen=True)
class ModelEvaluation:
    """Evaluation result for one trained model."""

    model_name: str
    rmse: float
    mae: float


def evaluate_model(
    model: RegressorMixin,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> ModelEvaluation:
    """Evaluate a trained regression model."""

    predictions = model.predict(X_test)

    rmse = (
        mean_squared_error(
            y_test,
            predictions,
        )
        ** 0.5
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    return ModelEvaluation(
        model_name=model_name,
        rmse=float(rmse),
        mae=float(mae),
    )


def select_best_model(
    evaluations: list[ModelEvaluation],
) -> ModelEvaluation:
    """Select the model with the lowest RMSE."""

    if not evaluations:
        raise ValueError("At least one model evaluation is required.")

    return min(
        evaluations,
        key=lambda evaluation: evaluation.rmse,
    )
