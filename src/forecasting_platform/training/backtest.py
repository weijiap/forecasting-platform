import numpy as np
import pandas as pd

from forecasting_platform.training.benchmark import ModelEvaluation
from forecasting_platform.training.models import Forecaster
from forecasting_platform.training.splitter import (
    TimeSeriesSplit,
    expanding_window_splits,
)


def make_backtest_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 3,
) -> list[TimeSeriesSplit]:
    """Create walk-forward folds that cover the end of the training data.

    The data are cut into ``n_folds + 2`` equal blocks. The first two blocks
    form the initial training window, and each later block is forecast once
    with all earlier data available for training.
    """

    if n_folds <= 0:
        raise ValueError("n_folds must be greater than zero.")

    fold_size = len(X) // (n_folds + 2)

    if fold_size < 1:
        raise ValueError("Not enough observations for walk-forward validation.")

    return expanding_window_splits(
        X,
        y,
        initial_train_size=len(X) - n_folds * fold_size,
        test_size=fold_size,
        step_size=fold_size,
    )


def backtest_forecaster(
    forecaster: Forecaster,
    splits: list[TimeSeriesSplit],
) -> ModelEvaluation:
    """Refit a forecaster on each fold and pool its out-of-sample errors."""

    if not splits:
        raise ValueError("At least one split is required.")

    errors: list[np.ndarray] = []

    for split in splits:
        forecaster.fit(split.X_train, split.y_train)
        predictions = forecaster.predict(split.X_test)
        errors.append(
            split.y_test.to_numpy(dtype=float) - predictions.to_numpy(dtype=float)
        )

    pooled = np.concatenate(errors)

    return ModelEvaluation(
        model_name=forecaster.name,
        rmse=float(np.sqrt(np.mean(pooled**2))),
        mae=float(np.mean(np.abs(pooled))),
    )
