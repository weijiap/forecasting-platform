from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

from forecasting_platform.features import lag_column_name
from forecasting_platform.training.trainer import TrainerConfig, train_random_forest

SEASON_LENGTH = 7


class Forecaster(Protocol):
    """Interface shared by baseline and machine-learning forecasters."""

    name: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    def predict(self, X: pd.DataFrame) -> pd.Series: ...


@dataclass
class LagForecaster:
    """Predict the value observed a fixed number of periods earlier.

    With lag 1 this is the naive forecast; with the seasonal lag it is the
    seasonal-naive forecast. Every model should beat these baselines.
    """

    name: str
    lag_column: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if self.lag_column not in X.columns:
            raise ValueError(f"Feature '{self.lag_column}' is required.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(
            X[self.lag_column].to_numpy(dtype=float),
            index=X.index,
            name="prediction",
        )


@dataclass
class RandomForestForecaster:
    """Random forest that predicts either levels or changes.

    Trees can only predict values seen during training, so a forest trained on
    levels cannot follow a trend beyond the training range. When
    ``difference_from`` is set, the forest predicts the change relative to that
    lag column, and the change is added back at prediction time.
    """

    name: str
    config: TrainerConfig = field(default_factory=TrainerConfig)
    difference_from: str | None = None
    _model: Any = field(default=None, init=False, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        target = y - X[self.difference_from] if self.difference_from else y
        self._model = train_random_forest(X, target, self.config).model

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError(f"Model '{self.name}' must be fitted before predicting.")

        predictions = pd.Series(
            self._model.predict(X),
            index=X.index,
            name="prediction",
        )

        if self.difference_from:
            predictions = predictions + X[self.difference_from].astype(float)

        return predictions.rename("prediction")


def build_default_forecasters(
    target_column: str,
    config: TrainerConfig | None = None,
) -> list[Forecaster]:
    """Return the candidate models compared by the pipeline."""

    trainer_config = config or TrainerConfig()
    last_value = lag_column_name(target_column, 1)

    return [
        LagForecaster(name="naive", lag_column=last_value),
        LagForecaster(
            name="seasonal_naive",
            lag_column=lag_column_name(target_column, SEASON_LENGTH),
        ),
        RandomForestForecaster(name="random_forest", config=trainer_config),
        RandomForestForecaster(
            name="random_forest_differenced",
            config=trainer_config,
            difference_from=last_value,
        ),
    ]
