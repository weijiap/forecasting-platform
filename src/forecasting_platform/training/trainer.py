from dataclasses import dataclass
from typing import Protocol

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from forecasting_platform.training.dataset import (
    TrainingResult,
)


class Regressor(Protocol):
    """Protocol implemented by forecasting models."""

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None: ...

    def predict(
        self,
        X: pd.DataFrame,
    ) -> list[float]: ...


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration for model training."""

    random_state: int = 42
    n_estimators: int = 200


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: TrainerConfig,
) -> TrainingResult:
    """Train a Random Forest forecasting model."""

    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        random_state=config.random_state,
    )

    model.fit(X_train, y_train)

    return TrainingResult(
        model=model,
        feature_columns=list(X_train.columns),
        target_column=(str(y_train.name) if y_train.name is not None else "target"),
        train_rows=len(X_train),
    )
