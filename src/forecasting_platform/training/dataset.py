from dataclasses import dataclass

import pandas as pd
from sklearn.base import RegressorMixin


@dataclass(frozen=True)
class TrainingResult:
    """Artifacts produced by a model training run."""

    model: RegressorMixin
    feature_columns: list[str]
    target_column: str
    train_rows: int


def prepare_training_data(
    df: pd.DataFrame,
    target_column: str,
    timestamp_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for model training."""

    feature_columns = [
        column
        for column in df.columns
        if column not in {target_column, timestamp_column}
    ]

    X = df[feature_columns]

    y = df[target_column]

    return X, y
