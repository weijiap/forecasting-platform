from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSeriesSplit:
    """One train-test split for time-series evaluation."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_time_series(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: int,
) -> TimeSeriesSplit:
    """Split time-series data while preserving chronological order."""

    if test_size <= 0:
        raise ValueError("test_size must be greater than zero.")

    if test_size >= len(X):
        raise ValueError("test_size must be smaller than the dataset.")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    split_index = len(X) - test_size

    return TimeSeriesSplit(
        X_train=X.iloc[:split_index].copy(),
        X_test=X.iloc[split_index:].copy(),
        y_train=y.iloc[:split_index].copy(),
        y_test=y.iloc[split_index:].copy(),
    )


def expanding_window_splits(
    X: pd.DataFrame,
    y: pd.Series,
    initial_train_size: int,
    test_size: int,
    step_size: int,
) -> list[TimeSeriesSplit]:
    """Create expanding-window train-test splits."""

    if initial_train_size <= 0:
        raise ValueError("initial_train_size must be greater than zero.")

    if test_size <= 0:
        raise ValueError("test_size must be greater than zero.")

    if step_size <= 0:
        raise ValueError("step_size must be greater than zero.")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    splits: list[TimeSeriesSplit] = []

    train_end = initial_train_size

    while train_end + test_size <= len(X):
        splits.append(
            TimeSeriesSplit(
                X_train=X.iloc[:train_end].copy(),
                X_test=X.iloc[train_end : train_end + test_size].copy(),
                y_train=y.iloc[:train_end].copy(),
                y_test=y.iloc[train_end : train_end + test_size].copy(),
            )
        )

        train_end += step_size

    return splits
