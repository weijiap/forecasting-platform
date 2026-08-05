import pandas as pd
import pytest

from forecasting_platform.training.splitter import (
    expanding_window_splits,
    split_time_series,
)


def test_split_time_series_preserves_order() -> None:
    """The final observations should form the test set."""

    X = pd.DataFrame(
        {
            "lag1": [1, 2, 3, 4, 5, 6],
        }
    )

    y = pd.Series(
        [2, 3, 4, 5, 6, 7],
        name="sales",
    )

    split = split_time_series(
        X,
        y,
        test_size=2,
    )

    assert split.X_train["lag1"].tolist() == [1, 2, 3, 4]
    assert split.X_test["lag1"].tolist() == [5, 6]
    assert split.y_train.tolist() == [2, 3, 4, 5]
    assert split.y_test.tolist() == [6, 7]


def test_split_time_series_rejects_invalid_test_size() -> None:
    """Invalid test sizes should raise ValueError."""

    X = pd.DataFrame({"lag1": [1, 2, 3]})
    y = pd.Series([2, 3, 4], name="sales")

    with pytest.raises(
        ValueError,
        match="test_size must be greater than zero.",
    ):
        split_time_series(
            X,
            y,
            test_size=0,
        )


def test_expanding_window_splits() -> None:
    """Expanding-window splits should grow the training sample."""

    X = pd.DataFrame(
        {
            "lag1": list(range(1, 11)),
        }
    )

    y = pd.Series(
        list(range(2, 12)),
        name="sales",
    )

    splits = expanding_window_splits(
        X,
        y,
        initial_train_size=4,
        test_size=2,
        step_size=2,
    )

    assert len(splits) == 3

    assert len(splits[0].X_train) == 4
    assert len(splits[1].X_train) == 6
    assert len(splits[2].X_train) == 8

    assert splits[0].X_test["lag1"].tolist() == [5, 6]
    assert splits[1].X_test["lag1"].tolist() == [7, 8]
    assert splits[2].X_test["lag1"].tolist() == [9, 10]
