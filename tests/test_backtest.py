import pandas as pd
import pytest

from forecasting_platform.training.backtest import (
    backtest_forecaster,
    make_backtest_splits,
)
from forecasting_platform.training.models import LagForecaster


def test_make_backtest_splits_walks_forward() -> None:
    """Folds should be contiguous, non-overlapping and end at the last row."""

    X = pd.DataFrame({"x": range(50)})
    y = pd.Series(range(50), name="y")

    splits = make_backtest_splits(X, y, n_folds=3)

    assert len(splits) == 3
    assert [len(s.X_test) for s in splits] == [10, 10, 10]
    assert splits[0].X_train.index.max() < splits[0].X_test.index.min()
    assert splits[1].X_test.index.min() == splits[0].X_test.index.max() + 1
    assert splits[-1].X_test.index.max() == 49


def test_make_backtest_splits_needs_enough_rows() -> None:
    """Too few rows for the requested folds should fail."""

    with pytest.raises(ValueError, match="Not enough observations"):
        make_backtest_splits(pd.DataFrame({"x": [1, 2]}), pd.Series([1, 2]))


def test_backtest_forecaster_pools_errors() -> None:
    """A perfect forecaster has zero error; a constant offset gives that offset."""

    X = pd.DataFrame({"lag": [float(i) for i in range(50)]})
    exact = pd.Series([float(i) for i in range(50)])
    offset = exact + 2.0

    splits_exact = make_backtest_splits(X, exact)
    splits_offset = make_backtest_splits(X, offset)
    forecaster = LagForecaster(name="naive", lag_column="lag")

    perfect = backtest_forecaster(forecaster, splits_exact)
    shifted = backtest_forecaster(forecaster, splits_offset)

    assert perfect.rmse == pytest.approx(0.0)
    assert shifted.rmse == pytest.approx(2.0)
    assert shifted.mae == pytest.approx(2.0)
    assert shifted.model_name == "naive"
