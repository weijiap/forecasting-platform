import numpy as np
import pandas as pd
import pytest

from forecasting_platform.training import TrainerConfig
from forecasting_platform.training.models import (
    LagForecaster,
    RandomForestForecaster,
    build_default_forecasters,
)


def trending_data(rows: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    """A target that rises steadily, with its previous value as a feature."""

    y = pd.Series(np.arange(rows, dtype=float) * 2.0 + 10.0, name="sales")
    X = pd.DataFrame({"sales_lag_1": y.shift(1)}).iloc[1:]

    return X, y.iloc[1:]


def test_lag_forecaster_returns_lag_column() -> None:
    """The naive forecast is the lagged value itself."""

    X = pd.DataFrame({"sales_lag_1": [1.0, 2.0, 3.0]}, index=[5, 6, 7])
    forecaster = LagForecaster(name="naive", lag_column="sales_lag_1")

    forecaster.fit(X, pd.Series([2.0, 3.0, 4.0]))
    predictions = forecaster.predict(X)

    assert predictions.tolist() == [1.0, 2.0, 3.0]
    assert predictions.index.tolist() == [5, 6, 7]


def test_lag_forecaster_requires_lag_column() -> None:
    """Fitting without the lag feature should fail clearly."""

    forecaster = LagForecaster(name="naive", lag_column="sales_lag_1")

    with pytest.raises(ValueError, match="sales_lag_1"):
        forecaster.fit(pd.DataFrame({"other": [1.0]}), pd.Series([1.0]))


def test_random_forest_requires_fit() -> None:
    """Predicting before fitting should raise an error."""

    forecaster = RandomForestForecaster(name="random_forest")

    with pytest.raises(RuntimeError, match="must be fitted"):
        forecaster.predict(pd.DataFrame({"sales_lag_1": [1.0]}))


def test_differencing_lets_trees_follow_a_trend() -> None:
    """A forest on levels plateaus; a forest on changes keeps rising."""

    X, y = trending_data()
    X_train, y_train = X.iloc[:40], y.iloc[:40]
    X_future = X.iloc[40:]

    levels = RandomForestForecaster(
        name="random_forest", config=TrainerConfig(n_estimators=20)
    )
    changes = RandomForestForecaster(
        name="random_forest_differenced",
        config=TrainerConfig(n_estimators=20),
        difference_from="sales_lag_1",
    )

    levels.fit(X_train, y_train)
    changes.fit(X_train, y_train)

    assert levels.predict(X_future).max() <= y_train.max()
    np.testing.assert_allclose(changes.predict(X_future), y.iloc[40:], atol=1e-6)


def test_build_default_forecasters_includes_baselines() -> None:
    """Every comparison should include naive and seasonal-naive baselines."""

    names = [f.name for f in build_default_forecasters("sales")]

    assert names == [
        "naive",
        "seasonal_naive",
        "random_forest",
        "random_forest_differenced",
    ]
