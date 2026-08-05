import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from forecasting_platform.forecasting import generate_predictions


def test_generate_predictions() -> None:
    """Predictions should preserve the feature matrix index."""

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

    X_test = pd.DataFrame(
        {
            "lag1": [6, 7],
        },
        index=[10, 11],
    )

    predictions = generate_predictions(
        model=model,
        X=X_test,
    )

    assert len(predictions) == 2
    assert predictions.name == "prediction"
    assert predictions.index.tolist() == [10, 11]
