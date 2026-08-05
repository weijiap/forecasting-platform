import pandas as pd
from sklearn.base import RegressorMixin


def generate_predictions(
    model: RegressorMixin,
    X: pd.DataFrame,
) -> pd.Series:
    """Generate predictions from a trained regression model."""

    predictions = model.predict(X)

    return pd.Series(
        predictions,
        index=X.index,
        name="prediction",
    )
