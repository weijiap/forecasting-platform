import numpy as np
import pandas as pd
import pytest

from forecasting_platform.pipeline import run_forecast_pipeline
from forecasting_platform.training import TrainerConfig
from forecasting_platform.training.models import Forecaster, build_default_forecasters


def make_series(periods: int = 120, seed: int = 0) -> pd.DataFrame:
    """Daily series with trend, weekly seasonality and noise."""

    rng = np.random.default_rng(seed)

    return pd.DataFrame(
        {
            "date": pd.date_range(
                start="2025-01-01",
                periods=periods,
                freq="D",
            ),
            "sales": [
                100 + index * 0.5 + (index % 7) * 2 + rng.normal(0, 1)
                for index in range(periods)
            ],
        }
    )


def fast_forecasters() -> list[Forecaster]:
    return build_default_forecasters("sales", TrainerConfig(n_estimators=20))


def test_run_forecast_pipeline_end_to_end() -> None:
    """The full pipeline should compare models and forecast the future."""

    periods = 120
    df = make_series(periods)

    result = run_forecast_pipeline(
        df=df,
        target_column="sales",
        timestamp_column="date",
        horizon=10,
        forecasters=fast_forecasters(),
    )

    assert result.validation_report.rows == periods
    assert result.validation_report.passed is True
    assert result.train_rows > 0

    model_names = {e.model_name for e in result.backtest}
    assert {"naive", "seasonal_naive", "random_forest_differenced"} <= model_names
    assert {e.model_name for e in result.holdout} == model_names
    assert result.selected_model in model_names
    assert result.evaluation.model_name == result.selected_model

    best_backtest = min(result.backtest, key=lambda e: e.rmse)
    assert result.selected_model == best_backtest.model_name

    assert len(result.predictions) == 30
    assert list(result.forecast.columns) == ["date", "actual", "prediction"]
    assert len(result.forecast) == 30
    assert result.forecast["date"].is_monotonic_increasing

    assert len(result.future_forecast) == 10
    assert result.future_forecast["date"].iloc[0] == df["date"].iloc[-1] + pd.Timedelta(
        days=1
    )
    assert result.future_forecast["forecast"].notna().all()


def test_pipeline_sorts_unsorted_input() -> None:
    """Shuffled rows should give the same result as sorted rows."""

    df = make_series()
    shuffled = df.sample(frac=1.0, random_state=1)

    sorted_result = run_forecast_pipeline(
        df, "sales", "date", horizon=3, forecasters=fast_forecasters()
    )
    shuffled_result = run_forecast_pipeline(
        shuffled, "sales", "date", horizon=3, forecasters=fast_forecasters()
    )

    pd.testing.assert_frame_equal(sorted_result.forecast, shuffled_result.forecast)


def test_pipeline_rejects_invalid_data() -> None:
    """Duplicate timestamps should stop the pipeline with a clear error."""

    df = make_series()
    df.loc[5, "date"] = df.loc[4, "date"]

    with pytest.raises(ValueError, match="duplicate timestamp"):
        run_forecast_pipeline(df, "sales", "date", forecasters=fast_forecasters())


def test_pipeline_rejects_short_series() -> None:
    """Too few rows for features, training and holdout should fail."""

    with pytest.raises(ValueError, match="rows are required"):
        run_forecast_pipeline(make_series(periods=40), "sales", "date")


def test_differenced_forest_beats_levels_on_trend() -> None:
    """On a trending series, predicting changes should beat predicting levels."""

    result = run_forecast_pipeline(
        make_series(periods=200),
        "sales",
        "date",
        horizon=1,
        forecasters=fast_forecasters(),
    )
    holdout = {e.model_name: e.rmse for e in result.holdout}

    assert holdout["random_forest_differenced"] < holdout["random_forest"]
    assert holdout["random_forest_differenced"] < holdout["naive"]
