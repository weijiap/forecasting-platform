import pandas as pd

from forecasting_platform.pipeline import run_forecast_pipeline


def test_run_forecast_pipeline_end_to_end() -> None:
    """The full forecasting pipeline should return predictions and metrics."""

    periods = 120

    df = pd.DataFrame(
        {
            "date": pd.date_range(
                start="2025-01-01",
                periods=periods,
                freq="D",
            ),
            "sales": [100 + index * 0.5 + (index % 7) * 2 for index in range(periods)],
        }
    )

    result = run_forecast_pipeline(
        df=df,
        target_column="sales",
        timestamp_column="date",
    )

    assert result.validation_report.rows == periods
    assert result.validation_report.passed is True
    assert result.training_result.train_rows > 0
    assert result.evaluation.model_name == "random_forest"
    assert result.evaluation.rmse >= 0
    assert result.evaluation.mae >= 0
    assert len(result.predictions) == 30
    assert list(result.forecast.columns) == [
        "date",
        "actual",
        "prediction",
    ]
    assert len(result.forecast) == 30
    assert result.forecast["date"].is_monotonic_increasing
