from dataclasses import dataclass

import pandas as pd

from forecasting_platform.features import FEATURE_WARMUP_ROWS, create_features
from forecasting_platform.forecasting import recursive_forecast
from forecasting_platform.frequency import infer_step
from forecasting_platform.preprocessing import prepare_series
from forecasting_platform.training import prepare_training_data
from forecasting_platform.training.backtest import (
    backtest_forecaster,
    make_backtest_splits,
)
from forecasting_platform.training.benchmark import (
    ModelEvaluation,
    evaluate_predictions,
    select_best_model,
)
from forecasting_platform.training.models import Forecaster, build_default_forecasters
from forecasting_platform.training.splitter import split_time_series
from forecasting_platform.validation import ValidationReport, validate_dataframe

MIN_TRAINING_ROWS = 30


@dataclass(frozen=True)
class ForecastPipelineResult:
    """Outputs produced by the forecasting pipeline."""

    validation_report: ValidationReport
    backtest: list[ModelEvaluation]
    holdout: list[ModelEvaluation]
    selected_model: str
    evaluation: ModelEvaluation
    predictions: pd.Series
    forecast: pd.DataFrame
    future_forecast: pd.DataFrame
    train_rows: int


def run_forecast_pipeline(
    df: pd.DataFrame,
    target_column: str,
    timestamp_column: str,
    test_size: int = 30,
    horizon: int = 14,
    forecasters: list[Forecaster] | None = None,
) -> ForecastPipelineResult:
    """Run the complete forecasting pipeline.

    1. Validate and sort the series, then build lag, rolling and calendar
       features.
    2. Hold out the last ``test_size`` rows.
    3. Compare candidate models (baselines included) with walk-forward
       validation on the training rows only, and select the lowest RMSE.
    4. Report every model on the untouched holdout.
    5. Refit the selected model on all rows and forecast ``horizon`` steps
       beyond the last observation.
    """

    if test_size <= 0:
        raise ValueError("test_size must be greater than zero.")

    if horizon <= 0:
        raise ValueError("horizon must be greater than zero.")

    validation_report = validate_dataframe(
        df,
        timestamp_column=timestamp_column,
        target_column=target_column,
        min_rows=FEATURE_WARMUP_ROWS + MIN_TRAINING_ROWS + test_size,
    )

    if not validation_report.passed:
        raise ValueError(" ".join(validation_report.issues))

    series = prepare_series(
        df,
        timestamp_column=timestamp_column,
        target_column=target_column,
    )

    feature_df = create_features(
        series,
        target_column=target_column,
        timestamp_column=timestamp_column,
    )

    feature_df = feature_df.dropna().reset_index(drop=True)

    X, y = prepare_training_data(
        feature_df,
        target_column=target_column,
        timestamp_column=timestamp_column,
    )

    split = split_time_series(
        X,
        y,
        test_size=test_size,
    )

    candidates = forecasters or build_default_forecasters(target_column)
    backtest_splits = make_backtest_splits(split.X_train, split.y_train)

    backtest = [
        backtest_forecaster(forecaster, backtest_splits) for forecaster in candidates
    ]
    selected_model = select_best_model(backtest).model_name
    selected = next(f for f in candidates if f.name == selected_model)

    holdout: list[ModelEvaluation] = []
    holdout_predictions: dict[str, pd.Series] = {}

    for forecaster in candidates:
        forecaster.fit(split.X_train, split.y_train)
        holdout_predictions[forecaster.name] = forecaster.predict(split.X_test)
        holdout.append(
            evaluate_predictions(
                forecaster.name,
                split.y_test,
                holdout_predictions[forecaster.name],
            )
        )

    predictions = holdout_predictions[selected_model]
    evaluation = next(e for e in holdout if e.model_name == selected_model)

    forecast = pd.DataFrame(
        {
            timestamp_column: feature_df.loc[
                split.X_test.index,
                timestamp_column,
            ].to_numpy(),
            "actual": split.y_test.to_numpy(),
            "prediction": predictions.to_numpy(),
        }
    )

    selected.fit(X, y)

    future_forecast = recursive_forecast(
        selected,
        history=series,
        target_column=target_column,
        timestamp_column=timestamp_column,
        feature_columns=list(X.columns),
        horizon=horizon,
        step=infer_step(series, timestamp_column),
    )

    return ForecastPipelineResult(
        validation_report=validation_report,
        backtest=backtest,
        holdout=holdout,
        selected_model=selected_model,
        evaluation=evaluation,
        predictions=predictions,
        forecast=forecast,
        future_forecast=future_forecast,
        train_rows=len(split.X_train),
    )
