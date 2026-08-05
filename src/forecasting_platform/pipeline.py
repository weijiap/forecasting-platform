from dataclasses import dataclass

import pandas as pd

from forecasting_platform.features import create_features
from forecasting_platform.forecasting import generate_predictions
from forecasting_platform.training import (
    TrainerConfig,
    prepare_training_data,
    train_random_forest,
)
from forecasting_platform.training.benchmark import (
    ModelEvaluation,
    evaluate_model,
)
from forecasting_platform.training.dataset import TrainingResult
from forecasting_platform.training.splitter import split_time_series
from forecasting_platform.validation import ValidationReport, validate_dataframe


@dataclass(frozen=True)
class ForecastPipelineResult:
    """Outputs produced by the forecasting pipeline."""

    validation_report: ValidationReport
    training_result: TrainingResult
    evaluation: ModelEvaluation
    predictions: pd.Series
    forecast: pd.DataFrame


def run_forecast_pipeline(
    df: pd.DataFrame,
    target_column: str,
    timestamp_column: str,
) -> ForecastPipelineResult:
    """Run the complete forecasting pipeline."""

    validation_report = validate_dataframe(
        df,
        timestamp_column=timestamp_column,
    )

    feature_df = create_features(
        df,
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
        test_size=30,
    )

    training_result = train_random_forest(
        split.X_train,
        split.y_train,
        TrainerConfig(),
    )

    evaluation = evaluate_model(
        model=training_result.model,
        model_name="random_forest",
        X_test=split.X_test,
        y_test=split.y_test,
    )

    predictions = generate_predictions(
        model=training_result.model,
        X=split.X_test,
    )

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

    return ForecastPipelineResult(
        validation_report=validation_report,
        training_result=training_result,
        evaluation=evaluation,
        predictions=predictions,
        forecast=forecast,
    )
