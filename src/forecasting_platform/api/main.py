from io import BytesIO
from typing import Annotated, cast

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from forecasting_platform.api.schemas import (
    MAX_HORIZON,
    ForecastPoint,
    ForecastRequest,
    ForecastSummaryResponse,
    ForecastUploadResponse,
    HealthResponse,
    ModelScore,
)
from forecasting_platform.pipeline import ForecastPipelineResult, run_forecast_pipeline

API_VERSION = "0.2.0"

app = FastAPI(
    title="AutoForecast Lab API",
    description="API for automated time-series forecasting.",
    version=API_VERSION,
)


def _summarize(
    result: ForecastPipelineResult,
    timestamp_column: str,
) -> ForecastSummaryResponse:
    """Convert pipeline output into the API response."""

    holdout = {evaluation.model_name: evaluation for evaluation in result.holdout}

    benchmarks = [
        ModelScore(
            model=backtest.model_name,
            backtest_rmse=backtest.rmse,
            backtest_mae=backtest.mae,
            holdout_rmse=holdout[backtest.model_name].rmse,
            holdout_mae=holdout[backtest.model_name].mae,
        )
        for backtest in result.backtest
    ]

    future_forecast = [
        ForecastPoint(timestamp=str(timestamp.date()), forecast=float(value))
        for timestamp, value in zip(
            result.future_forecast[timestamp_column],
            result.future_forecast["forecast"],
            strict=True,
        )
    ]

    return ForecastSummaryResponse(
        model=result.selected_model,
        rmse=result.evaluation.rmse,
        mae=result.evaluation.mae,
        forecast_rows=len(result.forecast),
        benchmarks=benchmarks,
        future_forecast=future_forecast,
    )


@app.get("/")
def read_root() -> dict[str, str]:
    """Return basic API information."""

    return {
        "message": "Welcome to the AutoForecast Lab API",
        "documentation": "/docs",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health_check() -> HealthResponse:
    """Return the health status of the API."""

    return HealthResponse(
        status="healthy",
        service="forecasting-platform",
        version=API_VERSION,
    )


@app.post(
    "/forecast/summary",
    response_model=ForecastSummaryResponse,
)
def forecast_summary(
    request: ForecastRequest,
) -> ForecastSummaryResponse:
    """Run the forecasting pipeline and return metrics and future forecasts."""

    if len(request.timestamps) != len(request.values):
        raise HTTPException(
            status_code=422,
            detail="timestamps and values must contain the same number of items.",
        )

    df = pd.DataFrame(
        {
            request.timestamp_column: request.timestamps,
            request.target_column: request.values,
        }
    )

    try:
        result = run_forecast_pipeline(
            df=df,
            target_column=request.target_column,
            timestamp_column=request.timestamp_column,
            horizon=request.horizon,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return _summarize(result, request.timestamp_column)


@app.post(
    "/forecast/upload",
    response_model=ForecastUploadResponse,
)
async def forecast_upload(
    file: Annotated[UploadFile, File()],
    timestamp_column: str = "date",
    target_column: str = "sales",
    horizon: Annotated[int, Query(ge=1, le=MAX_HORIZON)] = 14,
) -> ForecastUploadResponse:
    """Upload a CSV file and run the forecasting pipeline."""

    if file.filename is None or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    content = await file.read()

    try:
        df = pd.read_csv(BytesIO(content))

        result = run_forecast_pipeline(
            df=df,
            target_column=target_column,
            timestamp_column=timestamp_column,
            horizon=horizon,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    forecast_records = result.forecast.copy()
    forecast_records[timestamp_column] = forecast_records[timestamp_column].astype(str)

    forecast_data = cast(
        list[dict[str, str | float]],
        forecast_records.to_dict(orient="records"),
    )

    summary = _summarize(result, timestamp_column)

    return ForecastUploadResponse(
        **summary.model_dump(),
        forecast=forecast_data,
    )
