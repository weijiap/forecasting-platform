from io import BytesIO
from typing import Annotated, cast

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from forecasting_platform.api.schemas import (
    ForecastRequest,
    ForecastSummaryResponse,
    ForecastUploadResponse,
    HealthResponse,
)
from forecasting_platform.pipeline import run_forecast_pipeline

app = FastAPI(
    title="AutoForecast Lab API",
    description="API for automated time-series forecasting.",
    version="0.1.0",
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
        version="0.1.0",
    )


@app.post(
    "/forecast/summary",
    response_model=ForecastSummaryResponse,
)
def forecast_summary(
    request: ForecastRequest,
) -> ForecastSummaryResponse:
    """Run the forecasting pipeline and return summary metrics."""

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
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return ForecastSummaryResponse(
        model=result.evaluation.model_name,
        rmse=result.evaluation.rmse,
        mae=result.evaluation.mae,
        forecast_rows=len(result.forecast),
    )


@app.post(
    "/forecast/upload",
    response_model=ForecastUploadResponse,
)
async def forecast_upload(
    file: Annotated[UploadFile, File()],
    timestamp_column: str = "date",
    target_column: str = "sales",
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

    return ForecastUploadResponse(
        model=result.evaluation.model_name,
        rmse=result.evaluation.rmse,
        mae=result.evaluation.mae,
        forecast_rows=len(result.forecast),
        forecast=forecast_data,
    )
