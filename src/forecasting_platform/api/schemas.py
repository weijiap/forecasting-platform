from pydantic import BaseModel, Field

MAX_HORIZON = 365


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str


class ForecastRequest(BaseModel):
    """Time-series data submitted for forecasting."""

    timestamps: list[str]
    values: list[float]
    timestamp_column: str = "date"
    target_column: str = "value"
    horizon: int = Field(
        default=14,
        ge=1,
        le=MAX_HORIZON,
        description="Number of future periods to forecast.",
    )


class ModelScore(BaseModel):
    """Walk-forward and holdout accuracy for one candidate model."""

    model: str
    backtest_rmse: float
    backtest_mae: float
    holdout_rmse: float
    holdout_mae: float


class ForecastPoint(BaseModel):
    """One forecast beyond the last observation."""

    timestamp: str
    forecast: float


class ForecastSummaryResponse(BaseModel):
    """Forecast summary."""

    model: str
    rmse: float
    mae: float
    forecast_rows: int
    benchmarks: list[ModelScore]
    future_forecast: list[ForecastPoint]


class ForecastUploadResponse(ForecastSummaryResponse):
    """Forecast result returned after uploading a CSV file."""

    forecast: list[dict[str, str | float]]
