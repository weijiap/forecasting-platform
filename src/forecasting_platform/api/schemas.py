from pydantic import BaseModel


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


class ForecastSummaryResponse(BaseModel):
    """Forecast summary."""

    model: str
    rmse: float
    mae: float
    forecast_rows: int


class ForecastUploadResponse(BaseModel):
    """Forecast result returned after uploading a CSV file."""

    model: str
    rmse: float
    mae: float
    forecast_rows: int
    forecast: list[dict[str, str | float]]
