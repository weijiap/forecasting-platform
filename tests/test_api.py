import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from forecasting_platform.api.main import app

client = TestClient(app)


def make_payload(periods: int = 90, horizon: int = 5) -> dict[str, object]:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2025-01-01", periods=periods, freq="D")

    return {
        "timestamps": [str(d.date()) for d in dates],
        "values": [
            float(100 + i * 0.5 + (i % 7) * 2 + rng.normal(0, 1))
            for i in range(periods)
        ],
        "horizon": horizon,
    }


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_forecast_summary_returns_benchmarks_and_future() -> None:
    response = client.post("/forecast/summary", json=make_payload(horizon=5))

    assert response.status_code == 200

    body = response.json()
    models = {score["model"] for score in body["benchmarks"]}

    assert {"naive", "seasonal_naive"} <= models
    assert body["model"] in models
    assert body["forecast_rows"] == 30
    assert len(body["future_forecast"]) == 5
    assert body["future_forecast"][0]["timestamp"] == "2025-04-01"


def test_forecast_summary_rejects_mismatched_lengths() -> None:
    payload = make_payload()
    payload["values"] = [1.0, 2.0]

    response = client.post("/forecast/summary", json=payload)

    assert response.status_code == 422


def test_forecast_summary_rejects_short_series() -> None:
    response = client.post("/forecast/summary", json=make_payload(periods=20))

    assert response.status_code == 400
    assert "rows are required" in response.json()["detail"]


def test_forecast_summary_rejects_invalid_horizon() -> None:
    response = client.post("/forecast/summary", json=make_payload(horizon=0))

    assert response.status_code == 422


def test_forecast_upload_csv() -> None:
    payload = make_payload(horizon=3)
    csv = pd.DataFrame(
        {"date": payload["timestamps"], "sales": payload["values"]}
    ).to_csv(index=False)

    response = client.post(
        "/forecast/upload",
        params={"horizon": 3},
        files={"file": ("sales.csv", csv, "text/csv")},
    )

    assert response.status_code == 200

    body = response.json()

    assert len(body["forecast"]) == 30
    assert len(body["future_forecast"]) == 3


def test_forecast_upload_rejects_non_csv() -> None:
    response = client.post(
        "/forecast/upload",
        files={"file": ("sales.txt", "a,b", "text/plain")},
    )

    assert response.status_code == 400
