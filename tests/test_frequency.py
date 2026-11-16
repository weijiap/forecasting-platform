import pandas as pd
import pytest

from forecasting_platform.frequency import detect_frequency, infer_step


def test_detect_daily_frequency() -> None:
    """Daily timestamps should be detected."""

    df = pd.DataFrame(
        {
            "date": pd.date_range(
                start="2026-01-01",
                periods=5,
                freq="D",
            )
        }
    )

    frequency = detect_frequency(
        df,
        timestamp_column="date",
    )

    assert frequency == "D"


def test_infer_step_uses_pandas_frequency() -> None:
    """Regular monthly data should return the pandas frequency string."""

    df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=6, freq="MS")})

    assert infer_step(df, timestamp_column="date") == "MS"


def test_infer_step_falls_back_to_median_gap() -> None:
    """Irregular data should fall back to the median spacing."""

    df = pd.DataFrame(
        {"date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-04"])}
    )

    assert infer_step(df, timestamp_column="date") == pd.Timedelta(days=1.5)


def test_infer_step_needs_two_timestamps() -> None:
    with pytest.raises(ValueError, match="At least two"):
        infer_step(pd.DataFrame({"date": ["2026-01-01"]}), timestamp_column="date")
