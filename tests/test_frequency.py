import pandas as pd

from forecasting_platform.frequency import detect_frequency


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
