import pandas as pd
import pytest

from forecasting_platform.preprocessing import prepare_series


def test_prepare_series_sorts_and_types() -> None:
    """The series should be sorted by time with a float target."""

    df = pd.DataFrame(
        {
            "date": ["2026-01-03", "2026-01-01", "2026-01-02"],
            "sales": ["30", "10", "20"],
            "ignored": [1, 2, 3],
        }
    )

    series = prepare_series(df, timestamp_column="date", target_column="sales")

    assert list(series.columns) == ["date", "sales"]
    assert series["sales"].tolist() == [10.0, 20.0, 30.0]
    assert series["date"].is_monotonic_increasing


def test_prepare_series_requires_columns() -> None:
    with pytest.raises(ValueError, match="'sales'"):
        prepare_series(
            pd.DataFrame({"date": ["2026-01-01"]}),
            timestamp_column="date",
            target_column="sales",
        )
