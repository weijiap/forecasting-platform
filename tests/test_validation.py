import pandas as pd

from forecasting_platform.validation import validate_dataframe


def test_missing_values_are_counted() -> None:
    """Validation should correctly count missing values."""

    df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "sales": [100, None],
        }
    )

    report = validate_dataframe(
        df,
        timestamp_column="date",
    )

    assert report.missing_values == 1


def test_duplicate_timestamp_detection() -> None:
    """Duplicate timestamps should be counted."""

    df = pd.DataFrame(
        {
            "date": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-02",
            ],
            "sales": [100, 110, 120],
        }
    )

    report = validate_dataframe(
        df,
        timestamp_column="date",
    )

    assert report.duplicate_timestamps == 1
