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
    assert report.passed is False


def test_clean_data_passes() -> None:
    """A clean, sorted series should pass validation with no issues."""

    df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "sales": [100, 110, 120],
        }
    )

    report = validate_dataframe(df, timestamp_column="date", target_column="sales")

    assert report.passed is True
    assert report.is_sorted is True
    assert report.issues == ()


def test_unsorted_data_is_reported_but_passes() -> None:
    """The pipeline sorts the series, so unsorted rows are not an error."""

    df = pd.DataFrame(
        {
            "date": ["2026-01-03", "2026-01-01", "2026-01-02"],
            "sales": [120, 100, 110],
        }
    )

    report = validate_dataframe(df, timestamp_column="date", target_column="sales")

    assert report.is_sorted is False
    assert report.passed is True


def test_missing_and_non_numeric_target_fail() -> None:
    """Missing or non-numeric target values should fail validation."""

    df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "sales": [100, None, "abc"],
        }
    )

    report = validate_dataframe(df, timestamp_column="date", target_column="sales")

    assert report.passed is False
    assert any("2 target value" in issue for issue in report.issues)


def test_missing_columns_and_short_series_fail() -> None:
    """Missing columns and too few rows should each be reported."""

    df = pd.DataFrame({"date": ["2026-01-01"], "sales": [100]})

    missing_timestamp = validate_dataframe(df, timestamp_column="ds")
    missing_target = validate_dataframe(
        df, timestamp_column="date", target_column="y", min_rows=10
    )

    assert missing_timestamp.passed is False
    assert missing_target.passed is False
    assert len(missing_target.issues) == 2


def test_unparseable_timestamp_fails() -> None:
    """Timestamps that cannot be parsed should fail validation."""

    df = pd.DataFrame(
        {
            "date": ["2026-01-01", "not a date"],
            "sales": [100, 110],
        }
    )

    report = validate_dataframe(df, timestamp_column="date")

    assert report.passed is False
