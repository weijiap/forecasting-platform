from pathlib import Path

import pandas as pd
import pytest

from forecasting_platform.data_loader import load_data


def test_load_csv() -> None:
    """CSV files should load into a pandas DataFrame."""

    path = Path("data/sample_sales.csv")

    result = load_data(path)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["date", "sales"]
    assert len(result) == 120


def test_missing_file_raises_error() -> None:
    """Missing files should raise FileNotFoundError."""

    with pytest.raises(FileNotFoundError):
        load_data("data/missing.csv")


def test_unsupported_format_raises_error(tmp_path: Path) -> None:
    """Unsupported files should raise ValueError."""

    file_path = tmp_path / "sample.txt"
    file_path.write_text("example", encoding="utf-8")

    with pytest.raises(ValueError):
        load_data(file_path)
