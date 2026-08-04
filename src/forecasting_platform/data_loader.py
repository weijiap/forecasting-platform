from pathlib import Path

import pandas as pd

SUPPORTED_SUFFIXES = {".csv", ".parquet"}


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load a supported data file into a pandas DataFrame."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            "Supported formats are .csv and .parquet."
        )

    if suffix == ".csv":
        return pd.read_csv(path)

    return pd.read_parquet(path)
