"""Generate the example dataset in ``data/sample_sales.csv``.

One year of daily sales with a linear trend, yearly and weekly seasonality,
and autocorrelated noise. A fixed seed makes the file reproducible.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "sample_sales.csv"

# Average deviation from trend for Monday through Sunday.
WEEKDAY_EFFECT = np.array([-8.0, -3.0, 0.0, 2.0, 6.0, 14.0, -11.0])


def generate_sample_sales(periods: int = 365, seed: int = 7) -> pd.DataFrame:
    """Return a synthetic daily sales series."""

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=periods, freq="D")
    t = np.arange(periods)

    noise = np.zeros(periods)
    for i in range(1, periods):
        noise[i] = 0.5 * noise[i - 1] + rng.normal(0.0, 4.0)

    sales = (
        200.0
        + 0.2 * t
        + 10.0 * np.sin(2 * np.pi * t / 365)
        + WEEKDAY_EFFECT[dates.dayofweek]
        + noise
    )

    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "sales": sales.round(1)})


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    generate_sample_sales().to_csv(OUTPUT_PATH, index=False)
    print(f"Sample data saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
