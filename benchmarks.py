"""Benchmark comparison: compute metrics for multiple series side by side."""

import pandas as pd
import numpy as np
from metrics import compute_performance_metrics, MetricsParams


def load_benchmark(path: str, date_col: str = "date", value_col: str = "close") -> pd.Series:
    """Load a benchmark series from CSV. Returns Series indexed by date."""
    df = pd.read_csv(path, parse_dates=[date_col])
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    df = df.sort_values(date_col).set_index(date_col)
    return df[value_col]


def align_series(series_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """Align multiple series to a common date index (inner join)."""
    df = pd.DataFrame(series_dict)
    df = df.dropna()
    return df


def compare_strategies(
    series_dict: dict[str, pd.Series],
    params: MetricsParams | None = None,
) -> pd.DataFrame:
    """Compute performance metrics for each series and return comparison table.

    Args:
        series_dict: {"Strategy Name": equity_curve_series, "SPY": spy_series, ...}
        params: Metrics parameters.

    Returns:
        DataFrame with strategies as columns and metrics as rows.
    """
    if params is None:
        params = MetricsParams()

    aligned = align_series(series_dict)
    results = {}

    for name in aligned.columns:
        curve = aligned[name]
        # Normalize to start at same value for fair comparison
        normalized = curve / curve.iloc[0] * 100
        m = compute_performance_metrics(normalized, params=params)
        results[name] = m

    return pd.DataFrame(results)


def normalize_series(series_dict: dict[str, pd.Series], base: float = 100.0) -> pd.DataFrame:
    """Normalize all series to start at `base` for visual comparison."""
    aligned = align_series(series_dict)
    for col in aligned.columns:
        aligned[col] = aligned[col] / aligned[col].iloc[0] * base
    return aligned
