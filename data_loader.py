"""Data loading and harmonization for price and fundamental data."""

import pandas as pd
import numpy as np


def load_price_data(path: str) -> pd.DataFrame:
    """Load price CSV. Expects columns: date, ticker, close (minimum).
    Optional: open, high, low, volume, adj_close.
    Returns DataFrame with DatetimeIndex 'date' and multi-index or pivoted structure.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    required = {"date", "ticker", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in price data: {missing}")

    return df


def load_fundamental_data(path: str) -> pd.DataFrame:
    """Load fundamental CSV. Expects columns: date, ticker, plus fundamental fields.
    'date' = reporting/publication date of the fundamentals.
    Returns DataFrame sorted by date/ticker.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    required = {"date", "ticker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in fundamental data: {missing}")

    return df


def pivot_prices(df_prices: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """Pivot long-format prices to wide: index=date, columns=tickers, values=field."""
    return df_prices.pivot_table(index="date", columns="ticker", values=field)


def get_latest_fundamentals(df_funda: pd.DataFrame, as_of_date, pub_lag_months: int = 3) -> pd.DataFrame:
    """Get most recent fundamental row per ticker available before as_of_date minus pub_lag.
    This avoids look-ahead bias: we only use data published at least pub_lag_months ago.
    """
    cutoff = as_of_date - pd.DateOffset(months=pub_lag_months)
    available = df_funda[df_funda["date"] <= cutoff]
    if available.empty:
        return pd.DataFrame()
    latest = available.sort_values("date").groupby("ticker").last().reset_index()
    return latest


def generate_synthetic_data(
    n_tickers: int = 200,
    n_years: int = 10,
    freq: str = "ME",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic price and fundamental data for testing.
    Returns (df_prices, df_funda) in long format.
    """
    rng = np.random.default_rng(seed)
    tickers = [f"STOCK_{i:03d}" for i in range(n_tickers)]
    dates = pd.date_range("2013-01-31", periods=n_years * 12, freq=freq, tz="UTC")

    # --- prices ---
    rows_price = []
    for tk in tickers:
        drift = rng.uniform(0.02, 0.12)
        vol = rng.uniform(0.15, 0.45)
        log_returns = rng.normal(drift / 12, vol / np.sqrt(12), size=len(dates))
        prices = 100 * np.exp(np.cumsum(log_returns))
        for dt, px in zip(dates, prices):
            rows_price.append({"date": dt, "ticker": tk, "close": round(px, 2)})
    df_prices = pd.DataFrame(rows_price)

    # --- fundamentals (annual, published with ~3 month lag) ---
    annual_dates = dates[::12]
    rows_funda = []
    for tk in tickers:
        mcap_base = rng.uniform(0.5e9, 50e9)
        for dt in annual_dates:
            pub_date = dt + pd.DateOffset(months=3)
            pb = round(rng.uniform(0.3, 6.0), 2)
            f_score = int(rng.integers(0, 10))
            mcap = round(mcap_base * rng.uniform(0.8, 1.3), 0)
            rows_funda.append({
                "date": pub_date,
                "ticker": tk,
                "market_cap": mcap,
                "pb": pb,
                "roa": round(rng.uniform(-0.05, 0.20), 4),
                "roa_prev": round(rng.uniform(-0.05, 0.20), 4),
                "cfo": round(rng.uniform(-0.05, 0.25), 4),
                "delta_leverage": round(rng.uniform(-0.3, 0.3), 4),
                "delta_current_ratio": round(rng.uniform(-0.5, 0.5), 4),
                "shares_issued": int(rng.choice([0, 1], p=[0.7, 0.3])),
                "delta_gross_margin": round(rng.uniform(-0.1, 0.1), 4),
                "delta_asset_turnover": round(rng.uniform(-0.1, 0.1), 4),
            })
    df_funda = pd.DataFrame(rows_funda)

    return df_prices, df_funda
