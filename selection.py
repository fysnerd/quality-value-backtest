"""Quality/Value universe selection at a given rebalancing date."""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class SelectionParams:
    min_market_cap: float = 1e9
    pb_percentile_cut: float = 0.20  # keep bottom 20% P/B
    min_f_score: int = 7
    n_stocks: int = 40


def select_quality_value_universe(
    date: pd.Timestamp,
    df_prices: pd.DataFrame,
    df_funda: pd.DataFrame,
    params: SelectionParams,
    pub_lag_months: int = 3,
) -> list[str]:
    """Select tickers for the Quality/Value portfolio at a given rebalancing date.

    Steps:
    1. Get latest fundamentals with publication lag.
    2. Filter on market cap.
    3. Filter on available price data.
    4. Apply Value filter (low P/B percentile).
    5. Apply Quality filter (high F-Score).
    6. Rank by F-Score desc, then P/B asc.
    7. Return top N tickers.
    """
    from data_loader import get_latest_fundamentals

    # 1. Latest fundamentals with lag
    funda = get_latest_fundamentals(df_funda, date, pub_lag_months)
    if funda.empty:
        return []

    # 2. Market cap filter
    if "market_cap" in funda.columns:
        funda = funda[funda["market_cap"] >= params.min_market_cap]

    # 3. Available price data at this date
    latest_prices = df_prices[df_prices["date"] <= date]
    available_tickers = set(latest_prices["ticker"].unique())
    funda = funda[funda["ticker"].isin(available_tickers)]

    if funda.empty:
        return []

    # 4. Value filter: keep bottom pb_percentile_cut of P/B
    pb_threshold = funda["pb"].quantile(params.pb_percentile_cut)
    value_universe = funda[funda["pb"] <= pb_threshold]

    # 5. Quality filter: F-Score >= min_f_score
    quality_value = value_universe[value_universe["f_score"] >= params.min_f_score]

    if quality_value.empty:
        return []

    # 6. Rank: F-Score desc, P/B asc
    ranked = quality_value.sort_values(
        ["f_score", "pb"], ascending=[False, True]
    )

    # 7. Top N
    selected = ranked.head(params.n_stocks)["ticker"].tolist()
    return selected
