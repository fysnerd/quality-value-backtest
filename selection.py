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
    use_q_score_fallback: bool = False  # use Q-Score when F-Score missing
    min_q_score: int = 2  # minimum Q-Score (0-3) for fallback


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
    5. Apply Quality filter (high F-Score, with Q-Score fallback if enabled).
    6. Rank by F-Score desc, then P/B asc.
    7. Return top N tickers.
    """
    from data_loader import get_latest_fundamentals

    # 1. Latest fundamentals with lag
    funda = get_latest_fundamentals(df_funda, date, pub_lag_months)
    if funda.empty:
        return []

    # 2. Market cap filter (skip if all NaN or min_market_cap is 0)
    if "market_cap" in funda.columns and params.min_market_cap > 0:
        has_mcap = funda["market_cap"].notna()
        if has_mcap.any():
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

    # 5. Quality filter
    has_fscore = value_universe["f_score"].notna()
    quality_by_fscore = value_universe[has_fscore & (value_universe["f_score"] >= params.min_f_score)]

    if params.use_q_score_fallback and "q_score" in value_universe.columns:
        # Add stocks that lack F-Score but pass Q-Score
        no_fscore = value_universe[~has_fscore]
        quality_by_qscore = no_fscore[
            no_fscore["q_score"].notna() & (no_fscore["q_score"] >= params.min_q_score)
        ]
        # Give Q-Score stocks a synthetic f_score for ranking (scale q_score 0-3 to 0-9)
        if not quality_by_qscore.empty:
            quality_by_qscore = quality_by_qscore.copy()
            quality_by_qscore["f_score"] = quality_by_qscore["q_score"] * 3
        quality_value = pd.concat([quality_by_fscore, quality_by_qscore], ignore_index=True)
    else:
        quality_value = quality_by_fscore

    if quality_value.empty:
        return []

    # 6. Rank: F-Score desc, P/B asc
    ranked = quality_value.sort_values(
        ["f_score", "pb"], ascending=[False, True]
    )

    # 7. Top N
    selected = ranked.head(params.n_stocks)["ticker"].tolist()
    return selected
