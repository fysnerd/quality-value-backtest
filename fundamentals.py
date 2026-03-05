"""Compute fundamental scores: Price-to-Book and Piotroski F-Score."""

import pandas as pd
import numpy as np


def compute_pb(df_funda: pd.DataFrame, pb_col: str = "pb") -> pd.DataFrame:
    """Ensure P/B column exists. If raw book_value and price are provided, compute it.
    Otherwise, assumes 'pb' column already present.
    """
    if pb_col in df_funda.columns:
        return df_funda

    if "book_value_per_share" in df_funda.columns and "price" in df_funda.columns:
        df_funda = df_funda.copy()
        df_funda[pb_col] = df_funda["price"] / df_funda["book_value_per_share"].replace(0, np.nan)
        return df_funda

    raise ValueError(f"Cannot compute P/B: need '{pb_col}' column or 'book_value_per_share' + 'price'.")


def compute_f_score(df_funda: pd.DataFrame, f_score_col: str = "f_score") -> pd.DataFrame:
    """Compute Piotroski F-Score (0-9) from component columns.
    If f_score_col already exists, return as-is.

    Required component columns (if computing from scratch):
        roa, roa_prev, cfo, delta_leverage, delta_current_ratio,
        shares_issued, delta_gross_margin, delta_asset_turnover
    """
    if f_score_col in df_funda.columns:
        return df_funda

    required = [
        "roa", "roa_prev", "cfo", "delta_leverage",
        "delta_current_ratio", "shares_issued",
        "delta_gross_margin", "delta_asset_turnover",
    ]
    missing = set(required) - set(df_funda.columns)
    if missing:
        raise ValueError(f"Missing columns to compute F-Score: {missing}")

    df = df_funda.copy()

    # Profitability (4 points)
    df["_f1_roa"] = (df["roa"] > 0).astype(int)
    df["_f2_cfo"] = (df["cfo"] > 0).astype(int)
    df["_f3_delta_roa"] = (df["roa"] > df["roa_prev"]).astype(int)
    df["_f4_accrual"] = (df["cfo"] > df["roa"]).astype(int)

    # Leverage / Liquidity (3 points)
    df["_f5_leverage"] = (df["delta_leverage"] < 0).astype(int)
    df["_f6_liquidity"] = (df["delta_current_ratio"] > 0).astype(int)
    df["_f7_dilution"] = (df["shares_issued"] == 0).astype(int)

    # Operating efficiency (2 points)
    df["_f8_margin"] = (df["delta_gross_margin"] > 0).astype(int)
    df["_f9_turnover"] = (df["delta_asset_turnover"] > 0).astype(int)

    f_cols = [c for c in df.columns if c.startswith("_f")]
    df[f_score_col] = df[f_cols].sum(axis=1)
    df = df.drop(columns=f_cols)

    return df
