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
    If f_score_col exists but has NaN rows, recalculates those from components.
    If no components available, returns as-is.

    Component columns:
        roa, roa_prev, cfo, delta_leverage, delta_current_ratio,
        shares_issued, delta_gross_margin, delta_asset_turnover
    """
    components = [
        "roa", "roa_prev", "cfo", "delta_leverage",
        "delta_current_ratio", "shares_issued",
        "delta_gross_margin", "delta_asset_turnover",
    ]
    has_components = all(c in df_funda.columns for c in components)

    # If column exists and is fully populated, nothing to do
    if f_score_col in df_funda.columns and df_funda[f_score_col].notna().all():
        return df_funda

    # If no components to compute from, return as-is
    if not has_components:
        return df_funda

    df = df_funda.copy()

    # Compute from components
    df["_f1_roa"] = (df["roa"] > 0).astype(float).where(df["roa"].notna())
    df["_f2_cfo"] = (df["cfo"] > 0).astype(float).where(df["cfo"].notna())
    df["_f3_delta_roa"] = (df["roa"] > df["roa_prev"]).astype(float).where(
        df["roa"].notna() & df["roa_prev"].notna())
    df["_f4_accrual"] = (df["cfo"] > df["roa"]).astype(float).where(
        df["cfo"].notna() & df["roa"].notna())
    df["_f5_leverage"] = (df["delta_leverage"] < 0).astype(float).where(df["delta_leverage"].notna())
    df["_f6_liquidity"] = (df["delta_current_ratio"] > 0).astype(float).where(df["delta_current_ratio"].notna())
    df["_f7_dilution"] = (df["shares_issued"] == 0).astype(float)
    df["_f8_margin"] = (df["delta_gross_margin"] > 0).astype(float).where(df["delta_gross_margin"].notna())
    df["_f9_turnover"] = (df["delta_asset_turnover"] > 0).astype(float).where(df["delta_asset_turnover"].notna())

    f_cols = [c for c in df.columns if c.startswith("_f")]
    computed = df[f_cols].sum(axis=1)
    n_valid = df[f_cols].notna().sum(axis=1)
    # Only assign if at least 5 of 9 components are available
    computed = computed.where(n_valid >= 5)

    # Fill NaN f_score with computed values (preserve existing non-NaN)
    if f_score_col in df.columns:
        df[f_score_col] = df[f_score_col].fillna(computed)
    else:
        df[f_score_col] = computed

    df = df.drop(columns=f_cols)
    return df
