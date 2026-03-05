"""Compute fundamental scores: Price-to-Book, Piotroski F-Score, and Q-Score."""

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


def compute_f_score(
    df_funda: pd.DataFrame,
    f_score_col: str = "f_score",
    min_components: int = 3,
    scale_partial: bool = True,
) -> pd.DataFrame:
    """Compute Piotroski F-Score (0-9) from component columns.

    Supports partial F-Score: when fewer than 9 components are available,
    scales the raw sum to 0-9 range for comparability.

    Args:
        df_funda: DataFrame with fundamental data.
        f_score_col: Output column name.
        min_components: Minimum number of non-NaN components required (default 3).
        scale_partial: If True, scale partial scores to 0-9 range.
            E.g. 3/4 available components → scaled to 6.75/9.

    Also computes:
        f_score_components: number of available components (0-9)
        f_score_raw: unscaled sum of passing components
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

    # Compute 9 binary components
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
    raw_sum = df[f_cols].sum(axis=1)
    n_valid = df[f_cols].notna().sum(axis=1)

    # Store metadata
    df["f_score_components"] = n_valid
    df["f_score_raw"] = raw_sum.where(n_valid >= min_components)

    # Compute final score
    if scale_partial:
        # Scale to 0-9: if 4/5 components pass out of 5 available → 7.2/9
        computed = (9.0 * raw_sum / n_valid).where(n_valid >= min_components)
    else:
        computed = raw_sum.where(n_valid >= min_components)

    # Fill NaN f_score with computed values (preserve existing non-NaN)
    if f_score_col in df.columns:
        df[f_score_col] = df[f_score_col].fillna(computed)
    else:
        df[f_score_col] = computed

    df = df.drop(columns=f_cols)
    return df


def compute_q_score(df_funda: pd.DataFrame) -> pd.DataFrame:
    """Compute a simple Quality Score (0-3) from the 3 most robust XBRL fields.

    Q-Score uses only ROA, CFO, and leverage change — fields that are
    well-covered even in pre-2018 EDGAR data.

    Q1: ROA > 0 (profitable)
    Q2: CFO > 0 and CFO > ROA (cash quality, no accrual manipulation)
    Q3: Leverage not increasing (delta_leverage <= 0 or not available = pass)

    Returns DataFrame with added 'q_score' column (0-3).
    """
    df = df_funda.copy()

    q1 = (df["roa"] > 0).astype(float).where(df["roa"].notna()) if "roa" in df.columns else 0
    q2 = np.where(
        df["cfo"].notna() & df["roa"].notna(),
        ((df["cfo"] > 0) & (df["cfo"] > df["roa"])).astype(float),
        np.nan,
    ) if "cfo" in df.columns and "roa" in df.columns else 0
    q3 = np.where(
        df["delta_leverage"].notna(),
        (df["delta_leverage"] <= 0).astype(float),
        1.0,  # If leverage data missing, give benefit of the doubt
    ) if "delta_leverage" in df.columns else 1

    q_components = pd.DataFrame({"q1": q1, "q2": q2, "q3": q3})
    df["q_score"] = q_components.sum(axis=1).where(q_components.notna().sum(axis=1) >= 2)

    return df
