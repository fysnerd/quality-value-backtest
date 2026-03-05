"""Build fundamentals CSV from SimFin bulk data (free tier).

Downloads balance sheet, income statement, and cash flow data,
then computes P/B ratio and Piotroski F-Score components.

Usage:
    python data_simfin.py              # build fundamentals_simfin.csv
    python data_simfin.py --refresh    # re-download from SimFin first
"""

import argparse
import os
import pandas as pd
import numpy as np

SIMFIN_DIR = os.path.expanduser("~/.simfin_data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_simfin_bulk() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the 3 SimFin bulk CSVs (semicolon-separated)."""
    bs = pd.read_csv(os.path.join(SIMFIN_DIR, "us-balance-annual.csv"), sep=";")
    pl = pd.read_csv(os.path.join(SIMFIN_DIR, "us-income-annual.csv"), sep=";")
    cf = pd.read_csv(os.path.join(SIMFIN_DIR, "us-cashflow-annual.csv"), sep=";")
    return bs, pl, cf


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def build_fundamentals(
    bs: pd.DataFrame, pl: pd.DataFrame, cf: pd.DataFrame
) -> pd.DataFrame:
    """Merge SimFin bulk data and compute P/B + F-Score components.

    Returns a DataFrame with columns:
        ticker, date (= Publish Date), fiscal_year, report_date,
        pb, f_score, market_cap,
        roa, roa_prev, cfo, delta_leverage, delta_current_ratio,
        shares_issued, delta_gross_margin, delta_asset_turnover,
        book_value_per_share, total_assets, total_equity, revenue
    """
    join_keys = ["Ticker", "Fiscal Year"]

    # Select needed columns from each statement
    bs_cols = join_keys + [
        "Report Date", "Publish Date",
        "Shares (Basic)", "Total Assets", "Total Equity",
        "Total Current Assets", "Total Current Liabilities",
        "Long Term Debt",
    ]
    pl_cols = join_keys + [
        "Revenue", "Cost of Revenue", "Gross Profit", "Net Income",
    ]
    cf_cols = join_keys + [
        "Net Cash from Operating Activities",
    ]

    bs_sub = bs[bs_cols].copy()
    pl_sub = pl[pl_cols].copy()
    cf_sub = cf[cf_cols].copy()

    # Merge on Ticker + Fiscal Year (Fiscal Period differs: BS="Q4", PL/CF="FY")
    df = bs_sub.merge(pl_sub, on=join_keys, how="inner")
    df = df.merge(cf_sub, on=join_keys, how="inner")

    # Deduplicate: keep latest filing per Ticker + Fiscal Year
    df["Publish Date"] = pd.to_datetime(df["Publish Date"])
    df["Report Date"] = pd.to_datetime(df["Report Date"])
    df = df.sort_values("Publish Date").drop_duplicates(
        subset=["Ticker", "Fiscal Year"], keep="last"
    )

    # ---- Derived columns ----

    # Book value per share
    df["book_value_per_share"] = _safe_div(df["Total Equity"], df["Shares (Basic)"])

    # ROA = Net Income / Total Assets
    df["roa"] = _safe_div(df["Net Income"], df["Total Assets"])

    # CFO (scaled by total assets for comparability)
    df["cfo"] = _safe_div(
        df["Net Cash from Operating Activities"], df["Total Assets"]
    )

    # Gross margin = Gross Profit / Revenue
    df["gross_margin"] = _safe_div(df["Gross Profit"], df["Revenue"])

    # Asset turnover = Revenue / Total Assets
    df["asset_turnover"] = _safe_div(df["Revenue"], df["Total Assets"])

    # Leverage = Long Term Debt / Total Assets
    df["leverage"] = _safe_div(df["Long Term Debt"].fillna(0), df["Total Assets"])

    # Current ratio = Current Assets / Current Liabilities
    df["current_ratio"] = _safe_div(
        df["Total Current Assets"], df["Total Current Liabilities"]
    )

    # ---- Year-over-year deltas (need prev year) ----
    df = df.sort_values(["Ticker", "Fiscal Year"])

    for col in ["roa", "leverage", "current_ratio", "gross_margin", "asset_turnover", "Shares (Basic)"]:
        df[f"_prev_{col}"] = df.groupby("Ticker")[col].shift(1)

    df["roa_prev"] = df["_prev_roa"]
    df["delta_leverage"] = df["leverage"] - df["_prev_leverage"]
    df["delta_current_ratio"] = df["current_ratio"] - df["_prev_current_ratio"]
    df["delta_gross_margin"] = df["gross_margin"] - df["_prev_gross_margin"]
    df["delta_asset_turnover"] = df["asset_turnover"] - df["_prev_asset_turnover"]

    # Shares issued: 1 if new shares issued (dilution), 0 otherwise
    df["shares_issued"] = (
        df["Shares (Basic)"] > df["_prev_Shares (Basic)"]
    ).astype(float).where(df["_prev_Shares (Basic)"].notna())
    # Convention: 0 = no dilution (good), mapped to F-Score point later

    # ---- Piotroski F-Score (9 components) ----
    f1 = (df["roa"] > 0).astype(float)
    f2 = (df["cfo"] > 0).astype(float)
    f3 = (df["roa"] > df["roa_prev"]).astype(float).where(df["roa_prev"].notna())
    f4 = (df["cfo"] > df["roa"]).astype(float)  # accrual quality
    f5 = (df["delta_leverage"] < 0).astype(float).where(df["delta_leverage"].notna())
    f6 = (df["delta_current_ratio"] > 0).astype(float).where(df["delta_current_ratio"].notna())
    f7 = (df["shares_issued"] == 0).astype(float).where(df["shares_issued"].notna())
    f8 = (df["delta_gross_margin"] > 0).astype(float).where(df["delta_gross_margin"].notna())
    f9 = (df["delta_asset_turnover"] > 0).astype(float).where(df["delta_asset_turnover"].notna())

    f_components = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3, "f4": f4,
                                  "f5": f5, "f6": f6, "f7": f7, "f8": f8, "f9": f9})
    n_valid = f_components.notna().sum(axis=1)
    df["f_score"] = f_components.sum(axis=1).where(n_valid >= 5)

    # ---- Market cap proxy (need price data — will be joined later) ----
    # For now, set market_cap = NaN; the backtester uses it for filtering
    df["market_cap"] = np.nan

    # ---- Output columns ----
    result = pd.DataFrame({
        "date": df["Publish Date"],
        "ticker": df["Ticker"],
        "fiscal_year": df["Fiscal Year"],
        "report_date": df["Report Date"],
        "pb": np.nan,  # needs price — computed at backtest time or enriched below
        "f_score": df["f_score"],
        "market_cap": df["market_cap"],
        "book_value_per_share": df["book_value_per_share"],
        "total_assets": df["Total Assets"],
        "total_equity": df["Total Equity"],
        "revenue": df["Revenue"],
        "roa": df["roa"],
        "roa_prev": df["roa_prev"],
        "cfo": df["cfo"],
        "delta_leverage": df["delta_leverage"],
        "delta_current_ratio": df["delta_current_ratio"],
        "shares_issued": df["shares_issued"],
        "delta_gross_margin": df["delta_gross_margin"],
        "delta_asset_turnover": df["delta_asset_turnover"],
    })

    return result.reset_index(drop=True)


def enrich_with_prices(df_funda: pd.DataFrame, df_prices: pd.DataFrame) -> pd.DataFrame:
    """Add P/B and market_cap using price data via merge_asof (vectorized)."""
    df_funda = df_funda.copy()
    df_prices = df_prices.copy()

    df_funda["date"] = pd.to_datetime(df_funda["date"], utc=True)
    df_prices["date"] = pd.to_datetime(df_prices["date"], utc=True)

    # merge_asof: for each fundamental row, find closest price <= publish date
    df_funda = df_funda.sort_values("date")
    df_prices = df_prices.sort_values("date")

    merged = pd.merge_asof(
        df_funda,
        df_prices.rename(columns={"close": "_price"}),
        on="date",
        by="ticker",
        direction="nearest",
        tolerance=pd.Timedelta("90D"),
    )

    # Compute P/B and market_cap
    bvps = merged["book_value_per_share"]
    px = merged["_price"]
    valid = px.notna() & bvps.notna() & (bvps > 0)
    merged.loc[valid, "pb"] = px[valid] / bvps[valid]

    # Market cap = price * shares (shares = equity / bvps)
    shares = merged["total_equity"] / bvps
    valid_mc = valid & shares.notna() & (shares > 0)
    merged.loc[valid_mc, "market_cap"] = px[valid_mc] * shares[valid_mc]

    merged = merged.drop(columns=["_price"])

    return merged


def build_and_save(prices_path: str | None = None) -> pd.DataFrame:
    """Full pipeline: load SimFin → compute fundamentals → enrich with prices → save CSV."""
    print("Loading SimFin bulk data...")
    bs, pl, cf = load_simfin_bulk()
    print(f"  BS: {len(bs)} rows, PL: {len(pl)} rows, CF: {len(cf)} rows")

    print("Computing fundamentals + F-Score...")
    df_funda = build_fundamentals(bs, pl, cf)
    print(f"  {len(df_funda)} fundamental rows, {df_funda['ticker'].nunique()} tickers")
    print(f"  F-Score coverage: {df_funda['f_score'].notna().sum()}/{len(df_funda)} "
          f"({df_funda['f_score'].notna().mean():.0%})")

    # Enrich with prices if available
    if prices_path is None:
        prices_path = os.path.join(OUTPUT_DIR, "prices.csv")

    if os.path.exists(prices_path):
        print(f"Enriching with prices from {prices_path}...")
        df_prices = pd.read_csv(prices_path)
        df_funda = enrich_with_prices(df_funda, df_prices)
        pb_ok = df_funda["pb"].notna().sum()
        print(f"  P/B computed for {pb_ok}/{len(df_funda)} rows")
    else:
        print(f"  No prices file at {prices_path} — P/B will be NaN")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "fundamentals_simfin.csv")
    df_funda.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    # Summary stats
    print("\n=== Summary ===")
    print(f"Tickers: {df_funda['ticker'].nunique()}")
    if len(df_funda) > 0:
        print(f"Fiscal years: {int(df_funda['fiscal_year'].min())}-{int(df_funda['fiscal_year'].max())}")
        print(f"F-Score distribution:")
        print(df_funda["f_score"].value_counts().sort_index().to_string())
        print(f"\nP/B stats:")
        print(df_funda["pb"].describe().to_string())
    else:
        print("No data produced!")

    return df_funda


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build fundamentals from SimFin bulk data")
    parser.add_argument("--prices", default=None, help="Path to prices CSV")
    parser.add_argument("--refresh", action="store_true", help="Re-download SimFin data first")
    args = parser.parse_args()

    if args.refresh:
        import simfin as sf
        sf.set_api_key("6b03ff50-7a8e-4581-95e5-4a7ab7384750")
        sf.set_data_dir(SIMFIN_DIR)
        print("Downloading fresh SimFin data...")
        sf.load(dataset="us-balance", variant="annual", market="us")
        sf.load(dataset="us-income", variant="annual", market="us")
        sf.load(dataset="us-cashflow", variant="annual", market="us")

    build_and_save(prices_path=args.prices)
