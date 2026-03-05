"""Download fundamental data from SEC EDGAR XBRL API (free, no API key).

Covers US-listed companies from ~2008 to present.
Computes P/B and Piotroski F-Score from raw financial statements.

Usage:
    python data_sec_edgar.py                    # download all
    python data_sec_edgar.py --tickers AAPL MSFT  # specific tickers
    python data_sec_edgar.py --max-tickers 50   # limit universe size
"""

import os
import sys
import time
import argparse
import json
import pandas as pd
import numpy as np
import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SEC_BASE = "https://data.sec.gov/api/xbrl/companyfacts"
SEC_HEADERS = {"User-Agent": "QVBacktest quantresearch@example.com"}

# Same universe as data_download.py
SP500_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "UNH", "XOM", "JNJ",
    "JPM", "V", "PG", "MA", "AVGO", "HD", "CVX", "MRK", "ABBV", "LLY",
    "PEP", "KO", "COST", "TMO", "MCD", "WMT", "CSCO", "ABT", "CRM", "ACN",
    "BAC", "CMCSA", "PFE", "NFLX", "AMD", "DHR", "TXN", "ORCL", "LIN", "NKE",
    "UPS", "PM", "NEE", "UNP", "RTX", "LOW", "INTC", "QCOM", "INTU", "HON",
    "SPGI", "AMGN", "IBM", "AMAT", "GE", "CAT", "BA", "GS", "ELV", "DE",
    "BLK", "MDT", "ADP", "SYK", "GILD", "ISRG", "BKNG", "VRTX", "ADI", "MMC",
    "REGN", "LRCX", "MDLZ", "CB", "ZTS", "SCHW", "CI", "SO", "DUK", "MO",
    "BDX", "CL", "CME", "PLD", "MMM", "TGT", "BSX", "SLB", "AON", "NOC",
    "FDX", "EQIX", "WM", "APD", "ICE", "ITW", "GD", "EMR", "PNC", "HUM",
    "SHW", "MCK", "ORLY", "CCI", "AZO", "KLAC", "MPC", "PSX", "VLO", "CTSH",
    "ADSK", "MNST", "AIG", "F", "GM", "ALL", "TFC", "USB", "D", "AEP",
    "SRE", "EXC", "XEL", "WEC", "ES", "PEG", "ED", "AWK", "ETR", "DTE",
    "O", "SPG", "PSA", "AMT", "WELL", "EQR", "AVB", "DLR", "VTR", "MAA",
    "AFL", "MET", "PRU", "TRV", "AJG", "CINF", "L", "WRB", "GL", "BEN",
]


def _load_cik_map() -> dict[str, int]:
    """Load ticker -> CIK mapping from SEC."""
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=SEC_HEADERS)
    r.raise_for_status()
    data = r.json()
    return {v["ticker"]: int(v["cik_str"]) for v in data.values()}


def _get_company_facts(cik: int) -> dict | None:
    """Fetch all XBRL facts for a company from SEC EDGAR."""
    url = f"{SEC_BASE}/CIK{cik:010d}.json"
    r = requests.get(url, headers=SEC_HEADERS)
    if r.status_code == 200:
        return r.json()
    return None


def _extract_annual_values(facts: dict, field: str, unit: str = "USD") -> list[dict]:
    """Extract annual (10-K) values for a given XBRL field.
    Returns list of {end_date, value, filed_date}.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    if field not in us_gaap:
        return []

    units = us_gaap[field].get("units", {})
    entries = units.get(unit, [])

    # Collect all 10-K entries, deduplicate by end_date (keep latest filing)
    by_date = {}
    for e in entries:
        form = e.get("form", "")
        if form not in ("10-K", "10-K/A"):
            continue
        end = e["end"]
        filed = e.get("filed", end)
        # Keep the most recently filed value for each fiscal year end
        if end not in by_date or filed > by_date[end]["filed"]:
            by_date[end] = {"end_date": end, "value": e["val"], "filed": filed}

    return list(by_date.values())


def _build_annual_table(facts: dict) -> pd.DataFrame:
    """Build a table of annual fundamental data from XBRL facts."""
    # Fields we need and their XBRL tags (with alternatives)
    field_map = {
        "total_assets": ["Assets"],
        "stockholders_equity": [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ],
        "net_income": [
            "NetIncomeLoss",
            "ProfitLoss",
        ],
        "revenue": [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
        ],
        "cogs": [
            "CostOfGoodsAndServicesSold",
            "CostOfRevenue",
            "CostOfGoodsSold",
        ],
        "operating_cf": [
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByOperatingActivities",
        ],
        "long_term_debt": [
            "LongTermDebt",
            "LongTermDebtNoncurrent",
            "LongTermDebtAndCapitalLeaseObligations",
        ],
        "shares_outstanding": [
            "CommonStockSharesOutstanding",
            "EntityCommonStockSharesOutstanding",
            "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
        ],
        "current_assets": ["AssetsCurrent"],
        "current_liabilities": ["LiabilitiesCurrent"],
    }

    all_data = {}

    for our_field, xbrl_tags in field_map.items():
        for tag in xbrl_tags:
            # Try USD first, then shares
            for unit in ["USD", "shares", "pure"]:
                rows = _extract_annual_values(facts, tag, unit)
                if rows:
                    break
            if rows:
                for r in rows:
                    date = r["end_date"]
                    filed = r["filed"]
                    if date not in all_data:
                        all_data[date] = {"end_date": date, "filed_date": filed}
                    all_data[date][our_field] = r["value"]
                    # Keep most recent filed date
                    if filed > all_data[date].get("filed_date", ""):
                        all_data[date]["filed_date"] = filed
                break  # Use first tag that works

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(list(all_data.values()))
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["filed_date"] = pd.to_datetime(df["filed_date"])
    df = df.sort_values("end_date").reset_index(drop=True)

    return df


def _compute_fundamentals(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Compute P/B ratio components, F-Score components from raw annual data.
    Uses filing date as publication date (real-world availability).
    """
    if df.empty:
        return pd.DataFrame()

    rows = []
    for i in range(len(df)):
        r = df.iloc[i]
        row = {
            "ticker": ticker,
            # Use filing date as publication date (when investors actually see it)
            "date": r["filed_date"],
            "fiscal_year_end": r["end_date"],
        }

        ta = r.get("total_assets")
        eq = r.get("stockholders_equity")
        ni = r.get("net_income")
        rev = r.get("revenue")
        cogs = r.get("cogs")
        ocf = r.get("operating_cf")
        ltd = r.get("long_term_debt", 0) or 0
        shares = r.get("shares_outstanding")
        ca = r.get("current_assets")
        cl = r.get("current_liabilities")

        # Book value per share
        if eq and shares and shares > 0:
            row["book_value_per_share"] = eq / shares
        else:
            row["book_value_per_share"] = np.nan

        # Market cap placeholder (will be filled from price data)
        row["market_cap"] = np.nan

        # P/B placeholder (will be filled from price data)
        row["pb"] = np.nan

        # ROA
        row["roa"] = ni / ta if ni is not None and ta and ta > 0 else np.nan

        # CFO / Total Assets
        row["cfo"] = ocf / ta if ocf is not None and ta and ta > 0 else np.nan

        # Previous year values for deltas
        if i > 0:
            prev = df.iloc[i - 1]
            prev_ta = prev.get("total_assets")
            prev_ni = prev.get("net_income")
            prev_eq = prev.get("stockholders_equity")
            prev_ltd = prev.get("long_term_debt", 0) or 0
            prev_ca = prev.get("current_assets")
            prev_cl = prev.get("current_liabilities")
            prev_rev = prev.get("revenue")
            prev_cogs = prev.get("cogs")

            row["roa_prev"] = prev_ni / prev_ta if prev_ni is not None and prev_ta and prev_ta > 0 else np.nan

            # Delta leverage
            if ta and ta > 0 and prev_ta and prev_ta > 0:
                row["delta_leverage"] = (ltd / ta) - (prev_ltd / prev_ta)
            else:
                row["delta_leverage"] = np.nan

            # Delta current ratio
            if cl and cl > 0 and prev_cl and prev_cl > 0 and ca and prev_ca:
                row["delta_current_ratio"] = (ca / cl) - (prev_ca / prev_cl)
            else:
                row["delta_current_ratio"] = np.nan

            # Shares dilution
            prev_shares = prev.get("shares_outstanding")
            if shares and prev_shares and prev_shares > 0:
                row["shares_issued"] = 1 if shares > prev_shares * 1.01 else 0
            else:
                row["shares_issued"] = 0

            # Delta gross margin
            if rev and cogs and rev > 0 and prev_rev and prev_cogs and prev_rev > 0:
                gm = (rev - cogs) / rev
                gm_prev = (prev_rev - prev_cogs) / prev_rev
                row["delta_gross_margin"] = gm - gm_prev
            else:
                row["delta_gross_margin"] = np.nan

            # Delta asset turnover
            if rev and ta and ta > 0 and prev_rev and prev_ta and prev_ta > 0:
                row["delta_asset_turnover"] = (rev / ta) - (prev_rev / prev_ta)
            else:
                row["delta_asset_turnover"] = np.nan
        else:
            row["roa_prev"] = np.nan
            row["delta_leverage"] = np.nan
            row["delta_current_ratio"] = np.nan
            row["shares_issued"] = 0
            row["delta_gross_margin"] = np.nan
            row["delta_asset_turnover"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def _fill_pb_from_prices(df_funda: pd.DataFrame, df_prices: pd.DataFrame) -> pd.DataFrame:
    """Fill P/B and market_cap using price data at filing date."""
    df = df_funda.copy()

    # Build price lookup: for each ticker, get closest price to a date
    price_pivot = df_prices.pivot_table(index="date", columns="ticker", values="close")

    for i, row in df.iterrows():
        tk = row["ticker"]
        pub_date = row["date"]
        bvps = row.get("book_value_per_share")

        if tk not in price_pivot.columns or pd.isna(bvps) or bvps <= 0:
            continue

        # Find closest price on or before pub_date
        prices = price_pivot[tk].dropna()
        before = prices[prices.index <= pub_date]
        if before.empty:
            continue

        px = before.iloc[-1]
        df.loc[i, "pb"] = round(px / bvps, 3)

        # Rough market cap estimate (price * shares from last known)
        # Already have bvps and equity, can estimate shares
        if not pd.isna(row.get("book_value_per_share")) and row["book_value_per_share"] > 0:
            df.loc[i, "market_cap"] = px * (row.get("book_value_per_share", 1) / bvps) * px  # simplified

    return df


def _fill_market_cap_from_prices(df_funda: pd.DataFrame, df_prices: pd.DataFrame, df_shares: pd.DataFrame) -> pd.DataFrame:
    """Better market cap estimation using shares outstanding and price."""
    df = df_funda.copy()
    price_pivot = df_prices.pivot_table(index="date", columns="ticker", values="close")

    for i, row in df.iterrows():
        tk = row["ticker"]
        pub_date = row["date"]

        if tk not in price_pivot.columns:
            continue

        prices = price_pivot[tk].dropna()
        before = prices[prices.index <= pub_date]
        if before.empty:
            continue
        px = before.iloc[-1]

        # Get shares outstanding from the same row
        shares_rows = df_shares[
            (df_shares["ticker"] == tk) & (df_shares["date"] <= pub_date)
        ]
        if not shares_rows.empty:
            shares = shares_rows.iloc[-1].get("shares_outstanding", np.nan)
            if not pd.isna(shares) and shares > 0:
                df.loc[i, "market_cap"] = px * shares

    return df


def download_sec_fundamentals(
    tickers: list[str] | None = None,
    max_tickers: int | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Download fundamental data from SEC EDGAR for given tickers.

    Args:
        tickers: List of tickers. Defaults to SP500_TICKERS.
        max_tickers: Limit number of tickers (for testing).
        output_path: Output CSV path.

    Returns:
        DataFrame with columns compatible with the backtester.
    """
    if tickers is None:
        tickers = SP500_TICKERS
    if max_tickers:
        tickers = tickers[:max_tickers]

    print(f"Loading CIK mapping...")
    cik_map = _load_cik_map()

    print(f"Downloading fundamentals for {len(tickers)} tickers from SEC EDGAR...")
    all_rows = []
    errors = 0
    shares_data = []  # for market cap calc

    for i, tk in enumerate(tickers):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(tickers)}]")

        cik = cik_map.get(tk)
        if cik is None:
            # Try without dash (BRK-B -> BRKB)
            cik = cik_map.get(tk.replace("-", ""))
            if cik is None:
                errors += 1
                continue

        try:
            facts = _get_company_facts(cik)
            if facts is None:
                errors += 1
                continue

            annual = _build_annual_table(facts)
            if annual.empty:
                errors += 1
                continue

            funda = _compute_fundamentals(annual, tk)
            if not funda.empty:
                all_rows.append(funda)

                # Store shares data for market cap
                if "shares_outstanding" in annual.columns:
                    for _, ar in annual.iterrows():
                        if pd.notna(ar.get("shares_outstanding")):
                            shares_data.append({
                                "ticker": tk,
                                "date": ar["filed_date"],
                                "shares_outstanding": ar["shares_outstanding"],
                            })

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR {tk}: {e}")

        # SEC rate limit: 10 requests/sec
        time.sleep(0.15)

    if not all_rows:
        raise RuntimeError("No data downloaded from SEC EDGAR.")

    df = pd.concat(all_rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)

    print(f"\nRaw data: {len(df)} rows, {df['ticker'].nunique()} tickers")
    print(f"Errors: {errors}/{len(tickers)}")

    # Fill P/B and market cap from price data if available
    prices_path = DATA_DIR / "prices.csv"
    if prices_path.exists():
        print("Filling P/B and market cap from price data...")
        df_prices = pd.read_csv(prices_path, parse_dates=["date"])
        df_prices["date"] = pd.to_datetime(df_prices["date"], utc=True)

        # Fill P/B
        price_pivot = df_prices.pivot_table(index="date", columns="ticker", values="close")
        for i, row in df.iterrows():
            tk = row["ticker"]
            pub_date = row["date"]
            bvps = row.get("book_value_per_share")

            if tk not in price_pivot.columns or pd.isna(bvps) or bvps <= 0:
                continue

            prices = price_pivot[tk].dropna()
            before = prices[prices.index <= pub_date]
            if before.empty:
                continue
            px = before.iloc[-1]
            df.loc[i, "pb"] = round(px / bvps, 3)

        # Fill market cap
        df_shares = pd.DataFrame(shares_data)
        if not df_shares.empty:
            df_shares["date"] = pd.to_datetime(df_shares["date"], utc=True)
            for i, row in df.iterrows():
                tk = row["ticker"]
                pub_date = row["date"]
                if tk not in price_pivot.columns:
                    continue
                prices = price_pivot[tk].dropna()
                before = prices[prices.index <= pub_date]
                if before.empty:
                    continue
                px = before.iloc[-1]

                tk_shares = df_shares[
                    (df_shares["ticker"] == tk) & (df_shares["date"] <= pub_date)
                ]
                if not tk_shares.empty:
                    shares = tk_shares.iloc[-1]["shares_outstanding"]
                    if shares > 0:
                        df.loc[i, "market_cap"] = px * shares

    # Compute F-Score
    df = _compute_f_score(df)

    # Sort and save
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    if output_path is None:
        output_path = DATA_DIR / "fundamentals_sec.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # Stats
    print(f"\nSaved {len(df)} rows ({df['ticker'].nunique()} tickers) to {output_path}")
    print(f"Date range: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
    print(f"P/B coverage: {df['pb'].notna().sum()}/{len(df)}")
    print(f"F-Score coverage: {df['f_score'].notna().sum()}/{len(df)}")
    print(f"Market cap coverage: {df['market_cap'].notna().sum()}/{len(df)}")

    return df


def _compute_f_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Piotroski F-Score from components."""
    df = df.copy()

    components = []
    components.append((df["roa"] > 0).astype(float).where(df["roa"].notna()))
    components.append((df["cfo"] > 0).astype(float).where(df["cfo"].notna()))
    components.append(
        (df["roa"] > df["roa_prev"]).astype(float).where(
            df["roa"].notna() & df["roa_prev"].notna()
        )
    )
    components.append(
        (df["cfo"] > df["roa"]).astype(float).where(
            df["cfo"].notna() & df["roa"].notna()
        )
    )
    components.append(
        (df["delta_leverage"] < 0).astype(float).where(df["delta_leverage"].notna())
    )
    components.append(
        (df["delta_current_ratio"] > 0).astype(float).where(df["delta_current_ratio"].notna())
    )
    components.append((df["shares_issued"] == 0).astype(float))
    components.append(
        (df["delta_gross_margin"] > 0).astype(float).where(df["delta_gross_margin"].notna())
    )
    components.append(
        (df["delta_asset_turnover"] > 0).astype(float).where(df["delta_asset_turnover"].notna())
    )

    score_df = pd.concat(components, axis=1)
    n_valid = score_df.notna().sum(axis=1)
    raw_sum = score_df.sum(axis=1)
    df["f_score"] = np.where(n_valid >= 5, raw_sum, np.nan)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download fundamentals from SEC EDGAR")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers")
    parser.add_argument("--max-tickers", type=int, help="Limit number of tickers")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    download_sec_fundamentals(
        tickers=args.tickers,
        max_tickers=args.max_tickers,
        output_path=args.output,
    )
