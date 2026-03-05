"""Download and prepare real S&P 500 price + fundamental data for the backtester.

Usage:
    python data_download.py              # download everything
    python data_download.py --prices     # prices only
    python data_download.py --funda      # fundamentals only

Data is saved to data/prices.csv and data/fundamentals.csv.
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# ── S&P 500 tickers (hardcoded subset — reliable, no Wikipedia scraping needed) ──
# Top ~150 liquid large-caps. Extend as needed.
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

# Benchmark tickers
BENCHMARKS = ["SPY", "QQQ", "IWM"]


def download_prices(
    tickers: list[str] | None = None,
    start: str = "2010-01-01",
    end: str = "2025-12-31",
    interval: str = "1mo",
    output_path: str | None = None,
) -> pd.DataFrame:
    """Download monthly close prices from Yahoo Finance.

    Args:
        tickers: List of tickers. Defaults to SP500_TICKERS + BENCHMARKS.
        start/end: Date range.
        interval: '1d' for daily, '1mo' for monthly.
        output_path: Where to save CSV. Defaults to data/prices.csv.
    """
    if tickers is None:
        tickers = SP500_TICKERS + BENCHMARKS

    print(f"Downloading prices for {len(tickers)} tickers ({interval})...")

    # Download in batches to avoid rate limits
    batch_size = 50
    all_dfs = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_str = " ".join(batch)
        print(f"  Batch {i // batch_size + 1}/{(len(tickers) - 1) // batch_size + 1}: {len(batch)} tickers...")

        try:
            data = yf.download(
                batch_str,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )

            if data.empty:
                print(f"  WARNING: No data returned for batch")
                continue

            # Extract Close prices
            if isinstance(data.columns, pd.MultiIndex):
                closes = data["Close"]
            else:
                # Single ticker
                closes = data[["Close"]].rename(columns={"Close": batch[0]})

            # Melt to long format: date, ticker, close
            closes = closes.reset_index()
            melted = closes.melt(id_vars=["Date"], var_name="ticker", value_name="close")
            melted = melted.rename(columns={"Date": "date"})
            melted = melted.dropna(subset=["close"])
            all_dfs.append(melted)

        except Exception as e:
            print(f"  ERROR on batch: {e}")

        if i + batch_size < len(tickers):
            time.sleep(1)  # rate limit courtesy

    if not all_dfs:
        raise RuntimeError("No price data downloaded.")

    df = pd.concat(all_dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Save
    if output_path is None:
        output_path = DATA_DIR / "prices.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} price rows ({df['ticker'].nunique()} tickers) to {output_path}")

    return df


def download_fundamentals(
    tickers: list[str] | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Download fundamental data from Yahoo Finance for each ticker.

    Extracts: market_cap, pb (price-to-book), and builds a simplified F-Score
    from available financial statements.

    NOTE: yfinance provides limited fundamental history. For each ticker we get:
    - Annual balance sheet / income / cash flow (last 4 years typically)
    - We compute a simplified Piotroski F-Score from what's available.
    - For tickers where data is missing, we skip them.

    For production use, consider: SimFin, Sharadar/Quandl, or SEC EDGAR.
    """
    if tickers is None:
        tickers = SP500_TICKERS

    print(f"Downloading fundamentals for {len(tickers)} tickers...")
    rows = []
    errors = 0

    for i, tk in enumerate(tickers):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{len(tickers)}")

        try:
            stock = yf.Ticker(tk)

            # Balance sheet (annual)
            bs = stock.balance_sheet
            inc = stock.income_stmt
            cf = stock.cashflow

            if bs is None or bs.empty:
                continue

            # yfinance returns columns as dates, rows as items
            for col_date in bs.columns:
                pub_date = pd.Timestamp(col_date, tz="UTC") + pd.DateOffset(months=3)  # ~3mo pub lag

                row = {"date": pub_date, "ticker": tk}

                # Market cap (current only from yfinance — use as approximation)
                info = stock.info
                row["market_cap"] = info.get("marketCap", np.nan)

                # Price to Book
                book_value = _safe_get(bs, col_date, [
                    "Stockholders Equity", "Total Stockholder Equity",
                    "Stockholders' Equity", "Common Stock Equity",
                ])
                shares = _safe_get(bs, col_date, [
                    "Share Issued", "Shares Issued", "Ordinary Shares Number",
                    "Common Stock Shares Outstanding",
                ])

                if book_value and shares and shares > 0:
                    bvps = book_value / shares
                    # Get price near that date
                    price_at_date = _get_price_near_date(tk, col_date)
                    if price_at_date and bvps > 0:
                        row["pb"] = round(price_at_date / bvps, 3)
                    else:
                        row["pb"] = np.nan
                else:
                    row["pb"] = info.get("priceToBook", np.nan)

                # ── Simplified F-Score components ──
                # 1. ROA
                net_income = _safe_get(inc, col_date, ["Net Income", "Net Income Common Stockholders"])
                total_assets = _safe_get(bs, col_date, ["Total Assets"])
                roa = net_income / total_assets if net_income and total_assets and total_assets > 0 else None
                row["roa"] = round(roa, 4) if roa is not None else np.nan

                # 2. CFO / Total Assets
                cfo_val = _safe_get(cf, col_date, [
                    "Operating Cash Flow", "Total Cash From Operating Activities",
                    "Cash Flow From Continuing Operating Activities",
                ])
                cfo_ratio = cfo_val / total_assets if cfo_val and total_assets and total_assets > 0 else None
                row["cfo"] = round(cfo_ratio, 4) if cfo_ratio is not None else np.nan

                # Previous year values (for delta calculations)
                prev_dates = [d for d in bs.columns if d < col_date]
                if prev_dates:
                    prev_date = max(prev_dates)

                    prev_ni = _safe_get(inc, prev_date, ["Net Income", "Net Income Common Stockholders"])
                    prev_ta = _safe_get(bs, prev_date, ["Total Assets"])
                    roa_prev = prev_ni / prev_ta if prev_ni and prev_ta and prev_ta > 0 else None
                    row["roa_prev"] = round(roa_prev, 4) if roa_prev is not None else np.nan

                    # Delta leverage (long-term debt / total assets)
                    ltd = _safe_get(bs, col_date, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
                    ltd_prev = _safe_get(bs, prev_date, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
                    if ltd is not None and ltd_prev is not None and total_assets and prev_ta:
                        row["delta_leverage"] = round(ltd / total_assets - ltd_prev / prev_ta, 4)
                    else:
                        row["delta_leverage"] = np.nan

                    # Delta current ratio
                    ca = _safe_get(bs, col_date, ["Current Assets"])
                    cl = _safe_get(bs, col_date, ["Current Liabilities"])
                    ca_prev = _safe_get(bs, prev_date, ["Current Assets"])
                    cl_prev = _safe_get(bs, prev_date, ["Current Liabilities"])
                    if ca and cl and cl > 0 and ca_prev and cl_prev and cl_prev > 0:
                        row["delta_current_ratio"] = round(ca / cl - ca_prev / cl_prev, 4)
                    else:
                        row["delta_current_ratio"] = np.nan

                    # Delta gross margin
                    rev = _safe_get(inc, col_date, ["Total Revenue", "Revenue"])
                    cogs = _safe_get(inc, col_date, ["Cost Of Revenue", "Cost Of Goods Sold"])
                    rev_prev = _safe_get(inc, prev_date, ["Total Revenue", "Revenue"])
                    cogs_prev = _safe_get(inc, prev_date, ["Cost Of Revenue", "Cost Of Goods Sold"])
                    if rev and cogs and rev > 0 and rev_prev and cogs_prev and rev_prev > 0:
                        gm = (rev - cogs) / rev
                        gm_prev = (rev_prev - cogs_prev) / rev_prev
                        row["delta_gross_margin"] = round(gm - gm_prev, 4)
                    else:
                        row["delta_gross_margin"] = np.nan

                    # Delta asset turnover
                    if rev and total_assets and total_assets > 0 and rev_prev and prev_ta and prev_ta > 0:
                        row["delta_asset_turnover"] = round(rev / total_assets - rev_prev / prev_ta, 4)
                    else:
                        row["delta_asset_turnover"] = np.nan
                else:
                    row["roa_prev"] = np.nan
                    row["delta_leverage"] = np.nan
                    row["delta_current_ratio"] = np.nan
                    row["delta_gross_margin"] = np.nan
                    row["delta_asset_turnover"] = np.nan

                # Shares dilution (simplified: 0 = no new shares, 1 = diluted)
                if prev_dates:
                    prev_shares = _safe_get(bs, max(prev_dates), [
                        "Share Issued", "Shares Issued", "Ordinary Shares Number",
                    ])
                    if shares and prev_shares:
                        row["shares_issued"] = 1 if shares > prev_shares * 1.01 else 0
                    else:
                        row["shares_issued"] = 0
                else:
                    row["shares_issued"] = 0

                rows.append(row)

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR {tk}: {e}")

        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    if not rows:
        raise RuntimeError("No fundamental data downloaded.")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # Compute F-Score where we have all components
    df = _compute_f_score_from_components(df)

    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Save
    if output_path is None:
        output_path = DATA_DIR / "fundamentals.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} fundamental rows ({df['ticker'].nunique()} tickers) to {output_path}")
    print(f"  Errors: {errors}/{len(tickers)}")
    print(f"  F-Score coverage: {df['f_score'].notna().sum()}/{len(df)} rows")

    return df


def _safe_get(df: pd.DataFrame, col, keys: list) -> float | None:
    """Safely extract a value from a financial statement DataFrame."""
    if df is None or df.empty or col not in df.columns:
        return None
    for key in keys:
        if key in df.index:
            val = df.loc[key, col]
            if pd.notna(val):
                return float(val)
    return None


def _get_price_near_date(ticker: str, date, window_days: int = 30) -> float | None:
    """Get closing price near a given date (for P/B calculation)."""
    try:
        start = (pd.Timestamp(date) - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
        end = (pd.Timestamp(date) + pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
        hist = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if hist is not None and not hist.empty:
            # Flatten MultiIndex columns if present
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return None


def _compute_f_score_from_components(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Piotroski F-Score from individual components where available.
    Falls back to NaN where data is insufficient.
    """
    df = df.copy()

    components = []

    # F1: ROA > 0
    components.append((df["roa"] > 0).astype(float).where(df["roa"].notna()))
    # F2: CFO > 0
    components.append((df["cfo"] > 0).astype(float).where(df["cfo"].notna()))
    # F3: Delta ROA > 0
    has_both = df["roa"].notna() & df["roa_prev"].notna()
    components.append((df["roa"] > df["roa_prev"]).astype(float).where(has_both))
    # F4: CFO > ROA (accrual quality)
    has_both = df["cfo"].notna() & df["roa"].notna()
    components.append((df["cfo"] > df["roa"]).astype(float).where(has_both))
    # F5: Leverage decreased
    components.append((df["delta_leverage"] < 0).astype(float).where(df["delta_leverage"].notna()))
    # F6: Liquidity improved
    components.append((df["delta_current_ratio"] > 0).astype(float).where(df["delta_current_ratio"].notna()))
    # F7: No dilution
    components.append((df["shares_issued"] == 0).astype(float))
    # F8: Gross margin improved
    components.append((df["delta_gross_margin"] > 0).astype(float).where(df["delta_gross_margin"].notna()))
    # F9: Asset turnover improved
    components.append((df["delta_asset_turnover"] > 0).astype(float).where(df["delta_asset_turnover"].notna()))

    score_df = pd.concat(components, axis=1)
    # Count non-NaN components
    n_valid = score_df.notna().sum(axis=1)
    raw_sum = score_df.sum(axis=1)

    # Only assign F-Score if we have at least 6 out of 9 components
    df["f_score"] = np.where(n_valid >= 6, raw_sum.round(0).astype(int), np.nan)

    return df


def download_benchmark_prices(
    tickers: list[str] | None = None,
    start: str = "2010-01-01",
    end: str = "2025-12-31",
) -> None:
    """Download benchmark price series (SPY, QQQ, etc.) as separate CSVs."""
    if tickers is None:
        tickers = BENCHMARKS

    for tk in tickers:
        print(f"Downloading benchmark {tk}...")
        try:
            data = yf.download(tk, start=start, end=end, interval="1mo", auto_adjust=True, progress=False)
            if data.empty:
                print(f"  No data for {tk}")
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            df = data[["Close"]].reset_index()
            df.columns = ["date", "close"]
            df["date"] = pd.to_datetime(df["date"], utc=True)

            path = DATA_DIR / f"benchmark_{tk.lower()}.csv"
            df.to_csv(path, index=False)
            print(f"  Saved to {path}")
        except Exception as e:
            print(f"  ERROR: {e}")


# ── CLI ──
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download S&P 500 data for Quality/Value backtest")
    parser.add_argument("--prices", action="store_true", help="Download prices only")
    parser.add_argument("--funda", action="store_true", help="Download fundamentals only")
    parser.add_argument("--benchmarks", action="store_true", help="Download benchmark series")
    parser.add_argument("--start", default="2010-01-01", help="Start date")
    parser.add_argument("--end", default="2025-12-31", help="End date")
    parser.add_argument("--interval", default="1mo", choices=["1d", "1wk", "1mo"], help="Price interval")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    do_all = not (args.prices or args.funda or args.benchmarks)

    if args.prices or do_all:
        download_prices(start=args.start, end=args.end, interval=args.interval)

    if args.funda or do_all:
        download_fundamentals()

    if args.benchmarks or do_all:
        download_benchmark_prices(start=args.start, end=args.end)

    print("\nDone! Files saved in", DATA_DIR)
