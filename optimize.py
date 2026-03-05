"""CLI runner for Quality/Value parameter optimization.

Usage:
    python optimize.py                          # grid search on real data
    python optimize.py --search random --n 200  # random search, 200 samples
    python optimize.py --synthetic              # use synthetic data (for testing)
    python optimize.py --is-start 2012-01-01 --is-end 2019-12-31  # custom periods

Results are saved to data/results_param_search.csv.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import time
from pathlib import Path

from data_loader import load_price_data, load_fundamental_data, generate_synthetic_data
from optimization import (
    ParamSpace,
    OptimizationConfig,
    run_param_search,
    print_top_results,
    save_results,
)

DATA_DIR = Path(__file__).parent / "data"


def main():
    parser = argparse.ArgumentParser(description="Quality/Value parameter optimization")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (for testing)")
    parser.add_argument("--search", default="grid", choices=["grid", "random"], help="Search type")
    parser.add_argument("--n", type=int, default=300, help="Number of random samples")
    parser.add_argument("--is-start", default="2010-01-01", help="In-sample start")
    parser.add_argument("--is-end", default="2018-12-31", help="In-sample end")
    parser.add_argument("--oos-start", default="2019-01-01", help="Out-of-sample start")
    parser.add_argument("--oos-end", default="2024-12-31", help="Out-of-sample end")
    parser.add_argument("--top", type=int, default=20, help="Top N IS results to evaluate OOS")
    parser.add_argument("--rank-by", default="sharpe", help="IS metric to rank by (sharpe, calmar, cagr, sortino)")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    # ── Load data ──
    if args.synthetic:
        print("Using synthetic data (200 tickers, 15 years)...")
        df_prices, df_funda = generate_synthetic_data(n_tickers=200, n_years=15, seed=42)
    else:
        prices_path = DATA_DIR / "prices.csv"
        funda_path = DATA_DIR / "fundamentals.csv"

        if not prices_path.exists() or not funda_path.exists():
            print("ERROR: Data files not found. Run data_download.py first:")
            print(f"  python data_download.py")
            print(f"  Expected: {prices_path}")
            print(f"           {funda_path}")
            sys.exit(1)

        print(f"Loading prices from {prices_path}...")
        df_prices = load_price_data(str(prices_path))
        print(f"  {len(df_prices)} rows, {df_prices['ticker'].nunique()} tickers")

        print(f"Loading fundamentals from {funda_path}...")
        df_funda = load_fundamental_data(str(funda_path))
        print(f"  {len(df_funda)} rows, {df_funda['ticker'].nunique()} tickers")

    # ── Parameter space ──
    param_space = ParamSpace()
    print(f"\nParameter space: {param_space.total_combinations} total grid combinations")

    # ── Config ──
    config = OptimizationConfig(
        is_start=args.is_start,
        is_end=args.is_end,
        oos_start=args.oos_start,
        oos_end=args.oos_end,
        search_type=args.search,
        n_random_samples=args.n,
        top_n_for_oos=args.top,
        rank_metric=args.rank_by,
    )

    # ── Run optimization ──
    t0 = time.time()
    results = run_param_search(df_prices, df_funda, param_space, config)
    elapsed = time.time() - t0

    if results.empty:
        print("\nNo results. Check that data covers the specified periods.")
        sys.exit(1)

    # ── Print summary ──
    print_top_results(results, n=10)
    print(f"\nTotal time: {elapsed:.1f}s")

    # ── Save ──
    output_path = args.output or str(DATA_DIR / "results_param_search.csv")
    save_results(results, output_path)

    # ── Quick stats ──
    print(f"\n--- Summary ---")
    print(f"  Search: {args.search} ({len(results)} combos evaluated IS, top {args.top} OOS)")
    print(f"  IS period: {args.is_start} -> {args.is_end}")
    print(f"  OOS period: {args.oos_start} -> {args.oos_end}")
    if "sharpe_degradation" in results.columns:
        deg = results["sharpe_degradation"].dropna()
        if len(deg) > 0:
            print(f"  Avg Sharpe degradation: {deg.mean():.2f}x (median {deg.median():.2f}x)")
            robust_count = (deg >= 0.5).sum()
            print(f"  Robust configs (degradation >= 0.5x): {robust_count}/{len(deg)}")


if __name__ == "__main__":
    main()
