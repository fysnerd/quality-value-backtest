"""Example usage of the Quality/Value backtest framework with synthetic data."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import generate_synthetic_data
from backtest import run_backtest, BacktestParams
from selection import SelectionParams
from metrics import format_metrics
from benchmarks import compare_strategies, normalize_series
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():
    # 1. Generate synthetic data
    print("Generating synthetic data...")
    df_prices, df_funda = generate_synthetic_data(n_tickers=200, n_years=10, seed=42)

    print(f"  Prices: {len(df_prices)} rows, {df_prices['ticker'].nunique()} tickers")
    print(f"  Fundamentals: {len(df_funda)} rows")

    # 2. Run backtest with default parameters
    params = BacktestParams(
        start_date="2015-01-01",
        end_date="2022-12-31",
        rebalance_freq_months=6,
        transaction_cost=0.001,
        pub_lag_months=3,
        risk_free_rate=0.02,
        initial_capital=100_000,
        selection=SelectionParams(
            min_market_cap=1e9,
            pb_percentile_cut=0.20,
            min_f_score=7,
            n_stocks=40,
        ),
    )

    print("\nRunning backtest...")
    equity_curve, positions, trades, metrics = run_backtest(df_prices, df_funda, params)

    # 3. Display metrics
    print("\n=== Performance Metrics ===")
    df_metrics = format_metrics(metrics)
    print(df_metrics.to_string(index=False))

    # 4. Create a naive benchmark (equal-weight buy & hold of all stocks)
    print("\nBuilding equal-weight benchmark...")
    pivot = df_prices.pivot_table(index="date", columns="ticker", values="close")
    pivot = pivot[(pivot.index >= pd.Timestamp("2015-01-01", tz="UTC")) &
                  (pivot.index <= pd.Timestamp("2022-12-31", tz="UTC"))]
    benchmark = pivot.mean(axis=1)  # equal-weight all stocks
    benchmark = benchmark.dropna()

    # 5. Compare
    comparison = compare_strategies(
        {"Quality/Value": equity_curve, "EW Benchmark": benchmark},
    )
    print("\n=== Strategy Comparison ===")
    print(comparison.to_string())

    # 6. Plot
    normalized = normalize_series({"Quality/Value": equity_curve, "EW Benchmark": benchmark})
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

    axes[0].plot(normalized.index, normalized["Quality/Value"], label="Quality/Value", linewidth=2)
    axes[0].plot(normalized.index, normalized["EW Benchmark"], label="EW Benchmark", linewidth=1.5, alpha=0.7)
    axes[0].set_title("Quality/Value Strategy vs Benchmark")
    axes[0].set_ylabel("Normalized Value (base 100)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Drawdown
    peak = normalized["Quality/Value"].cummax()
    dd = (normalized["Quality/Value"] - peak) / peak
    axes[1].fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_title("Strategy Drawdown")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "backtest_result.png"), dpi=150)
    print("\nPlot saved to backtest_result.png")
    # plt.show()  # uncomment for interactive display

    # 7. Last holdings
    if len(positions) > 0:
        last = positions.iloc[-1]
        print(f"\nLast rebalance: {last['date']}")
        print(f"Holdings ({last['n_holdings']} stocks): {last['tickers'][:10]}...")


if __name__ == "__main__":
    main()
