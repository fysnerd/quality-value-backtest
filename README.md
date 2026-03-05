# Quality/Value Backtester

Backtesting framework for a **Low P/B + High Piotroski F-Score** equity strategy with parameter optimization and robustness validation.

## Features

- **Modular backtest engine** — data loading, fundamental scoring, universe selection, execution, metrics
- **Piotroski F-Score** (0-9) computed from financial statements
- **Parameter optimization** — grid search or random search with in-sample/out-of-sample validation
- **Robustness scoring** — Sharpe degradation ratio to detect overfitting
- **Interactive Streamlit dashboard** — single backtest + optimization tabs
- **Real data pipeline** — S&P 500 prices and fundamentals via yfinance

## Quick Start

```bash
pip install -r requirements.txt

# Run with synthetic data
python example.py

# Download real S&P 500 data
python data_download.py

# Run parameter optimization
python optimize.py --synthetic --search grid
python optimize.py --search random --n 200

# Launch interactive dashboard
streamlit run streamlit_app.py
```

## Project Structure

```
├── data_loader.py      # CSV loading + synthetic data generation
├── fundamentals.py     # P/B and Piotroski F-Score calculation
├── selection.py        # Quality/Value universe selection
├── backtest.py         # Core backtest engine
├── metrics.py          # CAGR, Sharpe, Sortino, MaxDD, Calmar, etc.
├── benchmarks.py       # Multi-benchmark comparison
├── data_download.py    # S&P 500 data download via yfinance
├── optimization.py     # Grid/random search with IS/OOS validation
├── optimize.py         # CLI optimization runner
├── streamlit_app.py    # Interactive dashboard
└── example.py          # Usage example with synthetic data
```

## Strategy Logic

1. Filter universe: US equities with market cap >= $1B
2. **Value filter**: keep bottom X% by Price-to-Book ratio
3. **Quality filter**: keep stocks with Piotroski F-Score >= Y
4. Rank by F-Score (desc) then P/B (asc), select top N
5. Equal-weight portfolio, rebalanced every R months
6. Publication lag enforced to prevent look-ahead bias

## Optimization

The optimizer tests parameter combinations and validates on out-of-sample data:

- **In-sample**: fit strategy parameters
- **Out-of-sample**: validate on unseen period
- **Robustness score**: `min(Sharpe_IS, Sharpe_OOS)` — penalizes large IS/OOS gaps
- **Sharpe degradation**: `Sharpe_OOS / Sharpe_IS` — values > 0.5 indicate robust configs
