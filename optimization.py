"""Parameter optimization for the Quality/Value strategy.

Supports grid search and random search with in-sample/out-of-sample validation
and robustness scoring to guard against overfitting.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import itertools
import time
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from backtest import run_backtest, BacktestParams
from selection import SelectionParams
from metrics import MetricsParams


@dataclass
class ParamSpace:
    """Parameter search space definition."""
    pb_percentile_cut: list[float] = field(default_factory=lambda: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    min_f_score: list[int] = field(default_factory=lambda: [5, 6, 7, 8])
    n_stocks: list[int] = field(default_factory=lambda: [20, 30, 40, 60])
    rebalance_months: list[int] = field(default_factory=lambda: [3, 6, 12])
    transaction_cost: list[float] = field(default_factory=lambda: [0.0005, 0.001, 0.002])

    @property
    def total_combinations(self) -> int:
        return (
            len(self.pb_percentile_cut)
            * len(self.min_f_score)
            * len(self.n_stocks)
            * len(self.rebalance_months)
            * len(self.transaction_cost)
        )

    def grid(self) -> list[dict]:
        """All combinations (full grid)."""
        keys = ["pb_percentile_cut", "min_f_score", "n_stocks", "rebalance_months", "transaction_cost"]
        values = [
            self.pb_percentile_cut,
            self.min_f_score,
            self.n_stocks,
            self.rebalance_months,
            self.transaction_cost,
        ]
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def random_sample(self, n: int, seed: int = 42) -> list[dict]:
        """Random sample from parameter space."""
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(n):
            samples.append({
                "pb_percentile_cut": rng.choice(self.pb_percentile_cut),
                "min_f_score": int(rng.choice(self.min_f_score)),
                "n_stocks": int(rng.choice(self.n_stocks)),
                "rebalance_months": int(rng.choice(self.rebalance_months)),
                "transaction_cost": rng.choice(self.transaction_cost),
            })
        return samples


@dataclass
class OptimizationConfig:
    """Configuration for the optimization run."""
    # In-sample period
    is_start: str = "2010-01-01"
    is_end: str = "2018-12-31"
    # Out-of-sample period
    oos_start: str = "2019-01-01"
    oos_end: str = "2024-12-31"
    # Search
    search_type: str = "grid"  # "grid" or "random"
    n_random_samples: int = 300
    random_seed: int = 42
    # Fixed params
    min_market_cap: float = 1e9
    pub_lag_months: int = 3
    risk_free_rate: float = 0.02
    initial_capital: float = 100_000.0
    # OOS evaluation
    top_n_for_oos: int = 20
    rank_metric: str = "sharpe"  # metric to rank IS results


def _build_backtest_params(combo: dict, config: OptimizationConfig, period: str = "is") -> BacktestParams:
    """Convert a parameter combo + config into BacktestParams."""
    start = config.is_start if period == "is" else config.oos_start
    end = config.is_end if period == "is" else config.oos_end

    return BacktestParams(
        start_date=start,
        end_date=end,
        rebalance_freq_months=int(combo["rebalance_months"]),
        transaction_cost=float(combo["transaction_cost"]),
        pub_lag_months=config.pub_lag_months,
        risk_free_rate=config.risk_free_rate,
        initial_capital=config.initial_capital,
        selection=SelectionParams(
            min_market_cap=config.min_market_cap,
            pb_percentile_cut=float(combo["pb_percentile_cut"]),
            min_f_score=int(combo["min_f_score"]),
            n_stocks=int(combo["n_stocks"]),
        ),
    )


def _run_single_backtest(df_prices, df_funda, combo, config, period="is") -> dict:
    """Run a single backtest and return metrics + params as a flat dict."""
    params = _build_backtest_params(combo, config, period)

    try:
        equity_curve, _, trades, metrics = run_backtest(df_prices, df_funda, params)

        if len(equity_curve) < 3:
            return None

        result = {**combo}
        suffix = "" if period == "is" else "_oos"
        for k, v in metrics.items():
            result[f"{k}{suffix}"] = v

        return result

    except Exception:
        return None


def run_param_search(
    df_prices: pd.DataFrame,
    df_funda: pd.DataFrame,
    param_space: ParamSpace | None = None,
    config: OptimizationConfig | None = None,
    progress_callback=None,
) -> pd.DataFrame:
    """Run parameter search (grid or random) with IS/OOS validation.

    Args:
        df_prices: Price data (long format).
        df_funda: Fundamental data.
        param_space: Search space. Defaults to standard grid.
        config: Optimization config. Defaults to standard.
        progress_callback: Optional callable(current, total, combo_dict) for progress updates.

    Returns:
        DataFrame with one row per parameter combination, columns = params + IS metrics + OOS metrics.
    """
    if param_space is None:
        param_space = ParamSpace()
    if config is None:
        config = OptimizationConfig()

    # Generate parameter combinations
    if config.search_type == "grid":
        combos = param_space.grid()
        print(f"Grid search: {len(combos)} combinations")
    else:
        combos = param_space.random_sample(config.n_random_samples, config.random_seed)
        print(f"Random search: {len(combos)} samples")

    # ── Phase 1: In-sample backtests ──
    print(f"\n{'='*60}")
    print(f"PHASE 1: In-Sample ({config.is_start} to {config.is_end})")
    print(f"{'='*60}")

    is_results = []
    t0 = time.time()

    for i, combo in enumerate(combos):
        result = _run_single_backtest(df_prices, df_funda, combo, config, "is")
        if result is not None:
            is_results.append(result)

        if progress_callback:
            progress_callback(i + 1, len(combos), combo)
        elif (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(combos) - i - 1)
            print(f"  [{i+1}/{len(combos)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    if not is_results:
        print("WARNING: No valid IS results. Check data coverage.")
        return pd.DataFrame()

    df_is = pd.DataFrame(is_results)
    print(f"\nIS complete: {len(df_is)} valid results out of {len(combos)} combos")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── Phase 2: OOS validation on top performers ──
    print(f"\n{'='*60}")
    print(f"PHASE 2: Out-of-Sample ({config.oos_start} to {config.oos_end})")
    print(f"{'='*60}")

    # Rank by IS metric, take top N
    rank_col = config.rank_metric
    if rank_col not in df_is.columns:
        rank_col = "sharpe"

    df_is_sorted = df_is.sort_values(rank_col, ascending=False)
    top_combos = df_is_sorted.head(config.top_n_for_oos)

    print(f"Evaluating top {len(top_combos)} IS performers on OOS...")

    param_cols = ["pb_percentile_cut", "min_f_score", "n_stocks", "rebalance_months", "transaction_cost"]
    df_final = top_combos.copy().reset_index(drop=True)

    for i, (idx, row) in enumerate(df_final.iterrows()):
        combo = {k: row[k] for k in param_cols}
        oos_result = _run_single_backtest(df_prices, df_funda, combo, config, "oos")

        if oos_result is not None:
            for k, v in oos_result.items():
                if k.endswith("_oos"):
                    df_final.loc[idx, k] = v

        if progress_callback:
            progress_callback(i + 1, len(df_final), combo)

    # ── Robustness scores ──
    df_final = _compute_robustness_scores(df_final)

    # Sort by robust_score
    df_final = df_final.sort_values("robust_score", ascending=False).reset_index(drop=True)

    print(f"\nOOS complete. Final results: {len(df_final)} parameter sets evaluated.")

    return df_final


def _compute_robustness_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute robustness metrics comparing IS vs OOS performance."""
    df = df.copy()

    # Robustness score 1: min(Sharpe_IS, Sharpe_OOS) — penalizes big IS/OOS gap
    sharpe_is = df.get("sharpe", pd.Series(dtype=float))
    sharpe_oos = df.get("sharpe_oos", pd.Series(dtype=float))

    if "sharpe_oos" in df.columns:
        # Score = min(IS, OOS) — rewards consistent performance
        df["robust_score"] = np.minimum(
            sharpe_is.fillna(0),
            sharpe_oos.fillna(0),
        )

        # Degradation ratio: how much does OOS degrade vs IS
        df["sharpe_degradation"] = np.where(
            sharpe_is > 0,
            sharpe_oos / sharpe_is,
            np.nan,
        )

        # CAGR stability
        if "cagr" in df.columns and "cagr_oos" in df.columns:
            df["cagr_stability"] = np.where(
                df["cagr"] > 0,
                df["cagr_oos"] / df["cagr"],
                np.nan,
            )
    else:
        df["robust_score"] = sharpe_is.fillna(0)
        df["sharpe_degradation"] = np.nan
        df["cagr_stability"] = np.nan

    return df


def print_top_results(df: pd.DataFrame, n: int = 10) -> None:
    """Print a formatted summary of top parameter sets."""
    print(f"\n{'='*80}")
    print(f"TOP {min(n, len(df))} PARAMETER SETS (ranked by robust_score)")
    print(f"{'='*80}")

    param_cols = ["pb_percentile_cut", "min_f_score", "n_stocks", "rebalance_months", "transaction_cost"]
    is_cols = ["cagr", "sharpe", "sortino", "max_drawdown", "calmar", "win_rate"]
    oos_cols = [f"{c}_oos" for c in is_cols if f"{c}_oos" in df.columns]
    meta_cols = ["robust_score", "sharpe_degradation"]

    for i, (_, row) in enumerate(df.head(n).iterrows()):
        print(f"\n--- #{i+1} (robust_score = {row.get('robust_score', 0):.3f}) ---")

        # Params
        print("  Params:", end="")
        for p in param_cols:
            val = row[p]
            if p == "pb_percentile_cut":
                print(f"  P/B<{val:.0%}", end="")
            elif p == "min_f_score":
                print(f"  F>={int(val)}", end="")
            elif p == "n_stocks":
                print(f"  N={int(val)}", end="")
            elif p == "rebalance_months":
                print(f"  Reb={int(val)}mo", end="")
            elif p == "transaction_cost":
                print(f"  TC={val:.2%}", end="")
        print()

        # IS metrics
        print("  IS  :", end="")
        for c in is_cols:
            if c in row and pd.notna(row[c]):
                if c in ["cagr", "max_drawdown", "win_rate"]:
                    print(f"  {c}={row[c]:.2%}", end="")
                else:
                    print(f"  {c}={row[c]:.2f}", end="")
        print()

        # OOS metrics
        if oos_cols:
            print("  OOS :", end="")
            for c in oos_cols:
                if c in row and pd.notna(row[c]):
                    name = c.replace("_oos", "")
                    if name in ["cagr", "max_drawdown", "win_rate"]:
                        print(f"  {name}={row[c]:.2%}", end="")
                    else:
                        print(f"  {name}={row[c]:.2f}", end="")
            print()

        # Degradation
        deg = row.get("sharpe_degradation")
        if pd.notna(deg):
            print(f"  Sharpe degradation: {deg:.2f}x (1.0 = perfect, <0.5 = red flag)")

    print(f"\n{'='*80}")


def save_results(df: pd.DataFrame, path: str | None = None) -> str:
    """Save optimization results to CSV."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "data", "results_param_search.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")
    return path
