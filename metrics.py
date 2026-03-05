"""Performance metrics for backtesting results."""

import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class MetricsParams:
    risk_free_rate: float = 0.02
    periods_per_year: int = 12  # 12 for monthly, 252 for daily


def compute_performance_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame | None = None,
    params: MetricsParams | None = None,
) -> dict:
    """Compute full set of performance metrics from an equity curve.

    Args:
        equity_curve: Series indexed by date with portfolio values.
        trades: Optional DataFrame with columns [date, ticker, pnl, value_traded].
        params: Risk-free rate and frequency.

    Returns:
        Dict of metrics.
    """
    if params is None:
        params = MetricsParams()

    N = params.periods_per_year
    rf = params.risk_free_rate
    V = equity_curve.values
    returns = equity_curve.pct_change().dropna()
    r = returns.values

    # CAGR — use actual calendar dates for accuracy
    if hasattr(equity_curve.index, 'dtype') and len(equity_curve) >= 2:
        try:
            n_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
        except (TypeError, AttributeError):
            n_years = len(equity_curve) / N
    else:
        n_years = len(equity_curve) / N
    cagr = (V[-1] / V[0]) ** (1 / n_years) - 1 if n_years > 0 and V[0] > 0 else 0.0

    # Annualized volatility
    vol_ann = np.std(r, ddof=1) * np.sqrt(N)

    # Sharpe
    excess = r - rf / N
    sharpe = (np.mean(excess) * N) / (np.std(r, ddof=1) * np.sqrt(N)) if np.std(r) > 0 else 0.0

    # Sortino
    downside = r[r < 0]
    downside_std = np.std(downside, ddof=1) * np.sqrt(N) if len(downside) > 1 else np.nan
    sortino = (np.mean(excess) * N) / downside_std if downside_std and downside_std > 0 else 0.0

    # Max Drawdown
    peak = np.maximum.accumulate(V)
    drawdown = (V - peak) / peak
    max_dd = np.min(drawdown)

    # Drawdown duration (in periods)
    in_dd = drawdown < 0
    if in_dd.any():
        # Count consecutive periods in drawdown
        changes = np.diff(in_dd.astype(int), prepend=0)
        dd_starts = np.where(changes == 1)[0]
        dd_ends = np.where(changes == -1)[0]
        if len(dd_ends) < len(dd_starts):
            dd_ends = np.append(dd_ends, len(in_dd))
        dd_lengths = dd_ends - dd_starts
        max_dd_duration = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0
    else:
        max_dd_duration = 0

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.inf

    # Trade-based metrics
    win_rate = np.nan
    profit_factor = np.nan
    n_trades = 0
    turnover = np.nan

    if trades is not None and len(trades) > 0:
        pnls = trades["pnl"].values
        n_trades = len(pnls)
        winners = pnls[pnls > 0]
        losers = pnls[pnls < 0]
        win_rate = len(winners) / n_trades if n_trades > 0 else 0.0
        profit_factor = (
            np.sum(winners) / abs(np.sum(losers))
            if len(losers) > 0 and np.sum(losers) != 0
            else np.inf
        )
        if "value_traded" in trades.columns:
            avg_portfolio = np.mean(V)
            turnover = trades["value_traded"].sum() / avg_portfolio if avg_portfolio > 0 else 0.0

    # Total return
    total_return = V[-1] / V[0] - 1 if V[0] > 0 else 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility_ann": vol_ann,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "max_dd_duration_periods": max_dd_duration,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": n_trades,
        "turnover": turnover,
    }


def format_metrics(metrics: dict) -> pd.DataFrame:
    """Format metrics dict into a readable DataFrame."""
    fmt = {
        "total_return": "{:.2%}",
        "cagr": "{:.2%}",
        "volatility_ann": "{:.2%}",
        "sharpe": "{:.2f}",
        "sortino": "{:.2f}",
        "max_drawdown": "{:.2%}",
        "max_dd_duration_periods": "{:.0f}",
        "calmar": "{:.2f}",
        "win_rate": "{:.2%}",
        "profit_factor": "{:.2f}",
        "n_trades": "{:.0f}",
        "turnover": "{:.2f}",
    }
    rows = []
    for k, v in metrics.items():
        pattern = fmt.get(k, "{:.4f}")
        try:
            formatted = pattern.format(v) if not (isinstance(v, float) and np.isnan(v)) else "N/A"
        except (ValueError, TypeError):
            formatted = str(v)
        rows.append({"metric": k, "value": formatted})
    return pd.DataFrame(rows)
