"""Core backtest engine for the Quality/Value strategy."""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from selection import SelectionParams, select_quality_value_universe
from fundamentals import compute_pb, compute_f_score, compute_q_score
from metrics import compute_performance_metrics, MetricsParams


@dataclass
class BacktestParams:
    start_date: str = "2014-01-01"
    end_date: str = "2023-12-31"
    rebalance_freq_months: int = 6
    transaction_cost: float = 0.001  # 0.1% per trade
    pub_lag_months: int = 3
    risk_free_rate: float = 0.02
    initial_capital: float = 100_000.0
    selection: SelectionParams = None

    def __post_init__(self):
        if self.selection is None:
            self.selection = SelectionParams()


def _generate_rebalance_dates(
    start: pd.Timestamp, end: pd.Timestamp, freq_months: int,
    available_dates: list | None = None,
) -> list[pd.Timestamp]:
    """Generate rebalancing dates aligned with available price dates.
    If available_dates is provided, snaps to the closest available date.
    """
    # Generate candidate dates at month-end
    candidates = pd.date_range(start, end, freq="ME", tz="UTC")[::freq_months]

    if available_dates is None or len(available_dates) == 0:
        return list(candidates)

    available = sorted(available_dates)
    snapped = []
    for c in candidates:
        # Find closest available date (prefer <= candidate, else next one)
        before = [d for d in available if d <= c]
        after = [d for d in available if d > c]
        if before:
            snapped.append(before[-1])
        elif after:
            snapped.append(after[0])
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for d in snapped:
        if d not in seen:
            seen.add(d)
            result.append(d)
    return result


def run_backtest(
    df_prices: pd.DataFrame,
    df_funda: pd.DataFrame,
    params: BacktestParams | None = None,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, dict]:
    """Run the Quality/Value backtest.

    Returns:
        equity_curve: Series of portfolio values indexed by date.
        positions_log: DataFrame logging holdings at each rebalance.
        trades_log: DataFrame of all trades with PnL.
        metrics: Dict of performance metrics.
    """
    if params is None:
        params = BacktestParams()

    # Prepare fundamentals
    df_funda = compute_pb(df_funda)
    df_funda = compute_f_score(df_funda, min_components=3, scale_partial=True)
    df_funda = compute_q_score(df_funda)

    start = pd.Timestamp(params.start_date, tz="UTC")
    end = pd.Timestamp(params.end_date, tz="UTC")

    # Price pivot: date x ticker
    price_dates = df_prices[
        (df_prices["date"] >= start) & (df_prices["date"] <= end)
    ]
    all_dates = sorted(price_dates["date"].unique())

    rebalance_dates = _generate_rebalance_dates(start, end, params.rebalance_freq_months, all_dates)

    # State
    capital = params.initial_capital
    holdings: dict[str, float] = {}  # ticker -> number of shares (fractional ok)
    entry_prices: dict[str, float] = {}  # ticker -> buy price (for PnL calc)
    equity_history = []
    positions_log = []
    trades_log = []

    # Price lookup: for each date, get dict ticker->close
    price_by_date = {}
    for dt, group in price_dates.groupby("date"):
        price_by_date[dt] = dict(zip(group["ticker"], group["close"]))

    rebalance_set = set(rebalance_dates)

    for dt in all_dates:
        prices_today = price_by_date.get(dt, {})

        # Portfolio value
        portfolio_value = capital
        for tk, shares in holdings.items():
            px = prices_today.get(tk)
            if px is not None:
                portfolio_value += shares * px

        # Rebalance?
        if dt in rebalance_set:
            selected = select_quality_value_universe(
                dt, df_prices, df_funda, params.selection, params.pub_lag_months
            )

            if selected:
                # Sell everything: convert to cash
                for tk, shares in holdings.items():
                    px = prices_today.get(tk)
                    if px is not None and shares > 0:
                        sell_value = shares * px
                        cost = sell_value * params.transaction_cost
                        capital += sell_value - cost

                        # PnL: sell price vs entry price
                        entry_px = entry_prices.get(tk, px)
                        pnl = shares * (px - entry_px) - cost

                        trades_log.append({
                            "date": dt,
                            "ticker": tk,
                            "action": "SELL",
                            "shares": shares,
                            "price": px,
                            "value": sell_value,
                            "cost": cost,
                            "pnl": pnl,
                            "value_traded": sell_value,
                        })

                holdings = {}

                # Buy new portfolio: equal weight
                total_value = capital
                weight = 1.0 / len(selected)
                target_value = total_value * weight

                for tk in selected:
                    px = prices_today.get(tk)
                    if px is not None and px > 0:
                        alloc = target_value
                        cost = alloc * params.transaction_cost
                        shares = (alloc - cost) / px
                        holdings[tk] = shares
                        entry_prices[tk] = px
                        capital -= alloc

                        trades_log.append({
                            "date": dt,
                            "ticker": tk,
                            "action": "BUY",
                            "shares": shares,
                            "price": px,
                            "value": alloc,
                            "cost": cost,
                            "pnl": 0.0,
                            "value_traded": alloc,
                        })

                # Log positions
                positions_log.append({
                    "date": dt,
                    "tickers": selected,
                    "n_holdings": len(selected),
                    "portfolio_value": total_value,
                })

        # Record equity
        # Recompute after possible rebalance
        eq_val = capital
        for tk, shares in holdings.items():
            px = prices_today.get(tk)
            if px is not None:
                eq_val += shares * px
        equity_history.append({"date": dt, "value": eq_val})

    # Build results
    equity_curve = pd.DataFrame(equity_history).set_index("date")["value"]
    df_trades = pd.DataFrame(trades_log) if trades_log else pd.DataFrame(
        columns=["date", "ticker", "action", "shares", "price", "value", "cost"]
    )
    df_positions = pd.DataFrame(positions_log) if positions_log else pd.DataFrame()

    # Filter only SELL trades for metrics (BUY trades have pnl=0)
    if not df_trades.empty:
        sell_trades = df_trades[df_trades["action"] == "SELL"].copy()
    else:
        sell_trades = df_trades

    # Metrics
    metrics_params = MetricsParams(
        risk_free_rate=params.risk_free_rate,
        periods_per_year=12,  # monthly data by default
    )
    metrics = compute_performance_metrics(
        equity_curve,
        sell_trades if not sell_trades.empty else None,
        metrics_params,
    )

    return equity_curve, df_positions, df_trades, metrics
