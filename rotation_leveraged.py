"""Volatility-managed rotation between leveraged Nasdaq and leveraged Gold.

Based on:
- Moreira & Muir (2017) — Volatility-managed portfolios
- Gold as safe haven in high-vol regimes (Baur & Lucey, 2010)
- Leveraged ETF rotation (LRS community research)

Features:
- Variable drag (Fed Funds proxy + fixed margin)
- Leverage profiles (Aggro / Normal / Light)
- Rolling/expanding quantile for vol threshold
- TQQQ buy & hold comparison with delta metrics
- Lever-down rule when DD exceeds threshold
- Sensitivity analysis across vol thresholds
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from metrics import compute_performance_metrics, MetricsParams


# ══════════════════════════════════════════════════════════════
# LEVERAGE PROFILES
# ══════════════════════════════════════════════════════════════

LEVERAGE_PROFILES = {
    "Aggro": {
        "leverage_equity": 3.0,
        "leverage_gold": 3.0,
        "max_dd_target": -0.50,
        "label": "Experimental — MaxDD can exceed -50%",
    },
    "Normal": {
        "leverage_equity": 2.0,
        "leverage_gold": 1.5,
        "max_dd_target": -0.40,
        "label": "Standard — DD target >= -40%",
    },
    "Light": {
        "leverage_equity": 1.5,
        "leverage_gold": 1.0,
        "max_dd_target": -0.35,
        "label": "Conservative — DD target >= -35%",
    },
}


# ══════════════════════════════════════════════════════════════
# FED FUNDS PROXY (for variable drag)
# ══════════════════════════════════════════════════════════════

# Simplified historical Fed Funds effective rate brackets (annual %).
# Source: FRED FEDFUNDS, approximated to yearly averages.
_FED_FUNDS_APPROX = [
    ("1975-01-01", 0.055), ("1977-01-01", 0.065), ("1979-01-01", 0.11),
    ("1981-01-01", 0.16), ("1983-01-01", 0.09), ("1985-01-01", 0.08),
    ("1987-01-01", 0.067), ("1989-01-01", 0.09), ("1991-01-01", 0.056),
    ("1993-01-01", 0.03), ("1995-01-01", 0.055), ("1997-01-01", 0.055),
    ("1999-01-01", 0.05), ("2001-01-01", 0.038), ("2002-01-01", 0.017),
    ("2004-01-01", 0.013), ("2005-01-01", 0.032), ("2006-01-01", 0.05),
    ("2007-01-01", 0.05), ("2008-01-01", 0.02), ("2009-01-01", 0.002),
    ("2016-01-01", 0.004), ("2017-01-01", 0.01), ("2018-01-01", 0.018),
    ("2019-01-01", 0.024), ("2020-01-01", 0.005), ("2020-04-01", 0.001),
    ("2022-04-01", 0.005), ("2022-07-01", 0.025), ("2023-01-01", 0.045),
    ("2023-07-01", 0.053), ("2024-01-01", 0.053), ("2024-10-01", 0.048),
    ("2025-01-01", 0.044),
]


def get_fed_funds_series(dates: pd.DatetimeIndex) -> pd.Series:
    """Build a daily Fed Funds rate proxy from approximate historical brackets."""
    ff_dates = [pd.Timestamp(d, tz="UTC") if dates.tz else pd.Timestamp(d) for d, _ in _FED_FUNDS_APPROX]
    ff_rates = [r for _, r in _FED_FUNDS_APPROX]
    ff = pd.Series(ff_rates, index=ff_dates).reindex(dates, method="ffill").fillna(0.02)
    return ff


def compute_variable_drag(
    dates: pd.DatetimeIndex,
    leverage: float,
    fixed_margin: float = 0.02,
    freq: int = 252,
) -> pd.Series:
    """Variable daily drag = (Fed Funds + fixed margin) * (leverage - 1) / freq.

    Only the financed portion (leverage - 1) pays the borrowing cost.
    """
    ff = get_fed_funds_series(dates)
    annual_drag = (ff + fixed_margin) * max(leverage - 1, 0)
    return annual_drag / freq


# ══════════════════════════════════════════════════════════════
# PARAMS
# ══════════════════════════════════════════════════════════════

@dataclass
class RotationParams:
    # Leverage
    leverage_equity: float = 3.0
    leverage_gold: float = 2.0
    max_exposure: float = 10.0  # notional cap in x — 10.0 = no cap, 1.5 = max 1.5x

    # Drag
    use_variable_drag: bool = True
    fixed_margin: float = 0.02  # added to Fed Funds for borrowing cost
    drag_equity_annual: float = 0.01  # flat drag fallback
    drag_gold_annual: float = 0.01

    # Volatility signal
    vol_window: int = 21
    vol_threshold_quantile: float = 0.70
    vol_threshold_absolute: float | None = None
    vol_quantile_mode: str = "expanding"  # "expanding", "rolling", or "global"
    vol_quantile_lookback: int = 756  # ~3 years rolling window (if mode=rolling)

    # Backtest
    start_date: str = "2010-01-01"
    end_date: str = "2025-12-31"
    initial_capital: float = 100_000.0
    risk_free_rate: float = 0.02
    transaction_cost: float = 0.001

    # Crash regime
    use_crash_regime: bool = False
    crash_vol_quantile: float = 0.95

    # Floor: minimum equity weight even in gold regime (0.0 = pure rotation)
    floor_eq_weight: float = 0.0

    # Cooldown: minimum months between regime switches (1 = no cooldown)
    min_months_between_switch: int = 1

    # Target-vol scaling (Moreira & Muir 2017)
    use_target_vol: bool = False
    target_vol: float = 0.25  # target annualized vol (e.g. 25%)
    target_vol_lookback: int = 126  # ~6 months for realized vol estimation
    target_vol_max_scale: float = 2.0  # max leverage multiplier from vol scaling

    # Dual vol thresholds (graduated regime: moderate stress → partial gold)
    use_dual_threshold: bool = False
    vol_threshold_moderate: float = 0.18  # moderate stress: partial gold
    vol_threshold_high: float = 0.25  # high stress: heavy gold
    dual_eq_weight_moderate: float = 0.60  # equity weight in moderate stress
    dual_eq_weight_high: float = 0.30  # equity weight in high stress

    # Trend filter: only allow gold when equity is below MA
    use_trend_filter: bool = False
    trend_ma_window: int = 200  # MA period in trading days

    # Risk management: lever-down
    use_lever_down: bool = False
    lever_down_dd_threshold: float = -0.40  # when DD hits this, go 1x
    lever_down_recovery: float = -0.20  # when DD recovers above this, resume leverage


# ══════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════

def build_leveraged_returns(
    daily_returns: pd.Series,
    leverage: float,
    variable_drag: pd.Series | None = None,
    flat_drag_annual: float = 0.01,
    freq: int = 252,
) -> pd.Series:
    """Construct synthetic leveraged ETF daily returns.

    r_lev_t = min(max_exposure, L) * r_t - drag_t
    """
    if variable_drag is not None:
        drag = variable_drag.reindex(daily_returns.index, method="ffill").fillna(flat_drag_annual / freq)
    else:
        drag = flat_drag_annual / freq
    return leverage * daily_returns - drag


def realized_vol(
    daily_returns: pd.Series,
    window: int = 21,
    freq: int = 252,
) -> pd.Series:
    """Annualized rolling realized volatility."""
    return daily_returns.rolling(window, min_periods=max(10, window // 2)).std() * np.sqrt(freq)


def compute_vol_regime(
    daily_returns: pd.Series,
    window: int = 21,
    threshold_quantile: float = 0.70,
    threshold_absolute: float | None = None,
    quantile_mode: str = "expanding",
    quantile_lookback: int = 756,
    crash_regime: bool = False,
    crash_quantile: float = 0.95,
) -> tuple[pd.Series, pd.Series]:
    """Compute volatility regime signal.

    Args:
        quantile_mode: "global" (full history), "expanding" (growing window),
                       or "rolling" (fixed lookback window).

    Returns:
        (regime, vol_threshold_series)
        regime: 0=equity, 1=gold, 2=cash. Shifted by 1 day.
        vol_threshold_series: the threshold at each point in time.
    """
    vol = realized_vol(daily_returns, window=window)

    if threshold_absolute is not None:
        thresh_series = pd.Series(threshold_absolute, index=vol.index)
    elif quantile_mode == "global":
        thresh_val = vol.quantile(threshold_quantile)
        thresh_series = pd.Series(thresh_val, index=vol.index)
    elif quantile_mode == "expanding":
        thresh_series = vol.expanding(min_periods=63).quantile(threshold_quantile)
    else:  # rolling
        thresh_series = vol.rolling(quantile_lookback, min_periods=63).quantile(threshold_quantile)

    regime = pd.Series(0, index=vol.index, dtype=int)
    regime[vol > thresh_series] = 1

    if crash_regime:
        if quantile_mode == "global":
            crash_thresh = vol.quantile(crash_quantile)
        elif quantile_mode == "expanding":
            crash_thresh = vol.expanding(min_periods=63).quantile(crash_quantile)
        else:
            crash_thresh = vol.rolling(quantile_lookback, min_periods=63).quantile(crash_quantile)
        regime[vol > crash_thresh] = 2

    return regime.shift(1), thresh_series  # shift to avoid look-ahead


def apply_lever_down(
    strat_ret: pd.Series,
    eq_ret: pd.Series,
    gold_ret: pd.Series,
    daily_regime: pd.Series,
    leverage_equity: float,
    leverage_gold: float,
    dd_threshold: float = -0.40,
    recovery_threshold: float = -0.20,
) -> pd.Series:
    """When current drawdown exceeds threshold, reduce to 1x until recovery.

    DD is computed incrementally from the ADJUSTED returns to avoid look-ahead bias.
    """
    adjusted = strat_ret.values.copy()
    eq_vals = eq_ret.values
    gold_vals = gold_ret.values
    regime_vals = daily_regime.values
    n = len(adjusted)

    cum_value = 1.0
    peak = 1.0
    in_lever_down = False

    for i in range(n):
        # Check DD from the actual adjusted path so far
        dd = (cum_value - peak) / peak if peak > 0 else 0.0

        if dd < dd_threshold:
            in_lever_down = True
        elif in_lever_down and dd > recovery_threshold:
            in_lever_down = False

        if in_lever_down:
            if regime_vals[i] == 0:
                adjusted[i] = eq_vals[i]
            elif regime_vals[i] == 1:
                adjusted[i] = gold_vals[i]
            # else cash, stays 0

        # Update cumulative value with the (possibly adjusted) return
        cum_value *= (1 + adjusted[i])
        if cum_value > peak:
            peak = cum_value

    return pd.Series(adjusted, index=strat_ret.index)


# ══════════════════════════════════════════════════════════════
# MAIN BACKTEST
# ══════════════════════════════════════════════════════════════

def run_rotation_backtest(
    eq_daily_returns: pd.Series,
    gold_daily_returns: pd.Series,
    params: RotationParams | None = None,
) -> dict:
    """Run the leveraged rotation backtest.

    Returns dict with: equity_curve, regime_series, monthly_returns, metrics,
    vol_series, vol_threshold_series, components, lever_down_active.
    """
    if params is None:
        params = RotationParams()

    # Align dates
    tz = eq_daily_returns.index.tz
    start = pd.Timestamp(params.start_date, tz=tz) if tz else pd.Timestamp(params.start_date)
    end = pd.Timestamp(params.end_date, tz=tz) if tz else pd.Timestamp(params.end_date)

    common_idx = eq_daily_returns.index.intersection(gold_daily_returns.index)
    common_idx = common_idx[(common_idx >= start) & (common_idx <= end)]

    eq_ret = eq_daily_returns.loc[common_idx]
    gold_ret = gold_daily_returns.loc[common_idx]

    # Cap effective leverage at max_exposure (notional cap: 1.5 means max 1.5x)
    eff_lev_eq = min(params.leverage_equity, params.max_exposure)
    eff_lev_gold = min(params.leverage_gold, params.max_exposure)

    # Variable or flat drag
    if params.use_variable_drag:
        eq_drag = compute_variable_drag(eq_ret.index, eff_lev_eq, params.fixed_margin)
        gold_drag = compute_variable_drag(gold_ret.index, eff_lev_gold, params.fixed_margin)
        eq_lev = build_leveraged_returns(eq_ret, eff_lev_eq, variable_drag=eq_drag)
        gold_lev = build_leveraged_returns(gold_ret, eff_lev_gold, variable_drag=gold_drag)
    else:
        eq_lev = build_leveraged_returns(eq_ret, eff_lev_eq, flat_drag_annual=params.drag_equity_annual)
        gold_lev = build_leveraged_returns(gold_ret, eff_lev_gold, flat_drag_annual=params.drag_gold_annual)

    # Regime signal
    regime, vol_thresh_series = compute_vol_regime(
        eq_ret,
        window=params.vol_window,
        threshold_quantile=params.vol_threshold_quantile,
        threshold_absolute=params.vol_threshold_absolute,
        quantile_mode=params.vol_quantile_mode,
        quantile_lookback=params.vol_quantile_lookback,
        crash_regime=params.use_crash_regime,
        crash_quantile=params.crash_vol_quantile,
    )

    # Monthly rebalance
    regime_monthly = regime.resample("ME").last().dropna()

    # Apply cooldown: enforce min_months_between_switch
    if params.min_months_between_switch > 1:
        cooled = regime_monthly.copy()
        months_since_switch = params.min_months_between_switch  # allow first switch
        prev_regime = cooled.iloc[0] if len(cooled) > 0 else 0
        for i in range(1, len(cooled)):
            months_since_switch += 1
            if cooled.iloc[i] != prev_regime:
                if months_since_switch >= params.min_months_between_switch:
                    prev_regime = cooled.iloc[i]
                    months_since_switch = 0
                else:
                    cooled.iloc[i] = prev_regime  # suppress switch
        regime_monthly = cooled

    daily_regime = regime_monthly.reindex(eq_lev.index, method="ffill")
    # Fill initial NaN days (before first month-end) with first known regime
    daily_regime = daily_regime.bfill()

    # ── Trend filter: gate gold regime with MA ──
    if params.use_trend_filter:
        eq_price = (1 + eq_ret).cumprod()
        ma = eq_price.rolling(params.trend_ma_window, min_periods=50).mean()
        below_ma = (eq_price < ma).shift(1).fillna(False)  # shift to avoid look-ahead
        # Monthly: allow gold only if equity was below MA at month-end
        below_ma_monthly = below_ma.resample("ME").last().reindex(daily_regime.index, method="ffill").fillna(False)
        # Override: if regime says gold (1) but equity is above MA → force equity (0)
        daily_regime = daily_regime.copy()
        force_equity = (daily_regime == 1) & (~below_ma_monthly)
        daily_regime[force_equity] = 0

    # ── Dual vol thresholds: graduated allocation ──
    if params.use_dual_threshold:
        vol = realized_vol(eq_ret, window=params.vol_window)
        vol_shifted = vol.shift(1)  # avoid look-ahead
        vol_monthly = vol_shifted.resample("ME").last()
        vol_daily = vol_monthly.reindex(eq_lev.index, method="ffill")

        # Compute daily equity weight based on vol level
        eq_weight = pd.Series(1.0, index=eq_lev.index)  # default: full equity
        moderate = vol_daily > params.vol_threshold_moderate
        high = vol_daily > params.vol_threshold_high
        eq_weight[moderate & ~high] = params.dual_eq_weight_moderate
        eq_weight[high] = params.dual_eq_weight_high

        strat_ret = eq_weight * eq_lev + (1 - eq_weight) * gold_lev
        mask_eq = eq_weight >= 0.5  # for stats: "mostly equity"
        mask_gold = eq_weight < 0.5
    else:
        # Standard binary regime (with floor_eq_weight)
        strat_ret = pd.Series(0.0, index=eq_lev.index)
        mask_eq = daily_regime == 0
        mask_gold = daily_regime == 1
        floor = params.floor_eq_weight

        strat_ret[mask_eq] = eq_lev[mask_eq]
        if floor > 0:
            strat_ret[mask_gold] = floor * eq_lev[mask_gold] + (1 - floor) * gold_lev[mask_gold]
        else:
            strat_ret[mask_gold] = gold_lev[mask_gold]

    # ── Target-vol scaling (Moreira & Muir) ──
    if params.use_target_vol:
        # Monthly realized vol of the strategy itself
        strat_vol = strat_ret.rolling(params.target_vol_lookback, min_periods=21).std() * np.sqrt(252)
        strat_vol_monthly = strat_vol.resample("ME").last()
        scale_monthly = (params.target_vol / strat_vol_monthly).clip(upper=params.target_vol_max_scale)
        scale_daily = scale_monthly.reindex(strat_ret.index, method="ffill").shift(1).fillna(1.0)  # t-1
        strat_ret = strat_ret * scale_daily

    # Lever-down rule
    lever_down_active = pd.Series(False, index=strat_ret.index)
    if params.use_lever_down:
        strat_ret = apply_lever_down(
            strat_ret, eq_ret, gold_ret, daily_regime,
            eff_lev_eq, eff_lev_gold,
            dd_threshold=params.lever_down_dd_threshold,
            recovery_threshold=params.lever_down_recovery,
        )
        # Recompute to track where lever-down was active
        temp_curve = (1 + strat_ret).cumprod()
        temp_peak = temp_curve.cummax()
        temp_dd = (temp_curve - temp_peak) / temp_peak
        # Approximate: lever-down active where return differs from leveraged
        lever_down_active = (strat_ret != 0) & (
            ((mask_eq) & (np.abs(strat_ret - eq_lev) > 1e-10)) |
            ((mask_gold) & (np.abs(strat_ret - gold_lev) > 1e-10))
        )

    # Transaction costs on regime switches
    regime_changes = daily_regime.diff().fillna(0) != 0
    month_ends = daily_regime.index[daily_regime.index.is_month_end | (daily_regime.index == daily_regime.index[0])]
    for dt in month_ends:
        if dt in regime_changes.index and regime_changes.loc[dt]:
            strat_ret.loc[dt] -= params.transaction_cost

    # Equity curve
    equity_curve = params.initial_capital * (1 + strat_ret).cumprod()

    # Monthly returns
    monthly_eq = equity_curve.resample("ME").last()
    monthly_returns = monthly_eq.pct_change().dropna()

    # Component curves
    eq_lev_curve = params.initial_capital * (1 + eq_lev).cumprod()
    gold_lev_curve = params.initial_capital * (1 + gold_lev).cumprod()
    eq_unlev_curve = params.initial_capital * (1 + eq_ret).cumprod()
    gold_unlev_curve = params.initial_capital * (1 + gold_ret).cumprod()

    # Metrics
    metrics_params = MetricsParams(risk_free_rate=params.risk_free_rate, periods_per_year=252)
    metrics = compute_performance_metrics(equity_curve, params=metrics_params)

    vol_series = realized_vol(eq_ret, window=params.vol_window)
    mask_cash = daily_regime == 2

    n_switches = int(regime_changes.sum())
    metrics.update({
        "n_regime_switches": n_switches,
        "time_in_equity_pct": float(mask_eq.mean()),
        "time_in_gold_pct": float(mask_gold.mean()),
        "time_in_cash_pct": float(mask_cash.mean()),
        "vol_threshold_used": float(vol_thresh_series.dropna().iloc[-1]) if not vol_thresh_series.dropna().empty else 0.0,
        "lever_down_pct": float(lever_down_active.mean()) if params.use_lever_down else 0.0,
    })

    return {
        "equity_curve": equity_curve,
        "regime_series": daily_regime,
        "monthly_returns": monthly_returns,
        "metrics": metrics,
        "vol_series": vol_series,
        "vol_threshold_series": vol_thresh_series,
        "lever_down_active": lever_down_active,
        "components": {
            "equity_leveraged": eq_lev_curve,
            "gold_leveraged": gold_lev_curve,
            "equity_unleveraged": eq_unlev_curve,
            "gold_unleveraged": gold_unlev_curve,
        },
    }


# ══════════════════════════════════════════════════════════════
# TQQQ COMPARISON
# ══════════════════════════════════════════════════════════════

def compute_tqqq_comparison(
    rotation_equity: pd.Series,
    tqqq_equity: pd.Series,
    risk_free_rate: float = 0.02,
) -> dict:
    """Compute delta metrics: rotation vs TQQQ buy & hold.

    Returns dict with delta_cagr, delta_sharpe, delta_max_dd,
    rolling_3y_outperformance_pct, rotation_useful (bool).
    """
    mp = MetricsParams(risk_free_rate=risk_free_rate, periods_per_year=252)

    # Align
    common = rotation_equity.index.intersection(tqqq_equity.index)
    rot = rotation_equity.loc[common]
    tqqq = tqqq_equity.loc[common]

    rot_m = compute_performance_metrics(rot, params=mp)
    tqqq_m = compute_performance_metrics(tqqq, params=mp)

    # Rolling 3-year outperformance
    rot_monthly = rot.resample("ME").last()
    tqqq_monthly = tqqq.resample("ME").last()
    if len(rot_monthly) >= 36:
        rot_roll = rot_monthly.pct_change().rolling(36).apply(lambda x: (1 + x).prod() - 1, raw=True)
        tqqq_roll = tqqq_monthly.pct_change().rolling(36).apply(lambda x: (1 + x).prod() - 1, raw=True)
        valid = rot_roll.dropna().index.intersection(tqqq_roll.dropna().index)
        if len(valid) > 0:
            outperform = (rot_roll.loc[valid] > tqqq_roll.loc[valid]).mean()
        else:
            outperform = np.nan
    else:
        outperform = np.nan

    delta_cagr = rot_m["cagr"] - tqqq_m["cagr"]
    delta_sharpe = rot_m["sharpe"] - tqqq_m["sharpe"]
    delta_max_dd = rot_m["max_drawdown"] - tqqq_m["max_drawdown"]  # less negative = better

    # Rotation is "useful" if it improves Sharpe OR MaxDD
    rotation_useful = (delta_sharpe > 0) or (delta_max_dd > 0)

    return {
        "rotation_cagr": rot_m["cagr"],
        "tqqq_cagr": tqqq_m["cagr"],
        "delta_cagr": delta_cagr,
        "rotation_sharpe": rot_m["sharpe"],
        "tqqq_sharpe": tqqq_m["sharpe"],
        "delta_sharpe": delta_sharpe,
        "rotation_max_dd": rot_m["max_drawdown"],
        "tqqq_max_dd": tqqq_m["max_drawdown"],
        "delta_max_dd": delta_max_dd,
        "rolling_3y_outperformance_pct": outperform,
        "rotation_useful": rotation_useful,
    }


# ══════════════════════════════════════════════════════════════
# SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════

def run_sensitivity_analysis(
    eq_ret: pd.Series,
    gold_ret: pd.Series,
    base_params: RotationParams,
    thresholds: list[float] | None = None,
    mode: str = "quantile",
) -> pd.DataFrame:
    """Run backtest for multiple vol thresholds.

    Args:
        mode: "quantile" → test different quantile values,
              "absolute" → test different absolute vol levels.
        thresholds: list of values to test.

    Returns:
        DataFrame with one row per threshold and key metrics.
    """
    if thresholds is None:
        if mode == "quantile":
            thresholds = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        else:
            thresholds = [0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.30]

    rows = []
    for t in thresholds:
        p = RotationParams(
            leverage_equity=base_params.leverage_equity,
            leverage_gold=base_params.leverage_gold,
            max_exposure=base_params.max_exposure,
            use_variable_drag=base_params.use_variable_drag,
            fixed_margin=base_params.fixed_margin,
            drag_equity_annual=base_params.drag_equity_annual,
            drag_gold_annual=base_params.drag_gold_annual,
            vol_window=base_params.vol_window,
            vol_threshold_quantile=t if mode == "quantile" else base_params.vol_threshold_quantile,
            vol_threshold_absolute=t if mode == "absolute" else None,
            vol_quantile_mode=base_params.vol_quantile_mode,
            vol_quantile_lookback=base_params.vol_quantile_lookback,
            start_date=base_params.start_date,
            end_date=base_params.end_date,
            initial_capital=base_params.initial_capital,
            risk_free_rate=base_params.risk_free_rate,
            transaction_cost=base_params.transaction_cost,
            use_crash_regime=base_params.use_crash_regime,
            crash_vol_quantile=base_params.crash_vol_quantile,
            use_lever_down=base_params.use_lever_down,
            lever_down_dd_threshold=base_params.lever_down_dd_threshold,
            lever_down_recovery=base_params.lever_down_recovery,
        )
        result = run_rotation_backtest(eq_ret, gold_ret, p)
        m = result["metrics"]
        rows.append({
            "threshold": t,
            "cagr": m["cagr"],
            "sharpe": m["sharpe"],
            "sortino": m["sortino"],
            "max_drawdown": m["max_drawdown"],
            "calmar": m["calmar"],
            "volatility_ann": m["volatility_ann"],
            "n_switches": m["n_regime_switches"],
            "time_equity": m["time_in_equity_pct"],
            "time_gold": m["time_in_gold_pct"],
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════

def download_rotation_data(
    start: str = "2010-01-01",
    end: str = "2025-12-31",
    equity_ticker: str = "^NDX",
    gold_ticker: str = "GC=F",
) -> tuple[pd.Series, pd.Series]:
    """Download daily returns for equity index and gold from yfinance."""
    import yfinance as yf

    eq_data = yf.download(equity_ticker, start=start, end=end, progress=False)
    if eq_data.empty or len(eq_data) < 252:
        eq_data = yf.download("^GSPC", start=start, end=end, progress=False)

    gold_data = yf.download(gold_ticker, start=start, end=end, progress=False)
    if gold_data.empty or len(gold_data) < 252:
        gold_data = yf.download("GLD", start=start, end=end, progress=False)

    eq_close = eq_data["Close"].squeeze()
    gold_close = gold_data["Close"].squeeze()

    if eq_close.index.tz is None:
        eq_close.index = eq_close.index.tz_localize("UTC")
    if gold_close.index.tz is None:
        gold_close.index = gold_close.index.tz_localize("UTC")

    eq_ret = eq_close.pct_change().dropna()
    gold_ret = gold_close.pct_change().dropna()
    eq_ret.name = "equity"
    gold_ret.name = "gold"

    return eq_ret, gold_ret


def download_tqqq_data(start: str = "2010-02-11", end: str = "2025-12-31") -> pd.Series:
    """Download TQQQ daily close for buy & hold comparison."""
    import yfinance as yf

    data = yf.download("TQQQ", start=start, end=end, progress=False)
    if data.empty:
        return pd.Series(dtype=float)
    close = data["Close"].squeeze()
    if close.index.tz is None:
        close.index = close.index.tz_localize("UTC")
    return close


# ══════════════════════════════════════════════════════════════
# FORMATTING
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# GRID OPTIMIZATION
# ══════════════════════════════════════════════════════════════

def evaluate_rotation_grid(
    eq_ret: pd.Series,
    gold_ret: pd.Series,
    tqqq_curve: pd.Series | None = None,
    base_params: RotationParams | None = None,
    grid: dict | None = None,
    filter_vs_tqqq: bool = True,
    cagr_tolerance: float = 0.05,
) -> pd.DataFrame:
    """Grid search over rotation parameters, with TQQQ comparison.

    Args:
        eq_ret: Daily equity returns.
        gold_ret: Daily gold returns.
        tqqq_curve: TQQQ equity curve (price series scaled to initial_capital).
            If None, builds synthetic 3x equity as proxy.
        base_params: Base params (dates, drag, risk-free, etc.).
        grid: Dict of param_name -> list of values to test. Defaults to standard grid.
        filter_vs_tqqq: If True, only keep configs that pass TQQQ filter.
        cagr_tolerance: Max CAGR shortfall vs TQQQ (default 5pp).

    Returns:
        DataFrame sorted by Sharpe desc, MaxDD asc, with TQQQ deltas.
    """
    import itertools
    import time

    if base_params is None:
        base_params = RotationParams()

    if grid is None:
        grid = {
            "leverage_equity": [2.0, 2.5, 3.0],
            "leverage_gold": [1.0, 1.5, 2.0],
            "floor_eq_weight": [0.0, 0.2, 0.3],
            "vol_threshold_absolute": [0.18, 0.20, 0.22],
            "min_months_between_switch": [1, 2, 3],
        }

    # Build TQQQ benchmark metrics
    mp = MetricsParams(risk_free_rate=base_params.risk_free_rate, periods_per_year=252)

    if tqqq_curve is None:
        # Synthetic 3x equity as TQQQ proxy
        if base_params.use_variable_drag:
            tz = eq_ret.index.tz
            start = pd.Timestamp(base_params.start_date, tz=tz) if tz else pd.Timestamp(base_params.start_date)
            end = pd.Timestamp(base_params.end_date, tz=tz) if tz else pd.Timestamp(base_params.end_date)
            mask = (eq_ret.index >= start) & (eq_ret.index <= end)
            eq_sub = eq_ret[mask]
            drag = compute_variable_drag(eq_sub.index, 3.0, base_params.fixed_margin)
            tqqq_ret = build_leveraged_returns(eq_sub, 3.0, variable_drag=drag)
        else:
            tz = eq_ret.index.tz
            start = pd.Timestamp(base_params.start_date, tz=tz) if tz else pd.Timestamp(base_params.start_date)
            end = pd.Timestamp(base_params.end_date, tz=tz) if tz else pd.Timestamp(base_params.end_date)
            mask = (eq_ret.index >= start) & (eq_ret.index <= end)
            eq_sub = eq_ret[mask]
            tqqq_ret = build_leveraged_returns(eq_sub, 3.0, flat_drag_annual=0.01)
        tqqq_curve = base_params.initial_capital * (1 + tqqq_ret).cumprod()

    tqqq_metrics = compute_performance_metrics(tqqq_curve, params=mp)

    # Generate all combos
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = [dict(zip(keys, c)) for c in itertools.product(*values)]
    print(f"Rotation grid: {len(combos)} combinations")

    rows = []
    t0 = time.time()

    for i, combo in enumerate(combos):
        p = RotationParams(
            leverage_equity=combo.get("leverage_equity", base_params.leverage_equity),
            leverage_gold=combo.get("leverage_gold", base_params.leverage_gold),
            max_exposure=base_params.max_exposure,
            use_variable_drag=base_params.use_variable_drag,
            fixed_margin=base_params.fixed_margin,
            drag_equity_annual=base_params.drag_equity_annual,
            drag_gold_annual=base_params.drag_gold_annual,
            vol_window=combo.get("vol_window", base_params.vol_window),
            vol_threshold_quantile=combo.get("vol_threshold_quantile", base_params.vol_threshold_quantile),
            vol_threshold_absolute=combo.get("vol_threshold_absolute", base_params.vol_threshold_absolute),
            vol_quantile_mode=combo.get("vol_quantile_mode", base_params.vol_quantile_mode),
            vol_quantile_lookback=base_params.vol_quantile_lookback,
            start_date=base_params.start_date,
            end_date=base_params.end_date,
            initial_capital=base_params.initial_capital,
            risk_free_rate=base_params.risk_free_rate,
            transaction_cost=base_params.transaction_cost,
            use_crash_regime=base_params.use_crash_regime,
            crash_vol_quantile=base_params.crash_vol_quantile,
            floor_eq_weight=combo.get("floor_eq_weight", base_params.floor_eq_weight),
            min_months_between_switch=combo.get("min_months_between_switch", base_params.min_months_between_switch),
            use_lever_down=base_params.use_lever_down,
            lever_down_dd_threshold=base_params.lever_down_dd_threshold,
            lever_down_recovery=base_params.lever_down_recovery,
        )

        try:
            result = run_rotation_backtest(eq_ret, gold_ret, p)
            m = result["metrics"]

            row = {**combo}
            row["cagr"] = m["cagr"]
            row["sharpe"] = m["sharpe"]
            row["sortino"] = m["sortino"]
            row["max_drawdown"] = m["max_drawdown"]
            row["volatility"] = m["volatility_ann"]
            row["calmar"] = m["calmar"]
            row["n_switches"] = m["n_regime_switches"]
            row["time_gold_pct"] = m["time_in_gold_pct"]
            row["time_equity_pct"] = m["time_in_equity_pct"]

            # Deltas vs TQQQ
            row["delta_cagr_vs_tqqq"] = m["cagr"] - tqqq_metrics["cagr"]
            row["delta_sharpe_vs_tqqq"] = m["sharpe"] - tqqq_metrics["sharpe"]
            row["delta_maxdd_vs_tqqq"] = m["max_drawdown"] - tqqq_metrics["max_drawdown"]

            rows.append(row)
        except Exception:
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(combos) - i - 1)
            print(f"  [{i+1}/{len(combos)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    print(f"Grid complete: {len(df)} valid results in {time.time() - t0:.0f}s")

    # TQQQ reference
    print(f"\nTQQQ B&H reference:")
    print(f"  CAGR: {tqqq_metrics['cagr']:.2%}  Sharpe: {tqqq_metrics['sharpe']:.2f}  "
          f"MaxDD: {tqqq_metrics['max_drawdown']:.2%}")

    # Filter vs TQQQ
    if filter_vs_tqqq:
        before = len(df)
        df_filtered = df[
            (df["cagr"] >= tqqq_metrics["cagr"] - cagr_tolerance) &
            (df["max_drawdown"] >= tqqq_metrics["max_drawdown"]) &  # less negative = better
            (df["sharpe"] >= tqqq_metrics["sharpe"])
        ].copy()
        print(f"\nFilter vs TQQQ: {len(df_filtered)}/{before} configs pass "
              f"(CAGR >= {tqqq_metrics['cagr'] - cagr_tolerance:.2%}, "
              f"MaxDD >= {tqqq_metrics['max_drawdown']:.2%}, "
              f"Sharpe >= {tqqq_metrics['sharpe']:.2f})")

        if df_filtered.empty:
            print("WARNING: No config passes all filters. Returning unfiltered, sorted by Sharpe.")
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()

    # Sort: Sharpe desc, MaxDD asc (less negative first), delta_sharpe desc
    df_filtered = df_filtered.sort_values(
        ["sharpe", "max_drawdown", "delta_sharpe_vs_tqqq"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return df_filtered


def evaluate_rotation_robustness(
    eq_ret: pd.Series,
    gold_ret: pd.Series,
    tqqq_curve_full: pd.Series | None = None,
    tqqq_curve_oos: pd.Series | None = None,
    grid: dict | None = None,
    is_start: str = "2010-01-01",
    is_end: str = "2020-12-31",
    oos_start: str = "2021-01-01",
    oos_end: str = "2025-12-31",
    top_n: int = 10,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run grid on IS period, then re-test top N configs on OOS.

    Returns:
        (is_results, oos_results) — both DataFrames with metrics + TQQQ deltas.
    """
    # IS
    print(f"\n{'='*60}")
    print(f"IN-SAMPLE: {is_start} -> {is_end}")
    print(f"{'='*60}")
    base_is = RotationParams(start_date=is_start, end_date=is_end, **kwargs)
    is_df = evaluate_rotation_grid(
        eq_ret, gold_ret, tqqq_curve=tqqq_curve_full,
        base_params=base_is, grid=grid, filter_vs_tqqq=False,
    )

    if is_df.empty:
        return is_df, pd.DataFrame()

    top_configs = is_df.head(top_n)

    # OOS
    print(f"\n{'='*60}")
    print(f"OUT-OF-SAMPLE: {oos_start} -> {oos_end}")
    print(f"{'='*60}")

    param_cols = [c for c in top_configs.columns if c in (grid or {}).keys()]
    if not param_cols:
        param_cols = ["leverage_equity", "leverage_gold", "floor_eq_weight",
                      "vol_threshold_absolute", "min_months_between_switch"]
        param_cols = [c for c in param_cols if c in top_configs.columns]

    base_oos = RotationParams(start_date=oos_start, end_date=oos_end, **kwargs)
    mp = MetricsParams(risk_free_rate=base_oos.risk_free_rate, periods_per_year=252)

    # Build OOS TQQQ
    if tqqq_curve_oos is not None:
        tqqq_m_oos = compute_performance_metrics(tqqq_curve_oos, params=mp)
    else:
        tqqq_m_oos = {"cagr": 0, "sharpe": 0, "max_drawdown": 0}

    oos_rows = []
    for _, row in top_configs.iterrows():
        combo = {c: row[c] for c in param_cols}
        p = RotationParams(
            leverage_equity=combo.get("leverage_equity", base_oos.leverage_equity),
            leverage_gold=combo.get("leverage_gold", base_oos.leverage_gold),
            max_exposure=base_oos.max_exposure,
            use_variable_drag=base_oos.use_variable_drag,
            fixed_margin=base_oos.fixed_margin,
            vol_window=base_oos.vol_window,
            vol_threshold_absolute=combo.get("vol_threshold_absolute"),
            vol_quantile_mode=base_oos.vol_quantile_mode,
            start_date=oos_start,
            end_date=oos_end,
            initial_capital=base_oos.initial_capital,
            risk_free_rate=base_oos.risk_free_rate,
            transaction_cost=base_oos.transaction_cost,
            floor_eq_weight=combo.get("floor_eq_weight", 0.0),
            min_months_between_switch=int(combo.get("min_months_between_switch", 1)),
        )
        try:
            result = run_rotation_backtest(eq_ret, gold_ret, p)
            m = result["metrics"]
            oos_row = {**combo}
            oos_row["cagr_oos"] = m["cagr"]
            oos_row["sharpe_oos"] = m["sharpe"]
            oos_row["max_drawdown_oos"] = m["max_drawdown"]
            oos_row["volatility_oos"] = m["volatility_ann"]
            oos_row["n_switches_oos"] = m["n_regime_switches"]
            oos_row["time_gold_pct_oos"] = m["time_in_gold_pct"]
            oos_row["delta_cagr_vs_tqqq_oos"] = m["cagr"] - tqqq_m_oos.get("cagr", 0)
            oos_row["delta_sharpe_vs_tqqq_oos"] = m["sharpe"] - tqqq_m_oos.get("sharpe", 0)

            # IS metrics for side-by-side
            oos_row["cagr_is"] = row["cagr"]
            oos_row["sharpe_is"] = row["sharpe"]
            oos_row["max_drawdown_is"] = row["max_drawdown"]
            oos_rows.append(oos_row)
        except Exception:
            continue

    oos_df = pd.DataFrame(oos_rows)
    if not oos_df.empty:
        oos_df = oos_df.sort_values("sharpe_oos", ascending=False).reset_index(drop=True)
        # Flag robust configs: Sharpe doesn't degrade more than 50%
        if "sharpe_is" in oos_df.columns:
            oos_df["sharpe_stability"] = np.where(
                oos_df["sharpe_is"] > 0,
                oos_df["sharpe_oos"] / oos_df["sharpe_is"],
                np.nan,
            )

    return is_df, oos_df


def format_rotation_metrics(metrics: dict) -> pd.DataFrame:
    """Format rotation-specific metrics for display."""
    fmt = {
        "total_return": ("{:.2%}", "Total Return"),
        "cagr": ("{:.2%}", "CAGR"),
        "volatility_ann": ("{:.2%}", "Volatility (ann.)"),
        "sharpe": ("{:.2f}", "Sharpe Ratio"),
        "sortino": ("{:.2f}", "Sortino Ratio"),
        "max_drawdown": ("{:.2%}", "Max Drawdown"),
        "max_dd_duration_periods": ("{:.0f}", "Max DD Duration (days)"),
        "calmar": ("{:.2f}", "Calmar Ratio"),
        "n_regime_switches": ("{:.0f}", "Regime Switches"),
        "time_in_equity_pct": ("{:.1%}", "Time in Equity"),
        "time_in_gold_pct": ("{:.1%}", "Time in Gold"),
        "time_in_cash_pct": ("{:.1%}", "Time in Cash"),
        "vol_threshold_used": ("{:.2%}", "Vol Threshold (ann.)"),
        "lever_down_pct": ("{:.1%}", "Time in Lever-Down"),
    }
    rows = []
    for k, v in metrics.items():
        if k in fmt:
            pattern, label = fmt[k]
            try:
                formatted = pattern.format(v) if not (isinstance(v, float) and np.isnan(v)) else "N/A"
            except (ValueError, TypeError):
                formatted = str(v)
            rows.append({"metric": label, "value": formatted})
    return pd.DataFrame(rows)
