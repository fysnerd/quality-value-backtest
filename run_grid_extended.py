"""Extended grid search for rotation strategy optimization.

Tests: vol_window, cooldown, floor, vol_thresh, target-vol, dual thresholds, trend filter.
Filters: Sharpe >= 1.0, MaxDD >= -40%, CAGR >= 25%.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import time
import itertools
from rotation_leveraged import (
    download_rotation_data, download_tqqq_data,
    run_rotation_backtest, RotationParams,
)
from metrics import compute_performance_metrics, MetricsParams

# ── Download ──
print("Downloading NDX, Gold, TQQQ...")
eq_ret, gold_ret = download_rotation_data(start="2010-01-01", end="2025-12-31")
tqqq_close = download_tqqq_data(start="2010-02-11", end="2025-12-31")
capital = 100_000

print(f"Data: eq {len(eq_ret)} days, gold {len(gold_ret)} days")

start_date = "2010-03-01"
end_date = "2025-12-31"
mp = MetricsParams(risk_free_rate=0.02, periods_per_year=252)

# TQQQ reference
tz = eq_ret.index.tz
tqqq_start = pd.Timestamp(start_date, tz=tz)
tqqq_end = pd.Timestamp(end_date, tz=tz)
tqqq_sub = tqqq_close[(tqqq_close.index >= tqqq_start) & (tqqq_close.index <= tqqq_end)]
tqqq_curve = tqqq_sub / tqqq_sub.iloc[0] * capital
tqqq_m = compute_performance_metrics(tqqq_curve, params=mp)
print(f"TQQQ ref: CAGR={tqqq_m['cagr']:.2%} Sharpe={tqqq_m['sharpe']:.2f} MaxDD={tqqq_m['max_drawdown']:.2%}")

rows = []

# ══════════════════════════════════════════════════════════════
# PASS 1: Base grid (binary regime + floor)
# ══════════════════════════════════════════════════════════════
grid = {
    "leverage_equity": [2.0, 2.5],
    "leverage_gold": [1.5, 2.0],
    "vol_window": [21, 42, 63],
    "cooldown": [2, 3, 4],
    "floor_eq": [0.2, 0.3, 0.4],
    "vol_thresh": [0.18, 0.20, 0.22],
}

keys = list(grid.keys())
values = [grid[k] for k in keys]
combos = [dict(zip(keys, c)) for c in itertools.product(*values)]
print(f"\n{'='*60}")
print(f"PASS 1: Base grid — {len(combos)} combos")
print(f"{'='*60}")

t0 = time.time()
for i, c in enumerate(combos):
    p = RotationParams(
        leverage_equity=c["leverage_equity"],
        leverage_gold=c["leverage_gold"],
        vol_window=c["vol_window"],
        vol_threshold_absolute=c["vol_thresh"],
        floor_eq_weight=c["floor_eq"],
        min_months_between_switch=c["cooldown"],
        use_variable_drag=True, fixed_margin=0.02,
        start_date=start_date, end_date=end_date,
        initial_capital=capital,
    )
    try:
        res = run_rotation_backtest(eq_ret, gold_ret, p)
        m = res["metrics"]
        rows.append({
            **c, "mode": "base",
            "cagr": m["cagr"], "sharpe": m["sharpe"], "sortino": m["sortino"],
            "max_drawdown": m["max_drawdown"], "volatility": m["volatility_ann"],
            "calmar": m["calmar"], "n_switches": m["n_regime_switches"],
            "time_gold": m["time_in_gold_pct"],
        })
    except Exception:
        pass
    if (i + 1) % 100 == 0:
        el = time.time() - t0
        print(f"  [{i+1}/{len(combos)}] {el:.0f}s")

n_base = len(rows)
print(f"Pass 1: {n_base} results in {time.time()-t0:.0f}s")

# ══════════════════════════════════════════════════════════════
# PASS 2: Target-vol on top 20
# ══════════════════════════════════════════════════════════════
df1 = pd.DataFrame(rows)
top20 = df1.nlargest(20, "sharpe")

print(f"\n{'='*60}")
print("PASS 2: Target-vol scaling on top 20")
print(f"{'='*60}")

for target_v in [0.20, 0.25, 0.30]:
    for _, row in top20.iterrows():
        p = RotationParams(
            leverage_equity=row["leverage_equity"],
            leverage_gold=row["leverage_gold"],
            vol_window=int(row["vol_window"]),
            vol_threshold_absolute=row["vol_thresh"],
            floor_eq_weight=row["floor_eq"],
            min_months_between_switch=int(row["cooldown"]),
            use_variable_drag=True, fixed_margin=0.02,
            start_date=start_date, end_date=end_date,
            initial_capital=capital,
            use_target_vol=True, target_vol=target_v,
        )
        try:
            res = run_rotation_backtest(eq_ret, gold_ret, p)
            m = res["metrics"]
            rows.append({
                "leverage_equity": row["leverage_equity"],
                "leverage_gold": row["leverage_gold"],
                "vol_window": int(row["vol_window"]),
                "cooldown": int(row["cooldown"]),
                "floor_eq": row["floor_eq"],
                "vol_thresh": row["vol_thresh"],
                "mode": f"tvol={target_v:.0%}",
                "cagr": m["cagr"], "sharpe": m["sharpe"], "sortino": m["sortino"],
                "max_drawdown": m["max_drawdown"], "volatility": m["volatility_ann"],
                "calmar": m["calmar"], "n_switches": m["n_regime_switches"],
                "time_gold": m["time_in_gold_pct"],
            })
        except Exception:
            pass

print(f"  +{len(rows) - n_base} target-vol configs")

# ══════════════════════════════════════════════════════════════
# PASS 3: Dual thresholds on top 20
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PASS 3: Dual thresholds on top 20")
print(f"{'='*60}")

n_before = len(rows)
for mod_w, high_w in [(0.60, 0.30), (0.50, 0.20), (0.70, 0.40)]:
    for _, row in top20.iterrows():
        p = RotationParams(
            leverage_equity=row["leverage_equity"],
            leverage_gold=row["leverage_gold"],
            vol_window=int(row["vol_window"]),
            floor_eq_weight=0.0,
            min_months_between_switch=int(row["cooldown"]),
            use_variable_drag=True, fixed_margin=0.02,
            start_date=start_date, end_date=end_date,
            initial_capital=capital,
            use_dual_threshold=True,
            vol_threshold_moderate=row["vol_thresh"],
            vol_threshold_high=row["vol_thresh"] + 0.07,
            dual_eq_weight_moderate=mod_w,
            dual_eq_weight_high=high_w,
        )
        try:
            res = run_rotation_backtest(eq_ret, gold_ret, p)
            m = res["metrics"]
            rows.append({
                "leverage_equity": row["leverage_equity"],
                "leverage_gold": row["leverage_gold"],
                "vol_window": int(row["vol_window"]),
                "cooldown": int(row["cooldown"]),
                "floor_eq": 0.0,
                "vol_thresh": row["vol_thresh"],
                "mode": f"dual({mod_w:.0%}/{high_w:.0%})",
                "cagr": m["cagr"], "sharpe": m["sharpe"], "sortino": m["sortino"],
                "max_drawdown": m["max_drawdown"], "volatility": m["volatility_ann"],
                "calmar": m["calmar"], "n_switches": m["n_regime_switches"],
                "time_gold": m["time_in_gold_pct"],
            })
        except Exception:
            pass

print(f"  +{len(rows) - n_before} dual-threshold configs")

# ══════════════════════════════════════════════════════════════
# PASS 4: Trend filter (MA200) on top 20
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PASS 4: Trend filter MA200 on top 20")
print(f"{'='*60}")

n_before = len(rows)
for _, row in top20.iterrows():
    p = RotationParams(
        leverage_equity=row["leverage_equity"],
        leverage_gold=row["leverage_gold"],
        vol_window=int(row["vol_window"]),
        vol_threshold_absolute=row["vol_thresh"],
        floor_eq_weight=row["floor_eq"],
        min_months_between_switch=int(row["cooldown"]),
        use_variable_drag=True, fixed_margin=0.02,
        start_date=start_date, end_date=end_date,
        initial_capital=capital,
        use_trend_filter=True, trend_ma_window=200,
    )
    try:
        res = run_rotation_backtest(eq_ret, gold_ret, p)
        m = res["metrics"]
        rows.append({
            "leverage_equity": row["leverage_equity"],
            "leverage_gold": row["leverage_gold"],
            "vol_window": int(row["vol_window"]),
            "cooldown": int(row["cooldown"]),
            "floor_eq": row["floor_eq"],
            "vol_thresh": row["vol_thresh"],
            "mode": "trend_MA200",
            "cagr": m["cagr"], "sharpe": m["sharpe"], "sortino": m["sortino"],
            "max_drawdown": m["max_drawdown"], "volatility": m["volatility_ann"],
            "calmar": m["calmar"], "n_switches": m["n_regime_switches"],
            "time_gold": m["time_in_gold_pct"],
        })
    except Exception:
        pass

print(f"  +{len(rows) - n_before} trend-filter configs")

# ══════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════
df = pd.DataFrame(rows)
print(f"\nTotal: {len(df)} configs tested in {time.time()-t0:.0f}s")

cols = ["mode", "leverage_equity", "leverage_gold", "vol_window", "cooldown",
        "floor_eq", "vol_thresh", "cagr", "sharpe", "sortino", "max_drawdown",
        "volatility", "calmar", "n_switches", "time_gold"]

def fmt_df(d):
    d = d[cols].copy()
    for c in ["cagr", "max_drawdown", "volatility", "time_gold"]:
        d[c] = d[c].apply(lambda x: f"{x:.1%}")
    for c in ["floor_eq", "vol_thresh"]:
        d[c] = d[c].apply(lambda x: f"{x:.0%}")
    for c in ["sharpe", "sortino", "calmar"]:
        d[c] = d[c].apply(lambda x: f"{x:.2f}")
    d["n_switches"] = d["n_switches"].astype(int)
    return d

# Filter 1: Sharpe >= 1.0, MaxDD >= -40%, CAGR >= 25%
f1 = df[(df["sharpe"] >= 1.0) & (df["max_drawdown"] >= -0.40) & (df["cagr"] >= 0.25)]
f1 = f1.sort_values("sharpe", ascending=False)
print(f"\nFiltered (Sharpe>=1.0, MaxDD>=-40%, CAGR>=25%): {len(f1)} configs")

print(f"\n{'='*120}")
print("TOP 15 CONFIGS")
print(f"{'='*120}")
print(fmt_df(f1.head(15)).to_string(index=False))

# Filter 2: elite
f2 = df[(df["sharpe"] >= 1.15) & (df["max_drawdown"] >= -0.40) & (df["cagr"] >= 0.25)]
f2 = f2.sort_values("sharpe", ascending=False)
print(f"\n{'='*120}")
print(f"ELITE (Sharpe>=1.15, MaxDD>=-40%, CAGR>=25%): {len(f2)} configs")
print(f"{'='*120}")
if not f2.empty:
    print(fmt_df(f2.head(10)).to_string(index=False))

# By mode breakdown
print(f"\n{'='*60}")
print("BEST SHARPE BY MODE")
print(f"{'='*60}")
for mode in df["mode"].unique():
    sub = df[df["mode"] == mode]
    best = sub.loc[sub["sharpe"].idxmax()]
    print(f"  {mode:20s}  Sharpe={best['sharpe']:.2f}  CAGR={best['cagr']:.1%}  "
          f"MaxDD={best['max_drawdown']:.1%}  Vol={best['volatility']:.1%}")

df.to_csv("data/rotation_grid_extended.csv", index=False)
print(f"\nSaved to data/rotation_grid_extended.csv")
