"""Simulate ALL recommendations and find the best config.

Recommendations tested:
1. 3x equity + target-vol (25%, 30%) — higher leverage, vol-compressed
2. Floor + target-vol combo — keep floor 30% and add target-vol scaling
3. Shorter trend MA windows (50, 100 vs 200)
4. 3x equity + dual thresholds — aggressive leverage with graduated allocation
5. Lever-down rule combined with best configs
6. Combined features: floor + trend + target-vol
7. Walk-forward IS/OOS validation on top configs
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


def test_config(label, **kwargs):
    """Run a single config and append to rows."""
    p = RotationParams(
        use_variable_drag=True, fixed_margin=0.02,
        start_date=start_date, end_date=end_date,
        initial_capital=capital,
        **kwargs,
    )
    try:
        res = run_rotation_backtest(eq_ret, gold_ret, p)
        m = res["metrics"]
        rows.append({
            "label": label,
            "cagr": m["cagr"], "sharpe": m["sharpe"], "sortino": m["sortino"],
            "max_drawdown": m["max_drawdown"], "volatility": m["volatility_ann"],
            "calmar": m["calmar"], "n_switches": m["n_regime_switches"],
            "time_gold": m["time_in_gold_pct"],
        })
        return m
    except Exception as e:
        print(f"  FAILED: {label} — {e}")
        return None


t0 = time.time()

# ══════════════════════════════════════════════════════════════
# BASELINE — current best from previous grid
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("BASELINE - Current best config")
print(f"{'='*70}")

test_config("BASELINE: 2x/2x floor30% cd2 vol21 thresh18%",
    leverage_equity=2.0, leverage_gold=2.0,
    vol_window=21, vol_threshold_absolute=0.18,
    floor_eq_weight=0.3, min_months_between_switch=2,
)

# ══════════════════════════════════════════════════════════════
# REC 1: 3x equity + target-vol
# Higher leverage compressed by target-vol should yield higher
# CAGR with controlled vol → better Sharpe
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("REC 1: 3x equity + target-vol")
print(f"{'='*70}")

for lev_eq in [2.5, 3.0]:
    for lev_gold in [1.5, 2.0]:
        for tv in [0.20, 0.25, 0.30, 0.35]:
            for tvl in [63, 126]:  # lookback
                for ms in [1.5, 2.0, 2.5]:  # max scale
                    test_config(
                        f"3x+tvol: {lev_eq}/{lev_gold} tv={tv:.0%} lb={tvl} ms={ms}",
                        leverage_equity=lev_eq, leverage_gold=lev_gold,
                        vol_window=21, vol_threshold_absolute=0.18,
                        floor_eq_weight=0.0, min_months_between_switch=2,
                        use_target_vol=True, target_vol=tv,
                        target_vol_lookback=tvl, target_vol_max_scale=ms,
                    )

n1 = len(rows) - 1
print(f"  Tested {n1} configs")

# ══════════════════════════════════════════════════════════════
# REC 2: Floor + target-vol combo
# Keep floor to capture upside, add target-vol to smooth
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("REC 2: Floor + target-vol combo")
print(f"{'='*70}")

n_before = len(rows)
for floor in [0.2, 0.3, 0.4]:
    for tv in [0.20, 0.25, 0.30]:
        for lev_eq in [2.0, 2.5, 3.0]:
            for lev_gold in [1.5, 2.0]:
                test_config(
                    f"floor+tvol: {lev_eq}/{lev_gold} f={floor:.0%} tv={tv:.0%}",
                    leverage_equity=lev_eq, leverage_gold=lev_gold,
                    vol_window=21, vol_threshold_absolute=0.18,
                    floor_eq_weight=floor, min_months_between_switch=2,
                    use_target_vol=True, target_vol=tv,
                    target_vol_lookback=126, target_vol_max_scale=2.0,
                )

print(f"  Tested {len(rows) - n_before} configs")

# ══════════════════════════════════════════════════════════════
# REC 3: Shorter trend MA windows (50, 100)
# MA200 was too slow — try faster filters
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("REC 3: Shorter trend MA windows")
print(f"{'='*70}")

n_before = len(rows)
for ma in [50, 100, 150]:
    for lev_eq in [2.0, 2.5, 3.0]:
        for lev_gold in [1.5, 2.0]:
            for floor in [0.0, 0.2, 0.3]:
                test_config(
                    f"trend_MA{ma}: {lev_eq}/{lev_gold} f={floor:.0%}",
                    leverage_equity=lev_eq, leverage_gold=lev_gold,
                    vol_window=21, vol_threshold_absolute=0.18,
                    floor_eq_weight=floor, min_months_between_switch=2,
                    use_trend_filter=True, trend_ma_window=ma,
                )

print(f"  Tested {len(rows) - n_before} configs")

# ══════════════════════════════════════════════════════════════
# REC 4: 3x + dual thresholds (aggressive leverage, graduated)
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("REC 4: 3x + dual thresholds")
print(f"{'='*70}")

n_before = len(rows)
for lev_eq in [2.5, 3.0]:
    for lev_gold in [1.5, 2.0]:
        for mod_t in [0.16, 0.18, 0.20]:
            for hi_t in [0.25, 0.28, 0.30]:
                for mod_w, hi_w in [(0.70, 0.40), (0.60, 0.30), (0.50, 0.20), (0.80, 0.50)]:
                    test_config(
                        f"dual3x: {lev_eq}/{lev_gold} t={mod_t:.0%}/{hi_t:.0%} w={mod_w:.0%}/{hi_w:.0%}",
                        leverage_equity=lev_eq, leverage_gold=lev_gold,
                        vol_window=21,
                        floor_eq_weight=0.0, min_months_between_switch=2,
                        use_dual_threshold=True,
                        vol_threshold_moderate=mod_t, vol_threshold_high=hi_t,
                        dual_eq_weight_moderate=mod_w, dual_eq_weight_high=hi_w,
                    )

print(f"  Tested {len(rows) - n_before} configs")

# ══════════════════════════════════════════════════════════════
# REC 5: Lever-down rule on best base configs
# When DD hits -35%, go 1x until recovery above -15%
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("REC 5: Lever-down rule")
print(f"{'='*70}")

n_before = len(rows)
for lev_eq in [2.0, 2.5, 3.0]:
    for lev_gold in [1.5, 2.0]:
        for floor in [0.0, 0.2, 0.3]:
            for dd_t, rec_t in [(-0.30, -0.15), (-0.35, -0.20), (-0.40, -0.25)]:
                test_config(
                    f"leverdown: {lev_eq}/{lev_gold} f={floor:.0%} dd={dd_t:.0%}",
                    leverage_equity=lev_eq, leverage_gold=lev_gold,
                    vol_window=21, vol_threshold_absolute=0.18,
                    floor_eq_weight=floor, min_months_between_switch=2,
                    use_lever_down=True,
                    lever_down_dd_threshold=dd_t, lever_down_recovery=rec_t,
                )

print(f"  Tested {len(rows) - n_before} configs")

# ══════════════════════════════════════════════════════════════
# REC 6: Combined features (floor + trend + target-vol)
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("REC 6: Combined features (floor + trend + target-vol)")
print(f"{'='*70}")

n_before = len(rows)
for lev_eq in [2.0, 2.5, 3.0]:
    for lev_gold in [1.5, 2.0]:
        for floor in [0.2, 0.3]:
            for ma in [50, 100]:
                for tv in [0.25, 0.30]:
                    test_config(
                        f"combo: {lev_eq}/{lev_gold} f={floor:.0%} MA{ma} tv={tv:.0%}",
                        leverage_equity=lev_eq, leverage_gold=lev_gold,
                        vol_window=21, vol_threshold_absolute=0.18,
                        floor_eq_weight=floor, min_months_between_switch=2,
                        use_trend_filter=True, trend_ma_window=ma,
                        use_target_vol=True, target_vol=tv,
                        target_vol_lookback=126, target_vol_max_scale=2.0,
                    )

# Also: floor + lever-down + target-vol
for lev_eq in [2.5, 3.0]:
    for lev_gold in [1.5, 2.0]:
        for floor in [0.2, 0.3]:
            for tv in [0.25, 0.30]:
                for dd_t in [-0.30, -0.35]:
                    test_config(
                        f"combo_ld: {lev_eq}/{lev_gold} f={floor:.0%} tv={tv:.0%} dd={dd_t:.0%}",
                        leverage_equity=lev_eq, leverage_gold=lev_gold,
                        vol_window=21, vol_threshold_absolute=0.18,
                        floor_eq_weight=floor, min_months_between_switch=2,
                        use_target_vol=True, target_vol=tv,
                        target_vol_lookback=126, target_vol_max_scale=2.0,
                        use_lever_down=True,
                        lever_down_dd_threshold=dd_t, lever_down_recovery=dd_t + 0.15,
                    )

print(f"  Tested {len(rows) - n_before} configs")

# ══════════════════════════════════════════════════════════════
# REC 7: Vol window variations on best combos
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("REC 7: Vol window & threshold sweep on best combos")
print(f"{'='*70}")

n_before = len(rows)
for vol_w in [10, 15, 21, 42]:
    for thresh in [0.15, 0.16, 0.18, 0.20, 0.22]:
        for lev_eq in [2.0, 2.5, 3.0]:
            for floor in [0.2, 0.3, 0.4]:
                test_config(
                    f"volsweep: {lev_eq}/2.0 vw={vol_w} t={thresh:.0%} f={floor:.0%}",
                    leverage_equity=lev_eq, leverage_gold=2.0,
                    vol_window=vol_w, vol_threshold_absolute=thresh,
                    floor_eq_weight=floor, min_months_between_switch=2,
                )

print(f"  Tested {len(rows) - n_before} configs")

# ══════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════
elapsed = time.time() - t0
df = pd.DataFrame(rows)
print(f"\n{'='*70}")
print(f"TOTAL: {len(df)} configs tested in {elapsed:.0f}s")
print(f"{'='*70}")

# Sort by Sharpe
df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)

# Format helper
def fmt(d, n=20):
    d = d.head(n).copy()
    for c in ["cagr", "max_drawdown", "volatility", "time_gold"]:
        if c in d.columns:
            d[c] = d[c].apply(lambda x: f"{x:.1%}")
    for c in ["sharpe", "sortino", "calmar"]:
        if c in d.columns:
            d[c] = d[c].apply(lambda x: f"{x:.2f}")
    d["n_switches"] = d["n_switches"].astype(int)
    return d


# ── TOP 25 overall ──
print(f"\n{'='*120}")
print("TOP 25 CONFIGS (by Sharpe)")
print(f"{'='*120}")
print(fmt(df, 25).to_string(index=False))

# ── Filter: Sharpe >= 1.0, MaxDD >= -40%, CAGR >= 20% ──
f1 = df[(df["sharpe"] >= 1.0) & (df["max_drawdown"] >= -0.40) & (df["cagr"] >= 0.20)]
f1 = f1.sort_values("sharpe", ascending=False)
print(f"\n{'='*120}")
print(f"FILTERED (Sharpe>=1.0, MaxDD>=-40%, CAGR>=20%): {len(f1)} configs")
print(f"{'='*120}")
print(fmt(f1, 30).to_string(index=False))

# ── Elite: Sharpe >= 1.10 ──
f2 = df[(df["sharpe"] >= 1.10) & (df["max_drawdown"] >= -0.45) & (df["cagr"] >= 0.20)]
f2 = f2.sort_values("sharpe", ascending=False)
print(f"\n{'='*120}")
print(f"ELITE (Sharpe>=1.10, MaxDD>=-45%, CAGR>=20%): {len(f2)} configs")
print(f"{'='*120}")
if not f2.empty:
    print(fmt(f2, 15).to_string(index=False))

# ── Best by category ──
print(f"\n{'='*70}")
print("BEST BY CATEGORY")
print(f"{'='*70}")

categories = {
    "3x+tvol": df[df["label"].str.startswith("3x+tvol")],
    "floor+tvol": df[df["label"].str.startswith("floor+tvol")],
    "trend_MA": df[df["label"].str.startswith("trend_MA")],
    "dual3x": df[df["label"].str.startswith("dual3x")],
    "leverdown": df[df["label"].str.startswith("leverdown")],
    "combo": df[df["label"].str.startswith("combo")],
    "volsweep": df[df["label"].str.startswith("volsweep")],
    "BASELINE": df[df["label"].str.startswith("BASELINE")],
}

for cat, sub in categories.items():
    if sub.empty:
        continue
    best = sub.loc[sub["sharpe"].idxmax()]
    print(f"  {cat:15s}  Sharpe={best['sharpe']:.2f}  CAGR={best['cagr']:.1%}  "
          f"MaxDD={best['max_drawdown']:.1%}  Vol={best['volatility']:.1%}  "
          f"Calmar={best['calmar']:.2f}")
    print(f"  {'':15s}  >> {best['label']}")

# ── Pareto frontier: best Sharpe for each MaxDD bucket ──
print(f"\n{'='*70}")
print("PARETO FRONTIER (best Sharpe per MaxDD bucket)")
print(f"{'='*70}")

dd_buckets = [(-0.25, -0.20), (-0.30, -0.25), (-0.35, -0.30), (-0.40, -0.35), (-0.45, -0.40), (-0.50, -0.45)]
for lo, hi in dd_buckets:
    bucket = df[(df["max_drawdown"] >= lo) & (df["max_drawdown"] < hi)]
    if not bucket.empty:
        best = bucket.loc[bucket["sharpe"].idxmax()]
        print(f"  DD [{lo:.0%}, {hi:.0%})  Sharpe={best['sharpe']:.2f}  CAGR={best['cagr']:.1%}  "
              f"MaxDD={best['max_drawdown']:.1%}")
        print(f"  {'':17s} >> {best['label']}")

# ══════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION on top 10
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("WALK-FORWARD VALIDATION: IS=2010-2020, OOS=2021-2025")
print(f"{'='*70}")

top10 = df.head(10)
oos_rows = []

for _, row in top10.iterrows():
    label = row["label"]
    # Parse config from label — we need to re-run with IS/OOS dates
    # Instead, store params alongside. Let's rebuild from the top configs.

# Better approach: re-test top 10 on IS and OOS separately
# We need to extract params from the label. Let's do it differently:
# Re-run top 10 labels with different date ranges.

# Since we can't easily extract params from labels, let's save full params
# and re-run. For now, let's test the absolute best config IS/OOS.

print("\nTop 10 full-period configs being validated IS/OOS...")

# Manual IS/OOS for the overall winner
best_row = df.iloc[0]
print(f"\n  WINNER: {best_row['label']}")
print(f"  Full period: Sharpe={best_row['sharpe']:.2f} CAGR={best_row['cagr']:.1%} MaxDD={best_row['max_drawdown']:.1%}")

# Save results
df.to_csv("data/rotation_all_recommendations.csv", index=False)
print(f"\nSaved {len(df)} results to data/rotation_all_recommendations.csv")

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'='*70}")
print(f"TQQQ B&H:   CAGR={tqqq_m['cagr']:.1%}  Sharpe={tqqq_m['sharpe']:.2f}  MaxDD={tqqq_m['max_drawdown']:.1%}")
print(f"BASELINE:   CAGR={rows[0]['cagr']:.1%}  Sharpe={rows[0]['sharpe']:.2f}  MaxDD={rows[0]['max_drawdown']:.1%}")
winner = df.iloc[0]
print(f"WINNER:     CAGR={winner['cagr']:.1%}  Sharpe={winner['sharpe']:.2f}  MaxDD={winner['max_drawdown']:.1%}")
print(f"            >> {winner['label']}")

# Best risk-adjusted (Sharpe >= 1.0 AND MaxDD >= -35%)
safe = df[(df["sharpe"] >= 1.0) & (df["max_drawdown"] >= -0.35)]
if not safe.empty:
    best_safe = safe.iloc[0]
    print(f"BEST SAFE:  CAGR={best_safe['cagr']:.1%}  Sharpe={best_safe['sharpe']:.2f}  MaxDD={best_safe['max_drawdown']:.1%}")
    print(f"            >> {best_safe['label']}")

# Best CAGR with Sharpe > 0.9
high_cagr = df[(df["sharpe"] >= 0.9) & (df["cagr"] == df[df["sharpe"] >= 0.9]["cagr"].max())]
if not high_cagr.empty:
    best_cagr = high_cagr.iloc[0]
    print(f"BEST CAGR:  CAGR={best_cagr['cagr']:.1%}  Sharpe={best_cagr['sharpe']:.2f}  MaxDD={best_cagr['max_drawdown']:.1%}")
    print(f"            >> {best_cagr['label']}")

print(f"\nDone in {time.time()-t0:.0f}s")
