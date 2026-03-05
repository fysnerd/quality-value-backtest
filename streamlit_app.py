"""Interactive Streamlit dashboard for Quality/Value backtest + optimization."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loader import generate_synthetic_data, load_price_data, load_fundamental_data
from backtest import run_backtest, BacktestParams
from selection import SelectionParams
from metrics import format_metrics, MetricsParams
from benchmarks import compare_strategies, normalize_series
from optimization import ParamSpace, OptimizationConfig, run_param_search

# --- Page config ---
st.set_page_config(page_title="Quality/Value Backtest", layout="wide")
st.title("Quality/Value Strategy Backtester")

# --- Tabs ---
tab_backtest, tab_optimize = st.tabs(["Single Backtest", "Parameter Optimization"])

# ══════════════════════════════════════════════════════════════
# SIDEBAR: shared data loading
# ══════════════════════════════════════════════════════════════
st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Data", ["Synthetic (demo)", "Upload CSV", "Load from data/"])

df_prices = None
df_funda = None

if data_source == "Synthetic (demo)":
    n_synth = st.sidebar.slider("Synthetic universe size", 50, 500, 200, 50)
    seed = st.sidebar.number_input("Random seed", value=42, step=1)

    @st.cache_data
    def get_synthetic(n, s):
        return generate_synthetic_data(n_tickers=n, n_years=15, seed=s)

    df_prices, df_funda = get_synthetic(n_synth, seed)
    st.sidebar.success(f"{df_prices['ticker'].nunique()} tickers loaded")

elif data_source == "Upload CSV":
    price_file = st.sidebar.file_uploader("Price CSV (date, ticker, close)", type="csv")
    funda_file = st.sidebar.file_uploader("Fundamentals CSV", type="csv")

    if price_file and funda_file:
        df_prices = pd.read_csv(price_file, parse_dates=["date"])
        df_prices["date"] = pd.to_datetime(df_prices["date"], utc=True)
        df_funda = pd.read_csv(funda_file, parse_dates=["date"])
        df_funda["date"] = pd.to_datetime(df_funda["date"], utc=True)
        st.sidebar.success(f"{df_prices['ticker'].nunique()} tickers loaded")
    else:
        st.sidebar.info("Upload price and fundamental CSVs.")

else:  # Load from data/
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    prices_path = os.path.join(data_dir, "prices.csv")
    funda_path = os.path.join(data_dir, "fundamentals.csv")

    if os.path.exists(prices_path) and os.path.exists(funda_path):
        @st.cache_data
        def load_local_data(pp, fp):
            return load_price_data(pp), load_fundamental_data(fp)

        df_prices, df_funda = load_local_data(prices_path, funda_path)
        st.sidebar.success(f"{df_prices['ticker'].nunique()} tickers from data/")
    else:
        st.sidebar.error("No data/ folder found. Run `python data_download.py` first.")


# ══════════════════════════════════════════════════════════════
# TAB 1: SINGLE BACKTEST
# ══════════════════════════════════════════════════════════════
with tab_backtest:
    if df_prices is None or df_funda is None:
        st.info("Load data from the sidebar first.")
    else:
        col_params, col_settings = st.columns(2)

        with col_params:
            st.subheader("Strategy Parameters")
            n_stocks = st.slider("Number of stocks (N)", 5, 100, 40, 5, key="bt_n")
            pb_pct = st.slider("P/B percentile cut", 0.05, 0.50, 0.20, 0.05, key="bt_pb")
            min_fscore = st.slider("Min F-Score", 0, 9, 7, 1, key="bt_fs")
            rebalance_freq = st.selectbox("Rebalance (months)", [1, 3, 6, 12], index=2, key="bt_reb")
            min_mcap = st.number_input("Min Market Cap ($)", value=1_000_000_000, step=100_000_000, format="%d", key="bt_mc")

        with col_settings:
            st.subheader("Backtest Settings")
            start_date = st.date_input("Start date", pd.Timestamp("2022-01-01"), key="bt_start")
            end_date = st.date_input("End date", pd.Timestamp("2025-12-31"), key="bt_end")
            tx_cost = st.slider("Transaction cost (%)", 0.0, 1.0, 0.1, 0.05, key="bt_tc") / 100
            risk_free = st.slider("Risk-free rate (%)", 0.0, 10.0, 2.0, 0.5, key="bt_rf") / 100
            initial_capital = st.number_input("Initial capital ($)", value=100_000, step=10_000, key="bt_cap")
            pub_lag = st.slider("Publication lag (months)", 0, 6, 0, 1, key="bt_lag",
                                help="Set to 0 if data_download.py already added the lag")

        # Apply from optimization results
        if "apply_params" in st.session_state and st.session_state["apply_params"] is not None:
            ap = st.session_state["apply_params"]
            st.info(f"Applied from optimization: P/B<{ap['pb_percentile_cut']:.0%}, F>={ap['min_f_score']}, "
                    f"N={ap['n_stocks']}, Reb={ap['rebalance_months']}mo, TC={ap['transaction_cost']:.2%}")
            n_stocks = int(ap["n_stocks"])
            pb_pct = ap["pb_percentile_cut"]
            min_fscore = int(ap["min_f_score"])
            rebalance_freq = int(ap["rebalance_months"])
            tx_cost = ap["transaction_cost"]
            st.session_state["apply_params"] = None

        run_bt = st.button("Run Backtest", type="primary", use_container_width=True, key="run_bt")

        if run_bt or "bt_results" in st.session_state:
            if run_bt:
                params = BacktestParams(
                    start_date=str(start_date),
                    end_date=str(end_date),
                    rebalance_freq_months=rebalance_freq,
                    transaction_cost=tx_cost,
                    pub_lag_months=pub_lag,
                    risk_free_rate=risk_free,
                    initial_capital=initial_capital,
                    selection=SelectionParams(
                        min_market_cap=min_mcap,
                        pb_percentile_cut=pb_pct,
                        min_f_score=min_fscore,
                        n_stocks=n_stocks,
                    ),
                )
                with st.spinner("Running backtest..."):
                    equity_curve, positions, trades, metrics = run_backtest(df_prices, df_funda, params)

                pivot = df_prices.pivot_table(index="date", columns="ticker", values="close")
                mask = (pivot.index >= pd.Timestamp(str(start_date), tz="UTC")) & \
                       (pivot.index <= pd.Timestamp(str(end_date), tz="UTC"))
                benchmark = pivot[mask].mean(axis=1).dropna()

                st.session_state["bt_results"] = {
                    "equity_curve": equity_curve, "positions": positions,
                    "trades": trades, "metrics": metrics, "benchmark": benchmark,
                }

            res = st.session_state["bt_results"]
            equity_curve = res["equity_curve"]
            benchmark = res["benchmark"]
            metrics = res["metrics"]
            positions = res["positions"]
            trades = res["trades"]

            # Charts
            col1, col2 = st.columns([3, 1])

            with col1:
                norm = normalize_series({"Quality/Value": equity_curve, "EW Benchmark": benchmark})
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                    row_heights=[0.7, 0.3], subplot_titles=["Equity Curve (base 100)", "Drawdown"])
                fig.add_trace(go.Scatter(x=norm.index, y=norm["Quality/Value"], name="Quality/Value",
                                         line=dict(color="#2196F3", width=2.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=norm.index, y=norm["EW Benchmark"], name="EW Benchmark",
                                         line=dict(color="#9E9E9E", width=1.5, dash="dash")), row=1, col=1)
                peak = norm["Quality/Value"].cummax()
                dd = (norm["Quality/Value"] - peak) / peak
                fig.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", name="Drawdown",
                                         line=dict(color="#F44336", width=1), fillcolor="rgba(244,67,54,0.2)"), row=2, col=1)
                fig.update_layout(height=500, template="plotly_white", showlegend=True,
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02))
                fig.update_yaxes(title_text="Value", row=1, col=1)
                fig.update_yaxes(title_text="DD %", tickformat=".0%", row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Metrics")
                df_m = format_metrics(metrics)
                for _, row in df_m.iterrows():
                    st.metric(row["metric"].replace("_", " ").title(), row["value"])

            # Comparison table
            st.subheader("Strategy vs Benchmark")
            comp = compare_strategies(
                {"Quality/Value": equity_curve, "EW Benchmark": benchmark},
                MetricsParams(risk_free_rate=risk_free, periods_per_year=12),
            )
            comp_display = comp.copy()
            for c in comp_display.columns:
                for idx in comp_display.index:
                    val = comp_display.loc[idx, c]
                    if idx in ["total_return", "cagr", "volatility_ann", "max_drawdown", "win_rate"]:
                        comp_display.loc[idx, c] = f"{val:.2%}"
                    elif idx in ["max_dd_duration_periods", "n_trades"]:
                        comp_display.loc[idx, c] = f"{val:.0f}"
                    else:
                        comp_display.loc[idx, c] = f"{val:.2f}" if not (isinstance(val, float) and np.isnan(val)) else "N/A"
            st.dataframe(comp_display, use_container_width=True)

            if len(positions) > 0:
                st.subheader("Rebalancing History")
                pos_display = positions.copy()
                pos_display["tickers"] = pos_display["tickers"].apply(
                    lambda x: ", ".join(x[:10]) + ("..." if len(x) > 10 else ""))
                st.dataframe(pos_display, use_container_width=True)

            # ── Export section ──
            st.divider()
            st.subheader("Export Results")

            export_col1, export_col2, export_col3 = st.columns(3)

            # 1. Copy-paste text summary
            summary_lines = [
                f"Quality/Value Backtest — {start_date} to {end_date}",
                f"P/B<{pb_pct:.0%} | F>={min_fscore} | N={n_stocks} | Reb={rebalance_freq}mo | TC={tx_cost:.2%}",
                "",
            ]
            for k, v in metrics.items():
                label = k.replace("_", " ").title()
                if k in ["total_return", "cagr", "volatility_ann", "max_drawdown", "win_rate"]:
                    summary_lines.append(f"{label}: {v:.2%}")
                elif k in ["max_dd_duration_periods", "n_trades"]:
                    summary_lines.append(f"{label}: {v:.0f}")
                elif isinstance(v, float) and not np.isnan(v):
                    summary_lines.append(f"{label}: {v:.2f}")
                else:
                    summary_lines.append(f"{label}: N/A")
            if len(positions) > 0:
                last = positions.iloc[-1]
                summary_lines.append(f"\nLast rebalance: {last['date'].strftime('%Y-%m-%d')}")
                summary_lines.append(f"Holdings ({last['n_holdings']}): {', '.join(last['tickers'])}")
            summary_text = "\n".join(summary_lines)

            with export_col1:
                st.text_area("Copy summary", summary_text, height=300, key="copy_summary")

            # 2. Download equity curve CSV
            with export_col2:
                eq_df = equity_curve.reset_index()
                eq_df.columns = ["date", "portfolio_value"]
                st.download_button(
                    "Download Equity Curve (.csv)",
                    eq_df.to_csv(index=False),
                    file_name="equity_curve.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                # Download metrics CSV
                metrics_df = pd.DataFrame([metrics])
                metrics_df.insert(0, "start_date", str(start_date))
                metrics_df.insert(1, "end_date", str(end_date))
                metrics_df.insert(2, "pb_pct", pb_pct)
                metrics_df.insert(3, "min_fscore", min_fscore)
                metrics_df.insert(4, "n_stocks", n_stocks)
                metrics_df.insert(5, "rebalance_months", rebalance_freq)
                metrics_df.insert(6, "tx_cost", tx_cost)
                st.download_button(
                    "Download Metrics (.csv)",
                    metrics_df.to_csv(index=False),
                    file_name="backtest_metrics.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # 3. Download trades + positions
            with export_col3:
                if len(trades) > 0:
                    st.download_button(
                        "Download Trades (.csv)",
                        trades.to_csv(index=False),
                        file_name="trades.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                if len(positions) > 0:
                    pos_export = positions.copy()
                    pos_export["tickers"] = pos_export["tickers"].apply(lambda x: ", ".join(x))
                    st.download_button(
                        "Download Positions (.csv)",
                        pos_export.to_csv(index=False),
                        file_name="positions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )


# ══════════════════════════════════════════════════════════════
# TAB 2: PARAMETER OPTIMIZATION
# ══════════════════════════════════════════════════════════════
with tab_optimize:
    if df_prices is None or df_funda is None:
        st.info("Load data from the sidebar first.")
    else:
        st.subheader("Parameter Search Configuration")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**Search Settings**")
            search_type = st.selectbox("Search type", ["grid", "random"], key="opt_search")
            n_random = st.number_input("Random samples", value=200, step=50, key="opt_n",
                                       disabled=(search_type == "grid"))
            top_n_oos = st.slider("Top N for OOS", 5, 50, 20, 5, key="opt_topn")
            rank_by = st.selectbox("Rank IS by", ["sharpe", "calmar", "sortino", "cagr"], key="opt_rank")

        with col_b:
            st.markdown("**In-Sample Period**")
            is_start = st.date_input("IS Start", pd.Timestamp("2010-01-01"), key="opt_is_s")
            is_end = st.date_input("IS End", pd.Timestamp("2018-12-31"), key="opt_is_e")
            st.markdown("**Out-of-Sample Period**")
            oos_start = st.date_input("OOS Start", pd.Timestamp("2019-01-01"), key="opt_oos_s")
            oos_end = st.date_input("OOS End", pd.Timestamp("2024-12-31"), key="opt_oos_e")

        with col_c:
            st.markdown("**Parameter Space**")
            st.caption("Customize the grid values")
            pb_values = st.multiselect("P/B percentile cuts", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
                                       default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30], key="opt_pb")
            fs_values = st.multiselect("Min F-Score", [3, 4, 5, 6, 7, 8, 9],
                                       default=[5, 6, 7, 8], key="opt_fs")
            ns_values = st.multiselect("N stocks", [10, 15, 20, 30, 40, 50, 60, 80],
                                       default=[20, 30, 40, 60], key="opt_ns")
            reb_values = st.multiselect("Rebalance months", [1, 3, 6, 12],
                                        default=[3, 6, 12], key="opt_reb")

        # Show grid size
        param_space = ParamSpace(
            pb_percentile_cut=sorted(pb_values) if pb_values else [0.20],
            min_f_score=sorted(fs_values) if fs_values else [7],
            n_stocks=sorted(ns_values) if ns_values else [40],
            rebalance_months=sorted(reb_values) if reb_values else [6],
            transaction_cost=[0.0005, 0.001, 0.002],
        )
        total = param_space.total_combinations
        effective = total if search_type == "grid" else min(n_random, total)
        st.info(f"Grid: {total} total combos. Will run: **{effective}** backtests IS + top {top_n_oos} OOS.")

        # Run button
        run_opt = st.button("Run Optimization", type="primary", use_container_width=True, key="run_opt")

        if run_opt:
            config = OptimizationConfig(
                is_start=str(is_start),
                is_end=str(is_end),
                oos_start=str(oos_start),
                oos_end=str(oos_end),
                search_type=search_type,
                n_random_samples=n_random,
                top_n_for_oos=top_n_oos,
                rank_metric=rank_by,
            )

            progress_bar = st.progress(0, text="Running IS backtests...")
            status_text = st.empty()

            def progress_cb(current, total, combo):
                pct = current / total
                progress_bar.progress(pct, text=f"IS backtest {current}/{total}")

            with st.spinner("Optimization running..."):
                results = run_param_search(df_prices, df_funda, param_space, config, progress_callback=progress_cb)

            progress_bar.progress(1.0, text="Complete!")
            st.session_state["opt_results"] = results

        # Display results
        if "opt_results" in st.session_state and not st.session_state["opt_results"].empty:
            results = st.session_state["opt_results"]

            st.subheader(f"Top {min(20, len(results))} Results")

            # Format for display
            display_cols = [
                "pb_percentile_cut", "min_f_score", "n_stocks", "rebalance_months", "transaction_cost",
                "cagr", "sharpe", "sortino", "max_drawdown", "calmar",
            ]
            oos_cols = ["cagr_oos", "sharpe_oos", "sortino_oos", "max_drawdown_oos", "calmar_oos"]
            meta_cols = ["robust_score", "sharpe_degradation"]

            available = [c for c in display_cols + oos_cols + meta_cols if c in results.columns]
            display_df = results[available].head(20).copy()

            # Format percentages
            pct_cols = [c for c in display_df.columns if any(k in c for k in ["cagr", "max_drawdown", "win_rate", "pb_percentile", "transaction_cost"])]
            for c in pct_cols:
                if c in display_df.columns:
                    display_df[c] = display_df[c].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")

            # Format ratios
            ratio_cols = [c for c in display_df.columns if any(k in c for k in ["sharpe", "sortino", "calmar", "robust", "degradation"])]
            for c in ratio_cols:
                if c in display_df.columns:
                    display_df[c] = display_df[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

            # Integer cols
            for c in ["min_f_score", "n_stocks", "rebalance_months"]:
                if c in display_df.columns:
                    display_df[c] = display_df[c].astype(int)

            st.dataframe(display_df, use_container_width=True, height=500)

            # Apply button
            st.subheader("Apply a configuration to Single Backtest")
            row_idx = st.number_input("Select row # to apply (0-indexed)", min_value=0,
                                      max_value=len(results) - 1, value=0, key="opt_apply_idx")
            if st.button("Apply to Backtest tab", key="opt_apply"):
                row = results.iloc[row_idx]
                st.session_state["apply_params"] = {
                    "pb_percentile_cut": row["pb_percentile_cut"],
                    "min_f_score": int(row["min_f_score"]),
                    "n_stocks": int(row["n_stocks"]),
                    "rebalance_months": int(row["rebalance_months"]),
                    "transaction_cost": row["transaction_cost"],
                }
                st.success(f"Config #{row_idx} applied! Switch to 'Single Backtest' tab and click Run.")

            # Scatter plot: IS Sharpe vs OOS Sharpe
            if "sharpe_oos" in results.columns:
                st.subheader("IS vs OOS Sharpe")
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=results["sharpe"], y=results["sharpe_oos"],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=results["robust_score"],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Robust Score"),
                    ),
                    text=[f"P/B<{r['pb_percentile_cut']:.0%} F>={int(r['min_f_score'])} N={int(r['n_stocks'])}"
                          for _, r in results.iterrows()],
                    hovertemplate="%{text}<br>IS Sharpe: %{x:.2f}<br>OOS Sharpe: %{y:.2f}<extra></extra>",
                ))
                # Diagonal line
                max_s = max(results["sharpe"].max(), results["sharpe_oos"].max()) * 1.1
                min_s = min(results["sharpe"].min(), results["sharpe_oos"].min()) * 0.9
                fig_scatter.add_trace(go.Scatter(
                    x=[min_s, max_s], y=[min_s, max_s],
                    mode="lines", line=dict(dash="dash", color="gray"),
                    showlegend=False,
                ))
                fig_scatter.update_layout(
                    xaxis_title="Sharpe (In-Sample)",
                    yaxis_title="Sharpe (Out-of-Sample)",
                    template="plotly_white",
                    height=450,
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            # Save results
            if st.button("Save results to CSV", key="opt_save"):
                path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results_param_search.csv")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                results.to_csv(path, index=False)
                st.success(f"Saved to {path}")
