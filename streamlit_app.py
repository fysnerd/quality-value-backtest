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
from rotation_leveraged import (
    RotationParams, run_rotation_backtest, download_rotation_data,
    download_tqqq_data, format_rotation_metrics, compute_tqqq_comparison,
    run_sensitivity_analysis, LEVERAGE_PROFILES,
)

# --- Page config ---
st.set_page_config(page_title="Quant Backtester", layout="wide")
st.title("Quant Strategy Backtester")

# --- Tabs ---
tab_backtest, tab_optimize, tab_rotation, tab_compare = st.tabs([
    "Quality/Value", "Parameter Optimization", "Leveraged Rotation", "Comparaison"
])

# ══════════════════════════════════════════════════════════════
# SIDEBAR: shared data loading
# ══════════════════════════════════════════════════════════════
st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Data", ["Synthetic (demo)", "Upload CSV", "Load from data/",
                                         "SimFin (2020-2024)", "Combined (2010-2025)"])

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

elif data_source == "Load from data/":
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

elif data_source == "SimFin (2020-2024)":
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    prices_path = os.path.join(data_dir, "prices.csv")
    simfin_path = os.path.join(data_dir, "fundamentals_simfin.csv")

    if os.path.exists(prices_path) and os.path.exists(simfin_path):
        @st.cache_data
        def load_simfin_data(pp, sp):
            p = load_price_data(pp)
            f = pd.read_csv(sp, parse_dates=["date"])
            f["date"] = pd.to_datetime(f["date"], utc=True)
            return p, f

        df_prices, df_funda = load_simfin_data(prices_path, simfin_path)
        n_with_pb = df_funda["pb"].notna().sum()
        n_with_fs = df_funda["f_score"].notna().sum()
        st.sidebar.success(
            f"{df_funda['ticker'].nunique()} tickers (SimFin)\n"
            f"P/B: {n_with_pb} rows | F-Score: {n_with_fs} rows"
        )
    else:
        st.sidebar.error("Run `python data_simfin.py` first to generate SimFin fundamentals.")

else:  # Combined (2010-2025)
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    prices_path = os.path.join(data_dir, "prices.csv")
    combined_path = os.path.join(data_dir, "fundamentals_combined.csv")

    if os.path.exists(prices_path) and os.path.exists(combined_path):
        @st.cache_data
        def load_combined_data(pp, cp):
            p = load_price_data(pp)
            f = pd.read_csv(cp, parse_dates=["date"])
            f["date"] = pd.to_datetime(f["date"], utc=True)
            return p, f

        df_prices, df_funda = load_combined_data(prices_path, combined_path)
        n_with_pb = df_funda["pb"].notna().sum()
        n_with_fs = df_funda["f_score"].notna().sum()
        st.sidebar.success(
            f"{df_funda['ticker'].nunique()} tickers (EDGAR+SimFin)\n"
            f"P/B: {n_with_pb} rows | F-Score: {n_with_fs} rows"
        )
    else:
        st.sidebar.error("Run `python data_sec_edgar.py` then `python data_simfin.py` first.")


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
            use_qscore = st.checkbox("Q-Score fallback (for sparse pre-2018 data)", value=False, key="bt_qs")
            min_qscore = st.slider("Min Q-Score (0-3)", 0, 3, 2, 1, key="bt_minqs", disabled=not use_qscore)

        with col_settings:
            st.subheader("Backtest Settings")
            default_start = pd.Timestamp("2012-01-01") if data_source == "Combined (2010-2025)" else pd.Timestamp("2022-01-01")
            start_date = st.date_input("Start date", default_start, key="bt_start")
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
                        use_q_score_fallback=use_qscore,
                        min_q_score=min_qscore,
                    ),
                )
                with st.spinner("Running backtest..."):
                    equity_curve, positions, trades, metrics = run_backtest(df_prices, df_funda, params)

                pivot = df_prices.pivot_table(index="date", columns="ticker", values="close")
                mask = (pivot.index >= pd.Timestamp(str(start_date), tz="UTC")) & \
                       (pivot.index <= pd.Timestamp(str(end_date), tz="UTC"))
                benchmark = pivot[mask].mean(axis=1).dropna()

                # Download SPY for comparison
                import yfinance as yf
                spy_data = yf.download("SPY", start=str(start_date), end=str(end_date),
                                        interval="1mo", progress=False)
                if not spy_data.empty:
                    spy_series = spy_data["Close"].squeeze()
                    spy_series.index = pd.to_datetime(spy_series.index, utc=True)
                    spy_series.name = "value"
                else:
                    spy_series = pd.Series(dtype=float)

                st.session_state["bt_results"] = {
                    "equity_curve": equity_curve, "positions": positions,
                    "trades": trades, "metrics": metrics, "benchmark": benchmark,
                    "spy": spy_series,
                }

            res = st.session_state["bt_results"]
            equity_curve = res["equity_curve"]
            benchmark = res["benchmark"]
            metrics = res["metrics"]
            positions = res["positions"]
            trades = res["trades"]
            spy_series = res.get("spy", pd.Series(dtype=float))

            # Charts
            col1, col2 = st.columns([3, 1])

            with col1:
                series_dict = {"Quality/Value": equity_curve}
                if not spy_series.empty:
                    series_dict["SPY"] = spy_series
                norm = normalize_series(series_dict)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                    row_heights=[0.7, 0.3], subplot_titles=["Equity Curve (base 100)", "Drawdown"])
                fig.add_trace(go.Scatter(x=norm.index, y=norm["Quality/Value"], name="Quality/Value",
                                         line=dict(color="#2196F3", width=2.5)), row=1, col=1)
                if "SPY" in norm.columns:
                    fig.add_trace(go.Scatter(x=norm.index, y=norm["SPY"], name="SPY",
                                             line=dict(color="#FF9800", width=2, dash="dot")), row=1, col=1)
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
            st.subheader("Strategy vs SPY")
            comp_dict = {"Quality/Value": equity_curve}
            if not spy_series.empty:
                comp_dict["SPY"] = spy_series
            comp = compare_strategies(
                comp_dict,
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


# ══════════════════════════════════════════════════════════════
# TAB 3: LEVERAGED ROTATION (Nasdaq/Gold vol-managed)
# ══════════════════════════════════════════════════════════════
with tab_rotation:
    st.markdown(
        "**Volatility-managed rotation** between leveraged Nasdaq and leveraged Gold. "
        "Signal: realized vol of equity index. High vol -> Gold, Low vol -> Nasdaq."
    )

    # ── Sub-tabs: Realistic vs Experimental ──
    rot_sub_real, rot_sub_exp, rot_sub_doc = st.tabs([
        "Realistic (2010-2025)", "Experimental (1985+ synthetic)", "Hypotheses & Limits"
    ])

    # ── Shared controls in sidebar ──
    st.sidebar.divider()
    st.sidebar.header("Rotation Settings")

    # Leverage profile
    rot_profile = st.sidebar.selectbox(
        "Leverage Profile",
        list(LEVERAGE_PROFILES.keys()),
        index=0,
        key="rot_profile",
        help="Aggro: 3x/3x | Normal: 2x/1.5x | Light: 1.5x/1x",
    )
    profile = LEVERAGE_PROFILES[rot_profile]
    st.sidebar.caption(profile["label"])

    rot_custom_lev = st.sidebar.checkbox("Custom leverage override", value=False, key="rot_custom")
    if rot_custom_lev:
        rot_lev_eq = st.sidebar.slider("Equity leverage (x)", 1.0, 5.0, profile["leverage_equity"], 0.5, key="rot_lev_eq")
        rot_lev_gold = st.sidebar.slider("Gold leverage (x)", 1.0, 5.0, profile["leverage_gold"], 0.5, key="rot_lev_gold")
    else:
        rot_lev_eq = profile["leverage_equity"]
        rot_lev_gold = profile["leverage_gold"]

    rot_max_exposure = st.sidebar.slider("Max exposure cap (x)", 1.0, 5.0, float(rot_lev_eq), 0.5, key="rot_max_exp",
                                          help="Notional cap — default = no cap (equals equity leverage)")

    # Drag
    rot_use_var_drag = st.sidebar.checkbox("Variable drag (Fed Funds + margin)", value=True, key="rot_var_drag")
    rot_fixed_margin = st.sidebar.slider("Fixed margin (% / year)", 0.5, 5.0, 2.0, 0.5, key="rot_margin",
                                          disabled=not rot_use_var_drag) / 100
    if not rot_use_var_drag:
        rot_drag_eq = st.sidebar.slider("Equity flat drag (%/yr)", 0.0, 3.0, 1.0, 0.1, key="rot_drag_eq") / 100
        rot_drag_gold = st.sidebar.slider("Gold flat drag (%/yr)", 0.0, 3.0, 1.0, 0.1, key="rot_drag_gold") / 100
    else:
        rot_drag_eq = 0.01
        rot_drag_gold = 0.01

    # Vol signal
    st.sidebar.divider()
    st.sidebar.subheader("Vol Signal")
    rot_vol_window = st.sidebar.slider("Vol window (days)", 5, 63, 21, 1, key="rot_vol_win")
    rot_use_abs = st.sidebar.checkbox("Absolute vol threshold", value=False, key="rot_abs")
    rot_abs_thresh = st.sidebar.slider("Abs threshold (% ann.)", 10, 50, 20, 1, key="rot_abs_v",
                                        disabled=not rot_use_abs) / 100
    rot_vol_q = st.sidebar.slider("Quantile threshold", 0.50, 0.95, 0.70, 0.05, key="rot_vol_q",
                                   disabled=rot_use_abs)
    rot_q_mode = st.sidebar.selectbox("Quantile mode", ["expanding", "rolling", "global"],
                                       key="rot_q_mode", disabled=rot_use_abs,
                                       help="expanding = growing window, rolling = fixed lookback, global = full history (prone to look-ahead)")
    rot_q_lookback = st.sidebar.slider("Rolling lookback (days)", 252, 1260, 756, 63,
                                        key="rot_q_lb", disabled=(rot_q_mode != "rolling"))

    # Risk management
    st.sidebar.divider()
    st.sidebar.subheader("Risk Management")
    rot_crash = st.sidebar.checkbox("Crash regime -> cash", value=False, key="rot_crash")
    rot_crash_q = st.sidebar.slider("Crash quantile", 0.90, 0.99, 0.95, 0.01, key="rot_crash_q",
                                     disabled=not rot_crash)
    rot_lever_down = st.sidebar.checkbox("Lever-down rule", value=False, key="rot_ld",
                                          help="Reduce to 1x when DD exceeds threshold")
    rot_ld_thresh = st.sidebar.slider("Lever-down DD threshold", -0.70, -0.20, -0.40, 0.05,
                                       key="rot_ld_t", disabled=not rot_lever_down)
    rot_ld_recovery = st.sidebar.slider("Resume leverage when DD >", -0.40, 0.0, -0.20, 0.05,
                                         key="rot_ld_r", disabled=not rot_lever_down)
    rot_floor_eq = st.sidebar.slider("Floor equity weight in gold regime", 0.0, 0.5, 0.0, 0.05,
                                      key="rot_floor_eq",
                                      help="Min equity allocation even in gold regime (0 = pure rotation)")
    rot_cooldown = st.sidebar.slider("Min months between switches", 1, 6, 1, 1,
                                      key="rot_cooldown",
                                      help="Cooldown: suppress regime switches within N months")

    # Common params
    rot_tx = st.sidebar.slider("Switch cost (%)", 0.0, 1.0, 0.1, 0.05, key="rot_tx") / 100
    rot_capital = st.sidebar.number_input("Initial capital ($)", value=100_000, step=10_000, key="rot_cap")
    rot_rf = st.sidebar.slider("Risk-free rate (%)", 0.0, 10.0, 2.0, 0.5, key="rot_rf") / 100

    # Helper: build params
    def _build_rot_params(start_str, end_str):
        return RotationParams(
            leverage_equity=rot_lev_eq,
            leverage_gold=rot_lev_gold,
            max_exposure=rot_max_exposure,
            use_variable_drag=rot_use_var_drag,
            fixed_margin=rot_fixed_margin,
            drag_equity_annual=rot_drag_eq,
            drag_gold_annual=rot_drag_gold,
            vol_window=rot_vol_window,
            vol_threshold_quantile=rot_vol_q,
            vol_threshold_absolute=rot_abs_thresh if rot_use_abs else None,
            vol_quantile_mode=rot_q_mode,
            vol_quantile_lookback=rot_q_lookback,
            start_date=start_str,
            end_date=end_str,
            initial_capital=rot_capital,
            risk_free_rate=rot_rf,
            transaction_cost=rot_tx,
            use_crash_regime=rot_crash,
            crash_vol_quantile=rot_crash_q,
            floor_eq_weight=rot_floor_eq,
            min_months_between_switch=rot_cooldown,
            use_lever_down=rot_lever_down,
            lever_down_dd_threshold=rot_ld_thresh,
            lever_down_recovery=rot_ld_recovery,
        )

    # Helper: render results
    def _render_rotation_results(res, session_key, lev_eq, lev_gold, rf, show_tqqq=True, is_experimental=False):
        rot_equity = res["equity_curve"]
        rot_metrics = res["metrics"]
        rot_regime = res["regime_series"]
        rot_vol = res["vol_series"]
        rot_vthresh = res.get("vol_threshold_series", pd.Series(dtype=float))
        comp = res["components"]
        spy_curve = res.get("spy_curve", pd.Series(dtype=float))
        tqqq_curve = res.get("tqqq_curve", pd.Series(dtype=float))

        # ── MaxDD warning ──
        max_dd = rot_metrics.get("max_drawdown", 0)
        dd_target = profile["max_dd_target"]
        if max_dd < -0.50:
            st.error(f"MaxDD = {max_dd:.1%} — exceeds -50%. This config is highly risky.")
        elif max_dd < dd_target:
            st.warning(f"MaxDD = {max_dd:.1%} — exceeds {rot_profile} target of {dd_target:.0%}.")

        if is_experimental:
            st.info("These results use synthetic leveraged ETFs reconstructed from indices. "
                    "Real leveraged ETFs did not exist before 2006-2010. Treat with caution.")

        # ── Chart: Equity + Drawdown + Regime ──
        chart_col, metrics_col = st.columns([3, 1])

        with chart_col:
            series_comp = {
                "Rotation": rot_equity,
                f"B&H {lev_eq:.0f}x Equity": comp["equity_leveraged"],
                f"B&H {lev_gold:.0f}x Gold": comp["gold_leveraged"],
                "1x Equity": comp["equity_unleveraged"],
            }
            if not spy_curve.empty:
                series_comp["SPY"] = spy_curve
            if not tqqq_curve.empty and show_tqqq:
                series_comp["TQQQ B&H"] = tqqq_curve
            norm = normalize_series(series_comp)

            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=["Equity Curves (base 100)", "Drawdown", "Regime"],
            )

            palette = {
                "Rotation": "#2196F3",
                f"B&H {lev_eq:.0f}x Equity": "#FF5722",
                f"B&H {lev_gold:.0f}x Gold": "#FFC107",
                "1x Equity": "#9E9E9E",
                "SPY": "#4CAF50",
                "TQQQ B&H": "#E91E63",
            }
            for name in norm.columns:
                color = palette.get(name, "#666666")
                dash = "solid" if name == "Rotation" else "dot"
                width = 2.5 if name == "Rotation" else 1.5
                fig.add_trace(go.Scatter(
                    x=norm.index, y=norm[name], name=name,
                    line=dict(color=color, width=width, dash=dash),
                ), row=1, col=1)

            peak = rot_equity.cummax()
            dd = (rot_equity - peak) / peak
            fig.add_trace(go.Scatter(
                x=dd.index, y=dd.values, fill="tozeroy", name="DD",
                line=dict(color="#F44336", width=1), fillcolor="rgba(244,67,54,0.2)",
                showlegend=False,
            ), row=2, col=1)

            regime_vals = rot_regime.dropna()
            regime_colors = regime_vals.map({0: "#2196F3", 1: "#FFC107", 2: "#9E9E9E"})
            regime_labels = regime_vals.map({0: "Equity", 1: "Gold", 2: "Cash"})
            fig.add_trace(go.Scatter(
                x=regime_vals.index, y=regime_vals.values, mode="markers",
                marker=dict(color=regime_colors.values, size=2),
                text=regime_labels.values,
                hovertemplate="%{x}: %{text}<extra></extra>",
                showlegend=False,
            ), row=3, col=1)

            fig.update_layout(
                height=700, template="plotly_white", showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            fig.update_yaxes(title_text="Value (log)", type="log", row=1, col=1)
            fig.update_yaxes(title_text="DD %", tickformat=".0%", row=2, col=1)
            fig.update_yaxes(tickvals=[0, 1, 2], ticktext=["Eq", "Au", "$"], row=3, col=1)
            st.plotly_chart(fig, use_container_width=True)

        with metrics_col:
            st.subheader("Metrics")
            df_rm = format_rotation_metrics(rot_metrics)
            for _, row in df_rm.iterrows():
                st.metric(row["metric"], row["value"])

        # ── Vol chart with dynamic threshold ──
        st.subheader("Realized Volatility & Threshold")
        fig_vol = go.Figure()
        vol_clean = rot_vol.dropna()
        fig_vol.add_trace(go.Scatter(
            x=vol_clean.index, y=vol_clean.values,
            name="Realized Vol", line=dict(color="#673AB7", width=1.5),
        ))
        if not rot_vthresh.empty:
            vt_clean = rot_vthresh.dropna()
            fig_vol.add_trace(go.Scatter(
                x=vt_clean.index, y=vt_clean.values,
                name="Threshold", line=dict(color="red", width=1.5, dash="dash"),
            ))
        else:
            thresh_val = rot_metrics.get("vol_threshold_used", 0)
            fig_vol.add_hline(y=thresh_val, line_dash="dash", line_color="red",
                              annotation_text=f"{thresh_val:.1%}")
        fig_vol.update_layout(
            height=300, template="plotly_white",
            yaxis_title="Annualized Vol", yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        # ── TQQQ comparison (realistic tab only) ──
        if show_tqqq and not tqqq_curve.empty:
            st.subheader("Rotation vs TQQQ Buy & Hold")
            tqqq_comp = compute_tqqq_comparison(rot_equity, tqqq_curve, rot_rf)

            tc1, tc2, tc3, tc4 = st.columns(4)
            tc1.metric("Delta CAGR", f"{tqqq_comp['delta_cagr']:+.2%}",
                        delta_color="normal" if tqqq_comp['delta_cagr'] > 0 else "inverse")
            tc2.metric("Delta Sharpe", f"{tqqq_comp['delta_sharpe']:+.2f}",
                        delta_color="normal" if tqqq_comp['delta_sharpe'] > 0 else "inverse")
            tc3.metric("Delta MaxDD", f"{tqqq_comp['delta_max_dd']:+.2%}",
                        help="Positive = rotation has less severe DD",
                        delta_color="normal" if tqqq_comp['delta_max_dd'] > 0 else "inverse")
            r3y = tqqq_comp['rolling_3y_outperformance_pct']
            tc4.metric("Rolling 3Y Win%", f"{r3y:.0%}" if not np.isnan(r3y) else "N/A")

            if not tqqq_comp["rotation_useful"]:
                st.warning("Rotation does NOT improve Sharpe or MaxDD vs TQQQ B&H. Config flagged as **rotation inutile**.")
            else:
                st.success("Rotation improves at least Sharpe or MaxDD vs TQQQ B&H.")

        # ── Comparison table ──
        st.subheader("Full Comparison")
        comp_series = {"Rotation": rot_equity}
        if not tqqq_curve.empty and show_tqqq:
            comp_series["TQQQ B&H"] = tqqq_curve
        if not spy_curve.empty:
            comp_series["SPY"] = spy_curve
        comp_series[f"{lev_eq:.0f}x Equity"] = comp["equity_leveraged"]
        comp_series[f"{lev_gold:.0f}x Gold"] = comp["gold_leveraged"]
        comp_series["1x Equity"] = comp["equity_unleveraged"]

        comp_table = compare_strategies(
            comp_series, MetricsParams(risk_free_rate=rf, periods_per_year=252),
        )
        comp_display = comp_table.copy()
        for c in comp_display.columns:
            for idx in comp_display.index:
                val = comp_display.loc[idx, c]
                if idx in ["total_return", "cagr", "volatility_ann", "max_drawdown", "win_rate"]:
                    comp_display.loc[idx, c] = f"{val:.2%}"
                elif idx in ["max_dd_duration_periods", "n_trades"]:
                    comp_display.loc[idx, c] = f"{val:.0f}"
                else:
                    comp_display.loc[idx, c] = (
                        f"{val:.2f}" if not (isinstance(val, float) and np.isnan(val)) else "N/A"
                    )
        st.dataframe(comp_display, use_container_width=True)

        # ── Export ──
        st.divider()
        exp1, exp2 = st.columns(2)
        with exp1:
            eq_df = rot_equity.reset_index()
            eq_df.columns = ["date", "portfolio_value"]
            st.download_button(
                "Download Equity Curve (.csv)", eq_df.to_csv(index=False),
                file_name=f"rotation_{session_key}.csv", mime="text/csv",
                use_container_width=True, key=f"dl_eq_{session_key}",
            )
        with exp2:
            regime_df = rot_regime.dropna().reset_index()
            regime_df.columns = ["date", "regime"]
            regime_df["label"] = regime_df["regime"].map({0: "Equity", 1: "Gold", 2: "Cash"})
            st.download_button(
                "Download Regime (.csv)", regime_df.to_csv(index=False),
                file_name=f"regime_{session_key}.csv", mime="text/csv",
                use_container_width=True, key=f"dl_reg_{session_key}",
            )

    # ══════════════════════════════════════════════════════════
    # SUB-TAB 1: REALISTIC (2010-2025)
    # ══════════════════════════════════════════════════════════
    with rot_sub_real:
        st.subheader("Realistic Backtest (2010-2025) — ETF levier existants")

        r_col1, r_col2 = st.columns(2)
        with r_col1:
            rot_real_start = st.date_input("Start", pd.Timestamp("2010-03-01"), key="rot_r_start",
                                            help="TQQQ inception: 2010-02-11")
            rot_real_end = st.date_input("End", pd.Timestamp("2025-12-31"), key="rot_r_end")
        with r_col2:
            rot_eq_ticker = st.selectbox("Equity", ["^NDX (Nasdaq 100)", "^GSPC (S&P 500)"], key="rot_r_eq")
            rot_gold_ticker = st.selectbox("Gold", ["GC=F (Gold Futures)", "GLD (Gold ETF)"], key="rot_r_gold")

        run_real = st.button("Run Realistic Backtest", type="primary", use_container_width=True, key="run_real")

        if run_real or "rot_real_results" in st.session_state:
            if run_real:
                eq_tk = rot_eq_ticker.split(" ")[0]
                gold_tk = rot_gold_ticker.split(" ")[0]

                with st.spinner(f"Downloading {eq_tk}, {gold_tk}, TQQQ, SPY..."):
                    import yfinance as yf
                    eq_ret, gold_ret = download_rotation_data(
                        start=str(rot_real_start), end=str(rot_real_end),
                        equity_ticker=eq_tk, gold_ticker=gold_tk,
                    )
                    # SPY
                    spy_data = yf.download("SPY", start=str(rot_real_start), end=str(rot_real_end), progress=False)
                    if not spy_data.empty:
                        spy_close = spy_data["Close"].squeeze()
                        if spy_close.index.tz is None:
                            spy_close.index = spy_close.index.tz_localize("UTC")
                        spy_curve = spy_close / spy_close.iloc[0] * rot_capital
                    else:
                        spy_curve = pd.Series(dtype=float)
                    # TQQQ
                    tqqq_close = download_tqqq_data(start=str(rot_real_start), end=str(rot_real_end))
                    if not tqqq_close.empty:
                        tqqq_curve = tqqq_close / tqqq_close.iloc[0] * rot_capital
                    else:
                        tqqq_curve = pd.Series(dtype=float)

                if eq_ret.empty or gold_ret.empty:
                    st.error("Download failed.")
                else:
                    st.info(f"Data: {eq_ret.index[0].strftime('%Y-%m-%d')} -> {eq_ret.index[-1].strftime('%Y-%m-%d')} "
                            f"({len(eq_ret)} days)")

                    rot_params = _build_rot_params(str(rot_real_start), str(rot_real_end))
                    with st.spinner("Running backtest..."):
                        result = run_rotation_backtest(eq_ret, gold_ret, rot_params)
                    result["spy_curve"] = spy_curve
                    result["tqqq_curve"] = tqqq_curve
                    result["eq_ret"] = eq_ret
                    result["gold_ret"] = gold_ret
                    st.session_state["rot_real_results"] = result

            if "rot_real_results" in st.session_state:
                _render_rotation_results(
                    st.session_state["rot_real_results"], "real",
                    rot_lev_eq, rot_lev_gold, rot_rf,
                    show_tqqq=True, is_experimental=False,
                )

                # ── Sensitivity analysis ──
                st.divider()
                st.subheader("Sensitivity Analysis — Vol Threshold")

                sens_col1, sens_col2 = st.columns(2)
                with sens_col1:
                    sens_mode = st.selectbox("Sweep mode", ["quantile", "absolute"], key="sens_mode")
                with sens_col2:
                    if sens_mode == "quantile":
                        sens_vals_str = st.text_input("Quantile values (comma-sep)",
                                                       "0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90",
                                                       key="sens_vals_q")
                    else:
                        sens_vals_str = st.text_input("Abs vol values % (comma-sep)",
                                                       "14, 16, 18, 20, 22, 25, 30",
                                                       key="sens_vals_a")

                if st.button("Run Sensitivity", key="run_sens"):
                    try:
                        vals = [float(x.strip()) for x in sens_vals_str.split(",")]
                        if sens_mode == "absolute":
                            vals = [v / 100 for v in vals]
                    except ValueError:
                        st.error("Invalid values.")
                        vals = None

                    if vals:
                        rr = st.session_state["rot_real_results"]
                        base_p = _build_rot_params(str(rot_real_start), str(rot_real_end))
                        with st.spinner(f"Running {len(vals)} backtests..."):
                            sens_df = run_sensitivity_analysis(
                                rr["eq_ret"], rr["gold_ret"], base_p,
                                thresholds=vals, mode=sens_mode,
                            )
                        st.session_state["sens_results"] = sens_df

                if "sens_results" in st.session_state:
                    sens_df = st.session_state["sens_results"]
                    disp = sens_df.copy()
                    for c in ["cagr", "max_drawdown", "volatility_ann", "time_equity", "time_gold"]:
                        if c in disp.columns:
                            disp[c] = disp[c].apply(lambda x: f"{x:.2%}")
                    for c in ["sharpe", "sortino", "calmar"]:
                        if c in disp.columns:
                            disp[c] = disp[c].apply(lambda x: f"{x:.2f}")
                    if "threshold" in disp.columns:
                        if sens_mode == "quantile":
                            disp["threshold"] = disp["threshold"].apply(lambda x: f"{x:.0%}")
                        else:
                            disp["threshold"] = disp["threshold"].apply(lambda x: f"{x:.1%}")
                    disp["n_switches"] = disp["n_switches"].astype(int)
                    st.dataframe(disp, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════
    # SUB-TAB 2: EXPERIMENTAL (1985+ synthetic leverage)
    # ══════════════════════════════════════════════════════════
    with rot_sub_exp:
        st.subheader("Experimental Long-Term (1985+ synthetic)")
        st.caption("Nasdaq 100: from 1985 | Gold: from 1975 | Leveraged ETFs: synthetic (did not exist)")

        e_col1, e_col2 = st.columns(2)
        with e_col1:
            rot_exp_start = st.date_input("Start", pd.Timestamp("1985-10-01"), key="rot_e_start",
                                           help="Nasdaq 100 index starts Oct 1985")
            rot_exp_end = st.date_input("End", pd.Timestamp("2025-12-31"), key="rot_e_end")
        with e_col2:
            rot_exp_eq = st.selectbox("Equity", [
                "^NDX (Nasdaq 100, from 1985)",
                "^GSPC (S&P 500, from 1975)",
            ], key="rot_e_eq")
            rot_exp_gold = st.selectbox("Gold", ["GC=F (Gold Futures, from ~1975)"], key="rot_e_gold")

        run_exp = st.button("Run Experimental Backtest", type="secondary", use_container_width=True, key="run_exp")

        if run_exp or "rot_exp_results" in st.session_state:
            if run_exp:
                eq_tk = rot_exp_eq.split(" ")[0]
                gold_tk = rot_exp_gold.split(" ")[0]

                # Enforce minimum dates
                min_start = pd.Timestamp("1985-10-01") if "NDX" in eq_tk else pd.Timestamp("1975-01-01")
                actual_start = max(pd.Timestamp(str(rot_exp_start)), min_start)

                with st.spinner(f"Downloading long history for {eq_tk} and {gold_tk}..."):
                    import yfinance as yf
                    eq_ret, gold_ret = download_rotation_data(
                        start=str(actual_start.date()), end=str(rot_exp_end),
                        equity_ticker=eq_tk, gold_ticker=gold_tk,
                    )
                    # SPY for comparison (only available from ~1993)
                    spy_data = yf.download("SPY", start=str(actual_start.date()), end=str(rot_exp_end), progress=False)
                    if not spy_data.empty:
                        spy_close = spy_data["Close"].squeeze()
                        if spy_close.index.tz is None:
                            spy_close.index = spy_close.index.tz_localize("UTC")
                        spy_curve = spy_close / spy_close.iloc[0] * rot_capital
                    else:
                        spy_curve = pd.Series(dtype=float)

                if eq_ret.empty or gold_ret.empty:
                    st.error("Download failed.")
                else:
                    st.info(f"Data: {eq_ret.index[0].strftime('%Y-%m-%d')} -> {eq_ret.index[-1].strftime('%Y-%m-%d')} "
                            f"({len(eq_ret)} days)")

                    rot_params = _build_rot_params(str(actual_start.date()), str(rot_exp_end))
                    with st.spinner("Running experimental backtest..."):
                        result = run_rotation_backtest(eq_ret, gold_ret, rot_params)
                    result["spy_curve"] = spy_curve
                    result["tqqq_curve"] = pd.Series(dtype=float)  # no TQQQ for experimental
                    st.session_state["rot_exp_results"] = result

            if "rot_exp_results" in st.session_state:
                _render_rotation_results(
                    st.session_state["rot_exp_results"], "exp",
                    rot_lev_eq, rot_lev_gold, rot_rf,
                    show_tqqq=False, is_experimental=True,
                )

    # ══════════════════════════════════════════════════════════
    # SUB-TAB 3: DOCUMENTATION
    # ══════════════════════════════════════════════════════════
    with rot_sub_doc:
        st.subheader("Hypotheses, Sources & Limitations")

        st.markdown("""
### Data Sources
| Asset | Source | Available From |
|-------|--------|----------------|
| Nasdaq 100 | ^NDX (Yahoo Finance) | Oct 1985 |
| S&P 500 | ^GSPC (Yahoo Finance) | ~1950 |
| Gold Spot | GC=F futures (Yahoo Finance) | ~1975 |
| GLD ETF | GLD (Yahoo Finance) | Nov 2004 |
| TQQQ (3x Nasdaq) | TQQQ (Yahoo Finance) | Feb 2010 |

### Leverage Construction (Synthetic)
For periods before leveraged ETFs existed, we construct synthetic daily returns:

**r_lev_t = L x r_underlying_t - drag_t**

Where `drag_t` depends on the drag model:
- **Variable drag** (recommended): `drag = (Fed Funds rate + fixed margin) x (L - 1) / 252`
  - Only the financed portion (L-1) pays borrowing cost
  - Fed Funds rate approximated from FRED FEDFUNDS historical averages
  - Fixed margin default: 2% (covers ETF fees + replication error)
- **Flat drag**: fixed annual percentage divided by 252

### Leverage Profiles
| Profile | Equity | Gold | MaxDD Target |
|---------|--------|------|--------------|
| **Aggro** | 3x | 3x | Can exceed -50% (experimental) |
| **Normal** | 2x | 1.5x | >= -40% |
| **Light** | 1.5x | 1x | >= -35% |

### Volatility Signal
- **Realized vol**: rolling std of daily returns x sqrt(252)
- **Quantile modes**:
  - `expanding`: growing window from start — no look-ahead, adapts to new vol regimes
  - `rolling`: fixed lookback (default 756 days ~3 years) — responsive to recent regime changes
  - `global`: full history quantile — WARNING: uses future information, use for reference only
- **Absolute threshold**: fixed level (e.g. 20% annualized) — simplest, no look-ahead risk

### Key Limitations
1. **Survivorship bias on Nasdaq**: Nasdaq 100 composition changed significantly over time
2. **Leveraged ETF path dependency**: synthetic leverage assumes daily rebalanced leverage, which suffers from
   volatility drag (variance drain). Real ETFs have additional tracking error.
3. **Gold pre-1975**: gold price was fixed ($35/oz) until 1971, free-floating from 1975.
   Backtests before 1975 are meaningless for gold.
4. **Transaction costs**: we model a fixed switch cost per rebalance, but real costs include
   bid-ask spread, market impact, and potential slippage on leveraged products
5. **Tax impact**: not modeled. Monthly switches on leveraged ETFs generate short-term capital gains.
6. **Fed Funds proxy**: our drag model uses approximate historical rates, not exact daily FEDFUNDS series

### References
- Moreira & Muir (2017) — *Volatility-Managed Portfolios*
- Baur & Lucey (2010) — *Is Gold a Hedge or a Safe Haven?*
- Risk parity literature (Bridgewater / RPAR / All Seasons)
- r/LETFs community — Leveraged Rotation Strategy (LRS) research
        """)

        st.info(
            f"**Current config**: {rot_profile} profile | "
            f"Equity {rot_lev_eq:.1f}x / Gold {rot_lev_gold:.1f}x | "
            f"Drag: {'variable (FF+margin)' if rot_use_var_drag else 'flat'} | "
            f"Vol: {rot_vol_window}d window, "
            f"{'abs ' + f'{rot_abs_thresh:.0%}' if rot_use_abs else f'q{rot_vol_q:.0%} ({rot_q_mode})'} | "
            f"Lever-down: {'ON' if rot_lever_down else 'OFF'}"
        )


# ══════════════════════════════════════════════════════════════
# TAB 4: COMPARAISON (all strategies head-to-head)
# ══════════════════════════════════════════════════════════════
with tab_compare:

    # ── Dark theme CSS ──
    st.markdown("""
    <style>
    .compare-container {
        background-color: #1a1a2e;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .tf-buttons {
        display: flex;
        gap: 0;
        margin-bottom: 1rem;
    }
    .tf-buttons button {
        flex: 1;
        background: #16213e;
        color: #e0e0e0;
        border: 1px solid #0f3460;
        padding: 0.5rem 1rem;
        cursor: pointer;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .tf-buttons button:first-child { border-radius: 8px 0 0 8px; }
    .tf-buttons button:last-child { border-radius: 0 8px 8px 0; }
    .tf-buttons button.active {
        background: #e94560;
        color: white;
        border-color: #e94560;
    }
    .metrics-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Segoe UI', sans-serif;
        font-size: 0.85rem;
    }
    .metrics-table th {
        background: #16213e;
        color: #a0a0b0;
        padding: 0.6rem 0.8rem;
        text-align: right;
        border-bottom: 2px solid #0f3460;
        font-weight: 600;
    }
    .metrics-table th:first-child { text-align: left; }
    .metrics-table td {
        padding: 0.5rem 0.8rem;
        text-align: right;
        border-bottom: 1px solid #0f3460;
        color: #e0e0e0;
    }
    .metrics-table td:first-child {
        text-align: left;
        font-weight: 600;
        color: #ffffff;
    }
    .metrics-table tr:hover { background: #16213e40; }
    .metrics-table .positive { color: #4ade80; }
    .metrics-table .negative { color: #f87171; }
    .section-title {
        color: #e0e0e0;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("## Backtest — Comparaison de Strategies")

    # ── Timeframe buttons ──
    tf_cols = st.columns(6)
    timeframes = {"Max": None, "1A": 1, "3A": 3, "5A": 5, "10A": 10, "20A": 20}
    selected_tf = st.session_state.get("compare_tf", "Max")
    for i, (label, _) in enumerate(timeframes.items()):
        if tf_cols[i].button(label, key=f"tf_{label}", use_container_width=True,
                              type="primary" if selected_tf == label else "secondary"):
            st.session_state["compare_tf"] = label
            selected_tf = label

    # ── Settings ──
    cmp_col1, cmp_col2 = st.columns([1, 3])
    with cmp_col1:
        cmp_capital = st.number_input("Capital initial ($)", value=100_000, step=10_000, key="cmp_cap")
        cmp_include_qv = st.checkbox("Quality/Value", value=True, key="cmp_qv")
        cmp_include_rot = st.checkbox("Rotation (2x/2x)", value=True, key="cmp_rot")
        cmp_include_spy = st.checkbox("SPY Buy & Hold", value=True, key="cmp_spy")
        cmp_include_tqqq = st.checkbox("TQQQ Buy & Hold", value=True, key="cmp_tqqq")
        cmp_include_ndx = st.checkbox("Nasdaq (1x)", value=False, key="cmp_ndx")

    run_cmp = st.button("Lancer la comparaison", type="primary", use_container_width=True, key="run_cmp")

    if run_cmp or "cmp_results" in st.session_state:
        if run_cmp:
            import yfinance as yf

            # Compute date range from timeframe
            end_date = pd.Timestamp.now(tz="UTC")
            tf_years = timeframes[selected_tf]
            if tf_years is not None:
                start_date = end_date - pd.DateOffset(years=tf_years)
            else:
                start_date = pd.Timestamp("2010-03-01", tz="UTC")

            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            strategies = {}

            with st.spinner(f"Downloading data ({selected_tf})..."):
                # ── SPY ──
                if cmp_include_spy:
                    spy_data = yf.download("SPY", start=start_str, end=end_str, progress=False)
                    if not spy_data.empty:
                        spy_close = spy_data["Close"].squeeze()
                        if spy_close.index.tz is None:
                            spy_close.index = spy_close.index.tz_localize("UTC")
                        strategies["SPY"] = {
                            "curve": spy_close / spy_close.iloc[0] * cmp_capital,
                            "color": "#4CAF50",
                            "trades": 0,
                        }

                # ── TQQQ ──
                if cmp_include_tqqq:
                    tqqq_data = yf.download("TQQQ", start=start_str, end=end_str, progress=False)
                    if not tqqq_data.empty:
                        tqqq_close = tqqq_data["Close"].squeeze()
                        if tqqq_close.index.tz is None:
                            tqqq_close.index = tqqq_close.index.tz_localize("UTC")
                        strategies["TQQQ B&H"] = {
                            "curve": tqqq_close / tqqq_close.iloc[0] * cmp_capital,
                            "color": "#E91E63",
                            "trades": 0,
                        }

                # ── Nasdaq 1x ──
                if cmp_include_ndx:
                    ndx_data = yf.download("^NDX", start=start_str, end=end_str, progress=False)
                    if not ndx_data.empty:
                        ndx_close = ndx_data["Close"].squeeze()
                        if ndx_close.index.tz is None:
                            ndx_close.index = ndx_close.index.tz_localize("UTC")
                        strategies["Nasdaq (1x)"] = {
                            "curve": ndx_close / ndx_close.iloc[0] * cmp_capital,
                            "color": "#9E9E9E",
                            "trades": 0,
                        }

                # ── Rotation ──
                if cmp_include_rot:
                    eq_ret, gold_ret = download_rotation_data(
                        start=start_str, end=end_str, equity_ticker="^NDX", gold_ticker="GC=F",
                    )
                    if not eq_ret.empty and not gold_ret.empty:
                        rot_params = RotationParams(
                            leverage_equity=2.0, leverage_gold=2.0,
                            floor_eq_weight=0.2,
                            vol_threshold_absolute=0.18,
                            min_months_between_switch=2,
                            use_variable_drag=True, fixed_margin=0.02,
                            start_date=start_str, end_date=end_str,
                            initial_capital=cmp_capital,
                        )
                        rot_result = run_rotation_backtest(eq_ret, gold_ret, rot_params)
                        strategies["Rotation (2x/2x)"] = {
                            "curve": rot_result["equity_curve"],
                            "color": "#2196F3",
                            "trades": rot_result["metrics"].get("n_regime_switches", 0),
                        }

                # ── Quality/Value ──
                if cmp_include_qv and df_prices is not None and df_funda is not None:
                    from backtest import run_backtest, BacktestParams
                    from selection import SelectionParams
                    qv_params = BacktestParams(
                        start_date=start_str, end_date=end_str,
                        rebalance_freq_months=12,
                        transaction_cost=0.001,
                        pub_lag_months=0,
                        initial_capital=cmp_capital,
                        selection=SelectionParams(
                            min_market_cap=0, pb_percentile_cut=0.20,
                            min_f_score=8, n_stocks=15,
                        ),
                    )
                    qv_eq, _, qv_trades, _ = run_backtest(df_prices, df_funda, qv_params)
                    n_qv_trades = len(qv_trades) if len(qv_trades) > 0 else 0
                    strategies["Quality/Value"] = {
                        "curve": qv_eq,
                        "color": "#FF9800",
                        "trades": n_qv_trades,
                    }

            # ── Compute metrics for each ──
            mp = MetricsParams(risk_free_rate=0.02, periods_per_year=252)
            mp_monthly = MetricsParams(risk_free_rate=0.02, periods_per_year=12)
            from metrics import compute_performance_metrics

            results = {}
            for name, data in strategies.items():
                curve = data["curve"]
                # Pick freq: daily if >50 points per year, else monthly
                n_per_year = len(curve) / max(1, (curve.index[-1] - curve.index[0]).days / 365.25)
                p = mp if n_per_year > 50 else mp_monthly
                m = compute_performance_metrics(curve, params=p)
                m["trades"] = data["trades"]
                m["final_value"] = float(curve.iloc[-1])
                results[name] = {"metrics": m, "curve": curve, "color": data["color"]}

            st.session_state["cmp_results"] = results
            st.session_state["cmp_start"] = start_str
            st.session_state["cmp_end"] = end_str
            st.session_state["cmp_tf_label"] = selected_tf
            st.session_state["cmp_capital_used"] = cmp_capital

        res = st.session_state["cmp_results"]
        tf_label = st.session_state.get("cmp_tf_label", "Max")
        cap_used = st.session_state.get("cmp_capital_used", 100_000)

        # ══════════════════════════════════════════════════
        # CHART: log scale equity curves (dark themed)
        # ══════════════════════════════════════════════════
        st.markdown(f'<div class="section-title">Evolution du portefeuille — {tf_label}</div>',
                    unsafe_allow_html=True)

        fig_cmp = go.Figure()
        for name, data in res.items():
            curve = data["curve"]
            fig_cmp.add_trace(go.Scatter(
                x=curve.index, y=curve.values,
                name=name,
                line=dict(color=data["color"], width=2.5),
                hovertemplate=f"{name}<br>Date: %{{x|%Y-%m-%d}}<br>Value: $%{{y:,.0f}}<extra></extra>",
            ))

        fig_cmp.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
            height=550,
            yaxis=dict(
                type="log",
                title="Valeur du portefeuille ($)",
                gridcolor="#0f3460",
                tickprefix="$",
                tickformat=",.0f",
            ),
            xaxis=dict(
                title="Date",
                gridcolor="#0f3460",
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                font=dict(size=12),
            ),
            margin=dict(l=60, r=20, t=40, b=40),
            font=dict(color="#e0e0e0"),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # ══════════════════════════════════════════════════
        # METRICS TABLE (HTML, styled like the screenshot)
        # ══════════════════════════════════════════════════
        st.markdown(f'<div class="section-title">Metriques de performance — {tf_label}</div>',
                    unsafe_allow_html=True)

        # Build HTML table
        columns = [
            ("Strategie", "name", None),
            ("Valeur finale", "final_value", "${:,.0f}"),
            ("CAGR", "cagr", "{:.1%}"),
            ("Max Drawdown", "max_drawdown", "{:.1%}"),
            ("Duree underwater", "max_dd_duration_periods", "{:.0f}j"),
            ("Volatilite", "volatility_ann", "{:.1%}"),
            ("Sharpe Ratio", "sharpe", "{:.2f}"),
            ("Sortino", "sortino", "{:.2f}"),
            ("Calmar", "calmar", "{:.2f}"),
            ("Trades", "trades", "{:.0f}"),
        ]

        header = "".join(f'<th>{c[0]}</th>' for c in columns)
        rows_html = ""

        for name, data in res.items():
            m = data["metrics"]
            cells = ""
            for col_name, key, fmt in columns:
                if key == "name":
                    val_html = f'<td style="color:{data["color"]};font-weight:700">{name}</td>'
                else:
                    raw = m.get(key, 0)
                    if isinstance(raw, float) and np.isnan(raw):
                        val_html = '<td>N/A</td>'
                    else:
                        formatted = fmt.format(raw) if fmt else str(raw)
                        # Color positive/negative
                        css_class = ""
                        if key in ["cagr", "sharpe", "sortino", "calmar"]:
                            css_class = "positive" if raw > 0 else "negative"
                        elif key == "max_drawdown":
                            css_class = "negative" if raw < -0.30 else ("positive" if raw > -0.15 else "")
                        val_html = f'<td class="{css_class}">{formatted}</td>'
                cells += val_html
            rows_html += f"<tr>{cells}</tr>"

        table_html = f"""
        <div class="compare-container">
            <table class="metrics-table">
                <thead><tr>{header}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """
        st.markdown(table_html, unsafe_allow_html=True)

        # ── Drawdown comparison chart ──
        st.markdown(f'<div class="section-title">Drawdown — {tf_label}</div>',
                    unsafe_allow_html=True)

        fig_dd = go.Figure()
        for name, data in res.items():
            curve = data["curve"]
            peak = curve.cummax()
            dd = (curve - peak) / peak
            fig_dd.add_trace(go.Scatter(
                x=dd.index, y=dd.values,
                name=name, fill="tozeroy",
                line=dict(color=data["color"], width=1.5),
                fillcolor=data["color"].replace(")", ",0.1)").replace("rgb", "rgba") if "rgb" in data["color"] else None,
            ))

        fig_dd.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
            height=300,
            yaxis=dict(title="Drawdown", tickformat=".0%", gridcolor="#0f3460"),
            xaxis=dict(gridcolor="#0f3460"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=60, r=20, t=30, b=30),
            font=dict(color="#e0e0e0"),
        )
        st.plotly_chart(fig_dd, use_container_width=True)
