
# --- robust import shim (works on Streamlit Cloud & locally)
import os, sys
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import engine as a module to avoid import list issues
import portfolio_engine as eng

# Matplotlib fallback
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False
    plt = None

st.set_page_config(page_title="Portfolio Strategy App", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.title("Portfolio Controls")
total_usd = st.sidebar.number_input("Total Portfolio (USD)", min_value=0.0, value=10000.0, step=100.0)
sector_cap = st.sidebar.slider("Sector cap", 0.10, 0.30, 0.20, 0.01)
position_cap = st.sidebar.slider("Position cap", 0.02, 0.08, 0.05, 0.01)

active_choice = st.sidebar.radio("Active Portfolio", ["Mid-D", "Mid-DV"], index=0)
if "Mid-D" not in st.session_state:
    st.session_state["Mid-D"] = eng.ensure_portfolio_df("Mid-D")
if "Mid-DV" not in st.session_state:
    st.session_state["Mid-DV"] = eng.ensure_portfolio_df("Mid-DV")

def get_active_df():
    return st.session_state[active_choice]

# ---------------- Tabs ----------------
tabs = st.tabs(["Overview", "Mid-D (Edit)", "Mid-DV (Edit)", "Fib Calculator", "Seasonality", "Balance Plan", "Rotation", "Optimizer", "Stress & MC", "Alerts & Notes"])
tab_overview, tab_mid_d, tab_mid_dv, tab_fib, tab_season, tab_balance, tab_rotation, tab_opt, tab_stress, tab_alerts = tabs

# ---------------- Overview ----------------
with tab_overview:
    st.header("Overview")
    st.write(f"**Active portfolio:** {active_choice}")
    df = get_active_df().copy()
    st.dataframe(df)

    tickers = df["ticker"].dropna().astype(str).str.upper().unique().tolist()
    prices = eng.fetch_current_prices(tickers) if tickers else {}
    perf = eng.summarize_portfolio(df, total_usd, current_prices=prices)

    st.subheader("By Stock (current vs target)")
    st.dataframe(perf["by_stock"])

    st.subheader("By Sector (current vs target)")
    st.dataframe(perf["by_sector"])

    ok_pos = eng.position_caps_ok(perf["by_stock"], cap=position_cap)
    ok_sec = eng.sector_caps_ok(perf["by_sector"], cap=sector_cap)
    st.info(f"Position caps OK: {ok_pos} | Sector caps OK: {ok_sec}")

# ---------------- Mid-D (Edit) ----------------
with tab_mid_d:
    st.header("Mid-D (Edit)")
    st.caption("Type tickers, names, sectors; edit allocations, shares.")
    st.session_state["Mid-D"] = st.data_editor(st.session_state["Mid-D"], num_rows="dynamic")

# ---------------- Mid-DV (Edit) ----------------
with tab_mid_dv:
    st.header("Mid-DV (Edit)")
    st.caption("Type tickers, names, sectors; edit allocations, shares.")
    st.session_state["Mid-DV"] = st.data_editor(st.session_state["Mid-DV"], num_rows="dynamic")

# ---------------- Fib Calculator ----------------
with tab_fib:
    st.header("Fib Calculator (12 months)")
    df = get_active_df().copy()
    tickers = df["ticker"].dropna().astype(str).str.upper().tolist()
    if not tickers:
        st.info("Add tickers to the active portfolio first.")
    else:
        px = eng.fetch_live_prices(tickers, period="1y", interval="1d")
        rows = []
        for t in tickers:
            dft = px[px["ticker"]==t].sort_values("date")
            if dft.empty: 
                continue
            fibs = eng.compute_fibs_12m(dft)
            cp = float(dft["close"].iloc[-1])
            wz = eng.weighted_entry_zone(cp, fibs)
            rows.append({"ticker":t, "last":cp, **fibs, **wz})
        st.dataframe(pd.DataFrame(rows))

# ---------------- Seasonality ----------------
with tab_season:
    st.header("Seasonality")
    df = get_active_df().copy()
    tickers = df["ticker"].dropna().astype(str).str.upper().tolist()
    if not tickers:
        st.info("Add tickers to the active portfolio first.")
    else:
        px = eng.fetch_live_prices(tickers, period="5y", interval="1d")
        stats = eng.monthly_seasonality(px)
        st.dataframe(stats)

# ---------------- Balance Plan ----------------
with tab_balance:
    st.header("Balance Plan (caps-aware)")
    df = get_active_df().copy()
    tickers = df["ticker"].dropna().astype(str).str.upper().tolist()
    prices = eng.fetch_current_prices(tickers) if tickers else {}
    plan_by_pos, plan_by_sector = eng.rebalance_plan_usd(df, prices, total_usd, sector_cap=sector_cap, pos_cap=position_cap)
    st.subheader("By Position")
    st.dataframe(plan_by_pos)
    st.subheader("By Sector")
    st.dataframe(plan_by_sector)

# ---------------- Rotation ----------------
with tab_rotation:
    st.header("Rotation (Predictive Tilt)")
    df = get_active_df().copy()
    tickers = df["ticker"].dropna().astype(str).str.upper().tolist()
    if not tickers:
        st.info("Add tickers to the active portfolio first.")
    else:
        # Simple sector bias from current month seasonality
        px5 = eng.fetch_live_prices(tickers, period="5y", interval="1d")
        cur_m = pd.Timestamp.today().month
        stats = eng.monthly_seasonality(px5) if not px5.empty else pd.DataFrame()
        mapping = df[["ticker","sector"]].dropna()
        sec_bias = {}
        if not stats.empty:
            sec_stats = eng.seasonality_by_sector(stats, mapping)
            cur = sec_stats[sec_stats["month"]==cur_m][["sector","avg_return"]].dropna()
            if not cur.empty:
                for _, r in cur.iterrows():
                    ar = float(r["avg_return"])
                    sec_bias[r["sector"]] = float(max(0.5, min(1.5, 1.0 + 2.0*ar)))
        for s in mapping["sector"].unique().tolist():
            sec_bias.setdefault(s, 1.0)
        # Apply predictive layer
        sectors_list = mapping["sector"].dropna().unique().tolist()
        sec_bias = eng.apply_predictive_bias_to_sectors(sec_bias, sectors_list)

        prices = eng.fetch_current_prices(tickers) if tickers else {}
        plan_by_pos, plan_by_sector = eng.rebalance_plan_usd(df, prices, total_usd, sector_cap=sector_cap, pos_cap=position_cap, sector_bias=sec_bias)
        st.subheader("Rotation Plan — By Position")
        st.dataframe(plan_by_pos)
        st.subheader("Rotation Plan — By Sector")
        st.dataframe(plan_by_sector)

# ---------------- Optimizer ----------------
with tab_opt:
    st.header("Mean-Variance Optimizer (cap-constrained)")
    df = get_active_df().copy()
    tickers = df["ticker"].dropna().astype(str).str.upper().tolist()
    if not tickers:
        st.info("Add tickers to the active portfolio first.")
    else:
        px2y = eng.fetch_live_prices(tickers, period="2y", interval="1d")
        if px2y.empty:
            st.info("Not enough price data for optimization.")
        else:
            risk_av = st.slider("Risk aversion (higher = safer)", 1.0, 10.0, 3.0, 0.5)
            raw = eng.mean_variance_opt(px2y, tickers, risk_aversion=risk_av)
            sec_map = df.set_index("ticker")["sector"].to_dict()
            w_cap = eng.cap_constrain(raw, sec_map, pos_cap=position_cap, sector_cap=sector_cap)
            st.subheader("Optimized Weights (cap-constrained)")
            st.dataframe(pd.DataFrame([w_cap]).T.rename(columns={0:"weight"}))

# ---------------- Stress & MC ----------------
with tab_stress:
    st.header("Stress Tests & Monte Carlo")
    df = get_active_df().copy()
    tickers = df["ticker"].dropna().astype(str).str.upper().tolist()
    if not tickers:
        st.info("Add tickers to the active portfolio first.")
    else:
        px2y = eng.fetch_live_prices(tickers, period="2y", interval="1d")
        prices = eng.fetch_current_prices(tickers)
        perf = eng.summarize_portfolio(df, total_usd, current_prices=prices)
        weights = perf["by_stock"].set_index("ticker")["current_%"].to_dict()
        if px2y.empty or not weights:
            st.info("Not enough data for stress tests.")
        else:
            st.subheader("Shock Scenarios")
            scenarios = eng.shock_scenarios(px2y, weights)
            st.dataframe(pd.DataFrame(list(scenarios.items()), columns=["Scenario","Est. Return"]))

            st.subheader("Monte Carlo (21 trading days)")
            mc = eng.monte_carlo_projection(px2y, weights, days=21, trials=1000)
            st.json(mc)

# ---------------- Alerts & Notes ----------------
with tab_alerts:
    st.header("Alerts & Notes")
    eng.init_notes("data/app.db")
    df = get_active_df().copy()
    tickers = df["ticker"].dropna().astype(str).str.upper().tolist()
    st.write("Add a Note")
    c1, c2 = st.columns([1,3])
    with c1:
        tsel = st.selectbox("Ticker", options=tickers if tickers else ["N/A"])
    with c2:
        note = st.text_input("Note text")
    if st.button("Save Note"):
        if note.strip() and tsel != "N/A":
            eng.add_note(tsel, note.strip(), db_path="data/app.db")
            st.success("Saved.")
    st.subheader("Notes")
    st.dataframe(eng.fetch_notes("data/app.db"))

st.caption("Research/education only. Not financial advice.")
