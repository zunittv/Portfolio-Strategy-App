
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from portfolio_engine import (
    compute_fibs_12m, compute_macd, sector_caps_ok, position_caps_ok,
    weighted_entry_zone, five_percent_flow, summarize_portfolio, suggest_rebalance,
    signals_from_purchase, suggest_candidates_stub, accuracy_tracker_stub
)

st.set_page_config(page_title="Portfolio Strategy App – Prototype", layout="wide")
st.title("📊 Portfolio Strategy App – Prototype")
st.caption("For research & education. Not financial advice.")

with st.sidebar:
    st.header("Inputs")
    total_cad = st.number_input("Total Portfolio (CAD)", value=4400.0, min_value=0.0, step=100.0)
    st.caption("Upload portfolio & price data below")

    port_file = st.file_uploader("Upload Portfolio CSV", type=["csv"])
    prices_file = st.file_uploader("Upload Prices CSV", type=["csv"])

st.write("### 1) Portfolio")
if port_file is not None:
    portfolio = pd.read_csv(port_file)
else:
    st.info("Using sample portfolio (upload to override).")
    portfolio = pd.read_csv("app/sample_portfolio.csv")

st.dataframe(portfolio, use_container_width=True)

st.write("### 2) Prices")
if prices_file is not None:
    prices = pd.read_csv(prices_file, parse_dates=["date"])
else:
    st.info("Using sample prices (upload to override).")
    prices = pd.read_csv("app/sample_prices.csv", parse_dates=["date"])
st.dataframe(prices.head(50), use_container_width=True)

# Compute % of total and sector totals
perf = summarize_portfolio(portfolio, total_cad)
st.write("#### Allocation Summary")
col1, col2 = st.columns(2)
with col1:
    st.dataframe(perf["by_stock"], use_container_width=True)
with col2:
    st.dataframe(perf["by_sector"], use_container_width=True)

# Check constraints
ok_sector = sector_caps_ok(perf["by_sector"])
ok_pos = position_caps_ok(perf["by_stock"])
st.write("#### Constraint Checks")
st.write(f"Sector caps (≤ 20% each): {'✅ OK' if ok_sector else '⚠️ Exceeds 20%'}")
st.write(f"Position caps (≤ 5% each): {'✅ OK' if ok_pos else '⚠️ Exceeds 5%'}")

st.write(\"---\")
st.write(\"### 3) Technicals & Zones\")
# Compute Fibs & MACD on last 252 trading days per ticker, plus current-weighted entry zone
tickers = portfolio[\"ticker\"].unique().tolist()
fib_frames = []

for t in tickers:
    df_t = prices.loc[prices[\"ticker\"]==t].sort_values(\"date\").copy()
    if len(df_t) < 30:
        continue
    fibs = compute_fibs_12m(df_t)
    macd_df = compute_macd(df_t)
    latest = df_t.iloc[-1][\"close\"]
    weighted = weighted_entry_zone(latest, fibs)
    fib_frames.append(pd.DataFrame({
        \"ticker\":[t],
        \"latest_close\":[latest],
        \"fib_23.6\":[fibs[\"23.6\"]],
        \"fib_38.2\":[fibs[\"38.2\"]],
        \"fib_50.0\":[fibs[\"50.0\"]],
        \"fib_61.8\":[fibs[\"61.8\"]],
        \"fib_78.6\":[fibs[\"78.6\"]],
        \"weighted_entry\":[weighted[\"weighted_entry\"]],
        \"dynamic_entry_zone\":[weighted[\"dynamic_entry_zone\"]],
        \"dynamic_support_zone\":[weighted[\"dynamic_support_zone\"]],
        \"dynamic_resistance_zone\":[weighted[\"dynamic_resistance_zone\"]]
    }))

if fib_frames:
    st.dataframe(pd.concat(fib_frames, ignore_index=True), use_container_width=True)

st.write(\"---\")
st.write(\"### 4) Strategy Engine\")
rebalance_suggestions = suggest_rebalance(portfolio, prices)
st.dataframe(rebalance_suggestions, use_container_width=True)

st.write(\"---\")
st.write(\"### 5) Signals from Purchase Dates\")
st.caption(\"Add a 'purchase_date' column (YYYY-MM-DD) and 'purchase_price' to your portfolio CSV to compare.\")
if {\"purchase_date\",\"purchase_price\"}.issubset(portfolio.columns):
    sigs = signals_from_purchase(portfolio, prices)
    st.dataframe(sigs, use_container_width=True)

st.write(\"---\")
st.write(\"### 6) Candidate Suggestions (Stub)\")
cands = suggest_candidates_stub(prices, min_days=120)
st.dataframe(cands, use_container_width=True)

st.write(\"---\")
st.write(\"### 7) Accuracy Tracker (Stub)\")
acc = accuracy_tracker_stub()
st.json(acc)
