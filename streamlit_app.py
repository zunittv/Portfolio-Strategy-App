
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from portfolio_engine import (
    fetch_live_prices, get_news_yf, get_news_newsapi, summarize_portfolio,
    sector_caps_ok, position_caps_ok, compute_fibs_12m, compute_macd,
    weighted_entry_zone, decide_action, rank_daily_strategy,
    init_db, log_signal, fetch_signals, evaluate_signals, ensure_portfolio_template
)

st.set_page_config(page_title="Portfolio Strategy App – v3", layout="wide")
st.title("📊 Portfolio Strategy App – v3")
st.caption("For research & education. Not financial advice.")

with st.sidebar:
    st.header("Settings")
    total_cad = st.number_input("Total Portfolio (CAD)", value=4400.0, min_value=0.0, step=100.0)
    sector_cap = st.number_input("Sector cap (max %)", value=20.0, min_value=5.0, max_value=50.0, step=1.0) / 100.0
    position_cap = st.number_input("Position cap (max %)", value=5.0, min_value=1.0, max_value=10.0, step=0.5) / 100.0
    up_trigger = st.number_input("Trim trigger (gain % vs purchase)", value=5.0, step=0.5) / 100.0
    down_trigger = - st.number_input("Add trigger (loss % vs purchase)", value=5.0, step=0.5) / 100.0
    st.divider()
    newsapi_key = st.text_input("NewsAPI key (optional)", type="password")

st.write("### 1) Portfolio (Editable)")
if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = ensure_portfolio_template()

edited = st.data_editor(
    st.session_state.portfolio_df,
    num_rows="dynamic",
    use_container_width=True,
    key="portfolio_editor"
)
st.session_state.portfolio_df = edited

perf = summarize_portfolio(edited, total_cad)
st.write("#### Allocation Summary")
c1, c2 = st.columns(2)
with c1:
    st.dataframe(perf["by_stock"], use_container_width=True)
with c2:
    st.dataframe(perf["by_sector"], use_container_width=True)
st.write(f"Sector caps (≤ {int(sector_cap*100)}%): {'✅ OK' if sector_caps_ok(perf['by_sector'], sector_cap) else '⚠️ Exceeds'}")
st.write(f"Position caps (≤ {int(position_cap*100)}%): {'✅ OK' if position_caps_ok(perf['by_stock'], position_cap) else '⚠️ Exceeds'}")

st.write("---")
st.write("### 2) Live Data, Technicals & Zones")
tickers = edited["ticker"].dropna().astype(str).str.upper().unique().tolist()
if tickers:
    px = fetch_live_prices(tickers, period="1y", interval="1d")
    st.caption("Latest 1y daily prices fetched via yfinance.")
    st.dataframe(px.groupby("ticker").tail(3), use_container_width=True)
else:
    px = pd.DataFrame(columns=["date","ticker","close","high","low"])
    st.info("Add tickers to your portfolio to fetch live data.")

tech_rows = []
decision_rows = []
for t in tickers:
    df_t = px.loc[px["ticker"]==t].sort_values("date").copy()
    if df_t.empty:
        continue
    fibs = compute_fibs_12m(df_t)
    macd_df = compute_macd(df_t)
    macd_cross = macd_df.attrs.get("crossover","none")
    latest = df_t.iloc[-1]["close"]
    try:
        pprice = float(edited.loc[edited["ticker"]==t, "purchase_price"].values[0])
    except Exception:
        pprice = None
    action, notes = decide_action(latest, weighted_entry_zone(latest, fibs)["weighted_entry"], macd_cross, pprice, up_trigger, down_trigger)
    wz = weighted_entry_zone(latest, fibs)
    tech_rows.append({
        "ticker":t, "latest_close":latest,
        "fib_23.6":fibs["23.6"], "fib_38.2":fibs["38.2"], "fib_50.0":fibs["50.0"], "fib_61.8":fibs["61.8"], "fib_78.6":fibs["78.6"],
        "weighted_entry":wz["weighted_entry"],
        "entry_zone":wz["dynamic_entry_zone"],
        "support_zone":wz["dynamic_support_zone"],
        "resistance_zone":wz["dynamic_resistance_zone"],
        "macd_cross":macd_cross,
        "suggestion":action, "notes":notes, "price":latest
    })
    decision_rows.append({"ticker":t, "suggestion":action, "notes":notes, "price":latest})

if tech_rows:
    st.dataframe(pd.DataFrame(tech_rows), use_container_width=True)

st.write("---")
st.write("### 3) Daily Strategy")
if tech_rows:
    ranked = rank_daily_strategy(pd.DataFrame(tech_rows))
    st.dataframe(ranked[["ticker","score","latest_close","weighted_entry","macd_cross","entry_zone","support_zone","resistance_zone"]], use_container_width=True)
else:
    st.info("No technical rows to rank yet.")

st.write("---")
st.write("### 4) News")
if tickers:
    tabs = st.tabs(tickers)
    for i, t in enumerate(tickers):
        with tabs[i]:
            news = get_news_newsapi(t, newsapi_key) if newsapi_key else []
            if not news:
                news = get_news_yf(t)
            if not news:
                st.write("No recent articles found.")
            else:
                for n in news[:5]:
                    title = n.get("title","(no title)")
                    pub = n.get("publisher") or n.get("source","")
                    when = n.get("providerPublishTime") or n.get("publishedAt","")
                    link = n.get("link") or n.get("url","")
                    st.markdown(f"- [{title}]({link}) — {pub} {when}")
else:
    st.info("Add tickers to view recent headlines.")

st.write("---")
st.write("### 5) Decision Logger & Accuracy Tracker")
from portfolio_engine import init_db, log_signal, fetch_signals, evaluate_signals
init_db("data/app.db")
colA, colB = st.columns(2)
with colA:
    if tech_rows and st.button("Log All Suggestions as Signals"):
        for row in tech_rows:
            log_signal(row["ticker"], row["suggestion"], float(row["price"]), db_path="data/app.db")
        st.success("Logged current suggestions.")
    sigs = fetch_signals("data/app.db")
    st.write("Recent Signals")
    st.dataframe(sigs.head(25), use_container_width=True)
with colB:
    st.write("Evaluate Signals (1d/1w/1m/1y)")
    if st.button("Run Evaluation Now"):
        sigs = fetch_signals("data/app.db")
        results = evaluate_signals(sigs)
        st.dataframe(results, use_container_width=True)
        if not results.empty:
            st.download_button("Download Results CSV", data=results.to_csv(index=False), file_name="signal_accuracy.csv", mime="text/csv")

st.write("---")
st.caption("This app is for research/education. Not financial advice.")
