
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from portfolio_engine import (
    MID_D, MID_DV, ensure_portfolio_df, fetch_live_prices, fetch_current_prices, get_news_yf, get_news_newsapi, fetch_earnings_calendar,
    summarize_portfolio, sector_caps_ok, position_caps_ok,
    compute_fibs_12m, compute_macd, compute_rsi, compute_adx, weighted_entry_zone,
    monthly_seasonality, strong_weak_months, seasonality_by_sector,
    simple_sentiment, risk_score, composite_rank_row,
    rebalance_plan_usd, sector_position_targets_biased,
    init_db, log_signal, fetch_signals, evaluate_signals
)

st.set_page_config(page_title="Portfolio Strategy App – v5.1 (USD)", layout="wide")
st.title("📊 Portfolio Strategy App – v5.1 (USD)")
st.caption("For research & education. Not financial advice.")

with st.sidebar:
    st.header("Global Settings (USD)")
    total_usd = st.number_input("Total Portfolio (USD)", value=4400.0, min_value=0.0, step=100.0)
    sector_cap = st.number_input("Sector cap (max %)", value=20.0, min_value=5.0, max_value=50.0, step=1.0) / 100.0
    position_cap = st.number_input("Position cap (max %)", value=5.0, min_value=1.0, max_value=10.0, step=0.5) / 100.0
    up_trigger = st.number_input("Trim trigger (gain % vs purchase)", value=5.0, step=0.5) / 100.0
    down_trigger = - st.number_input("Add trigger (loss % vs purchase)", value=5.0, step=0.5) / 100.0
    st.divider()
    newsapi_key = st.text_input("NewsAPI key (optional)", type="password")
    bias_k = st.slider("Seasonality bias strength (k)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)

tabs = st.tabs(["Overview", "Mid-D", "Mid-DV", "Fib Calculator", "Seasonality", "Risk & Scores", "News & Calendar", "Balance Plan (Biased)", "Signals & Accuracy"])
tab_overview, tab_mid_d, tab_mid_dv, tab_fib, tab_season, tab_risk, tab_news, tab_balance, tab_signals = tabs

# ============ Overview ============
with tab_overview:
    st.subheader("Quick Start: Choose a Portfolio")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Load Mid-D"):
            st.session_state.portfolio_df = ensure_portfolio_df("Mid-D")
    with c2:
        if st.button("Load Mid-DV"):
            st.session_state.portfolio_df = ensure_portfolio_df("Mid-DV")

    st.write("### Editable Portfolio (USD)")
    if "portfolio_df" not in st.session_state:
        st.session_state.portfolio_df = ensure_portfolio_df("Mid-D")
    edited = st.data_editor(
        st.session_state.portfolio_df,
        num_rows="dynamic",
        use_container_width=True,
        key="portfolio_editor_v51"
    )
    st.session_state.portfolio_df = edited

    tickers = edited["ticker"].dropna().astype(str).str.upper().unique().tolist()
    current_prices = fetch_current_prices(tickers) if tickers else {}
    perf = summarize_portfolio(edited, total_usd, current_prices=current_prices)

    st.write("#### Allocation Summary (Targets & Current Values)")
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(perf["by_stock"], use_container_width=True)
    with c2:
        st.dataframe(perf["by_sector"], use_container_width=True)

    st.write(f"Sector caps (≤ {int(sector_cap*100)}%): {'✅ OK' if sector_caps_ok(perf['by_sector'], sector_cap) else '⚠️ Exceeds'}")
    st.write(f"Position caps (≤ {int(position_cap*100)}%): {'✅ OK' if position_caps_ok(perf['by_stock'], position_cap) else '⚠️ Exceeds'}")

def pull_prices_for_current(period="5y"):
    tickers = st.session_state.portfolio_df["ticker"].dropna().astype(str).str.upper().unique().tolist()
    if not tickers:
        return pd.DataFrame(columns=["date","ticker","close","high","low"])
    return fetch_live_prices(tickers, period=period, interval="1d")

# ============ Mid-D / Mid-DV ============
with tab_mid_d:
    st.subheader("Mid-D Stocks")
    st.write(pd.DataFrame(MID_D, columns=["Ticker","Company","Sector"]))

with tab_mid_dv:
    st.subheader("Mid-DV Stocks")
    st.write(pd.DataFrame(MID_DV, columns=["Ticker","Company","Sector"]))

# ============ Fib Calculator ============
with tab_fib:
    st.subheader("Fibonacci Calculator (12M from historical pricing)")
    px = pull_prices_for_current(period="2y")
    if px.empty:
        st.info("Add tickers in Overview to compute.")
    else:
        out = []
        for t in st.session_state.portfolio_df["ticker"].dropna().astype(str).str.upper().tolist():
            df_t = px[px["ticker"]==t].sort_values("date").copy()
            if df_t.empty: continue
            fibs = compute_fibs_12m(df_t)
            latest = df_t.iloc[-1]["close"]
            wz = weighted_entry_zone(latest, fibs)
            out.append({
                "ticker":t, "latest_close":latest,
                "fib_23.6":fibs["23.6"], "fib_38.2":fibs["38.2"], "fib_50.0":fibs["50.0"], "fib_61.8":fibs["61.8"], "fib_78.6":fibs["78.6"],
                "weighted_entry":wz["weighted_entry"], "entry_zone":wz["dynamic_entry_zone"],
                "support_zone":wz["dynamic_support_zone"], "resistance_zone":wz["dynamic_resistance_zone"]
            })
        st.dataframe(pd.DataFrame(out), use_container_width=True)

# ============ Seasonality ============
with tab_season:
    st.subheader("Strong/Weak Months per Stock and Sector (5y)")
    px = pull_prices_for_current(period="5y")
    if px.empty:
        st.info("Add tickers in Overview to compute seasonality.")
    else:
        stats = monthly_seasonality(px)
        st.write("Monthly returns table:")
        st.dataframe(stats, use_container_width=True)
        st.write("Top/Bottom months per ticker:")
        sw = strong_weak_months(stats, top_n=3)
        st.json(sw)
        mapping = st.session_state.portfolio_df[["ticker","sector"]].dropna()
        sector_stats = seasonality_by_sector(stats, mapping)
        st.write("Sector-level seasonality:")
        st.dataframe(sector_stats, use_container_width=True)

# ============ Risk & Scores ============
with tab_risk:
    st.subheader("Risk, Scores & Composite Rank")
    px = pull_prices_for_current(period="1y")
    if px.empty:
        st.info("Add tickers in Overview first.")
    else:
        bench = fetch_live_prices(["SPY"], period="1y", interval="1d")
        rows = []
        ticker_scores = {}
        for t in st.session_state.portfolio_df["ticker"].dropna().astype(str).str.upper().tolist():
            df_t = px[px["ticker"]==t].sort_values("date").copy()
            if df_t.empty: continue
            latest = df_t.iloc[-1]["close"]
            fibs = compute_fibs_12m(df_t)
            wz = weighted_entry_zone(latest, fibs)
            macd_df = compute_macd(df_t); macd_cross = macd_df.attrs.get("crossover","none")
            rsi = compute_rsi(df_t)
            adx = compute_adx(df_t)
            news = get_news_yf(t)
            rscore = risk_score(df_t, bench, news)
            comp = composite_rank_row(latest, wz["weighted_entry"], macd_cross, rsi, adx, rscore)
            ticker_scores[t] = float(comp)
            rows.append({
                "ticker":t, "latest_close":latest, "risk_0_10":round(rscore,2),
                "macd_cross":macd_cross, "rsi":round(rsi,2) if not np.isnan(rsi) else np.nan,
                "adx":round(adx,2) if not np.isnan(adx) else np.nan, "composite_score":round(comp,3)
            })
        st.dataframe(pd.DataFrame(rows).sort_values("composite_score", ascending=False), use_container_width=True)
        st.session_state.ticker_scores = ticker_scores

# ============ News & Calendar ============
with tab_news:
    st.subheader("News & Upcoming Earnings")
    tickers = st.session_state.portfolio_df["ticker"].dropna().astype(str).str.upper().unique().tolist()
    if not tickers:
        st.info("Add tickers in Overview to view headlines.")
    else:
        cal = fetch_earnings_calendar(tickers)
        st.write("Upcoming Earnings (if available):")
        st.dataframe(cal, use_container_width=True)
        st.write("Headlines:")
        tabs = st.tabs(tickers)
        for i, t in enumerate(tickers):
            with tabs[i]:
                news = get_news_newsapi(t, st.session_state.get("newsapi_key","")) if st.session_state.get("newsapi_key") else []
                if not news:
                    news = get_news_yf(t)
                if not news:
                    st.write("No recent articles found.")
                else:
                    for n in news[:10]:
                        title = n.get("title","(no title)")
                        pub = n.get("publisher") or n.get("source","")
                        when = n.get("providerPublishTime") or n.get("publishedAt","")
                        link = n.get("link") or n.get("url","")
                        st.markdown(f"- [{title}]({link}) — {pub} {when}")

# ============ Balance Plan (Biased) ============
with tab_balance:
    st.subheader("Balance Plan (USD) — Biased by Composite Scores & Sector Seasonality")
    port = st.session_state.portfolio_df.copy()
    if port.empty:
        st.info("Load a portfolio on the Overview tab.")
    else:
        tickers = port["ticker"].dropna().astype(str).str.upper().unique().tolist()
        current_prices = fetch_current_prices(tickers) if tickers else {}
        # Compute seasonality bias for the CURRENT month at sector level
        px5 = fetch_live_prices(tickers, period="5y", interval="1d")
        stats = monthly_seasonality(px5) if not px5.empty else pd.DataFrame()
        mapping = port[["ticker","sector"]].dropna()
        sec_stats = seasonality_by_sector(stats, mapping) if not stats.empty else pd.DataFrame(columns=["sector","month","avg_return"])
        cur_m = pd.Timestamp.today().month
        # bias = 1 + k * avg_return_month (clamped)
        sector_bias = {}
        if not sec_stats.empty:
            cur = sec_stats[sec_stats["month"]==cur_m][["sector","avg_return"]].dropna()
            if not cur.empty:
                for _, r in cur.iterrows():
                    ar = float(r["avg_return"])
                    sector_bias[r["sector"]] = float(max(0.5, min(1.5, 1.0 + st.session_state.get("bias_k",2.0) * ar)))
        # fallback: neutral for missing sectors
        for s in mapping["sector"].unique().tolist():
            sector_bias.setdefault(s, 1.0)

        # ticker scores from Risk & Scores tab (if computed), else neutral
        ticker_scores = st.session_state.get("ticker_scores", {t:1.0 for t in tickers})

        plan_by_pos, plan_by_sector = rebalance_plan_usd(
            port, current_prices, total_usd,
            sector_cap=sector_cap, pos_cap=position_cap,
            ticker_scores=ticker_scores, sector_bias=sector_bias
        )
        st.write("Sector Bias (current month):")
        st.json(sector_bias)
        st.write("Per-Position Targets & Deltas (USD):")
        st.dataframe(plan_by_pos, use_container_width=True)
        st.write("Per-Sector Summary (USD):")
        st.dataframe(plan_by_sector, use_container_width=True)
        st.caption("Positive delta = suggested buy; negative delta = suggested trim. Targets reflect score- and seasonality-aware biases within your caps.")

# ============ Signals & Accuracy ============
with tab_signals:
    st.subheader("Signals & Accuracy Tracker")
    init_db("data/app.db")
    if st.button("Log Current Balance Plan as Signals"):
        port = st.session_state.portfolio_df.copy()
        tickers = port["ticker"].dropna().astype(str).str.upper().unique().tolist()
        current_prices = fetch_current_prices(tickers) if tickers else {}
        # compute unbiased ticker_scores/sector_bias if not available
        ticker_scores = st.session_state.get("ticker_scores", {t:1.0 for t in tickers})
        px5 = fetch_live_prices(tickers, period="5y", interval="1d")
        stats = monthly_seasonality(px5) if not px5.empty else pd.DataFrame()
        mapping = port[["ticker","sector"]].dropna()
        sec_stats = seasonality_by_sector(stats, mapping) if not stats.empty else pd.DataFrame(columns=["sector","month","avg_return"])
        cur_m = pd.Timestamp.today().month
        sector_bias = {}
        if not sec_stats.empty:
            cur = sec_stats[sec_stats["month"]==cur_m][["sector","avg_return"]].dropna()
            for _, r in cur.iterrows():
                ar = float(r["avg_return"])
                sector_bias[r["sector"]] = float(max(0.5, min(1.5, 1.0 + st.session_state.get("bias_k",2.0) * ar)))
        for s in mapping["sector"].unique().tolist():
            sector_bias.setdefault(s, 1.0)

        plan_by_pos, _ = rebalance_plan_usd(
            port, current_prices, total_usd,
            sector_cap=sector_cap, pos_cap=position_cap,
            ticker_scores=ticker_scores, sector_bias=sector_bias
        )
        for _, r in plan_by_pos.iterrows():
            action = "Add" if r["suggested_delta_usd"]>0 else ("Trim" if r["suggested_delta_usd"]<0 else "Hold")
            log_signal(r["ticker"], action, float(r["last_price"]), db_path="data/app.db")
        st.success("Logged signals from biased balance plan.")
    sigs = fetch_signals("data/app.db")
    st.write("Recent Signals:")
    st.dataframe(sigs.head(50), use_container_width=True)
    if st.button("Evaluate Signals (1d/1w/1m/1y)"):
        sigs = fetch_signals("data/app.db")
        res = evaluate_signals(sigs)
        st.dataframe(res, use_container_width=True)

st.write("---")
st.caption("This app is for research/education. Not financial advice.")
