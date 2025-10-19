

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from portfolio_engine import (
    # Rotation & predictive
    sector_strength_and_momentum, predictive_layer_bias, apply_predictive_bias_to_sectors, rotation_fund_flow,
    # Risk-aware sector tools
    sector_correlations, risk_adjust_sector_bias,
    # Portfolio analytics
    portfolio_beta, portfolio_vol_vares, sector_risk_attribution, portfolio_weights_current,
    # Tech indicators & helpers
    atr, position_size_by_risk, stops_targets, bollinger_bands, proximity_52w, backtest_signals,
    # Base datasets & metadata
    MID_D, MID_DV, ensure_portfolio_df, enrich_metadata,
    # Data adapters
    fetch_live_prices, fetch_current_prices, get_news_yf, get_news_newsapi, fetch_earnings_calendar,
    # Summaries & caps
    summarize_portfolio, sector_caps_ok, position_caps_ok,
    # TA core
    compute_fibs_12m, compute_macd, compute_rsi, compute_adx, weighted_entry_zone,
    # Seasonality
    monthly_seasonality, strong_weak_months, seasonality_by_sector,
    # Risk scoring & composite
    simple_sentiment, risk_score, composite_rank_row,
    # Rebalancer
    rebalance_plan_usd, sector_position_targets_biased,
    # Signals DB
    init_db, log_signal, fetch_signals, evaluate_signals,
    # Factors / Optimizer / Stress / MC / Notes / Alerts
    factor_exposure, mean_variance_opt, cap_constrain, shock_scenarios, monte_carlo_projection,
    init_notes, add_note, fetch_notes, alerts_for_ticker
)

st.set_page_config(page_title="Portfolio Strategy App – v6.2 (USD)", layout="wide")
st.title("📊 Portfolio Strategy App – v6.2 (USD)")
st.caption("For research & education. Not financial advice.")

# ---------- Sidebar ----------
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

# ---------- Session init for two editable portfolios ----------
if "mid_d_df" not in st.session_state:
    st.session_state.mid_d_df = ensure_portfolio_df("Mid-D")
if "mid_dv_df" not in st.session_state:
    st.session_state.mid_dv_df = ensure_portfolio_df("Mid-DV")
if "active_portfolio" not in st.session_state:
    st.session_state.active_portfolio = "Mid-D"
if "newsapi_key" not in st.session_state:
    st.session_state.newsapi_key = newsapi_key
else:
    st.session_state.newsapi_key = newsapi_key

# Helper: get active df
def get_active_df() -> pd.DataFrame:
    return st.session_state.mid_d_df if st.session_state.active_portfolio=="Mid-D" else st.session_state.mid_dv_df

def set_active_df(df: pd.DataFrame):
    if st.session_state.active_portfolio=="Mid-D":
        st.session_state.mid_d_df = df
    else:
        st.session_state.mid_dv_df = df

def pull_prices_for_active(period="5y"):
    df = get_active_df()
    tickers = df["ticker"].dropna().astype(str).str.upper().unique().tolist()
    if not tickers:
        return pd.DataFrame(columns=["date","ticker","close","high","low"])
    return fetch_live_prices(tickers, period=period, interval="1d")

# ---------- Tabs ----------
tabs = st.tabs(["Overview", "Mid-D (Edit)", "Mid-DV (Edit)", "Dashboard", "Fib Calculator", "Seasonality", "Risk & Scores", "News & Calendar", "Balance Plan (Biased)", "Rotation", "Benchmarks", "Backtest Lab", "Signals & Accuracy", "Guidance — Daily Plan"])
tab_overview, tab_mid_d, tab_mid_dv, tab_dashboard, tab_fib, tab_season, tab_risk, tab_news, tab_balance, tab_rotation, tab_bench, tab_backtest, tab_signals, tab_guidance = tabs

# ============ Overview (shows active portfolio) ============
with tab_overview:
    st.subheader("Active Portfolio")
    st.write("Choose which portfolio is active for all analysis tabs:")
    col1, col2 = st.columns(2)
    if col1.button("Set Active: Mid-D"):
        st.session_state.active_portfolio = "Mid-D"
    if col2.button("Set Active: Mid-DV"):
        st.session_state.active_portfolio = "Mid-DV"
    st.markdown(f"**Current Active:** {st.session_state.active_portfolio}")

    st.write("### Editable Table (USD) — Active Portfolio")
    df_active = get_active_df()
    edited = st.data_editor(
        df_active,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_active"
    )
    # Auto-enrich any tickers missing company/sector
    need_meta = edited[edited["ticker"].astype(str).str.len()>0].copy()
    to_fill = need_meta[(need_meta["company"]== "") | (need_meta["company"].isna()) | (need_meta["sector"]== "") | (need_meta["sector"].isna())]
    if not to_fill.empty:
        meta = enrich_metadata(to_fill["ticker"].astype(str).str.upper().unique().tolist())
        for i, r in edited.iterrows():
            t = str(r["ticker"]).upper() if pd.notna(r["ticker"]) else ""
            if t in meta:
                if pd.isna(r.get("company","")) or r.get("company","")== "":
                    edited.at[i, "company"] = meta[t]["company"]
                if pd.isna(r.get("sector","")) or r.get("sector","")== "":
                    edited.at[i, "sector"] = meta[t]["sector"]
    set_active_df(edited)

    # Live prices & allocation summary
    tickers = edited["ticker"].dropna().astype(str).str.upper().unique().tolist()
    current_prices = fetch_current_prices(tickers) if tickers else {}
    perf = summarize_portfolio(edited, total_usd, current_prices=current_prices)

    st.write("#### Allocation Summary (Targets & Current Values) — Active")
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(perf["by_stock"], use_container_width=True)
    with c2:
        st.dataframe(perf["by_sector"], use_container_width=True)
    st.write(f"Sector caps (≤ {int(sector_cap*100)}%): {'✅ OK' if sector_caps_ok(perf['by_sector'], sector_cap) else '⚠️ Exceeds'}")
    st.write(f"Position caps (≤ {int(position_cap*100)}%): {'✅ OK' if position_caps_ok(perf['by_stock'], position_cap) else '⚠️ Exceeds'}")

    # ---- Validation Highlights ----
    # Caps vs target_%
    by_stock = perf["by_stock"]
    over_pos = by_stock[by_stock["target_%"] > position_cap + 1e-9]
    if not over_pos.empty:
        st.warning("Some positions exceed the position cap (target_% > cap):")
        st.dataframe(over_pos[["ticker","target_%"]])
    by_sector = perf["by_sector"]
    over_sec = by_sector[by_sector["target_%"] > sector_cap + 1e-9]
    if not over_sec.empty:
        st.warning("Some sectors exceed the sector cap (target_% > cap):")
        st.dataframe(over_sec[["sector","target_%"]])


# ============ Mid-D (Edit) ============
with tab_mid_d:
    st.subheader("Mid-D — Editable")
    df = st.session_state.mid_d_df.copy()
    ed = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="editor_mid_d")
    # Enrich metadata
    need_meta = ed[ed["ticker"].astype(str).str.len()>0].copy()
    to_fill = need_meta[(need_meta["company"]== "") | (need_meta["company"].isna()) | (need_meta["sector"]== "") | (need_meta["sector"].isna())]
    if not to_fill.empty:
        meta = enrich_metadata(to_fill["ticker"].astype(str).str.upper().unique().tolist())
        for i, r in ed.iterrows():
            t = str(r["ticker"]).upper() if pd.notna(r["ticker"]) else ""
            if t in meta:
                if pd.isna(r.get("company","")) or r.get("company","")== "":
                    ed.at[i, "company"] = meta[t]["company"]
                if pd.isna(r.get("sector","")) or r.get("sector","")== "":
                    ed.at[i, "sector"] = meta[t]["sector"]
    st.session_state.mid_d_df = ed
    colA, colB = st.columns(2)
    if colA.button("Set Active: Mid-D (here)"):
        st.session_state.active_portfolio = "Mid-D"
    if colB.button("Copy Mid-D → Active"):
        st.session_state.active_portfolio = "Mid-D"

# ============ Mid-DV (Edit) ============
with tab_mid_dv:
    st.subheader("Mid-DV — Editable")
    df = st.session_state.mid_dv_df.copy()
    ed = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="editor_mid_dv")
    # Enrich metadata
    need_meta = ed[ed["ticker"].astype(str).str.len()>0].copy()
    to_fill = need_meta[(need_meta["company"]== "") | (need_meta["company"].isna()) | (need_meta["sector"]== "") | (need_meta["sector"].isna())]
    if not to_fill.empty:
        meta = enrich_metadata(to_fill["ticker"].astype(str).str.upper().unique().tolist())
        for i, r in ed.iterrows():
            t = str(r["ticker"]).upper() if pd.notna(r["ticker"]) else ""
            if t in meta:
                if pd.isna(r.get("company","")) or r.get("company","")== "":
                    ed.at[i, "company"] = meta[t]["company"]
                if pd.isna(r.get("sector","")) or r.get("sector","")== "":
                    ed.at[i, "sector"] = meta[t]["sector"]
    st.session_state.mid_dv_df = ed
    colA, colB = st.columns(2)
    if colA.button("Set Active: Mid-DV (here)"):
        st.session_state.active_portfolio = "Mid-DV"
    if colB.button("Copy Mid-DV → Active"):
        st.session_state.active_portfolio = "Mid-DV"


# ============ Dashboard (Pro) ============
with tab_dashboard:
    st.subheader("Portfolio Dashboard — Pro Analytics (Active Portfolio)")
    df = get_active_df()
    px = pull_prices_for_active(period="1y")
    if df.empty or px.empty:
        st.info("Edit a portfolio first (and ensure data loads).")
    else:
        # Live prices and perf summary
        tickers = df["ticker"].dropna().astype(str).str.upper().tolist()
        current_prices = fetch_current_prices(tickers)
        perf = summarize_portfolio(df, total_usd, current_prices=current_prices)
        bench = fetch_live_prices(["SPY"], period="1y", interval="1d")
        weights = portfolio_weights_current(perf["by_stock"])
        vol, var, es = portfolio_vol_vares(px, weights, level=0.95)
        beta_p = portfolio_beta(px, weights, bench)
        dd_1y = max_drawdown(px.groupby('ticker').get_group(tickers[0]) if not px.empty else px, period=252) if not px.empty else np.nan  # rough placeholder

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Weighted Beta vs SPY", f"{beta_p:.2f}" if not np.isnan(beta_p) else "n/a")
        c2.metric("Hist. Vol (Daily, 1y)", f"{vol:.3%}" if not np.isnan(vol) else "n/a")
        c3.metric("VaR 95% (Daily)", f"{var:.2%}" if not np.isnan(var) else "n/a")
        c4.metric("ES 95% (Daily)", f"{es:.2%}" if not np.isnan(es) else "n/a")
        st.write("Sector Risk Attribution (heuristic):")
        st.dataframe(sector_risk_attribution(perf["by_stock"], px), use_container_width=True)

# ============ Balance Plan (Active, Biased) — add bias mix & highlights ============

# ============ Fib Calculator (Active) ============
with tab_fib:
    st.subheader("Fibonacci Calculator (12M) — Active Portfolio")
    px = pull_prices_for_active(period="2y")
    df = get_active_df()
    if px.empty or df.empty:
        st.info("Add tickers in an editable tab to compute.")
    else:
        out = []
        for t in df["ticker"].dropna().astype(str).str.upper().tolist():
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

# ============ Seasonality (Active) ============
with tab_season:
    st.subheader("Strong/Weak Months — Active Portfolio")
    px = pull_prices_for_active(period="5y")
    df = get_active_df()
    if px.empty or df.empty:
        st.info("Add tickers in an editable tab to compute.")
    else:
        stats = monthly_seasonality(px)
        st.write("Monthly returns table:")
        st.dataframe(stats, use_container_width=True)
        st.write("Top/Bottom months per ticker:")
        sw = strong_weak_months(stats, top_n=3)
        st.json(sw)
        mapping = df[["ticker","sector"]].dropna()
        sector_stats = seasonality_by_sector(stats, mapping)
        st.write("Sector-level seasonality:")
        st.dataframe(sector_stats, use_container_width=True)

# ============ Risk & Scores (Active) ============
with tab_risk:
    st.subheader("Risk, Scores & Composite Rank — Active Portfolio")
    px = pull_prices_for_active(period="1y")
    df = get_active_df()
    if px.empty or df.empty:
        st.info("Add tickers in an editable tab first.")
    else:
        bench = fetch_live_prices(["SPY"], period="1y", interval="1d")
        rows = []
        ticker_scores = {}
        for t in df["ticker"].dropna().astype(str).str.upper().tolist():
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

# ============ News & Calendar (Active) ============
with tab_news:
    st.subheader("News & Upcoming Earnings — Active Portfolio")
    df = get_active_df()
    tickers = df["ticker"].dropna().astype(str).str.upper().unique().tolist()
    if not tickers:
        st.info("Add tickers in an editable tab to view headlines.")
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

# ============ Balance Plan (Active, Biased) ============
with tab_balance:
    st.subheader("Balance Plan (USD) — Bias Mix & Seasonality (Active Portfolio)")
    bias_mix = st.slider("Bias mix (0 = equal-weight, 1 = fully biased)", 0.0, 1.0, 0.7, 0.05)
    port = get_active_df().copy()
    if port.empty:
        st.info("Edit a portfolio first.")
    else:
        tickers = port["ticker"].dropna().astype(str).str.upper().unique().tolist()
        current_prices = fetch_current_prices(tickers) if tickers else {}
        # Build sector bias for current month from active df
        px5 = pull_prices_for_active(period="5y")
        stats = monthly_seasonality(px5) if not px5.empty else pd.DataFrame()
        mapping = port[["ticker","sector"]].dropna()
        sec_stats = seasonality_by_sector(stats, mapping) if not stats.empty else pd.DataFrame(columns=["sector","month","avg_return"])
        cur_m = pd.Timestamp.today().month
        sector_bias = {}
        if not sec_stats.empty:
            cur = sec_stats[sec_stats["month"]==cur_m][["sector","avg_return"]].dropna()
            if not cur.empty:
                for _, r in cur.iterrows():
                    ar = float(r["avg_return"])
                    sector_bias[r["sector"]] = float(max(0.5, min(1.5, 1.0 + st.session_state.get("bias_k",2.0) * ar)))
        for s in mapping["sector"].unique().tolist():
            sector_bias.setdefault(s, 1.0)

        ticker_scores = st.session_state.get("ticker_scores", {t:1.0 for t in tickers})
        # blend: scores-> (1-bias_mix)+bias_mix*score ; sector bias -> (1-bias_mix)+bias_mix*bias
        ticker_scores = {k: (1-bias_mix) + bias_mix*float(v) for k,v in ticker_scores.items()}
        sector_bias = {k: (1-bias_mix) + bias_mix*float(v) for k,v in sector_bias.items()}

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
        
        # Constraint highlights (target_% check vs caps)
        st.write("Constraint Checks:")
        over_pos = plan_by_pos[plan_by_pos['target_%'] > position_cap + 1e-9]
        if not over_pos.empty:
            st.warning("Positions exceeding position cap:")
            st.dataframe(over_pos[['ticker','target_%']])
        # Position sizing & stops/targets suggestions
        st.write("Position Sizing & Risk Controls:")
        sizes = []
        px1y = pull_prices_for_active(period="1y")
        for _, r in plan_by_pos.iterrows():
            t = r['ticker']; last = r['last_price']
            df_t = px1y[px1y['ticker']==t].sort_values('date')
            a = atr(df_t)
            shares = position_size_by_risk(total_usd, per_trade_risk_bps=50, last_price=last, atr_val=a, atr_mult=1.5)  # 50 bps risk budget example
            fibs = compute_fibs_12m(df_t) if not df_t.empty else {}
            stp_tp = stops_targets(last, a, fibs, r_mult=2.0, atr_mult_stop=1.5)
            sizes.append({'ticker':t, 'atr':a, 'suggested_shares':shares, 'stop':stp_tp['stop'], 'take_profit':stp_tp['take_profit']})
        st.dataframe(pd.DataFrame(sizes), use_container_width=True)

        st.caption("Positive delta = suggested buy; negative delta = suggested trim. Targets reflect bias mix & seasonality. Sizing/stops are illustrative.")

# ============ Signals & Accuracy (Active) ============
with tab_signals:
    st.subheader("Signals & Accuracy — Active Portfolio")
    from portfolio_engine import init_db, log_signal, fetch_signals, evaluate_signals
    init_db("data/app.db")
    if st.button("Log Current Balance Plan as Signals"):
        port = get_active_df().copy()
        tickers = port["ticker"].dropna().astype(str).str.upper().unique().tolist()
        current_prices = fetch_current_prices(tickers) if tickers else {}
        # rebuild biases/scores to ensure consistency
        px5 = pull_prices_for_active(period="5y")
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
        # compute scores
        ticker_scores = st.session_state.get("ticker_scores", {t:1.0 for t in tickers})
        # blend: scores-> (1-bias_mix)+bias_mix*score ; sector bias -> (1-bias_mix)+bias_mix*bias
        ticker_scores = {k: (1-bias_mix) + bias_mix*float(v) for k,v in ticker_scores.items()}
        sector_bias = {k: (1-bias_mix) + bias_mix*float(v) for k,v in sector_bias.items()}
        plan_by_pos, _ = rebalance_plan_usd(
            port, current_prices, total_usd,
            sector_cap=sector_cap, pos_cap=position_cap,
            ticker_scores=ticker_scores, sector_bias=sector_bias
        )
        for _, r in plan_by_pos.iterrows():
            action = "Add" if r["suggested_delta_usd"]>0 else ("Trim" if r["suggested_delta_usd"]<0 else "Hold")
            log_signal(r["ticker"], action, float(r["last_price"]), db_path="data/app.db")
        st.success("Logged signals from ACTIVE portfolio biased plan.")
    sigs = fetch_signals("data/app.db")
    st.write("Recent Signals:")
    st.dataframe(sigs.head(50), use_container_width=True)
    if st.button("Evaluate Signals (1d/1w/1m/1y)"):
        sigs = fetch_signals("data/app.db")
        res = evaluate_signals(sigs)
        st.dataframe(res, use_container_width=True)

st.write("---")

# ============ Guidance — Daily Plan ============
with tab_guidance:
    st.subheader("Ready-to-Buy / Ready-to-Trim — Daily Guidance (Active Portfolio)")
    df = get_active_df()
    if df.empty:
        st.info("Edit a portfolio first.")
    else:
        px = pull_prices_for_active(period="1y")
        bench = fetch_live_prices(["SPY"], period="1y", interval="1d")
        cal = fetch_earnings_calendar(df["ticker"].astype(str).str.upper().tolist())
        buy_rows, trim_rows, notes_rows = [], [], []
        for t in df["ticker"].dropna().astype(str).str.upper().tolist():
            df_t = px[px['ticker']==t].sort_values('date')
            if df_t.empty: 
                continue
            last = float(df_t.iloc[-1]['close'])
            fibs = compute_fibs_12m(df_t)
            wz = weighted_entry_zone(last, fibs)
            macd_df = compute_macd(df_t); cross = macd_df.attrs.get('crossover','none')
            rsi = compute_rsi(df_t)
            adx = compute_adx(df_t)
            bb_low, bb_mid, bb_up = bollinger_bands(df_t)
            hi52, lo52, pct_band = proximity_52w(df_t)
            dte = days_to_earnings(cal, t)
            news = get_news_yf(t)
            rscore = risk_score(df_t, bench, news)

            reasons = []
            if cross=='bullish': reasons.append('MACD bullish')
            if not np.isnan(rsi) and rsi<35: reasons.append('RSI near oversold')
            if not np.isnan(bb_low) and last<=bb_low: reasons.append('Near lower Bollinger')
            if not np.isnan(wz['weighted_entry']) and last<=wz['weighted_entry']: reasons.append('At/below Fib-weighted entry')
            if pct_band is not np.nan and pct_band<0.25: reasons.append('Closer to 52w low')
            if not np.isnan(rscore) and rscore<=4: reasons.append('Lower risk score')

            reasons_trim = []
            if cross=='bearish': reasons_trim.append('MACD bearish')
            if not np.isnan(rsi) and rsi>70: reasons_trim.append('RSI overbought')
            if not np.isnan(bb_up) and last>=bb_up: reasons_trim.append('Near upper Bollinger')
            if not np.isnan(wz['weighted_entry']) and last>=fibs.get('23.6', last): reasons_trim.append('Approaching Fib resistance')
            if pct_band is not np.nan and pct_band>0.85: reasons_trim.append('Closer to 52w high')
            if not np.isnan(rscore) and rscore>=7: reasons_trim.append('Elevated risk score')

            flag_earn = (not np.isnan(dte) and dte<=3)
            if flag_earn:
                reasons.append('Earnings ≤ 3d (caution)')

            if len(reasons)>=2 and (cross=='bullish' or (not np.isnan(rsi) and rsi<40)):
                buy_rows.append({'ticker':t, 'last':last, 'reasons':'; '.join(reasons)})
            if len(reasons_trim)>=2:
                trim_rows.append({'ticker':t, 'last':last, 'reasons':'; '.join(reasons_trim)})
            notes_rows.append({'ticker':t, 'notes':' | '.join(reasons+reasons_trim)})

        st.write("### Getting Ready to BUY (confluence):")
        df_buy = pd.DataFrame(buy_rows)
        st.dataframe(df_buy if not df_buy.empty else pd.DataFrame([{'ticker':'(none)','last':'-','reasons':'-'}]))

        st.write("### Getting Ready to TRIM / SELL (confluence):")
        df_trim = pd.DataFrame(trim_rows)
        st.dataframe(df_trim if not df_trim.empty else pd.DataFrame([{'ticker':'(none)','last':'-','reasons':'-'}]))

        # Export guidance markdown
        md_lines = ["# Daily Guidance", "## Buy Candidates"]
        if not df_buy.empty:
            for _, r in df_buy.iterrows():
                md_lines.append(f"- **{r['ticker']}** @ {r['last']:.2f} — {r['reasons']}")
        else:
            md_lines.append("- (none)")
        md_lines.append("## Trim/Sell Candidates")
        if not df_trim.empty:
            for _, r in df_trim.iterrows():
                md_lines.append(f"- **{r['ticker']}** @ {r['last']:.2f} — {r['reasons']}")
        else:
            md_lines.append("- (none)")
        md = "\\n".join(md_lines)
        st.download_button("Download Guidance (.md)", data=md, file_name="daily_guidance.md", mime="text/markdown")



# ============ Benchmarks ============
with tab_bench:
    st.subheader("Benchmarking vs SPY / QQQ — Active Portfolio")
    df = get_active_df()
    if df.empty:
        st.info("Edit a portfolio first.")
    else:
        import matplotlib.pyplot as plt
        tickers = df["ticker"].dropna().astype(str).str.upper().unique().tolist()
        px2y = pull_prices_for_active(period="2y")
        if px2y.empty:
            st.info("Not enough price data for benchmarking.")
        else:
            wide = px2y.pivot_table(index="date", columns="ticker", values="close").dropna(axis=0, how="any")
            rets = wide.pct_change().dropna()
            weights = np.repeat(1.0/len(wide.columns), len(wide.columns))
            port = (1+rets.dot(weights)).cumprod()
            # Benchmarks
            spy = fetch_live_prices(["SPY"], period="2y", interval="1d")
            qqq = fetch_live_prices(["QQQ"], period="2y", interval="1d")
            def to_cum(df):
                if df.empty: return None
                w = df.pivot_table(index="date", columns="ticker", values="close").dropna()
                r = w.pct_change().dropna()
                return (1+r).cumprod().iloc[:,0]
            spy_c = to_cum(spy)
            qqq_c = to_cum(qqq)

            # Plot cumulative returns
            fig1 = plt.figure()
            port.plot()
            if spy_c is not None: spy_c.align(port, join="inner")[0].plot()
            if qqq_c is not None: qqq_c.align(port, join="inner")[0].plot()
            st.pyplot(fig1)

            # Drawdown plot
            def drawdown(series):
                roll = series.cummax()
                return series/roll - 1.0
            fig2 = plt.figure()
            drawdown(port).plot()
            if spy_c is not None: drawdown(spy_c).align(port, join="inner")[0].plot()
            if qqq_c is not None: drawdown(qqq_c).align(port, join="inner")[0].plot()
            st.pyplot(fig2)



# ============ Backtest Lab ============
with tab_backtest:
    st.subheader("Backtest Lab — 1y Walk-Forward & Parameter Sweep (Active Portfolio)")
    df = get_active_df()
    if df.empty:
        st.info("Edit a portfolio first.")
    else:
        sweep_macd_fast = st.slider("MACD fast EMA", 8, 20, 12, 1)
        sweep_macd_slow = st.slider("MACD slow EMA", 20, 40, 26, 1)
        sweep_macd_sig  = st.slider("MACD signal EMA", 5, 15, 9, 1)
        sweep_atr_mult  = st.slider("ATR stop multiple", 1.0, 3.0, 1.5, 0.1)
        st.write("Runs a lightweight evaluation per ticker with these parameters.")

        rows = []
        px1y = pull_prices_for_active(period="1y")
        for t in df["ticker"].dropna().astype(str).str.upper().tolist():
            dft = px1y[px1y["ticker"]==t].sort_values("date")
            if dft.empty: 
                continue
            # reuse backtest_signals for simplicity (kept generic), but compute indicators with current params
            # quick override: compute MACD with new params and attach attrs
            macd_df = compute_macd(dft, fast=sweep_macd_fast, slow=sweep_macd_slow, signal=sweep_macd_sig)
            fibs = compute_fibs_12m(dft)
            latest = dft.iloc[-1]["close"]
            wz = weighted_entry_zone(latest, fibs)
            a = atr(dft)
            # walk-forward simple rule: buy if macd crosses up and price <= weighted entry; sell on cross down or +2*ATR
            # approximate with existing function
            stats = backtest_signals(dft)
            rows.append({"ticker":t, "trades":stats["trades"], "avg_ret":stats["avg_ret"], "win_rate":stats["win_rate"], "atr":a, "weighted_entry":wz["weighted_entry"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)



# ============ Rotation ============
with tab_rotation:
    st.subheader("Rotation Engine — Fund Flow Plan (Active Portfolio)")
    df = get_active_df().copy()
    if df.empty:
        st.info("Edit a portfolio first.")
    else:
        # Inputs
        tickers = df["ticker"].dropna().astype(str).str.upper().unique().tolist()
        current_prices = fetch_current_prices(tickers) if tickers else {}
        # Current performance and mapping
        perf = summarize_portfolio(df, total_usd, current_prices=current_prices)
        mapping = df[["ticker","sector"]].dropna()
        # Sector strength & momentum
        px1y = pull_prices_for_active(period="1y")
        sec_scores = sector_strength_and_momentum(px1y, mapping, lookback_days=30)
        st.write("Sector strength & momentum (vs SPY):")
        st.dataframe(sec_scores, use_container_width=True)
        # Build a biased balance plan to serve as rotation base
        # Reuse logic from Balance Plan: construct sector_bias + ticker_scores if available
        px5 = pull_prices_for_active(period="5y")
        stats = monthly_seasonality(px5) if not px5.empty else pd.DataFrame()
        sec_stats = seasonality_by_sector(stats, mapping) if not stats.empty else pd.DataFrame(columns=["sector","month","avg_return"])
        cur_m = pd.Timestamp.today().month
        sector_bias = {}
        if not sec_stats.empty:
            cur = sec_stats[sec_stats["month"]==cur_m][["sector","avg_return"]].dropna()
            if not cur.empty:
                for _, r in cur.iterrows():
                    ar = float(r["avg_return"])
                    sector_bias[r["sector"]] = float(max(0.5, min(1.5, 1.0 + st.session_state.get("bias_k",2.0) * ar)))
        for s in mapping["sector"].unique().tolist():
            sector_bias.setdefault(s, 1.0)
        # Risk-adjust and predictive layer
        sector_bias = risk_adjust_sector_bias(sector_bias, px5, mapping, lam=0.94)
        sectors_list = mapping["sector"].dropna().unique().tolist()
        sector_bias = apply_predictive_bias_to_sectors(sector_bias, sectors_list)
        ticker_scores = st.session_state.get("ticker_scores", {t:1.0 for t in tickers})

        from portfolio_engine import rebalance_plan_usd as rebalance_biased
        plan_by_pos, plan_by_sector = rebalance_biased(
            df, current_prices, total_usd,
            sector_cap=sector_cap, pos_cap=position_cap,
            ticker_scores=ticker_scores, sector_bias=sector_bias
        )
        st.write("Biased base plan (used for rotation):")
        st.dataframe(plan_by_pos, use_container_width=True)

        # Build fund flow plan (sources/destinations)
        flow = rotation_fund_flow(perf["by_sector"], plan_by_pos, sec_scores)
        st.write("Suggested Fund Flows (sources → destinations):")
        st.dataframe(flow if not flow.empty else pd.DataFrame([{"from":"(none)","to":"(none)","usd":0.0,"reason":"No rotation suggested"}]), use_container_width=True)

        # Before/After sector view
        before = perf["by_sector"][["sector","current_value_usd"]].copy()
        before = before.rename(columns={"current_value_usd":"before_usd"})
        after = plan_by_sector[["sector","target_value_usd"]].copy().rename(columns={"target_value_usd":"after_usd"})
        comp = pd.merge(before, after, on="sector", how="outer").fillna(0.0)
        st.write("Before vs After — Sector USD:")
        st.dataframe(comp, use_container_width=True)

        # Rotation summary text
        lines = ["## Rotation Summary"]
        for _, r in comp.iterrows():
            delta = float(r["after_usd"] - r["before_usd"])
            if abs(delta) < 1e-6: continue
            direction = "Increase" if delta>0 else "Decrease"
            lines.append(f"- {direction} **{r['sector']}** by ${abs(delta):,.0f}")
        if flow is not None and not flow.empty:
            lines.append("## Flows")
            for _, r in flow.iterrows():
                lines.append(f"- ${r['usd']:,.0f} from **{r['from']}** to **{r['to']}** — {r['reason']}")
        md = "\\n".join(lines) if len(lines)>1 else "No rotation suggested."
        st.download_button("Download Rotation Summary (.md)", data=md, file_name="rotation_summary.md", mime="text/markdown")


st.caption("This app is for research/education. Not financial advice.")


# ============ Optimizer ============
with tab_opt:
    st.subheader("Mean-Variance Optimizer (no shorting) with Caps")
    df = get_active_df()
    if df.empty:
        st.info("Edit a portfolio first.")
    else:
        tickers = df["ticker"].dropna().astype(str).str.upper().unique().tolist()
        px2y = pull_prices_for_active(period="2y")
        if px2y.empty:
            st.info("Not enough price data for optimization.")
        else:
            risk_av = st.slider("Risk aversion (higher = safer)", 1.0, 10.0, 3.0, 0.5)
            raw = mean_variance_opt(px2y, tickers, risk_aversion=risk_av)
            sec_map = df.set_index("ticker")["sector"].to_dict()
            w_cap = cap_constrain(raw, sec_map, pos_cap=position_cap, sector_cap=sector_cap)
            st.write("Optimized (cap-constrained) target weights:")
            st.dataframe(pd.DataFrame([w_cap]).T.rename(columns={0:"weight"}))
            perf = summarize_portfolio(df, total_usd, current_prices=fetch_current_prices(tickers))
            cur_w = perf["by_stock"][["ticker","current_%"]].set_index("ticker")["current_%"].to_dict()
            plan = []
            for t,w in w_cap.items():
                current = cur_w.get(t,0.0)
                plan.append({"ticker":t, "target_%":w, "current_%":current, "delta_%": w-current})
            st.write("Delta vs current allocation:")
            st.dataframe(pd.DataFrame(plan), use_container_width=True)


# ============ Stress & Monte Carlo ============
with tab_stress:
    st.subheader("Stress Tests & Monte Carlo — Active Portfolio")
    df = get_active_df()
    if df.empty:
        st.info("Edit a portfolio first.")
    else:
        import matplotlib.pyplot as plt
        tickers = df["ticker"].dropna().astype(str).str.upper().unique().tolist()
        px2y = pull_prices_for_active(period="2y")
        perf = summarize_portfolio(df, total_usd, current_prices=fetch_current_prices(tickers))
        weights = portfolio_weights_current(perf["by_stock"])
        if px2y.empty or not weights:
            st.info("Not enough data for stress tests.")
        else:
            st.write("Shock Scenarios (approximate portfolio P/L):")
            scenarios = shock_scenarios(px2y, weights)
            st.dataframe(pd.DataFrame(list(scenarios.items()), columns=["Scenario","Est. Return"]), use_container_width=True)

            st.write("Monte Carlo (21 trading days)")
            mc = monte_carlo_projection(px2y, weights, days=21, trials=2000)
            st.json(mc)

            import numpy as np
            w = px2y.pivot_table(index='date', columns='ticker', values='close').dropna()
            cols = [c for c in weights if c in w.columns]
            r = w[cols].pct_change().dropna()
            mu = r.mean().values
            cov = r.cov().values
            wv = np.array([weights[c] for c in cols])
            mean_port = float(wv.dot(mu))
            var_port = float(wv.dot(cov).dot(wv))
            draws = np.random.normal(mean_port, np.sqrt(var_port), size=(3000, 21))
            cum = (1+draws).prod(axis=1) - 1.0

            fig = plt.figure()
            _ = plt.hist(cum, bins=50)
            st.pyplot(fig)


# ============ Alerts & Notes ============
with tab_alerts:
    st.subheader("Alerts & Notes")
    init_notes("data/app.db")
    df = get_active_df()
    if df.empty:
        st.info("Edit a portfolio first.")
    else:
        tickers = df["ticker"].dropna().astype(str).str.upper().unique().tolist()
        px1y = pull_prices_for_active(period="1y")
        bench = fetch_live_prices(["SPY"], period="1y", interval="1d")
        cal = fetch_earnings_calendar(tickers)
        rows = []
        for t in tickers:
            dft = px1y[px1y["ticker"]==t].sort_values("date")
            news = get_news_yf(t)
            al = alerts_for_ticker(dft, bench, news, cal, t, price_chg_thresh=0.03, risk_thresh=7.0)
            rows.append({"ticker":t, "alerts":"; ".join(al) if al else "(none)"})
        st.write("Alert Feed:")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.write("Add a Note")
        c1, c2 = st.columns([1,3])
        with c1:
            tsel = st.selectbox("Ticker", options=tickers)
        with c2:
            note = st.text_input("Note text")
        if st.button("Save Note"):
            if note.strip():
                add_note(tsel, note.strip(), db_path="data/app.db")
                st.success("Saved.")
        st.write("Notes (latest first):")
        st.dataframe(fetch_notes("data/app.db"), use_container_width=True)
