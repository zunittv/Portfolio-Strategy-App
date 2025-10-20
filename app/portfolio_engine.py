
import pandas as pd
import numpy as np
import yfinance as yf
import requests, sqlite3, math
from datetime import datetime, timedelta
from typing import List, Optional, Dict

# =========================
# Preset portfolios
# =========================

MID_D = [
    ("NVDA","NVIDIA Corp.","Technology"),
    ("MSFT","Microsoft Corp.","Technology"),
    ("AAPL","Apple Inc.","Technology"),
    ("GOOGL","Alphabet Inc. (A)","Communication Services"),
    ("UNH","UnitedHealth Group","Healthcare"),
    ("PFE","Pfizer Inc.","Healthcare"),
    ("HON","Honeywell International","Industrials"),
    ("CAT","Caterpillar Inc.","Industrials"),
    ("XOM","Exxon Mobil Corp.","Energy"),
    ("JPM","JPMorgan Chase & Co.","Financials"),
    ("PG","Procter & Gamble Co.","Consumer Staples"),
]

MID_DV = [
    ("UAL","United Airlines","Travel/Airlines"),
    ("DAL","Delta Air Lines","Travel/Airlines"),
    ("EXPE","Expedia Group","Travel/Airlines"),
    ("BKNG","Booking Holdings","Travel/Airlines"),
    ("VLO","Valero Energy","Energy/Refining"),
    ("MPC","Marathon Petroleum","Energy/Refining"),
    ("XLE","Energy Select Sector SPDR","Energy/Refining"),
    ("SU","Suncor Energy","Energy/Refining"),
    ("AEM","Agnico Eagle Mines","Gold/Precious Metals"),
    ("GOLD","Barrick Gold","Gold/Precious Metals"),
    ("GDX","VanEck Gold Miners ETF","Gold/Precious Metals"),
    ("FNV","Franco-Nevada","Gold/Precious Metals"),
    ("IWM","iShares Russell 2000 ETF","Small-Caps/High Beta"),
    ("PLTR","Palantir Technologies","Small-Caps/High Beta"),
    ("UPST","Upstart Holdings","Small-Caps/High Beta"),
    ("RIVN","Rivian Automotive","Small-Caps/High Beta"),
    ("COIN","Coinbase Global","Crypto/FinTech"),
    ("SQ","Block, Inc.","Crypto/FinTech"),
    ("MARA","Marathon Digital Holdings","Crypto/FinTech"),
    ("SOFI","SoFi Technologies","Crypto/FinTech"),
]

def ensure_portfolio_df(preset: str="Mid-D") -> pd.DataFrame:
    rows = MID_D if preset=="Mid-D" else MID_DV
    return pd.DataFrame([
        {"ticker":t, "company":c, "sector":s, "allocation_usd":0.0, "shares_held":0.0, "purchase_date":"", "purchase_price":""}
        for t,c,s in rows
    ])

# =========================
# Data adapters
# =========================

def fetch_live_prices(tickers: List[str], period="5y", interval="1d") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["date","ticker","close","high","low"])
    df = yf.download(tickers, period=period, interval=interval, group_by="ticker", auto_adjust=False, threads=True, progress=False)
    rows = []
    try:
        if isinstance(tickers, str) or (isinstance(tickers, list) and len(tickers)==1):
            t = tickers[0] if isinstance(tickers, list) else tickers
            dft = df.copy().reset_index()
            for _, r in dft.iterrows():
                rows.append([r["Date"], t, float(r.get("Close", np.nan)), float(r.get("High", np.nan)), float(r.get("Low", np.nan))])
        else:
            df = df.swaplevel(axis=1).sort_index(axis=1)
            for t in tickers:
                if (t, "Close") not in df.columns: 
                    continue
                dft = df[t].reset_index()
                for _, r in dft.iterrows():
                    rows.append([r["Date"], t, float(r.get("Close", np.nan)), float(r.get("High", np.nan)), float(r.get("Low", np.nan))])
    except Exception:
        return pd.DataFrame(columns=["date","ticker","close","high","low"])
    out = pd.DataFrame(rows, columns=["date","ticker","close","high","low"]).dropna()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["ticker","date"])

def fetch_current_prices(tickers: List[str]) -> Dict[str, float]:
    if not tickers: return {}
    df = yf.download(tickers, period="5d", interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    last = {}
    try:
        if isinstance(tickers, str) or (isinstance(tickers, list) and len(tickers)==1):
            t = tickers[0] if isinstance(tickers, list) else tickers
            last[t] = float(df["Close"].dropna().iloc[-1])
        else:
            df = df.swaplevel(axis=1).sort_index(axis=1)
            for t in tickers:
                if (t,"Close") in df.columns:
                    last[t] = float(df[(t,"Close")].dropna().iloc[-1])
    except Exception:
        pass
    return last

def get_news_yf(ticker: str):
    try:
        return yf.Ticker(ticker).news or []
    except Exception:
        return []

def fetch_earnings_calendar(tickers: List[str]) -> pd.DataFrame:
    rows=[]
    for t in tickers:
        try:
            cal = yf.Ticker(t).calendar
            if cal is not None and not cal.empty and "Earnings Date" in cal.index:
                dt = cal.loc["Earnings Date"].values[0]
                rows.append({"ticker":t,"event":"Earnings","date":pd.to_datetime(dt)})
        except Exception:
            continue
    return pd.DataFrame(rows)

# =========================
# Summaries & caps
# =========================

def summarize_portfolio(portfolio_df: pd.DataFrame, total_usd: float, current_prices: Dict[str,float]=None):
    df = portfolio_df.copy()
    if "allocation_usd" not in df.columns: df["allocation_usd"]=0.0
    if "shares_held" not in df.columns: df["shares_held"]=0.0
    current_prices = current_prices or {}
    df["last_price"] = df["ticker"].map(current_prices).fillna(0.0)
    df["current_value_usd"] = df["shares_held"] * df["last_price"]
    total = float(total_usd or 0.0)
    df["target_%"] = (df["allocation_usd"]/total) if total>0 else 0.0
    cur_total = df["current_value_usd"].sum()
    df["current_%"] = (df["current_value_usd"]/cur_total) if cur_total>0 else 0.0

    by_stock = df[["ticker","company","sector","allocation_usd","target_%","shares_held","last_price","current_value_usd","current_%"]].copy()
    by_sector = by_stock.groupby("sector", as_index=False).agg(allocation_usd=("allocation_usd","sum"),
                                                               current_value_usd=("current_value_usd","sum"))
    by_sector["target_%"] = (by_sector["allocation_usd"]/total) if total>0 else 0.0
    sect_total = by_sector["current_value_usd"].sum()
    by_sector["current_%"] = (by_sector["current_value_usd"]/sect_total) if sect_total>0 else 0.0
    return {"by_stock":by_stock, "by_sector":by_sector}

def sector_caps_ok(by_sector: pd.DataFrame, cap=0.20):
    if by_sector.empty: return True
    return (by_sector["target_%"] <= cap + 1e-9).all()

def position_caps_ok(by_stock: pd.DataFrame, cap=0.05):
    if by_stock.empty: return True
    return (by_stock["target_%"] <= cap + 1e-9).all()

# =========================
# Indicators
# =========================

def compute_fibs_12m(df_t: pd.DataFrame):
    d = df_t.tail(252) if len(df_t)>=252 else df_t.copy()
    if d.empty: 
        return {"23.6":np.nan,"38.2":np.nan,"50.0":np.nan,"61.8":np.nan,"78.6":np.nan,"high":np.nan,"low":np.nan}
    hi = d["high"].max(); lo = d["low"].min(); rng = hi-lo if (pd.notna(hi) and pd.notna(lo)) else np.nan
    if not pd.notna(rng) or rng==0:
        return {"23.6":np.nan,"38.2":np.nan,"50.0":np.nan,"61.8":np.nan,"78.6":np.nan,"high":hi,"low":lo}
    return {
        "23.6": lo + rng*0.236, "38.2": lo + rng*0.382, "50.0": lo + rng*0.5,
        "61.8": lo + rng*0.618, "78.6": lo + rng*0.786, "high": hi, "low": lo
    }

def compute_macd(df_t: pd.DataFrame, fast=12, slow=26, signal=9):
    d = df_t.sort_values("date").copy()
    px = d["close"].astype(float)
    ema_fast = px.ewm(span=fast, adjust=False).mean()
    ema_slow = px.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - macd_signal
    out = d[["date","ticker","close"]].copy()
    out["macd"] = macd.values; out["macd_signal"] = macd_signal.values; out["macd_hist"] = hist.values
    cross="none"
    if len(out)>=2:
        prev = out.iloc[-2]["macd"] - out.iloc[-2]["macd_signal"]
        last = out.iloc[-1]["macd"] - out.iloc[-1]["macd_signal"]
        if prev<=0 and last>0: cross="bullish"
        elif prev>=0 and last<0: cross="bearish"
    out.attrs["crossover"]=cross
    return out

def compute_rsi(df_t: pd.DataFrame, period=14):
    d = df_t.sort_values("date").copy()
    delta = d["close"].diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / down.replace(0,np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if len(rsi) else np.nan

def atr(df_t: pd.DataFrame, period=14):
    if df_t.empty: return np.nan
    d = df_t.copy()
    d["prev_close"] = d["close"].shift(1)
    tr = pd.concat([(d["high"]-d["low"]), (d["high"]-d["prev_close"]).abs(), (d["low"]-d["prev_close"]).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])

def weighted_entry_zone(current_price: float, fibs: dict):
    levels = {k:fibs[k] for k in ["38.2","50.0","61.8","78.6"] if pd.notna(fibs.get(k))}
    if not levels:
        return {"weighted_entry":np.nan, "dynamic_entry_zone":"","dynamic_support_zone":"","dynamic_resistance_zone":""}
    eps = max(0.01, 0.005*current_price)
    weights = {k: 1.0/(abs(current_price-v)+eps) for k,v in levels.items()}
    sumw = sum(weights.values())
    w_entry = sum(weights[k]*levels[k] for k in weights)/sumw if sumw>0 else np.nan
    lev_sorted = sorted(levels.values()); steps=[lev_sorted[i+1]-lev_sorted[i] for i in range(len(lev_sorted)-1)]
    avg_step = np.mean(steps) if steps else np.nan
    if np.isnan(avg_step):
        return {"weighted_entry":w_entry, "dynamic_entry_zone":"","dynamic_support_zone":"","dynamic_resistance_zone":""}
    entry_low = w_entry - 0.25*avg_step; entry_high = w_entry + 0.25*avg_step
    support_low = min(w_entry, levels.get("61.8", w_entry)); support_high = levels.get("78.6", w_entry)
    resist_low = fibs.get("23.6", fibs.get("38.2", w_entry)); resist_high = max(w_entry, levels.get("50.0", w_entry))
    return {
        "weighted_entry": w_entry,
        "dynamic_entry_zone": f"{entry_low:.2f} – {entry_high:.2f}",
        "dynamic_support_zone": f"{support_low:.2f} – {support_high:.2f}",
        "dynamic_resistance_zone": f"{resist_low:.2f} – {resist_high:.2f}"
    }

# =========================
# Seasonality
# =========================

def monthly_seasonality(px: pd.DataFrame) -> pd.DataFrame:
    if px.empty: return pd.DataFrame(columns=["ticker","month","avg_return","win_rate"])
    df = px.copy()
    df["ret"] = df.groupby("ticker")["close"].pct_change()
    df["month"] = df["date"].dt.month
    g = df.groupby(["ticker","month"])["ret"]
    stats = g.agg(avg_return="mean", count="count").reset_index()
    wins = g.apply(lambda s:(s>0).sum()).reset_index(name="wins")
    stats = stats.merge(wins, on=["ticker","month"], how="left")
    stats["win_rate"] = stats["wins"]/stats["count"]
    return stats[["ticker","month","avg_return","win_rate"]]

def strong_weak_months(stats: pd.DataFrame, top_n=3):
    out={}
    if stats.empty: return out
    for t,g in stats.groupby("ticker"):
        g_sorted = g.sort_values("avg_return", ascending=False)
        out[t] = {"strong":g_sorted.head(top_n)["month"].tolist(),
                  "weak":g_sorted.tail(top_n)["month"].tolist()}
    return out

def seasonality_by_sector(stats: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    if stats.empty or mapping.empty: return pd.DataFrame(columns=["sector","month","avg_return","win_rate"])
    merged = stats.merge(mapping[["ticker","sector"]].drop_duplicates(), on="ticker", how="left")
    agg = merged.groupby(["sector","month"]).agg(avg_return=("avg_return","mean"), win_rate=("win_rate","mean")).reset_index()
    return agg

# =========================
# Risk & Scores
# =========================

SENT_POS = {"beat","strong","surge","up","record","growth","raise","upgrade","profit","positive","buyback"}
SENT_NEG = {"miss","weak","plunge","down","cut","downgrade","loss","negative","lawsuit","selloff"}

def simple_sentiment(headlines):
    if not headlines: return 0.0
    score=0; cnt=0
    for h in headlines:
        text=(h.get("title") or "").lower()
        if not text: continue
        cnt += 1
        pos=sum(w in text for w in SENT_POS); neg=sum(w in text for w in SENT_NEG)
        score += (pos - neg)
    return score/max(cnt,1)

def risk_score(df_t: pd.DataFrame, bench: pd.DataFrame, headlines) -> float:
    # ATR%, daily vol, beta, max drawdown, 200D gap, sentiment
    def atr_percent(d: pd.DataFrame, period=14):
        if d.empty: return np.nan
        x = d.copy()
        x["prev_close"] = x["close"].shift(1)
        tr = pd.concat([(x["high"]-x["low"]), (x["high"]-x["prev_close"]).abs(), (x["low"]-x["prev_close"]).abs()], axis=1).max(axis=1)
        atr_val = tr.rolling(period).mean().iloc[-1]
        return (atr_val/x["close"].iloc[-1]) if pd.notna(atr_val) else np.nan
    def daily_vol(d: pd.DataFrame, period=20):
        if d.empty: return np.nan
        return d["close"].pct_change().tail(period).std()
    def beta_vs_bench(d: pd.DataFrame, b: pd.DataFrame, period=60):
        if d.empty or b.empty: return np.nan
        a = d.sort_values("date").tail(period).set_index("date")["close"].pct_change().dropna()
        bb = b.sort_values("date").tail(period).set_index("date")["close"].pct_change().dropna()
        j = a.to_frame("a").join(bb.to_frame("b"), how="inner")
        if j["b"].var()==0 or j.dropna().empty: return np.nan
        return j["a"].cov(j["b"]) / j["b"].var()
    def max_dd(d: pd.DataFrame, period=90):
        if d.empty: return np.nan
        x = d.sort_values("date").tail(period).copy()
        roll_max = x["close"].cummax()
        return (x["close"]/roll_max - 1.0).min()
    def ma_gap(d: pd.DataFrame, ma=200):
        if len(d)<ma: return np.nan
        m = d.sort_values("date")["close"].rolling(ma).mean().iloc[-1]
        c = d.sort_values("date")["close"].iloc[-1]
        return (c-m)/m if m else np.nan

    atrp = atr_percent(df_t); vol = daily_vol(df_t)
    spy = bench
    beta = beta_vs_bench(df_t, spy)
    dd = max_dd(df_t); gap = ma_gap(df_t)
    sent = simple_sentiment(headlines)
    comps=[]
    if pd.notna(atrp): comps.append(min(max(atrp*100,0),4))
    if pd.notna(vol): comps.append(min(max(vol*100,0),3))
    if pd.notna(beta): comps.append(min(abs(beta),1.5))
    if pd.notna(dd): comps.append(min(abs(dd)*5,1.5))
    if pd.notna(gap): comps.append(min(abs(gap)*5,1.0))
    comps.append(0.5 - 0.5*np.tanh(sent))
    raw = sum(comps)
    return float(max(0.0, min(10.0, raw)))

def composite_rank_row(latest_close, weighted_entry, macd_cross, rsi, risk_score_val):
    score=0.0
    if macd_cross=="bullish": score+=2.0
    if macd_cross=="bearish": score-=1.0
    if not np.isnan(weighted_entry) and not np.isnan(latest_close):
        dist = abs(latest_close-weighted_entry)
        score += 1.0/(1.0+dist)
    if not np.isnan(rsi):
        if 40<=rsi<=60: score+=0.5
        elif rsi<30: score+=0.2
        elif rsi>70: score-=0.2
    if not np.isnan(risk_score_val):
        score += (1.0 - min(1.0, risk_score_val/10.0))
    return float(score)

# =========================
# Rebalancer (caps aware)
# =========================

def _sector_position_targets_biased(portfolio: pd.DataFrame,
                                   sector_cap=0.20,
                                   pos_cap=0.05,
                                   ticker_scores: Dict[str,float]=None,
                                   sector_bias: Dict[str,float]=None):
    df = portfolio.copy()
    sectors = df["sector"].dropna().unique().tolist()
    if sector_bias is None: sector_bias = {s:1.0 for s in sectors}
    w = {s: sector_bias.get(s,1.0) for s in sectors}
    total_w = sum(w.values()) if w else 1.0
    w = {s: (val/total_w) for s,val in w.items()}
    w = {s: min(val, sector_cap) for s,val in w.items()}
    leftover = 1.0 - sum(w.values())
    if leftover > 1e-9:
        free = [s for s in sectors if w[s] < sector_cap - 1e-9]
        if free:
            add = leftover/len(free)
            for s in free:
                w[s] = min(sector_cap, w[s] + add)
    tot = sum(w.values()) if w else 1.0
    w = {k: v/tot for k,v in w.items()}

    pos_targets = {}
    for s in sectors:
        members = df[df["sector"]==s]["ticker"].tolist()
        if not members: continue
        scores = {t: (ticker_scores.get(t,1.0) if ticker_scores else 1.0) for t in members}
        sum_scores = sum(scores.values()) if scores else 1.0
        for t in members:
            base = w[s] * (scores[t]/sum_scores)
            pos_targets[t] = min(base, pos_cap)
    total_pos = sum(pos_targets.values()) if pos_targets else 1.0
    pos_targets = {k: v/total_pos for k,v in pos_targets.items()}
    return w, pos_targets

def rebalance_plan_usd(portfolio: pd.DataFrame,
                       current_prices: Dict[str,float],
                       total_usd: float,
                       sector_cap=0.20,
                       pos_cap=0.05,
                       ticker_scores: Dict[str,float]=None,
                       sector_bias: Dict[str,float]=None):
    df = portfolio.copy()
    if "shares_held" not in df.columns: df["shares_held"] = 0.0
    df["last_price"] = df["ticker"].map(current_prices).fillna(0.0)
    df["current_value_usd"] = df["shares_held"] * df["last_price"]

    sector_targets, pos_targets = _sector_position_targets_biased(df, sector_cap, pos_cap, ticker_scores, sector_bias)

    rows = []
    for _, r in df.iterrows():
        t = r["ticker"]; s = r["sector"]
        target_pct = pos_targets.get(t, 0.0)
        target_value = target_pct * total_usd
        delta = target_value - r["current_value_usd"]
        rows.append({
            "ticker": t,
            "sector": s,
            "last_price": r["last_price"],
            "shares_held": r["shares_held"],
            "current_value_usd": r["current_value_usd"],
            "target_%": target_pct,
            "target_value_usd": target_value,
            "suggested_delta_usd": delta,
            "suggested_shares_delta": (delta / r["last_price"]) if r["last_price"] else 0.0
        })
    out = pd.DataFrame(rows)
    sector_plan = out.groupby("sector").agg(current_value_usd=("current_value_usd","sum"),
                                            target_value_usd=("target_value_usd","sum"),
                                            delta_usd=("suggested_delta_usd","sum")).reset_index()
    return out.sort_values("suggested_delta_usd"), sector_plan

# =========================
# Rotation helpers
# =========================

SECTOR_SPIDER_MAP = {
    "Technology": "XLK",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Gold/Precious Metals": "XLB",
    "Crypto/FinTech": "XLF",
    "Travel/Airlines": "XLI",
    "Small-Caps/High Beta": "IWM",
    "Energy/Refining": "XLE"
}

EARLY_CYCLE = {"Technology","Communication Services","Consumer Discretionary","Industrials","Materials","Small-Caps/High Beta"}
LATE_CYCLE  = {"Utilities","Healthcare","Consumer Staples","Energy","Real Estate"}

def map_sector_etf(sector: str) -> str:
    return SECTOR_SPIDER_MAP.get(sector, "SPY")

def predictive_layer_bias():
    # Use VIX and yield curve proxy (10Y - 5Y) to tilt risk-on/off
    try:
        vix_df = fetch_live_prices(["^VIX"], period="6mo", interval="1d")
        vix = float(vix_df[vix_df["ticker"]=="^VIX"]["close"].iloc[-1]) if not vix_df.empty else np.nan
    except Exception:
        vix = np.nan
    try:
        tnx = fetch_live_prices(["^TNX"], period="6mo", interval="1d")
        fvx = fetch_live_prices(["^FVX"], period="6mo", interval="1d")
        y10 = float(tnx[tnx["ticker"]=="^TNX"]["close"].iloc[-1]) if not tnx.empty else np.nan
        y5  = float(fvx[fvx["ticker"]=="^FVX"]["close"].iloc[-1]) if not fvx.empty else np.nan
        curve = y10 - y5 if not (np.isnan(y10) or np.isnan(y5)) else np.nan
    except Exception:
        curve = np.nan
    bias = {"risk_on":1.0, "risk_off":1.0, "neutral":1.0}
    if not np.isnan(vix):
        if vix <= 15:  # calm
            bias["risk_on"] *= 1.05
        elif vix >= 25:  # high vol
            bias["risk_off"] *= 1.10
    if not np.isnan(curve):
        if curve > 0.0:
            bias["risk_on"] *= 1.05
        else:
            bias["risk_off"] *= 1.05
    return bias

def apply_predictive_bias_to_sectors(sector_bias: Dict[str,float], sectors: List[str]) -> Dict[str,float]:
    regime = predictive_layer_bias()
    out = {}
    for s in sectors:
        b = sector_bias.get(s, 1.0)
        if s in EARLY_CYCLE:
            b *= regime.get("risk_on", 1.0)
        elif s in LATE_CYCLE:
            b *= regime.get("risk_off", 1.0)
        else:
            b *= regime.get("neutral", 1.0)
        out[s] = float(max(0.5, min(1.6, b)))
    return out

# =========================
# Factors / Optimizer / Stress & Monte Carlo
# =========================

def factor_matrix(px: pd.DataFrame):
    fac_tickers = ['SPY','QQQ','IWM','TLT','GLD','USO','UUP']
    fac = fetch_live_prices(fac_tickers, period='2y', interval='1d')
    if fac.empty or px.empty: return pd.DataFrame()
    w = fac.pivot_table(index='date', columns='ticker', values='close').dropna()
    return w.pct_change().dropna()

def mean_variance_opt(px: pd.DataFrame, tickers: list, risk_aversion: float = 3.0):
    w = px.pivot_table(index='date', columns='ticker', values='close').dropna()
    cols = [t for t in tickers if t in w.columns]
    if len(cols) < 2: 
        return {t: (1.0 if i==0 else 0.0) for i,t in enumerate(cols)}
    r = w[cols].pct_change().dropna()
    mu = r.mean().values
    cov = r.cov().values
    inv = np.linalg.pinv(cov)
    w_raw = inv.dot(mu) / max(1e-8, risk_aversion)
    w_long = np.clip(w_raw, 0, None)
    if w_long.sum() == 0:
        w_long = np.ones_like(w_long)
    w_norm = w_long / w_long.sum()
    return {t: float(w_norm[i]) for i,t in enumerate(cols)}

def cap_constrain(weights: dict, sector_map: dict, pos_cap=0.05, sector_cap=0.20):
    import copy
    w = copy.deepcopy(weights)
    for k in list(w.keys()):
        if w[k] > pos_cap:
            w[k] = pos_cap
    def sec_sum(s):
        return sum(w[t] for t,sec in sector_map.items() if sec==s and t in w)
    changed = True
    while changed:
        changed = False
        for s in set(sector_map.values()):
            if sec_sum(s) > sector_cap + 1e-12:
                members = [t for t,sec in sector_map.items() if sec==s and t in w]
                total = sum(w[t] for t in members)
                if total>0:
                    scale = sector_cap/total
                    for t in members:
                        w[t] *= scale
                    changed = True
    tot = sum(w.values())
    if tot>0:
        w = {k: v/tot for k,v in w.items()}
    return w

def shock_scenarios(px: pd.DataFrame, weights: dict):
    fac = factor_matrix(px)
    if fac.empty or not weights:
        return {}
    w = px.pivot_table(index='date', columns='ticker', values='close').dropna()
    r = w.pct_change().dropna()
    port = r.dot(pd.Series(weights).reindex(w.columns).fillna(0.0))
    df = port.to_frame('y').join(fac, how='inner').dropna()
    if df.empty: return {}
    Y = df['y'].values
    X = df.drop(columns=['y']).values
    import numpy as np
    X = np.concatenate([np.ones((len(X),1)), X], axis=1)
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    keys = ['alpha'] + list(df.drop(columns=['y']).columns)
    b = {k: float(v) for k,v in zip(keys,beta)}
    scenarios = {
        "Market -10% (SPY)": -0.10 * b.get('SPY',0.8),
        "Market +10% (SPY)":  0.10 * b.get('SPY',0.8),
        "Oil +5% (USO)":      0.05 * b.get('USO',0.0),
        "USD +10% (UUP)":     0.10 * b.get('UUP',0.0),
        "Rates + (TLT -5%)": -0.05 * b.get('TLT',0.0),
        "Gold +5% (GLD)":     0.05 * b.get('GLD',0.0)
    }
    return scenarios

def monte_carlo_projection(px: pd.DataFrame, weights: dict, days=21, trials=1000):
    w = px.pivot_table(index='date', columns='ticker', values='close').dropna()
    cols = [c for c in weights if c in w.columns]
    if not cols:
        return {}
    r = w[cols].pct_change().dropna()
    mu = r.mean().values
    cov = r.cov().values
    import numpy as np
    wv = np.array([weights[c] for c in cols])
    mean_port = float(wv.dot(mu))
    var_port = float(wv.dot(cov).dot(wv))
    daily_draws = np.random.normal(mean_port, np.sqrt(var_port), size=(trials, days))
    cum = (1+daily_draws).prod(axis=1) - 1.0
    return {
        "mean": float(np.mean(cum)),
        "p05": float(np.quantile(cum, 0.05)),
        "p50": float(np.quantile(cum, 0.50)),
        "p95": float(np.quantile(cum, 0.95))
    }

# =========================
# Notes / Alerts
# =========================

def init_notes(db_path: str = "data/app.db"):
    import os
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, ticker TEXT, note TEXT)")
    conn.commit()
    conn.close()

def add_note(ticker: str, note: str, db_path: str = "data/app.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO notes(timestamp,ticker,note) VALUES(?,?,?)", (ts, ticker, note))
    conn.commit()
    conn.close()

def fetch_notes(db_path: str = "data/app.db") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM notes ORDER BY id DESC", conn)
    conn.close()
    return df
