
import pandas as pd
import numpy as np
import yfinance as yf
import requests, sqlite3, math
from datetime import datetime, timedelta
from typing import List, Optional, Dict

# =========================
# Baseline datasets
# =========================

MID_D = [
    ("NVDA","NVIDIA Corp.","Technology"),
    ("MSFT","Microsoft Corp.","Technology"),
    ("AAPL","Apple Inc.","Technology"),
    ("GOOGL","Alphabet Inc. (A)","Communication Services"),
    ("UNH","UnitedHealth Group","Healthcare"),
    ("PFE","Pfizer Inc.","Healthcare"),
    ("CRSP","CRISPR Therapeutics","Biotech"),
    ("HON","Honeywell International","Industrials"),
    ("CAT","Caterpillar Inc.","Industrials"),
    ("XOM","Exxon Mobil Corp.","Energy"),
    ("JPM","JPMorgan Chase & Co.","Financials"),
    ("PG","Procter & Gamble Co.","Consumer Staples"),
]

MID_DV = [
    ("UAL","United Airlines Holdings","Travel/Airlines"),
    ("DAL","Delta Air Lines","Travel/Airlines"),
    ("EXPE","Expedia Group","Travel/Airlines"),
    ("BKNG","Booking Holdings","Travel/Airlines"),
    ("VLO","Valero Energy","Energy/Refining"),
    ("MPC","Marathon Petroleum","Energy/Refining"),
    ("XLE","Energy Select Sector SPDR (ETF)","Energy/Refining"),
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

def ensure_portfolio_df(preset: str = "Mid-D") -> pd.DataFrame:
    rows = MID_D if preset == "Mid-D" else MID_DV
    return pd.DataFrame([
        {"ticker": t, "company": c, "sector": s, "allocation_usd": 0.0, "shares_held": 0.0, "purchase_date": "", "purchase_price": ""}
        for t,c,s in rows
    ])

# -------- Metadata enrichment (ticker -> company, sector) --------
def enrich_metadata(tickers: List[str]) -> Dict[str, Dict[str,str]]:
    """Return dict[ticker] = {company, sector} using yfinance; resilient to missing fields."""
    out: Dict[str, Dict[str,str]] = {}
    for t in tickers:
        if not t:
            continue
        try:
            info = yf.Ticker(t).get_info()
        except Exception:
            info = {}
        company = info.get("longName") or info.get("shortName") or info.get("symbol") or t
        sector = info.get("sector") or info.get("industry") or "Unknown"
        out[t] = {"company": company, "sector": sector}
    return out

# =========================
# Data & Live Adapters
# =========================

def fetch_live_prices(tickers: List[str], period="5y", interval="1d") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["date","ticker","close","high","low"])
    df = yf.download(tickers, period=period, interval=interval, group_by="ticker", auto_adjust=False, threads=True, progress=False)
    rows = []
    try:
        if isinstance(tickers, str) or (isinstance(tickers, list) and len(tickers) == 1):
            t = tickers[0] if isinstance(tickers, list) else tickers
            dft = df.copy().reset_index()
            for _, r in dft.iterrows():
                rows.append([r["Date"], t, float(r.get("Close", np.nan)), float(r.get("High", np.nan)), float(r.get("Low", np.nan))])
        else:
            df = df.swaplevel(axis=1).sort_index(axis=1)
            for t in tickers:
                if t not in df.columns.get_level_values(0):
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
    if not tickers:
        return {}
    df = yf.download(tickers, period="5d", interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    last: Dict[str,float] = {}
    try:
        if isinstance(tickers, str) or (isinstance(tickers, list) and len(tickers) == 1):
            t = tickers[0] if isinstance(tickers, list) else tickers
            price = float(df["Close"][-1])
            last[t] = price
        else:
            df = df.swaplevel(axis=1).sort_index(axis=1)
            for t in tickers:
                try:
                    price = float(df[(t, "Close")].dropna().iloc[-1])
                    last[t] = price
                except Exception:
                    continue
    except Exception:
        pass
    return last

def get_news_yf(ticker: str):
    try:
        t = yf.Ticker(ticker)
        return t.news or []
    except Exception:
        return []

def get_news_newsapi(ticker: str, api_key: Optional[str]):
    if not api_key:
        return []
    try:
        url = "https://newsapi.org/v2/everything"
        params = {"q": ticker, "pageSize": 10, "sortBy": "publishedAt", "language": "en", "apiKey": api_key}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json().get("articles", [])
        mapped = []
        for a in data:
            mapped.append({
                "title": a.get("title"),
                "publisher": a.get("source",{}).get("name"),
                "providerPublishTime": a.get("publishedAt"),
                "link": a.get("url")
            })
        return mapped
    except Exception:
        return []

def fetch_earnings_calendar(tickers: List[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            cal = yf.Ticker(t).calendar
            if cal is not None and not cal.empty and "Earnings Date" in cal.index:
                dt = cal.loc["Earnings Date"].values[0]
                rows.append({"ticker":t, "event":"Earnings", "date": pd.to_datetime(dt)})
        except Exception:
            continue
    return pd.DataFrame(rows)

# =========================
# Allocation & Constraints
# =========================

def summarize_portfolio(portfolio_df: pd.DataFrame, total_usd: float, current_prices: Dict[str, float] = None):
    df = portfolio_df.copy()
    if "allocation_usd" not in df.columns:
        df["allocation_usd"] = 0.0
    if "shares_held" not in df.columns:
        df["shares_held"] = 0.0
    if current_prices is None:
        current_prices = {}
    df["last_price"] = df["ticker"].map(current_prices).fillna(0.0)
    df["current_value_usd"] = df["shares_held"] * df["last_price"]
    total = total_usd if total_usd else 0.0
    df["target_%"] = (df["allocation_usd"] / total) if total > 0 else 0.0
    cur_total = df["current_value_usd"].sum()
    df["current_%"] = (df["current_value_usd"] / cur_total) if cur_total > 0 else 0.0

    by_stock = df[["ticker","company","sector","allocation_usd","target_%","shares_held","last_price","current_value_usd","current_%"]].copy()
    sector_agg = by_stock.groupby("sector", as_index=False).agg(
        allocation_usd=("allocation_usd","sum"),
        current_value_usd=("current_value_usd","sum")
    )
    sector_agg["target_%"] = (sector_agg["allocation_usd"] / total) if total > 0 else 0.0
    sector_total = sector_agg["current_value_usd"].sum()
    sector_agg["current_%"] = (sector_agg["current_value_usd"] / sector_total) if sector_total > 0 else 0.0
    return {"by_stock": by_stock, "by_sector": sector_agg}

def sector_caps_ok(by_sector: pd.DataFrame, cap=0.20):
    if by_sector.empty: return True
    return (by_sector["target_%"] <= cap + 1e-9).all()

def position_caps_ok(by_stock: pd.DataFrame, cap=0.05):
    if by_stock.empty: return True
    return (by_stock["target_%"] <= cap + 1e-9).all()

# =========================
# Technical Indicators
# =========================

def compute_fibs_12m(df_t: pd.DataFrame):
    d = df_t.tail(252) if len(df_t) >= 252 else df_t.copy()
    hi = d["high"].max()
    lo = d["low"].min()
    rng = hi - lo if pd.notna(hi) and pd.notna(lo) else np.nan
    if rng is None or rng == 0 or np.isnan(rng):
        return {"23.6":np.nan,"38.2":np.nan,"50.0":np.nan,"61.8":np.nan,"78.6":np.nan,"high":hi,"low":lo}
    return {
        "23.6": lo + rng*0.236,
        "38.2": lo + rng*0.382,
        "50.0": lo + rng*0.500,
        "61.8": lo + rng*0.618,
        "78.6": lo + rng*0.786,
        "high": hi, "low": lo
    }

def compute_macd(df_t: pd.DataFrame, fast=12, slow=26, signal=9):
    d = df_t.copy().sort_values("date")
    px = d["close"].astype(float)
    ema_fast = px.ewm(span=fast, adjust=False).mean()
    ema_slow = px.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - macd_signal
    out = d[["date","ticker","close"]].copy()
    out["macd"] = macd.values
    out["macd_signal"] = macd_signal.values
    out["macd_hist"] = hist.values
    cross = "none"
    if len(out) >= 2:
        prev = out.iloc[-2]["macd"] - out.iloc[-2]["macd_signal"]
        last = out.iloc[-1]["macd"] - out.iloc[-1]["macd_signal"]
        if prev <= 0 and last > 0:
            cross = "bullish"
        elif prev >= 0 and last < 0:
            cross = "bearish"
    out.attrs["crossover"] = cross
    return out

def compute_rsi(df_t: pd.DataFrame, period=14):
    d = df_t.sort_values("date").copy()
    delta = d["close"].diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if len(rsi) else np.nan

def compute_adx(df_t: pd.DataFrame, period=14):
    d = df_t.sort_values("date").copy()
    if len(d) < period+1: return np.nan
    high, low, close = d["high"], d["low"], d["close"]
    plus_dm = (high.diff().clip(lower=0)).where((high.diff() > low.diff().abs()), 0.0)
    minus_dm = (low.diff(-1).abs().clip(lower=0)).where((low.diff().abs() > high.diff()), 0.0)
    tr1 = (high - low)
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0,np.nan)
    adx = dx.rolling(period).mean()
    return adx.iloc[-1] if len(adx) else np.nan

def weighted_entry_zone(current_price: float, fibs: dict):
    levels = {k:fibs[k] for k in ["38.2","50.0","61.8","78.6"] if pd.notna(fibs.get(k))}
    if not levels:
        return {"weighted_entry":np.nan, "dynamic_entry_zone":"", "dynamic_support_zone":"", "dynamic_resistance_zone":""}
    eps = max(0.01, 0.005*current_price)
    weights = {}
    for k,v in levels.items():
        dist = abs(current_price - v)
        weights[k] = 1.0 / (dist + eps)
    sumw = sum(weights.values())
    w_entry = sum(weights[k]*levels[k] for k in weights)/sumw if sumw>0 else np.nan
    lev_sorted = sorted(levels.values())
    steps = [lev_sorted[i+1]-lev_sorted[i] for i in range(len(lev_sorted)-1)]
    avg_step = np.mean(steps) if steps else np.nan
    if np.isnan(avg_step):
        return {"weighted_entry":w_entry, "dynamic_entry_zone":"", "dynamic_support_zone":"", "dynamic_resistance_zone":""}
    entry_low = w_entry - 0.25*avg_step
    entry_high = w_entry + 0.25*avg_step
    support_low = min(w_entry, levels.get("61.8", w_entry))
    support_high = levels.get("78.6", w_entry)
    resist_low = fibs.get("23.6", fibs.get("38.2", w_entry))
    resist_high = max(w_entry, levels.get("50.0", w_entry))
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
    wins = g.apply(lambda s: (s>0).sum()).reset_index(name="wins")
    stats = stats.merge(wins, on=["ticker","month"], how="left")
    stats["win_rate"] = stats["wins"]/stats["count"]
    return stats[["ticker","month","avg_return","win_rate"]]

def strong_weak_months(stats: pd.DataFrame, top_n=3):
    out = {}
    if stats.empty: return out
    for t, g in stats.groupby("ticker"):
        g_sorted = g.sort_values("avg_return", ascending=False)
        strong = g_sorted.head(top_n)["month"].tolist()
        weak = g_sorted.tail(top_n)["month"].tolist()
        out[t] = {"strong": strong, "weak": weak}
    return out

def seasonality_by_sector(stats: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    if stats.empty or mapping.empty: return pd.DataFrame(columns=["sector","month","avg_return","win_rate"])
    merged = stats.merge(mapping[["ticker","sector"]].drop_duplicates(), on="ticker", how="left")
    agg = merged.groupby(["sector","month"]).agg(avg_return=("avg_return","mean"), win_rate=("win_rate","mean")).reset_index()
    return agg

# =========================
# Risk / Sentiment / Scores
# =========================

SENT_POS = set(["beat","strong","surge","up","record","growth","raise","upgrade","profit","positive","buyback"])
SENT_NEG = set(["miss","weak","plunge","down","cut","downgrade","loss","negative","lawsuit","selloff"])

def simple_sentiment(headlines):
    if not headlines: return 0.0
    score = 0
    cnt = 0
    for h in headlines:
        text = (h.get("title") or "").lower()
        if not text: continue
        cnt += 1
        pos = sum(w in text for w in SENT_POS)
        neg = sum(w in text for w in SENT_NEG)
        score += (pos - neg)
    return score / max(cnt,1)

def atr_percent(df_t: pd.DataFrame, period=14) -> float:
    if df_t.empty: return np.nan
    d = df_t.copy()
    d["prev_close"] = d["close"].shift(1)
    m1 = d["high"] - d["low"]
    m2 = (d["high"] - d["prev_close"]).abs()
    m3 = (d["low"] - d["prev_close"]).abs()
    d["tr"] = pd.concat([m1, m2, m3], axis=1).max(axis=1)
    atr = d["tr"].rolling(period).mean().iloc[-1]
    return (atr / d["close"].iloc[-1]) if pd.notna(atr) else np.nan

def daily_vol(df_t: pd.DataFrame, period=20) -> float:
    if df_t.empty: return np.nan
    r = df_t["close"].pct_change().tail(period)
    return r.std()

def beta_vs_benchmark(df_t: pd.DataFrame, bench: pd.DataFrame, period=60) -> float:
    if df_t.empty or bench.empty: return np.nan
    a = df_t.sort_values("date").tail(period).copy()
    b = bench.sort_values("date").tail(period).copy()
    a = a.set_index("date")["close"].pct_change().dropna()
    b = b.set_index("date")["close"].pct_change().dropna()
    j = a.to_frame("a").join(b.to_frame("b"), how="inner")
    if j["b"].var() == 0 or j.dropna().empty:
        return np.nan
    return j["a"].cov(j["b"]) / j["b"].var()

def max_drawdown(df_t: pd.DataFrame, period=90) -> float:
    if df_t.empty: return np.nan
    d = df_t.sort_values("date").tail(period).copy()
    roll_max = d["close"].cummax()
    drawdown = d["close"]/roll_max - 1.0
    return drawdown.min()

def ma_gap(df_t: pd.DataFrame, ma=200) -> float:
    if len(df_t) < ma: return np.nan
    d = df_t.sort_values("date").copy()
    m = d["close"].rolling(ma).mean().iloc[-1]
    return (d["close"].iloc[-1] - m) / m if m and not math.isclose(m,0) else np.nan

def risk_score(df_t: pd.DataFrame, bench: pd.DataFrame, headlines) -> float:
    atrp = atr_percent(df_t)
    vol = daily_vol(df_t)
    beta = beta_vs_benchmark(df_t, bench)
    dd = max_drawdown(df_t)
    gap = ma_gap(df_t)
    sent = simple_sentiment(headlines)
    comps = []
    if pd.notna(atrp): comps.append(min(max(atrp*100,0), 4))
    if pd.notna(vol): comps.append(min(max(vol*100,0), 3))
    if pd.notna(beta): comps.append(min(abs(beta), 1.5))
    if pd.notna(dd): comps.append(min(abs(dd)*5, 1.5))
    if pd.notna(gap): comps.append(min(abs(gap)*5, 1.0))
    comps.append(0.5 - 0.5*np.tanh(sent))
    raw = sum(comps)
    return float(max(0.0, min(10.0, raw)))

def composite_rank_row(latest_close, weighted_entry, macd_cross, rsi, adx, risk_score_val):
    score = 0.0
    if macd_cross == "bullish": score += 2.0
    if macd_cross == "bearish": score -= 1.0
    if not np.isnan(weighted_entry) and not np.isnan(latest_close):
        dist = abs(latest_close - weighted_entry)
        score += 1.0 / (1.0 + dist)
    if not np.isnan(rsi):
        if 40 <= rsi <= 60: score += 0.5
        elif rsi < 30: score += 0.2
        elif rsi > 70: score -= 0.2
    if not np.isnan(adx):
        if adx < 15: score -= 0.2
        elif adx > 25: score += 0.2
    if not np.isnan(risk_score_val):
        score += (1.0 - min(1.0, risk_score_val/10.0))
    return score

# =========================
# Rebalancer
# =========================

def sector_position_targets_biased(portfolio: pd.DataFrame,
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

    sector_targets, pos_targets = sector_position_targets_biased(df, sector_cap, pos_cap, ticker_scores, sector_bias)

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
# DB / Signals
# =========================

def init_db(db_path: str = "data/app.db"):
    import os
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        ticker TEXT,
        action TEXT,
        price REAL
    )""")
    conn.commit()
    conn.close()

def log_signal(ticker: str, action: str, price: float, db_path: str = "data/app.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO signals(timestamp,ticker,action,price) VALUES(?,?,?,?)", (ts, ticker, action, price))
    conn.commit()
    conn.close()

def fetch_signals(db_path: str = "data/app.db") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM signals ORDER BY id DESC", conn)
    conn.close()
    return df

def evaluate_signals(signals_df: pd.DataFrame, horizons = {"1d":1, "1w":5, "1m":21, "1y":252}) -> pd.DataFrame:
    if signals_df.empty:
        return pd.DataFrame(columns=["signal_id","ticker","horizon","ret"])
    out_rows = []
    for _, sig in signals_df.iterrows():
        t = sig["ticker"]
        try:
            ts = pd.to_datetime(sig["timestamp"])
        except Exception:
            continue
        start = (ts - timedelta(days=2)).strftime("%Y-%m-%d")
        end = (ts + timedelta(days=400)).strftime("%Y-%m-%d")
        df = yf.download(t, start=start, end=end, interval="1d", progress=False)
        if df.empty:
            continue
        df = df.reset_index().rename(columns={"Date":"date","Close":"close"})
        df = df[["date","close"]].sort_values("date").reset_index(drop=True)
        idx0 = df[df["date"] >= ts].index.min()
        if pd.isna(idx0):
            continue
        p0 = float(df.loc[idx0, "close"])
        for hname, hdays in horizons.items():
            idxh = idx0 + hdays
            if idxh >= len(df):
                continue
            ph = float(df.loc[idxh, "close"])
            ret = (ph - p0) / p0
            out_rows.append({"signal_id": int(sig["id"]), "ticker": t, "horizon": hname, "ret": ret})
    return pd.DataFrame(out_rows)

# =========================
# Additional Indicators & Analytics
# =========================

def bollinger_bands(df_t: pd.DataFrame, period=20, n_std=2.0):
    d = df_t.sort_values('date').copy()
    if len(d) < period: 
        return np.nan, np.nan, np.nan
    ma = d['close'].rolling(period).mean()
    std = d['close'].rolling(period).std()
    upper = ma + n_std*std
    lower = ma - n_std*std
    return float(lower.iloc[-1]), float(ma.iloc[-1]), float(upper.iloc[-1])

def proximity_52w(df_t: pd.DataFrame):
    d = df_t.tail(252).copy()
    if d.empty: return np.nan, np.nan, np.nan
    hi = d['close'].max(); lo = d['close'].min(); last = d['close'].iloc[-1]
    pct_band = (last - lo)/(hi - lo) if (hi - lo) else np.nan
    return float(hi), float(lo), float(pct_band)

def atr(df_t: pd.DataFrame, period=14):
    if df_t.empty: return np.nan
    d = df_t.copy()
    d['prev_close'] = d['close'].shift(1)
    tr1 = d['high'] - d['low']
    tr2 = (d['high'] - d['prev_close']).abs()
    tr3 = (d['low'] - d['prev_close']).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])

def position_size_by_risk(total_usd: float, per_trade_risk_bps: float, last_price: float, atr_val: float, atr_mult: float=1.5, min_shares: int=0):
    if not last_price or not atr_val or np.isnan(atr_val): 
        return 0
    risk_per_share = atr_val * atr_mult
    risk_dollars = total_usd * (per_trade_risk_bps/10000.0)
    shares = int(max(min_shares, risk_dollars / risk_per_share)) if risk_per_share>0 else 0
    return shares

def stops_targets(last_price: float, atr_val: float, fibs: dict, r_mult: float=2.0, atr_mult_stop: float=1.5):
    if not last_price or not atr_val or np.isnan(atr_val): 
        return {"stop": np.nan, "take_profit": np.nan}
    stop = last_price - atr_mult_stop*atr_val
    res = max([fibs.get(k, np.nan) for k in ["38.2","23.6"] if pd.notna(fibs.get(k, np.nan))] or [np.nan])
    if pd.isna(res) or res <= last_price:
        tp = last_price + r_mult*atr_mult_stop*atr_val
    else:
        tp = res
    return {"stop": float(stop), "take_profit": float(tp)}

def backtest_signals(df_t: pd.DataFrame):
    if df_t.empty: 
        return {"trades":0, "avg_ret":np.nan, "win_rate":np.nan}
    d = df_t.sort_values('date').copy()
    fibs = compute_fibs_12m(d)
    latest = d.iloc[-1]['close']
    wz = weighted_entry_zone(latest, fibs)
    macd_df = compute_macd(d)
    macd_df = macd_df.set_index('date')
    closes = d.set_index('date')['close']
    buy = False; entry=0.0; pnl=[]; tp = stops_targets(latest, atr(d), fibs)['take_profit']
    for dt, px in closes.tail(252).items():
        idxs = macd_df.index.get_indexer([dt], method='nearest')
        idx = idxs[0] if len(idxs)>0 and idxs[0]>=0 else None
        if idx is not None and idx>0:
            prev = macd_df.iloc[idx-1]['macd'] - macd_df.iloc[idx-1]['macd_signal']
            last = macd_df.iloc[idx]['macd'] - macd_df.iloc[idx]['macd_signal']
            bull = prev<=0 and last>0
            bear = prev>=0 and last<0
        else:
            bull = bear = False
        if not buy:
            if bull and (px <= (wz['weighted_entry'] if not np.isnan(wz['weighted_entry']) else px)):
                buy=True; entry=px
        else:
            if bear or (tp and px>=tp):
                pnl.append((px-entry)/entry)
                buy=False; entry=0.0
    trades = len(pnl)
    avg_ret = float(np.mean(pnl)) if pnl else np.nan
    win_rate = float(np.mean([1 if x>0 else 0 for x in pnl])) if pnl else np.nan
    return {"trades":trades, "avg_ret":avg_ret, "win_rate":win_rate}

# =========================
# Portfolio analytics
# =========================

def portfolio_weights_current(by_stock_df: pd.DataFrame):
    cur_total = by_stock_df['current_value_usd'].sum()
    if cur_total <= 0: 
        return {r['ticker']:0.0 for _,r in by_stock_df.iterrows()}
    return {r['ticker']: (r['current_value_usd']/cur_total) for _, r in by_stock_df.iterrows()}

def portfolio_beta(px: pd.DataFrame, weights: dict, bench: pd.DataFrame):
    betas = {}
    for t in weights:
        df_t = px[px['ticker']==t]
        b = beta_vs_benchmark(df_t, bench, period=60)
        if pd.isna(b): b = 1.0
        betas[t] = b
    return sum(weights[t]*betas[t] for t in weights)

def portfolio_vol_vares(px: pd.DataFrame, weights: dict, level=0.95):
    if px.empty or not weights: 
        return np.nan, np.nan, np.nan
    dates = sorted(px['date'].unique())
    if len(dates) < 60: 
        return np.nan, np.nan, np.nan
    w = px.pivot_table(index='date', columns='ticker', values='close').dropna(axis=0, how='any')
    rets = w.pct_change().dropna()
    port_ret = rets.dot(pd.Series(weights).reindex(w.columns).fillna(0.0))
    vol = float(port_ret.std())
    q = np.quantile(port_ret, 1-level)
    var = float(-q)
    es = float(-port_ret[port_ret <= q].mean()) if (port_ret <= q).any() else np.nan
    return vol, var, es

def sector_risk_attribution(by_stock_df: pd.DataFrame, px: pd.DataFrame):
    if px.empty or by_stock_df.empty:
        return pd.DataFrame(columns=['sector','risk_contrib'])
    w = portfolio_weights_current(by_stock_df)
    sectors = by_stock_df.set_index('ticker')['sector'].to_dict()
    wide = px.pivot_table(index='date', columns='ticker', values='close').dropna(axis=0, how='any')
    rets = wide.pct_change().dropna()
    sec_weights = {}
    for t, wt in w.items():
        sec = sectors.get(t, 'Unknown')
        sec_weights[sec] = sec_weights.get(sec, 0.0) + wt
    ssum = sum(sec_weights.values()) or 1.0
    sec_weights = {k:v/ssum for k,v in sec_weights.items()}
    sectors_list = sorted(set(sectors.values()))
    rows = []
    for s in sectors_list:
        members = [t for t in w if sectors.get(t)==s and t in rets.columns]
        if not members: 
            continue
        sub = rets[members].mean(axis=1)
        rows.append({'sector': s, 'risk_contrib': float(sub.std()*sec_weights.get(s,0.0))})
    return pd.DataFrame(rows).sort_values('risk_contrib', ascending=False)

# =========================
# Risk-aware Bias Helpers
# =========================

def ewma_vol(series: pd.Series, lam: float=0.94):
    r = series.pct_change().dropna()
    if r.empty: return np.nan
    var = 0.0
    for x in r[::-1]:
        var = lam*var + (1-lam)*(x**2)
    return float(np.sqrt(var))

def sector_correlations(px: pd.DataFrame, mapping: pd.DataFrame):
    if px.empty or mapping.empty:
        return pd.DataFrame()
    wide = px.pivot_table(index='date', columns='ticker', values='close').dropna(axis=0, how='any')
    rets = wide.pct_change().dropna()
    sec_map = mapping.set_index('ticker')['sector'].to_dict()
    sectors = sorted(set(sec_map.values()))
    sec_ret = {}
    for s in sectors:
        members = [t for t,v in sec_map.items() if v==s and t in rets.columns]
        if not members: 
            continue
        sec_ret[s] = rets[members].mean(axis=1)
    if not sec_ret:
        return pd.DataFrame()
    df = pd.DataFrame(sec_ret).dropna()
    return df.corr()

def risk_adjust_sector_bias(sector_bias: Dict[str,float], px: pd.DataFrame, mapping: pd.DataFrame, lam: float=0.94):
    if px.empty or mapping.empty:
        return sector_bias
    wide = px.pivot_table(index='date', columns='ticker', values='close').dropna(axis=0, how='any')
    rets = wide.pct_change().dropna()
    sec_map = mapping.set_index('ticker')['sector'].to_dict()
    sectors = sorted(set(sec_map.values()))
    vol_sec = {}
    for s in sectors:
        members = [t for t,v in sec_map.items() if v==s and t in wide.columns]
        if not members:
            continue
        eq = rets[members].mean(axis=1)
        vol_sec[s] = ewma_vol(eq, lam=lam)
    vols = [v for v in vol_sec.values() if v is not None and not np.isnan(v)]
    if not vols:
        return sector_bias
    vmin, vmax = min(vols), max(vols)
    adj = {}
    for s,b in sector_bias.items():
        v = vol_sec.get(s, np.nan)
        if np.isnan(v) or vmax==vmin:
            adj[s] = b
        else:
            scale = 1.2 - 0.4*((v - vmin)/(vmax - vmin))  # maps to [0.8,1.2]
            adj[s] = float(b*scale)
    return {k: max(0.5, min(1.5, v)) for k,v in adj.items()}

# =========================
# Sector Rotation & Predictive Layer
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
    "Biotech": "XLV",
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

def sector_strength_and_momentum(px: pd.DataFrame, mapping: pd.DataFrame, lookback_days: int = 30):
    if px.empty or mapping.empty:
        return pd.DataFrame(columns=["sector","rs","momentum","etf"])
    sectors = mapping["sector"].dropna().unique().tolist()
    spy = fetch_live_prices(["SPY"], period="1y", interval="1d")
    spy = spy.rename(columns={"close":"spy_close"})[["date","spy_close"]].drop_duplicates("date")
    rows = []
    for s in sectors:
        etf = map_sector_etf(s)
        etf_px = fetch_live_prices([etf], period="1y", interval="1d")
        if etf_px.empty or spy.empty:
            rows.append({"sector":s, "rs":np.nan, "momentum":np.nan, "etf":etf})
            continue
        e = etf_px.rename(columns={"close":"etf_close"})[["date","etf_close"]].drop_duplicates("date")
        j = pd.merge(e, spy, on="date", how="inner").dropna()
        if len(j) < lookback_days+5:
            rows.append({"sector":s, "rs":np.nan, "momentum":np.nan, "etf":etf})
            continue
        j["ret_etf"] = j["etf_close"].pct_change()
        j["ret_spy"] = j["spy_close"].pct_change()
        rs = (1 + j["ret_etf"].tail(20)).prod() - (1 + j["ret_spy"].tail(20)).prod()
        mom = (j["etf_close"].iloc[-1] / j["etf_close"].iloc[-lookback_days]) - 1.0
        rows.append({"sector":s, "rs":float(rs), "momentum":float(mom), "etf":etf})
    return pd.DataFrame(rows)

def predictive_layer_bias():
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
        if vix <= 15:
            bias["risk_on"] *= 1.05
        elif vix >= 25:
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

def rotation_fund_flow(perf_by_sector: pd.DataFrame,
                       plan_by_pos: pd.DataFrame,
                       sector_scores: pd.DataFrame,
                       add_threshold: float = -0.05,
                       trim_threshold: float = 0.05,
                       max_moves: int = 5):
    flows = []
    if plan_by_pos.empty:
        return pd.DataFrame(columns=["from","to","usd","reason"])
    sources = plan_by_pos[plan_by_pos["suggested_delta_usd"] < 0].copy()
    dests   = plan_by_pos[plan_by_pos["suggested_delta_usd"] > 0].copy()
    sec_score_map = {}
    for _, r in sector_scores.iterrows():
        s = r["sector"]; rs = r.get("rs", 0.0); mom = r.get("momentum",0.0)
        sec_score_map[s] = float( max(-1.0, min(1.0, 0.5*rs + 0.5*mom)) )
    total_score = sum(max(0.0, sec_score_map.get(s,0.0)) for s in dests["sector"]) or 1.0
    for _, src in sources.head(max_moves).iterrows():
        amt = abs(float(src["suggested_delta_usd"]))
        for _, dst in dests.head(max_moves).iterrows():
            s = dst["sector"]
            w = max(0.0, sec_score_map.get(s,0.0)) / total_score
            usd = amt * w
            if usd <= 1e-6: 
                continue
            flows.append({
                "from": f"{src['ticker']} ({src['sector']})",
                "to":   f"{dst['ticker']} ({dst['sector']})",
                "usd": float(usd),
                "reason": f"Rotate from trim source to higher-scored sector {s} (rs/mom)"
            })
    return pd.DataFrame(flows)

# =========================
# Factors, Optimizer, Stress/MC, Alerts/Notes
# =========================

def factor_matrix(px: pd.DataFrame):
    fac_tickers = ['SPY','QQQ','IWM','TLT','GLD','USO','UUP']
    fac = fetch_live_prices(fac_tickers, period='2y', interval='1d')
    if fac.empty or px.empty: return pd.DataFrame()
    w = fac.pivot_table(index='date', columns='ticker', values='close').dropna()
    return w.pct_change().dropna()

def factor_exposure(px: pd.DataFrame, ticker: str):
    w = px.pivot_table(index='date', columns='ticker', values='close').dropna()
    if ticker not in w.columns: 
        return {}
    r = w.pct_change().dropna()
    y = r[ticker].copy()
    Xf = factor_matrix(px)
    if Xf.empty:
        return {}
    df = y.to_frame('y').join(Xf, how='inner').dropna()
    if df.empty: return {}
    Y = df['y'].values
    X = df.drop(columns=['y']).values
    X = np.concatenate([np.ones((len(X),1)), X], axis=1)
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    keys = ['alpha'] + list(df.drop(columns=['y']).columns)
    return {k: float(v) for k,v in zip(keys,beta)}

def mean_variance_opt(px: pd.DataFrame, tickers: list, target_risk: float = None, risk_aversion: float = 3.0):
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

# Alerts & Notes helpers
def pct_change_last(df_t: pd.DataFrame, days=1):
    d = df_t.sort_values('date').tail(days+1).copy()
    if len(d) < days+1: return np.nan
    return float(d['close'].pct_change().iloc[-1])

def macd_cross_signal(df_t: pd.DataFrame):
    macd_df = compute_macd(df_t)
    return macd_df.attrs.get('crossover','none')

def days_to_earnings(cal_df: pd.DataFrame, ticker: str):
    if cal_df is None or cal_df.empty: return np.nan
    now = pd.Timestamp.utcnow().normalize()
    fut = cal_df[(cal_df['ticker']==ticker) & (cal_df['date']>=now)]
    if fut.empty: return np.nan
    dt = fut['date'].min()
    return (dt - now).days

def earnings_proximity_flag(cal_df: pd.DataFrame, ticker: str, window_days=3):
    dte = days_to_earnings(cal_df, ticker)
    if np.isnan(dte): return False
    return dte <= window_days

def alerts_for_ticker(df_t: pd.DataFrame, bench: pd.DataFrame, headlines, cal_df: pd.DataFrame, ticker: str, price_chg_thresh=0.03, risk_thresh=7.0):
    if df_t.empty: 
        return []
    alerts = []
    chg1d = pct_change_last(df_t, days=1)
    if not np.isnan(chg1d) and abs(chg1d) >= price_chg_thresh:
        alerts.append(f"Price move {chg1d:.2%} (≥ {price_chg_thresh:.0%})")
    cross = macd_cross_signal(df_t)
    if cross in ['bullish','bearish']:
        alerts.append(f"MACD {cross} crossover")
    rscore = risk_score(df_t, bench, headlines)
    if not np.isnan(rscore) and rscore >= risk_thresh:
        alerts.append(f"High risk score {rscore:.1f} (≥ {risk_thresh})")
    if earnings_proximity_flag(cal_df, ticker, window_days=3):
        alerts.append("Earnings ≤ 3 days (event risk)")
    return alerts

# Notes DB
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
