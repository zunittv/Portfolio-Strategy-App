
import pandas as pd
import numpy as np
import yfinance as yf
import requests, sqlite3, math
from datetime import datetime, timedelta
from typing import List, Optional, Dict

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

# -------- Data & Live Adapters --------

def fetch_live_prices(tickers: List[str], period="5y", interval="1d") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["date","ticker","close","high","low"])
    df = yf.download(tickers, period=period, interval=interval, group_by="ticker", auto_adjust=False, threads=True, progress=False)
    rows = []
    try:
        if isinstance(tickers, str) or len(tickers) == 1:
            t = tickers[0] if not isinstance(tickers, str) else tickers
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
    last = {}
    try:
        if isinstance(tickers, str) or len(tickers) == 1:
            t = tickers[0] if not isinstance(tickers, str) else tickers
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

# -------- Allocation & Constraints --------

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

# -------- Technical Indicators --------

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

# -------- Seasonality --------

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

# -------- Sentiment / Risk --------

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

# -------- Rebalancer (Score + Seasonality aware) --------

def sector_position_targets_biased(portfolio: pd.DataFrame,
                                   sector_cap=0.20,
                                   pos_cap=0.05,
                                   ticker_scores: Dict[str,float]=None,
                                   sector_bias: Dict[str,float]=None):
    """Compute targets biased by composite scores and sector seasonality.
       - sector_bias: multiplier per sector (e.g., >1 if current month historically strong).
       - ticker_scores: weight positions within a sector.
    """
    df = portfolio.copy()
    sectors = df["sector"].dropna().unique().tolist()
    if sector_bias is None: sector_bias = {s:1.0 for s in sectors}
    # Base equal-sector weight * bias, then cap and renormalize
    w = {s: sector_bias.get(s,1.0) for s in sectors}
    # Normalize weights to sum 1, then apply sector_cap
    total_w = sum(w.values()) if w else 1.0
    w = {s: (val/total_w) for s,val in w.items()}
    # Cap per sector and re-spread excess
    # First clamp
    w = {s: min(val, sector_cap) for s,val in w.items()}
    # Re-distribute leftover to uncapped sectors
    leftover = 1.0 - sum(w.values())
    if leftover > 1e-9:
        free = [s for s in sectors if w[s] < sector_cap - 1e-9]
        if free:
            add = leftover/len(free)
            for s in free:
                w[s] = min(sector_cap, w[s] + add)
    # Renormalize slight errors
    tot = sum(w.values()) if w else 1.0
    w = {k: v/tot for k,v in w.items()}

    # Within sector, allocate by composite score (fallback equal) while respecting pos_cap
    pos_targets = {}
    for s in sectors:
        members = df[df["sector"]==s]["ticker"].tolist()
        if not members: continue
        scores = {t: (ticker_scores.get(t,1.0) if ticker_scores else 1.0) for t in members}
        sum_scores = sum(scores.values()) if scores else 1.0
        for t in members:
            base = w[s] * (scores[t]/sum_scores)
            pos_targets[t] = min(base, pos_cap)
    # Normalize position targets to sum to 1
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

# -------- DB / Signals --------

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

def ensure_portfolio_df(preset="Mid-D") -> pd.DataFrame:
    rows = MID_D if preset=="Mid-D" else MID_DV
    return pd.DataFrame([{"ticker":t,"company":c,"sector":s,"allocation_usd":0.0,"shares_held":0.0,"purchase_date":"","purchase_price":""} for t,c,s in rows])
