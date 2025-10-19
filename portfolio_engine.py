
import pandas as pd
import numpy as np
import yfinance as yf
import requests, sqlite3
from datetime import datetime, timedelta
from typing import List, Optional

# =========================
# Data & Live Adapters
# =========================

def fetch_live_prices(tickers: List[str], period="1y", interval="1d") -> pd.DataFrame:
    """Return long DataFrame with columns: date,ticker,close,high,low"""
    if not tickers:
        return pd.DataFrame(columns=["date","ticker","close","high","low"])
    df = yf.download(tickers, period=period, interval=interval, group_by="ticker", auto_adjust=False, threads=True, progress=False)
    rows = []
    try:
        if isinstance(tickers, str) or len(tickers) == 1:
            t = tickers[0] if not isinstance(tickers, str) else tickers
            dft = df.copy().reset_index()
            for _, r in dft.iterrows():
                rows.append([r["Date"], t, float(r["Close"]), float(r["High"]), float(r["Low"])])
        else:
            df = df.swaplevel(axis=1).sort_index(axis=1)
            for t in tickers:
                if t not in df.columns.get_level_values(0):
                    continue
                dft = df[t].reset_index()
                for _, r in dft.iterrows():
                    rows.append([r["Date"], t, float(r["Close"]), float(r["High"]), float(r["Low"])])
    except Exception:
        return pd.DataFrame(columns=["date","ticker","close","high","low"])
    out = pd.DataFrame(rows, columns=["date","ticker","close","high","low"]).dropna()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["ticker","date"])

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
        import requests
        url = "https://newsapi.org/v2/everything"
        params = {"q": ticker, "pageSize": 5, "sortBy": "publishedAt", "language": "en", "apiKey": api_key}
        r = requests.get(url, params=params, timeout=8)
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

# =========================
# Allocation & Constraints
# =========================

def summarize_portfolio(portfolio_df: pd.DataFrame, total_cad: float):
    df = portfolio_df.copy()
    if "allocation_cad" not in df.columns:
        df["allocation_cad"] = 0.0
    total = total_cad if total_cad else 0.0
    df["%"] = (df["allocation_cad"] / total) if total > 0 else 0.0
    by_stock = df[["ticker","company","sector","allocation_cad","%"]].copy()
    by_sector = by_stock.groupby("sector", as_index=False).agg({"allocation_cad":"sum"})
    by_sector["%"] = (by_sector["allocation_cad"] / total) if total > 0 else 0.0
    return {"by_stock": by_stock, "by_sector": by_sector}

def sector_caps_ok(by_sector: pd.DataFrame, cap=0.20):
    if by_sector.empty: return True
    return (by_sector["%"] <= cap + 1e-9).all()

def position_caps_ok(by_stock: pd.DataFrame, cap=0.05):
    if by_stock.empty: return True
    return (by_stock["%"] <= cap + 1e-9).all()

# =========================
# Technicals
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
# Strategy & Decisions
# =========================

def decide_action(current_price, fib_weighted_entry, macd_cross, purchase_price=None, up_trigger=0.05, down_trigger=-0.05):
    notes = []
    action = "Hold"
    if macd_cross == "bullish":
        action = "Consider Buy"
        notes.append("MACD bullish crossover")
    elif macd_cross == "bearish":
        action = "Consider Sell/Trim"
        notes.append("MACD bearish crossover")
    if pd.notna(fib_weighted_entry) and current_price is not None:
        if current_price <= fib_weighted_entry:
            notes.append("At/below weighted entry")
            if action == "Hold":
                action = "Consider Buy"
        elif current_price >= fib_weighted_entry:
            notes.append("Above weighted entry")
    if purchase_price:
        chg = (current_price - purchase_price) / purchase_price
        if chg >= up_trigger:
            action = "Trim (5% rule)"
            notes.append("Gain ≥ +5% vs purchase")
        elif chg <= down_trigger:
            if action != "Trim (5% rule)":
                action = "Add (5% rule)"
                notes.append("Loss ≤ -5% vs purchase")
    return action, "; ".join(notes)

def rank_daily_strategy(tech_df: pd.DataFrame) -> pd.DataFrame:
    if tech_df.empty:
        return tech_df
    df = tech_df.copy()
    df["score"] = 0.0
    df["score"] += (df["macd_cross"]=="bullish") * 2.0
    df["score"] += (df["macd_cross"]=="bearish") * -1.0
    df["dist_to_entry"] = abs(df["latest_close"] - df["weighted_entry"])
    m = df["dist_to_entry"].max() if df["dist_to_entry"].max() > 0 else 1.0
    df["score"] += (1.0 - df["dist_to_entry"]/m)
    return df.sort_values("score", ascending=False)

# =========================
# Accuracy Tracker (SQLite)
# =========================

def init_db(db_path: str = "data/app.db"):
    import os, sqlite3
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
    cur.execute("""CREATE TABLE IF NOT EXISTS outcomes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id INTEGER,
        horizon TEXT,
        ret REAL
    )""")
    conn.commit()
    conn.close()

def log_signal(ticker: str, action: str, price: float, db_path: str = "data/app.db"):
    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO signals(timestamp,ticker,action,price) VALUES(?,?,?,?)", (ts, ticker, action, price))
    conn.commit()
    conn.close()

def fetch_signals(db_path: str = "data/app.db") -> pd.DataFrame:
    import sqlite3
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
# Utilities
# =========================

def ensure_portfolio_template() -> pd.DataFrame:
    return pd.DataFrame([
        {"ticker":"AAPL","company":"Apple Inc.","sector":"Technology","allocation_cad":220,"purchase_date":"","purchase_price":""},
        {"ticker":"MSFT","company":"Microsoft Corp.","sector":"Technology","allocation_cad":220,"purchase_date":"","purchase_price":""},
        {"ticker":"NVDA","company":"NVIDIA Corp.","sector":"Technology","allocation_cad":220,"purchase_date":"","purchase_price":""},
        {"ticker":"JPM","company":"JPMorgan Chase","sector":"Financials","allocation_cad":176,"purchase_date":"","purchase_price":""},
        {"ticker":"PG","company":"Procter & Gamble","sector":"Consumer Staples","allocation_cad":176,"purchase_date":"","purchase_price":""},
        {"ticker":"XOM","company":"Exxon Mobil","sector":"Energy","allocation_cad":176,"purchase_date":"","purchase_price":""},
    ])
