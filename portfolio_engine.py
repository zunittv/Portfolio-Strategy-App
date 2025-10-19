
import pandas as pd
import numpy as np
from datetime import datetime

def summarize_portfolio(portfolio_df: pd.DataFrame, total_cad: float):
    df = portfolio_df.copy()
    if "allocation_cad" not in df.columns:
        df["allocation_cad"] = 0.0
    df["%"] = df["allocation_cad"] / total_cad if total_cad else 0.0
    by_stock = df[["ticker","company","sector","allocation_cad","%"]].copy()
    by_sector = by_stock.groupby("sector", as_index=False).agg({"allocation_cad":"sum"})
    by_sector["%"] = by_sector["allocation_cad"] / total_cad if total_cad else 0.0
    return {"by_stock": by_stock, "by_sector": by_sector}

def sector_caps_ok(by_sector: pd.DataFrame, cap=0.20):
    if by_sector.empty: return True
    return (by_sector["%"] <= cap + 1e-9).all()

def position_caps_ok(by_stock: pd.DataFrame, cap=0.05):
    if by_stock.empty: return True
    return (by_stock["%"] <= cap + 1e-9).all()

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
    support_low = min(w_entry, fibs.get("61.8", w_entry))
    support_high = fibs.get("78.6", w_entry)
    resist_low = fibs.get("23.6", fibs.get("38.2", w_entry))
    resist_high = max(w_entry, fibs.get("50.0", w_entry))
    return {
        "weighted_entry": w_entry,
        "dynamic_entry_zone": f"{entry_low:.2f} – {entry_high:.2f}",
        "dynamic_support_zone": f"{support_low:.2f} – {support_high:.2f}",
        "dynamic_resistance_zone": f"{resist_low:.2f} – {resist_high:.2f}"
    }

def five_percent_flow(portfolio_df: pd.DataFrame, latest_prices: dict, purchase_prices: dict):
    rows = []
    for _,row in portfolio_df.iterrows():
        t = row["ticker"]
        if t not in latest_prices or t not in purchase_prices: 
            continue
        change = (latest_prices[t] - purchase_prices[t]) / purchase_prices[t]
        action = "Hold"
        if change >= 0.05:
            action = "Trim winner; reallocate to laggards"
        elif change <= -0.05:
            action = "Add to loser (if sector cap allows)"
        rows.append({"ticker":t,"chg_since_purchase":change,"action":action})
    return pd.DataFrame(rows)

def suggest_rebalance(portfolio_df: pd.DataFrame, prices_df: pd.DataFrame):
    last = prices_df.sort_values("date").groupby("ticker").tail(1).set_index("ticker")
    hist = prices_df.sort_values("date").groupby("ticker")["close"].tail(10).groupby(level=0).mean()
    rows = []
    for _,r in portfolio_df.iterrows():
        t = r["ticker"]
        if t not in last.index or t not in hist.index:
            continue
        pct = (last.loc[t,"close"] - hist.loc[t]) / hist.loc[t]
        if pct > 0.05:
            decision = "Trim (5% rule)"
        elif pct < -0.05:
            decision = "Add (5% rule)"
        else:
            decision = "Hold"
        rows.append({"ticker":t,"last_close":last.loc[t,"close"],"vs_10d_avg":pct,"suggestion":decision})
    return pd.DataFrame(rows)

def signals_from_purchase(portfolio_df: pd.DataFrame, prices_df: pd.DataFrame):
    out = []
    for _,r in portfolio_df.iterrows():
        if not {"purchase_date","purchase_price"}.issubset(r.index):
            continue
        t = r["ticker"]
        try:
            d = pd.to_datetime(r["purchase_date"])
        except Exception:
            continue
        px = prices_df[(prices_df["ticker"]==t) & (prices_df["date"]>=d)].sort_values("date")
        if px.empty: 
            continue
        fibs = compute_fibs_12m(px)
        macd = compute_macd(px).iloc[-1]
        cur = px.iloc[-1]["close"]
        change = (cur - r["purchase_price"])/r["purchase_price"] if r["purchase_price"] else np.nan
        out.append({
            "ticker":t,"purchase_date":str(d.date()),"purchase_price":r["purchase_price"],
            "current_price":cur,"return_since_purchase":change,
            "macd":macd["macd"],"macd_signal":macd["macd_signal"]
        })
    return pd.DataFrame(out)

def suggest_candidates_stub(prices_df: pd.DataFrame, min_days=120):
    rows = []
    for t, grp in prices_df.sort_values("date").groupby("ticker"):
        if len(grp) < min_days:
            continue
        close = grp["close"]
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1] if len(close)>=200 else np.nan
        last = close.iloc[-1]
        if pd.notna(ma200) and last > ma50 > ma200:
            rows.append({"ticker":t,"last_close":last,"ma50":ma50,"ma200":ma200,"note":"Uptrend candidate"})
    return pd.DataFrame(rows)

def accuracy_tracker_stub():
    return {
        "schema": {
            "signal_id":"uuid",
            "ticker":"str",
            "timestamp":"datetime",
            "signal_type":"str (buy/sell/trim/add)",
            "horizons":["1d","1w","1m","1y"],
            "metric":"return vs benchmark, precision/recall, brier, etc."
        },
        "notes":"Record signals daily, evaluate outcomes after horizons, update reliability score."
    }
