
# Portfolio Strategy App – Prototype (Local)

**What this is:** a working prototype you can run locally. It:
- Lets you load a portfolio of tickers
- Computes **Fibonacci levels** (last 252 trading days) and **MACD** signals
- Applies your **5% trim / add** rule and sector caps (≤ 20% per sector, ≤ 5% per stock)
- Generates **suggested entry/exit zones** (based on Fib + current price weighting) and shows real‑time (session) signal checks
- Compares entry/exit from purchase date to now
- Provides a basic **suggestion engine** (placeholder) and **news/NLP hooks** (stubs to extend)
- Includes a basic **accuracy tracker** scaffold (day/week/month/year)

> This is for **research & education**. No financial advice.

## How to Run

1. Install dependencies:
```bash
pip install streamlit pandas numpy
```
2. Start the app:
```bash
streamlit run app/streamlit_app.py
```
3. In the UI:
   - Upload a **portfolio CSV** like `app/sample_portfolio.csv`
   - Upload a **prices CSV** like `app/sample_prices.csv` (or your own with columns: `date,ticker,close,high,low`)
   - Optionally set a **purchase date** per holding to compare historical entry/exit

## CSV Formats

### `sample_portfolio.csv`
```
ticker,company,sector,allocation_cad
AAPL,Apple Inc.,Technology,220
MSFT,Microsoft Corp.,Technology,220
XOM,Exxon Mobil Corp.,Energy,176
PG,Procter & Gamble,Consumer Staples,176
NVDA,NVIDIA Corp.,Technology,220
JPM,JPMorgan Chase,Financials,176
```
- `allocation_cad` is what you currently plan to allocate. The app computes % of total and sector totals.

### `sample_prices.csv`
```
date,ticker,close,high,low
2025-06-02,AAPL,201.2,203.5,198.9
...
```
- Include 252+ trading days of history for best Fib accuracy.

## Extend to Live Data
- Add your data adapters to pull live prices and fundamentals from your chosen provider
- Implement the `news_sentiment_stub` to fetch headlines via API and compute basic sentiment
- Add a scheduler (e.g., cron) to run daily analytics and write to local storage

**Disclaimer:** Not Financial Advice, Only for Research Purposes And May Sometimes Be Inaccurate.
