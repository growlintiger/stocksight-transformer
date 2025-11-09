import yfinance as yf
import os

tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "NVDA", "GOOGL", "META", "JPM", "NFLX", "AMD"]
os.makedirs("data", exist_ok=True)

for ticker in tickers:
    data = yf.Ticker(ticker).history(period="3y")
    data.to_csv(f"data/{ticker}.csv")
    print(f"✅ Saved {ticker}.csv with {len(data)} rows")
