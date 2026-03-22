import yfinance as yf
import os

os.makedirs("data", exist_ok=True)

# Gold
gold = yf.download("GC=F", period="10y")
gold.reset_index(inplace=True)
gold.to_csv("data/gold.csv", index=False)

# Silver
silver = yf.download("SI=F", period="10y")
silver.reset_index(inplace=True)
silver.to_csv("data/silver.csv", index=False)

print("Gold & Silver data downloaded!")