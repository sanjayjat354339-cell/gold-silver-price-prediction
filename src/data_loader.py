import yfinance as yf
import pandas as pd
import os

def download_gold():
    if not os.path.exists("data"):
        os.makedirs("data")

    gold = yf.download("GC=F", start="2010-01-01")
    gold.to_csv("data/gold.csv")
    print("Gold data saved successfully!")

if __name__ == "__main__":
    download_gold()
    