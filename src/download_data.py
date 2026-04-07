import yfinance as yf
import pandas as pd
import os
from datetime import datetime


DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

ASSETS = {
    "gold": "GC=F",
    "silver": "SI=F"
}

PERIOD = "10y"
INTERVAL = "1d"


# DOWNLOAD FUNCTION 
def download_asset(name, symbol):
    try:
        print(f"📥 Downloading {name.upper()} data...")

        df = yf.download(
            symbol,
            period=PERIOD,
            interval=INTERVAL,
            progress=False
        )

        if df.empty:
            print(f"❌ No data for {name}")
            return

        df.reset_index(inplace=True)

       
        df["Asset"] = name.capitalize()
        df["Downloaded_At"] = datetime.now()

        file_path = os.path.join(DATA_DIR, f"{name}.csv")
        df.to_csv(file_path, index=False)

        print(f"✅ {name.capitalize()} saved to {file_path}")

    except Exception as e:
        print(f"⚠️ Error downloading {name}: {e}")


def main():
    print("🚀 Starting data download pipeline...\n")

    for name, symbol in ASSETS.items():
        download_asset(name, symbol)

    print("\n🎉 All data downloaded successfully!")



if __name__ == "__main__":
    main()