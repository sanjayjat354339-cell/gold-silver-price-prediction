import pandas as pd
import yfinance as yf


# FETCH LIVE DATA
def fetch_live_data(symbol="GC=F"):
    df = yf.download(symbol, period="2y", interval="1d")

    # Handle MultiIndex columns (IMPORTANT)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    return df


#  MAIN PREPROCESS 
def load_and_prepare(source, is_live=False):

    # LOAD 
    if is_live:
        df = fetch_live_data(source)
    else:
        df = pd.read_csv(source)

    # FIX MULTIINDEX 
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    #  CLEAN 
    df.columns = df.columns.str.strip()

    if "Close" not in df.columns:
        raise ValueError("❌ 'Close' column not found")

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # FEATURE ENGINEERING 

    # Returns
    df["Return"] = df["Close"].pct_change()

    # Relative Moving Averages (normalized)
    df["MA7"] = df["Close"].rolling(7).mean() / df["Close"]
    df["MA30"] = df["Close"].rolling(30).mean() / df["Close"]

    # REAL Moving Averages (for visualization)
    df["MA7_plot"] = df["Close"].rolling(7).mean()
    df["MA30_plot"] = df["Close"].rolling(30).mean()
    # Momentum (normalized)
    df["Momentum"] = df["Close"].pct_change(5)

    # Lag Features
    df["Lag1"] = df["Return"].shift(1)
    df["Lag2"] = df["Return"].shift(2)
    df["Lag3"] = df["Return"].shift(3)

    # Volatility
    df["Volatility7"] = df["Return"].rolling(7).std()

    # Trend Feature (NEW)
    df["Trend"] = df["MA7"] - df["MA30"]

    # Target: next day return
    df["Target"] = df["Return"].shift(-1)

    #  CLEAN NA 
    df.dropna(inplace=True)

    return df