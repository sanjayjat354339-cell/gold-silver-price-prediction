import pandas as pd

def load_and_prepare(file_path):

    df = pd.read_csv(file_path)

    df.columns = df.columns.str.strip()

    # Clean Close
    df["Close"] = (
        df["Close"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # Feature Engineering
    df["Return"] = df["Close"].pct_change()

    df["MA7"] = df["Return"].rolling(7).mean().shift(1)
    df["MA30"] = df["Return"].rolling(30).mean().shift(1)

    df["Lag1"] = df["Return"].shift(1)
    df["Lag2"] = df["Return"].shift(2)
    df["Lag3"] = df["Return"].shift(3)

    df["Volatility7"] = df["Return"].rolling(7).std().shift(1)

    df["Target"] = df["Return"].shift(-1)

    df.dropna(inplace=True)

    return df