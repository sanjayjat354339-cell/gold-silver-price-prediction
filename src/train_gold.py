import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from preprocess import load_and_prepare


#  CHANGE SYMBOL
SYMBOL = "GC=F"  


df = load_and_prepare(SYMBOL, is_live=True)


# EXTRA FEATURES (NEW) 

# RSI (Relative Strength Index)
delta = df["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

# Price trend strength
df["TrendStrength"] = df["Close"].pct_change(10)


#  FINAL FEATURES
features = [
    "MA7", "MA30", "Momentum",
    "Lag1", "Lag2", "Lag3",
    "Volatility7", "Trend",
    "RSI", "TrendStrength"
]

df = df.dropna()

X = df[features].values
y = df["Target"].values


#  SCALE
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# MODEL (UPGRADED) 
model = RandomForestRegressor(
    n_estimators=400,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_scaled, y)


#  EVALUATE 
pred = model.predict(X_scaled)
r2 = r2_score(y, pred)

print("✅ R2 Score:", r2)


#  SAVE 
joblib.dump(model, f"models/{SYMBOL}_model.pkl")
joblib.dump(scaler, f"models/{SYMBOL}_scaler.pkl")

print("💾 Model saved successfully")