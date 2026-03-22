from src.preprocess import load_and_prepare
from sklearn.linear_model import LinearRegression
import joblib
import os

df = load_and_prepare("data/gold.csv")

X = df[[
    "MA7", "MA30", "Lag1", "Lag2", "Lag3", "Volatility7"
]]
y = df["Target"]

model = LinearRegression()
model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/gold_model.pkl")

print("Gold model trained!") 