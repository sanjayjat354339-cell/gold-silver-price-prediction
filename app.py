import streamlit as st
import joblib
from src.preprocess import load_and_prepare

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Gold & Silver Dashboard", layout="centered")

# ---------- TITLE ----------
st.title("Gold Price Prediction Dashboard")

# ---------- SELECT ----------
section = st.radio("Select Market", ["Gold", "Silver"])

# ---------- LOAD ----------
def load_all(data_path, model_path):
    df = load_and_prepare(data_path)
    model = joblib.load(model_path)
    return df, model

if section == "Gold":
    df, model = load_all("data/gold.csv", "models/gold_model.pkl")
else:
    df, model = load_all("data/silver.csv", "models/silver_model.pkl")

# ---------- GRAPH (EXACT LIKE YOUR IMAGE) ----------
st.subheader(f"{section} Price History")

# IMPORTANT: reset index for clean X-axis like your image
df_plot = df.tail(300).reset_index(drop=True)

# ONLY CLOSE → clean white line
st.line_chart(df_plot["Close"])

# ---------- LATEST ----------
latest = df.iloc[-1]

st.subheader("Latest Market Data")
st.write(f"Latest Close Price: {latest['Close']:.2f}")

# ---------- PREDICTION ----------
features = [[
    latest["MA7"],
    latest["MA30"],
    latest["Lag1"],
    latest["Lag2"],
    latest["Lag3"],
    latest["Volatility7"]
]]

prediction = model.predict(features)[0]

if prediction > 0:
    st.success("🟢 BUY")
else:
    st.error("🔴 SELL")

# ---------- ABOUT ----------
st.markdown("---")
st.subheader("About")
st.write("Built by Sanjay 🚀")