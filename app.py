import streamlit as st
import joblib
from src.preprocess import load_and_prepare

st.set_page_config(page_title="Gold Prediction Dashboard")

st.title("Gold Price Prediction Dashboard")

model = joblib.load("models/gold_model.pkl")

df = load_and_prepare()

latest_row = df.iloc[-1]

latest_close = latest_row["Close"]
latest_ma7 = latest_row["MA7"]
latest_ma30 = latest_row["MA30"]

st.subheader("Gold Price History")
st.line_chart(df["Close"])

st.subheader("Latest Market Data")
st.write(f"Latest Close Price: {latest_close:.2f}")
st.write(f"7-Day Moving Avg (returns): {latest_ma7:.6f}")
st.write(f"30-Day Moving Avg (returns): {latest_ma30:.6f}")

if st.button("Predict Next Day Return"):

    features = [[
        latest_row["MA7"],
        latest_row["MA30"],
        latest_row["Lag1"],
        latest_row["Lag2"],
        latest_row["Lag3"],
        latest_row["Volatility7"]
    ]]

    prediction = model.predict(features)[0]

    st.success(f"Predicted Next Day Return: {prediction:.6f}")

    if prediction > 0:
        st.info("Suggestion: BUY (Expected Positive Return)")
    else:
        st.warning("Suggestion: SELL (Expected Negative Return)")