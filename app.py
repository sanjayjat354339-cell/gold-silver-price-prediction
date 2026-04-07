import streamlit as st
import joblib
import plotly.graph_objects as go
import numpy as np
from src.evaluate import evaluate_model
from src.preprocess import load_and_prepare
from src.evaluate import evaluate_model
from sklearn.metrics import r2_score

#  CONFIG 
st.set_page_config(page_title="Trading Dashboard", layout="wide")

#  STYLE 
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

#  SIDEBAR 
st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Price Calculator", "History"]
)

market = st.sidebar.radio("Select Market", ["Gold", "Silver"])

#  GRAPH SELECT 
st.sidebar.markdown("### 📊 Graph Settings")
graph_type = st.sidebar.radio(
    "Choose View",
    ["Price Chart", "Moving Averages"]
)

# note
st.sidebar.markdown('⚠️ Note: This application is for educational purposes only and does not constitute financial advice.')
#  MARKET CONFIG 
if market == "Gold":
    symbol = "GC=F"
    model_path = "GC=F_model.pkl"
    scaler_path = "GC=F_scaler.pkl"
    icon = "🥇"
else:
    symbol = "SI=F"
    model_path = "SI=F_model.pkl"
    scaler_path = "SI=F_scaler.pkl"
    icon = "🥈"

#  LOAD DATA 
@st.cache_data(ttl=300)
def load_data(sym):
    return load_and_prepare(sym, is_live=True)

df = load_data(symbol)

#  ADD SAME FEATURES AS TRAINING 
# RSI
delta = df["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

# Trend Strength
df["TrendStrength"] = df["Close"].pct_change(10)

df.dropna(inplace=True)

#  FEATURES 
features = [
    "MA7", "MA30", "Momentum",
    "Lag1", "Lag2", "Lag3",
    "Volatility7", "Trend",
    "RSI", "TrendStrength"
]

X = df[features].values
y = df["Target"].values

#  LOAD MODEL + SCALER 
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

X_scaled = scaler.transform(X)

#  PREDICTION 
latest_features = [X_scaled[-1]]
pred_return = model.predict(latest_features)[0]

latest_price = df["Close"].iloc[-1]
pred_price = latest_price * (1 + pred_return)

#  MODEL ACCURACY 
y_pred = model.predict(X_scaled)
r2 = r2_score(y, y_pred)

#  CONFIDENCE 
last_actual = y[-1]
last_pred = y_pred[-1]

error = abs(last_actual - last_pred)
vol = df.iloc[-1]["Volatility7"]

confidence = 1 / (1 + (error / (vol + 1e-6)))
confidence = max(0.2, min(confidence, 0.95))

#  LATEST PRICE 
latest = df.iloc[-1]
prev = df.iloc[-2]

change = latest["Close"] - prev["Close"]
percent = (change / prev["Close"]) * 100

# =====
# 🟢 DASHBOARD
if page == "Dashboard":

    st.title(f"{icon} {market}  Trading Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Live Price", f"${latest['Close']:.2f}", f"{percent:.2f}%")
    c2.metric("Prediction (Next Day)", f"${pred_price:.2f}")
    c3.metric("Confidence", f"{confidence:.2f}")
    c4.metric("Accuracy (R²)", f"{r2:.2f}")

    left, right = st.columns([3, 1])

    df_plot = df.tail(200).reset_index()
    x_axis = df_plot["Date"] if "Date" in df_plot else df_plot.index

    fig = go.Figure()

    if graph_type == "Price Chart":
        fig.add_trace(go.Scatter(x=x_axis, y=df_plot["Close"], name="Price"))

    elif graph_type == "Moving Averages":
        fig.add_trace(go.Scatter(x=x_axis, y=df_plot["Close"], name="Price"))
        fig.add_trace(go.Scatter(x=x_axis, y=df_plot["MA7_plot"], mode='lines', name="MA7", line=dict(width=3)))
        fig.add_trace(go.Scatter(x=x_axis, y=df_plot["MA30_plot"], mode='lines', name="MA30", line=dict(width=3, dash='dash')))

    # Glow dot
    fig.add_trace(go.Scatter(
        x=[x_axis.iloc[-1]],
        y=[df_plot["Close"].iloc[-1]],
        mode="markers",
        marker=dict(size=15, color="rgba(0,255,0,0.3)"),
        showlegend=False
    ))

    # Main dot
    fig.add_trace(go.Scatter(
        x=[x_axis.iloc[-1]],
        y=[df_plot["Close"].iloc[-1]],
        mode="markers",
        marker=dict(size=8, color="lime", line=dict(width=2, color="white")),
        name="Live"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=500,
        hovermode="x unified"
    )
    #main chart
    st.plotly_chart(fig, use_container_width=True)

    #  SIGNAL 
    st.markdown("### 🤖 Trading Recommendation")

    if pred_price > latest["Close"]:
        st.success("BUY 🟢")
    else:
        st.error("SELL 🔴")

    st.progress(confidence)
    st.caption(f"Model Confidence: {confidence:.2f}")
    #left right
    left_col, right_col = st.columns([2, 1])
    with left_col:
        st.markdown("### 📊 Model Evaluation")
        metrics=evaluate_model(model,X_scaled,y)
        col1, col2, col3 = st.columns(3)

        col1.metric("MAE", f"{metrics['MAE']:.4f}")
        col2.metric("RMSE", f"{metrics['RMSE']:.4f}")
        col3.metric("R²", f"{metrics['R2']:.2f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Precision", f"{metrics['Precision']:.2f}")
        col5.metric("Recall", f"{metrics['Recall']:.2f}")
        col6.metric("F1 Score", f"{metrics['F1']:.2f}")
    with right_col:
        st.markdown("### ⚙️ Live Features")

        st.info(f"MA7: {latest['MA7']:.2f}")
        st.info(f"MA30: {latest['MA30']:.2f}")
        st.info(f"RSI: {latest['RSI']:.2f}")
        st.info(f"TrendStrength: {latest['TrendStrength']:.4f}")
        st.info(f"Volatility: {latest['Volatility7']:.4f}")

# ======
# 🟡 PRICE CALCULATOR
# ======
elif page == "Price Calculator":

    st.title("⚖️ Price Calculator")

    weight = st.number_input("Enter Weight (grams)", min_value=1.0, step=1.0)

    price_per_gram = latest["Close"] / 31.1
    total_price = weight * price_per_gram

    st.success(f"💰 {weight}g {market} = ${total_price:.2f}")
    st.info("📌 Based on latest live price")

# =========
# 🔵 HISTORY PAGE
# =========
elif page == "History":

    st.title("📜 Market History Dashboard")

    df_hist = df.tail(100).reset_index()

    st.subheader("📉 Closing Price History")
    st.line_chart(df_hist["Close"])

    st.subheader("📊 Buy/Sell History")

    y_pred = model.predict(X_scaled)

    for i in range(len(y_pred)-20, len(y_pred)):
        signal_hist = "BUY" if y_pred[i] > 0 else "SELL"
        st.write(f"{signal_hist} → ${df['Close'].iloc[i]:.2f}")

# ========
# ℹ️ ABOUT
# ========
with st.expander("ℹ️ About Project & Developer"):
    st.markdown("""
### 🚀 Gold And Silver Trading Dashboard

A real-time machine learning powered trading assistant.

#### 👨‍💻 Developer:
Sanjay | Machine Learning + Trading Enthusiast
""")

st.markdown("---")
st.caption("🚀 Built with ML + Streamlit")        