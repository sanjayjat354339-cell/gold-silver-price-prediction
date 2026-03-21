import joblib
import pandas as pd
from preprocess import load_and_prepare

def predict_next():
    # Load trained model
    model = joblib.load("models/gold_model.pkl")

    # Load data
    df = load_and_prepare()

    # Get latest close price
    latest_close = df.iloc[-1]["Close"]

    # Predict next day
    prediction = model.predict([[latest_close]])

    print("Latest Close Price:", latest_close)
    print("Predicted Next Day Price:", prediction[0])

if __name__ == "__main__":
    predict_next()