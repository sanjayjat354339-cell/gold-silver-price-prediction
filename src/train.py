from src.preprocess import load_and_prepare
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import os


def train_model():

    df = load_and_prepare()

    X = df[[
        "MA7",
        "MA30",
        "Lag1",
        "Lag2",
        "Lag3",
        "Volatility7"
    ]]

    y = df["Target"]

    split_index = int(len(df) * 0.8)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance:")
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/gold_model.pkl")

    print("\nModel trained and saved!")


if __name__ == "__main__":
    train_model()