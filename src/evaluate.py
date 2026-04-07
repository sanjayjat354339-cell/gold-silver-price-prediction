from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def evaluate_model(model, X_scaled, y):

    preds = model.predict(X_scaled)

    # Regression
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
   

    # Convert to classification
    y_class = (y[1:] > y[:-1]).astype(int)
    pred_class = (preds[1:] > preds[:-1]).astype(int)

    precision = precision_score(y_class, pred_class)
    recall = recall_score(y_class, pred_class)
    f1 = f1_score(y_class, pred_class)
   

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }
    