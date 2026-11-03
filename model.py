import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

MODEL_PATH = "saved_models/random_forest_model.joblib"
CAT_COLS = ['City', 'Locality', 'Property_Type', 'Furnished_Status', 'Parking_Space']
NUM_COLS = ['Size_in_SqFt', 'BHK', 'Bathrooms']
TARGET = 'Price_in_Lakhs'

def clean_data(df):
    """Clean data: remove nulls/duplicates and format data types."""
    df = df.drop_duplicates().dropna(subset=['City', 'Size_in_SqFt', TARGET]).copy()
    df['BHK'] = pd.to_numeric(df['BHK'], errors='coerce').fillna(1).astype(int)
    df['Bathrooms'] = pd.to_numeric(df.get('Bathrooms', df['BHK']), errors='coerce').fillna(1).astype(int)
    df['Size_in_SqFt'] = pd.to_numeric(df['Size_in_SqFt'], errors='coerce').fillna(1000.0)
    df['Price_in_Lakhs'] = pd.to_numeric(df['Price_in_Lakhs'], errors='coerce').fillna(50.0)
    for col in CAT_COLS:
        df[col] = df[col].astype(str).str.strip()
    return df

def train_and_evaluate(df):
    """Train Random Forest on 80/20 train/test split of 1,000 records."""
    df = clean_data(df)
    X, y = df[CAT_COLS + NUM_COLS], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ('prep', ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS),
            ('num', StandardScaler(), NUM_COLS)
        ])),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=14, random_state=42))
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
    accuracy = round(max(0.0, min(100.0, 100.0 - mape)), 2)
    mse = float(mean_squared_error(y_test, y_pred))
    metrics = {
        "model_name": "Random Forest Regressor",
        "train_records": len(X_train),
        "test_records": len(X_test),
        "accuracy": accuracy,
        "r2_score": round(float(r2_score(y_test, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 2),
        "mse": round(mse, 2),
        "rmse": round(float(np.sqrt(mse)), 2)
    }

    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    return metrics, pipe

def predict_price(input_dict):
    """Predict price in Lakhs and INR using the trained Random Forest model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not trained yet. Train the model first.")
    pipe = joblib.load(MODEL_PATH)
    pred_lakhs = max(1.0, round(float(pipe.predict(pd.DataFrame([input_dict]))[0]), 2))
    return pred_lakhs, round(pred_lakhs * 100000, 2), "Random Forest Regressor"
