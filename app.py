import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from PIL import Image
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

DATASET_DIR = "dataset"
MODELS_DIR = "models and scalers"
PREDICTIONS_DIR = "predictions"

@st.cache_data
def load_data():
    df = pd.read_csv(f"{DATASET_DIR}/TCS_stock_history.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

@st.cache_resource
def load_models_and_scalers():
    lstm_model = load_model(f"{MODELS_DIR}/tcs_lstm_model.h5", compile=False)
    rf_model = joblib.load(f"{MODELS_DIR}/tcs_rf_model.pkl")
    xgb_model = joblib.load(f"{MODELS_DIR}/tcs_xgb_model.pkl")

    scaler_lstm = joblib.load(f"{MODELS_DIR}/scaler_lstm.pkl")
    scaler_rf = joblib.load(f"{MODELS_DIR}/scaler_rf.pkl")
    scaler_xgb = joblib.load(f"{MODELS_DIR}/scaler_xgb.pkl")

    return lstm_model, rf_model, xgb_model, scaler_lstm, scaler_rf, scaler_xgb

def preprocess_features(df):
    df = df.copy()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['MA_7'] = df['Close'].rolling(7).mean()
    df['MA_30'] = df['Close'].rolling(30).mean()
    df['Vol_7'] = df['Log_Return'].rolling(7).std() * np.sqrt(7)
    df['Vol_30'] = df['Log_Return'].rolling(30).std() * np.sqrt(30)
    df['Price_Change'] = df['Close'].diff()
    df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
    df['OHLC_Avg'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    for lag in [1, 2, 3, 7]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
    df.dropna(inplace=True)
    return df

def predict_lstm(lstm_model, scaler_lstm, df):
    sequence_length = 60
    data = df[['Close']].values[-sequence_length:]
    data_scaled = scaler_lstm.transform(data)
    X = np.reshape(data_scaled, (1, sequence_length, 1))
    pred_scaled = lstm_model.predict(X)
    pred = scaler_lstm.inverse_transform(pred_scaled)[0][0]
    return pred

def predict_rf_xgb(model, scaler, df):
    features = ['Open', 'High', 'Low', 'Volume', 'MA_7', 'MA_30', 'Vol_7', 'Vol_30',
                'Price_Change', 'High_Low_Pct', 'OHLC_Avg', 'Close_Lag_1', 'Close_Lag_2',
                'Close_Lag_3', 'Close_Lag_7']
    latest = df[features].iloc[-1:].values
    latest_scaled = scaler.transform(latest)
    pred = model.predict(latest_scaled)[0]
    return pred

def main():
    st.set_page_config(page_title="TCS Stock Dashboard", layout="wide")
    st.title("📈 TCS Stock Price Prediction Dashboard")

    # Load data and models
    df = load_data()
    lstm_model, rf_model, xgb_model, scaler_lstm, scaler_rf, scaler_xgb = load_models_and_scalers()
    df_processed = preprocess_features(df)

    # Sidebar navigation
    menu = st.sidebar.radio(
        "Navigate",
        ["Dashboard", "Model Predictions", "Model Performance", "SHAP Importance", "Raw Data","Model Details"]
    )

    # -----------------------------
    if menu == "Dashboard":
        st.subheader("📊 Stock Closing Price History")
        st.line_chart(df['Close'])

        st.subheader("📌 Latest Market Snapshot")
        st.write(df.tail(5))

    # -----------------------------
    elif menu == "Model Predictions":
        st.subheader("🧠 Prediction for Next Day")

        if st.button("🔮 Predict with LSTM", key="predict_lstm"):
            pred = predict_lstm(lstm_model, scaler_lstm, df)
            st.success(f"LSTM Predicted Close Price: ₹{pred:.2f}")

        if st.button("🌲 Predict with Random Forest", key="predict_rf"):
            pred = predict_rf_xgb(rf_model, scaler_rf, df_processed)
            st.success(f"Random Forest Predicted Close Price: ₹{pred:.2f}")

        if st.button("📦 Predict with XGBoost", key="predict_xgb"):
            pred = predict_rf_xgb(xgb_model, scaler_xgb, df_processed)
            st.success(f"XGBoost Predicted Close Price: ₹{pred:.2f}")

    # -----------------------------
    elif menu == "Model Performance":
        st.subheader("📊 Model Performance Metrics")

        metrics_data = {
            "Model": ["XGBoost", "Random Forest", "LSTM"],
            "MAE": [836.449393, 829.211660, 127.080590],
            "MSE": [1.047715e+06, 1.035629e+06, 2.787096e+04],
            "R2": [-1.974819, -1.940504, 0.919695],
        }

        metrics_df = pd.DataFrame(metrics_data)
        metrics_df["MAE"] = metrics_df["MAE"].map("{:.2f}".format)
        metrics_df["MSE"] = metrics_df["MSE"].map(lambda x: f"{x:.2e}")
        metrics_df["R2"] = metrics_df["R2"].map("{:.6f}".format)

        st.table(metrics_df.set_index("Model"))

    # -----------------------------
    elif menu == "SHAP Importance":
        st.subheader("📉 SHAP Feature Importance")

        shap_image_path = f"{PREDICTIONS_DIR}/shap_summary.png"

        st.image(shap_image_path, caption="SHAP Feature Importance for XGBoost", width=500)

        with open(shap_image_path, "rb") as file:
            st.download_button(
                label="📥 Download SHAP Plot",
                data=file,
                file_name="shap_summary.png",
                mime="image/png"
            )

    # -----------------------------
    elif menu == "Raw Data":
        st.subheader("📁 Raw Stock Data")
        st.dataframe(df)

        st.download_button("📥 Download CSV", df.to_csv().encode(), "tcs_stock_data.csv", "text/csv")
   
    elif menu == "Model Details":
        st.subheader("🧠 Model Information Cards")

        with st.expander("🔮 LSTM (Long Short-Term Memory)"):
            st.markdown("""
            - **Type**: Deep learning (RNN-based)  
            - **Input**: Last 60 days of Close prices  
            - **Architecture**: 
                - 2 LSTM layers  
                - 1 Dense output layer  
            - **Loss Function**: Mean Squared Error (MSE)  
            - **Optimizer**: Adam  
            - **Pros**:
                - Captures temporal dependencies
                - Good at sequential forecasting  
            - **Cons**:
                - Slower to train
                - Needs large data to perform well
            """)

        with st.expander("🌲 Random Forest Regressor"):
            st.markdown("""
            - **Type**: Ensemble machine learning model  
            - **Input Features**: OHLC, Moving Averages, Volatility, Lagged values, etc.  
            - **Training**: 100+ decision trees on bootstrapped samples  
            - **Pros**:
                - Fast and interpretable
                - Handles non-linear data well  
            - **Cons**:
                - Can overfit if not tuned
                - No sequence memory
            """)

        with st.expander("📦 XGBoost Regressor"):
            st.markdown("""
            - **Type**: Gradient Boosted Decision Trees  
            - **Input Features**: Similar to RF  
            - **Training**: Optimized with gradient descent and regularization  
            - **Pros**:
                - Excellent performance on tabular data  
                - Built-in regularization prevents overfitting  
            - **Cons**:
                - Requires hyperparameter tuning  
                - Less intuitive than RF
            """)


if __name__ == "__main__":
    main()
