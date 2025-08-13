import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import warnings

# Suppress sklearn warnings about feature names
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

# Directory paths
MODELS_DIR = "models and scalers"

# Feature names (must match training order)
FEATURE_NAMES = [
    'Open', 'High', 'Low', 'Volume',
    'MA_7', 'MA_30', 'Vol_7', 'Vol_30',
    'Price_Change', 'High_Low_Pct', 'OHLC_Avg',
    'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_7'
]

# ================================
# Load Data from Local CSV
# ================================
@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv("dataset/tcs_stock_data.csv", parse_dates=True, index_col=0)
        st.success(f"✅ Loaded data: {len(df)} rows, up to {df.index[-1].date()}")
        return df
    except Exception as e:
        st.error(f"Failed to load local CSV: {e}")
        return None

# ================================
# Load Models and Scalers
# ================================
@st.cache_resource
def load_models_and_scalers():
    try:
        lstm_model = load_model(f"{MODELS_DIR}/tcs_lstm_model.h5", compile=False)
        rf_model = joblib.load(f"{MODELS_DIR}/tcs_rf_model.pkl")
        xgb_model = joblib.load(f"{MODELS_DIR}/tcs_xgb_model.pkl")
        scaler_lstm = joblib.load(f"{MODELS_DIR}/scaler_lstm.pkl")
        scaler_rf = joblib.load(f"{MODELS_DIR}/scaler_rf.pkl")
        scaler_xgb = joblib.load(f"{MODELS_DIR}/scaler_xgb.pkl")
        return lstm_model, rf_model, xgb_model, scaler_lstm, scaler_rf, scaler_xgb
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# ================================
# Feature Engineering
# ================================
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

# ================================
# LSTM Prediction for Historical Date
# ================================
def predict_lstm_for_date(lstm_model, scaler_lstm, df, date):
    if date not in df.index:
        return None
    idx = df.index.get_loc(date)
    if idx < 60:
        return None
    seq = df['Close'].iloc[idx-60:idx].values.reshape(-1, 1)
    seq_scaled = scaler_lstm.transform(seq)
    X = np.reshape(seq_scaled, (1, 60, 1))
    pred_scaled = lstm_model.predict(X, verbose=0)
    return scaler_lstm.inverse_transform(pred_scaled)[0][0]

# ================================
# RF/XGBoost Prediction for Historical Date
# ================================
def predict_rf_xgb_for_date(model, scaler, df, date):
    if date not in df.index:
        return None
    row = df.loc[date]
    X = row[FEATURE_NAMES].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)[0]

# ================================
# Predict Tomorrow
# ================================
def predict_tomorrow(lstm_model, rf_model, xgb_model, scaler_lstm, scaler_rf, scaler_xgb, df_processed):
    last_date = df_processed.index[-1]
    st.info(f"Using data up to: {last_date.strftime('%Y-%m-%d')}")

    # LSTM: Predict next close
    close_seq = df_processed['Close'].tail(60).values.reshape(-1, 1)
    close_scaled = scaler_lstm.transform(close_seq)
    X_lstm = np.reshape(close_scaled, (1, 60, 1))
    pred_lstm_scaled = lstm_model.predict(X_lstm, verbose=0)
    pred_lstm = scaler_lstm.inverse_transform(pred_lstm_scaled)[0][0]

    # RF/XGBoost: Build future feature vector
    today = df_processed.iloc[-1]
    prev_close = df_processed['Close'].iloc[-2]

    future_row = {
        'Open': today['Close'],
        'High': today['Close'] * 1.01,
        'Low': today['Close'] * 0.99,
        'Volume': today['Volume'],
        'MA_7': today['MA_7'],
        'MA_30': today['MA_30'],
        'Vol_7': today['Vol_7'],
        'Vol_30': today['Vol_30'],
        'Price_Change': today['Close'] - prev_close,
        'High_Low_Pct': (today['High'] - today['Low']) / today['Close'],
        'OHLC_Avg': today[['Open', 'High', 'Low', 'Close']].mean(),
        'Close_Lag_1': today['Close'],
        'Close_Lag_2': df_processed['Close_Lag_2'].iloc[-1],
        'Close_Lag_3': df_processed['Close_Lag_3'].iloc[-1],
        'Close_Lag_7': df_processed['Close_Lag_7'].iloc[-1]
    }

    X_df = pd.DataFrame([list(future_row[f] for f in FEATURE_NAMES)], columns=FEATURE_NAMES)
    X_rf_scaled = scaler_rf.transform(X_df)
    X_xgb_scaled = scaler_xgb.transform(X_df)
    pred_rf = rf_model.predict(X_rf_scaled)[0]
    pred_xgb = xgb_model.predict(X_xgb_scaled)[0]

    return pred_lstm, pred_rf, pred_xgb

# ================================
# Main App
# ================================
def main():
    st.set_page_config(page_title="TCS Stock Prediction", layout="wide")
    st.title("TCS Stock Price Prediction Dashboard")

    df = load_data()
    if df is None:
        return

    df_processed = preprocess_features(df)
    model_data = load_models_and_scalers()
    if model_data is None:
        return
    lstm_model, rf_model, xgb_model, scaler_lstm, scaler_rf, scaler_xgb = model_data

    menu = st.sidebar.radio("Navigation", [
        "Dashboard",
        "Model Predictions",
        "Model Performance",
        "Raw Data",
        "Model Details"
    ])

    if menu == "Dashboard":
        st.subheader("Stock Closing Price History")
        st.line_chart(df['Close'])
        st.subheader("Latest Market Data")
        st.dataframe(df.tail())

    elif menu == "Model Predictions":
        st.subheader("Stock Price Predictions")
        option = st.radio("Prediction Mode", ["Predict Tomorrow", "Predict Historical Date"])
        if option == "Predict Historical Date":
            available_dates = df_processed.index.strftime("%Y-%m-%d").tolist()
            selected_date_str = st.selectbox("Select Date", available_dates, index=len(available_dates)-1)
            selected_date = pd.to_datetime(selected_date_str)

            st.write(f"### Predictions for {selected_date_str}")
            pred_lstm = predict_lstm_for_date(lstm_model, scaler_lstm, df_processed, selected_date)
            pred_rf = predict_rf_xgb_for_date(rf_model, scaler_rf, df_processed, selected_date)
            pred_xgb = predict_rf_xgb_for_date(xgb_model, scaler_xgb, df_processed, selected_date)

            if pred_lstm is not None:
                st.write(f"**LSTM Prediction**: ₹{pred_lstm:.2f}")
            else:
                st.write("LSTM: Not enough history (requires 60 prior days)")
            st.write(f"Random Forest Prediction: ₹{pred_rf:.2f}")
            st.write(f"XGBoost Prediction: ₹{pred_xgb:.2f}")

        else:
            tomorrow_date = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            st.subheader(f"Predicting Closing Price for: {tomorrow_date}")
            with st.spinner("Computing predictions..."):
                pred_lstm, pred_rf, pred_xgb = predict_tomorrow(
                    lstm_model, rf_model, xgb_model,
                    scaler_lstm, scaler_rf, scaler_xgb, df_processed
                )
                st.write(f"**LSTM Prediction (Recommended)**: ₹{pred_lstm:.2f}")
                st.write(f"Random Forest Prediction: ₹{pred_rf:.2f}")
                st.write(f"XGBoost Prediction: ₹{pred_xgb:.2f}")
            st.info(f"📌 Forecast based on data up to {df_processed.index[-1].strftime('%Y-%m-%d')}")

    elif menu == "Model Performance":
        st.subheader("Model Evaluation Metrics")
        metrics = pd.DataFrame({
            "Model": ["XGBoost", "Random Forest", "LSTM"],
            "MAE": [836.45, 829.21, 127.08],
            "MSE": [1.05e6, 1.04e6, 2.79e4],
            "R²": [-1.97, -1.94, 0.92]
        }).set_index("Model")
        st.table(metrics.round(4))

    elif menu == "Raw Data":
        st.subheader("Stock Data")
        st.dataframe(df)
        st.download_button(
            label="Download CSV",
            data=df.to_csv(),
            file_name="tcs_stock_data.csv",
            mime="text/csv"
        )

    elif menu == "Model Details":
        st.subheader("Model Information")
        st.write("**LSTM Model**")
        st.write("- Type: Deep Learning (Recurrent Neural Network)")
        st.write("- Input: 60 days of closing prices")
        st.write("- Architecture: 2 LSTM layers, 1 dense output")
        st.write("- Captures temporal patterns in stock prices")
        st.write("**Random Forest**")
        st.write("- Type: Ensemble Tree-Based Model")
        st.write("- Features: OHLC, volume, moving averages, volatility, lagged values")
        st.write("- Robust to outliers and non-linear relationships")
        st.write("**XGBoost**")
        st.write("- Type: Gradient Boosted Trees")
        st.write("- Optimized for performance and regularization")
        st.write("- Often outperforms Random Forest on structured data")

if __name__ == "__main__":
    main()
