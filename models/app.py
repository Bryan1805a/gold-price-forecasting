import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import yfinance as yf
import matplotlib.pyplot as plt
import os

# Web configuration
st.set_page_config(page_title="Gold Price Prediction AI")
st.title("Gold Price Prediction")
st.write("LSTM Model for Gold Price Prediction Using Historical Data")

# Load Model and Scaler (Using cache)
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('gold_price_lstm_model.keras')
    scaler = joblib.load('gold_price_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_resources()
    st.success("Loaded Model and Scaler")
except Exception as e:
    st.error(f"Can not find model: {e}")
    st.stop()

# Feature Engineering
def engineer_features(df):
    data = df[['Close']].copy()

    data['MA_15'] = data['Close'].rolling(window=15).mean()
    data['MA_60'] = data['Close'].rolling(window=60).mean()
    data['Lag_1'] = data['Close'].shift(1)
    data['Lag_2'] = data['Close'].shift(2)
    data['Lag_7'] = data['Close'].shift(7)
    data['Volatility_30'] = data['Close'].rolling(window=30).std()
    data['Pct_Change'] = data['Close'].pct_change()

    data.dropna(inplace=True)
    return data

# Use offline data if fail to download online
@st.cache_data(ttl=3600)
def fetch_data(ticker_symbol="GC=F"):
    try:
        data = yf.download(ticker_symbol, period="2y", progress=False)

        if len(data) > 0:
            return data, "Online (Yahoo Finance)"
        else:
            raise Exception("Empty")
    
    except Exception as e:
        if os.path.exists(r'..\data\raw\gold_price_dataset.csv'):
            data = pd.read_csv(r'..\data\raw\gold_price_dataset.csv', index_col='Date', parse_dates=True)
            return data, "Offline (Backup data)"
        else:
            return None, "Error"

# UI
st.header("1. Update market data")
days_to_fetch = st.slider("How many days?", 100, 365, 200)

if st.button("Get data and Predict"):
    with st.spinner("Connecting..."):
        raw_data, source = fetch_data()

        if raw_data is None:
            st.error("Error: Cannot download data from Yahoo Finance and Backup CSV File")
            st.stop()
        
        if "Offline" in source:
            st.warning(f"Warning: Yahoo Finance is being block (Rate Limit). App will now use Offline data")
        else:
            st.success(f"Connection successful! Source data: {source}")
            
        # Feature Engineering
        processed_data = engineer_features(raw_data)

        # Get the number of days
        display_data = processed_data.tail(days_to_fetch)

        # Draw chart
        st.subheader("Current gold price chart")
        st.line_chart(display_data['Close'])

        # Prediction
        st.header("2. Predict the next day")
        last_60_days = processed_data.tail(60)

        if len(last_60_days) < 60:
            st.error("Insufficient data")
        else:
            input_data = scaler.transform(last_60_days)

            # Reshape (1, 60, 6)
            X_future = np.array([input_data])

            # Predict
            prediction_scaled = model.predict(X_future)

            # Inverse Scaled
            dummy = np.zeros((1, scaler.n_features_in_))
            dummy[:, 0] = prediction_scaled.flatten()
            prediction_real = scaler.inverse_transform(dummy)[:, 0]

            # Show result
            current_price = last_60_days['Close'].iloc[-1]
            predicted_price = prediction_real[0]
            delta = predicted_price - current_price

            col1, col2 = st.columns(2)
            col1.metric("Close price today", f"${current_price:.2f}")
            col2.metric("Tomorrow price forecast", f"${predicted_price:.2f}", f"{delta:.2f} USD")

            if delta > 0:
                st.balloons()
                st.success("AI recommendation: Upward trend")
            else:
                st.warning("AI recommendation: Downtrend")