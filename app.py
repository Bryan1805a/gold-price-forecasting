import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import os

# Web configuration
st.set_page_config(page_title="Gold Price Prediction AI", page_icon="💰")
st.title("💰 AI Gold Price Prediction")
st.caption("Server-side Training")

# Data loading
@st.cache_data(ttl=3600)
def fetch_data(ticker_symbol="GC=F"):
    try:
        data = yf.download(ticker_symbol, period="5y", progress=False)
        if len(data) > 0:
            return data, "Online (Yahoo Finance)"
        else:
            raise Exception("Empty Data")
    except Exception as e:
        csv_path = 'gold_price_dataset.csv' 
        if os.path.exists(csv_path):
            data = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
            return data, "Offline (Backup CSV)"
        else:
            return None, "Error"

# Train model
@st.cache_resource 
def build_and_train_model(df):
    # Prepare data
    data = df[['Close']].copy()
    data = data.dropna()
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    time_steps = 60
    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i : i + time_steps])
        y.append(scaled_data[i + time_steps, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Reshape for LTSM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build model
    model = tf.keras.models.Sequential()
    # Input layer
    model.add(tf.keras.layers.Input(shape=(X.shape[1], 1)))
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Training
    with st.spinner('Training model...'):
        model.fit(X, y, epochs=20, batch_size=32, verbose=0)
    
    return model, scaler

# UI

# Get data
raw_data, source = fetch_data()

if raw_data is None:
    st.error("Cannot find data")
    st.stop()
else:
    st.success(f"Data source: {source}")

# Train model in Streamlit server
try:
    model, scaler = build_and_train_model(raw_data)
    st.success("Model ready")
except Exception as e:
    st.error(f"Error while training: {e}")
    st.stop()

# Feature Engineering
def engineer_features(df):
    data = df[['Close']].copy()
    data['MA_15'] = data['Close'].rolling(window=15).mean()
    data['MA_60'] = data['Close'].rolling(window=60).mean()
    return data

processed_data = engineer_features(raw_data)

st.header("1. Market chart")
days = st.slider("How many days?", 100, 500, 200)
st.line_chart(processed_data['Close'].tail(days))

# Predict
st.header("2. Tomorrow prediction")
if st.button("Predict now"):
    # Get last 60 days
    last_60_days = raw_data[['Close']].tail(60)
    
    if len(last_60_days) < 60:
        st.error("Not enough 60 days.")
    else:
        # Scale
        input_data = scaler.transform(last_60_days)
        # Reshape
        X_future = np.array([input_data])
        # Predict
        prediction_scaled = model.predict(X_future)
        # Inverse
        prediction = scaler.inverse_transform(prediction_scaled)
        
        real_price_today = last_60_days.iloc[-1].item()
        pred_price = prediction[0][0]
        delta = pred_price - real_price_today
        
        c1, c2 = st.columns(2)
        c1.metric("Today price", f"${real_price_today:.2f}")
        c2.metric("AI Prediction", f"${pred_price:.2f}", f"{delta:.2f}")
        
        if delta > 0:
            st.balloons()