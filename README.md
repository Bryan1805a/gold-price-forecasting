# Gold Price Prediction App (End-to-End LSTM Project)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gold-price-forecasting-test.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

An End-to-End Machine Learning Web Application that forecasts Gold Futures (GC=F) prices using Long Short-Term Memory (LSTM) neural networks. The app fetches real-time data from Yahoo Finance, processes technical indicators, and performs time-series forecasting.

## Live Demo
Check out the live app here: **https://gold-price-forecasting-test.streamlit.app/**

## Key Features

- **Hybrid Data Pipeline:**
  - **Online Mode:** Fetches real-time data using `yfinance` API.
  - **Offline Fallback:** Automatically switches to a local backup dataset (`.csv`) if the API fails or hits rate limits (Robust Error Handling).
  
- **Advanced Feature Engineering:**
  - Uses a **60-day Sliding Window** approach.
  - Integrates technical indicators:
    - Moving Averages (MA15, MA60) for trend detection.
    - Volatility (Standard Deviation) for risk assessment.
    - Lag Features & Daily Returns.

- **Deep Learning Model (LSTM):**
  - Implemented using **TensorFlow/Keras**.
  - **Server-side Training:** The model is trained *on-the-fly* when the app starts. This ensures 100% compatibility with the deployment environment (Streamlit Cloud) and eliminates "version mismatch" issues between local and cloud environments.

- **Interactive Dashboard:**
  - Built with **Streamlit**.
  - Visualize historical trends and real-time predictions.

## Tech Stack

- **Core:** Python 3.10
- **Deep Learning:** TensorFlow, Keras
- **Data Manipulation:** Pandas, NumPy, Scikit-learn (MinMaxScaler)
- **Data Source:** Yfinance (Yahoo Finance API)
- **Visualization:** Matplotlib, Streamlit Charts

## Project Structure

```bash
.
|-- data/
|     |-- raw/
|-- models/
|-- notebooks/
|-- app.py
|-- convert_model.py
|-- README.md
|-- requirements.txt
```

## How to Run Locally
- 1.Clone the repository:
```base
git clone [https://github.com/Bryan1805a/gold-price-forecasting.git](https://github.com/Bryan1805a/gold-price-forecasting.git)
cd gold-price-forecasting
```
- 2.Create a virtual environment (Recommended):
```base
conda create -n gold-ai python=3.10
conda activate gold-ai
```

- 3.Install dependencies:
```base
pip install -r requirements.txt
```

- 4.Run the App:
```base
streamlit run app.py
```

## Model Architecture
The project uses a Stacked LSTM architecture designed for time-series data:

1.Input Layer: Sequence of 60 days (Features: Close Price).
2.LSTM Layer 1: 50 Units (Return Sequences = True).
3.Dropout: 20% (To prevent overfitting).
4.LSTM Layer 2: 50 Units.
5.Dense Output Layer: Predicts the next day's closing price.

## Limitations & Future Work
- Current Limitation: The model relies heavily on historical price action (Technical Analysis) and does not yet account for macroeconomic news (e.g., Fed rates, Inflation data).

- Future Improvements:
  - Integrate Sentiment Analysis from financial news.
  - Deploy model as a FastAPI microservice.
  - Add Backtesting module to simulate trading strategies.

# Disclaimer:
This project is for educational purposes only. Do not use this for real financial trading.
