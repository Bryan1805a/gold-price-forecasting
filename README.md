# Gold Price Forecasting with Machine Learning
Time series forecasting with Gold Futures (GC=F) using Yahoo Finance data
and machine learning models in Jupyter Notebook

## Motivation
Gold price is influenced by macroeconomic factors and market sentiment.
This project explores whether historical price data alone can be used
to forecast short-term gold futures prices.

## Tech Stack
- Python 3.10 (Miniconda)
- yfinance
- pandas, numpy
- scikit-learn
- matplotlib / seaborn
- Jupyter Notebook

## Data Source
- Yahoo Finance
- Ticker: `GC=F` (Gold Futures)
- Frequency: Daily

## Project Structure
.
├── data/
│   └── raw/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── models/
├── environment.yml
└── README.md

## Modeling Approach
- Baseline: Moving Average
- Machine Learning Models:
  - Linear Regression
  - Random Forest
- Evaluation metrics:
  - RMSE
  - MAE

## Results
The Random Forest model outperformed the baseline moving average,
achieving lower RMSE on the test set.

## Limitations
- Only historical price data is used
- No macroeconomic indicators included
- Financial markets are inherently noisy and non-stationary

## Future Work
- Add macroeconomic indicators (USD index, interest rates)
- Try LSTM / Transformer models
- Walk-forward validation

## How to Run
```bash
conda env create -f environment.yml
conda activate gold-forecast
jupyter notebook