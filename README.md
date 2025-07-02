# Stock Market Analysis and Prediction

## Description

This project analyzes and predicts the stock market behavior of NASDAQ and S&P 500 indices, particularly focusing on the period around the 2008 recession. The analysis involves time series forecasting using ARIMA and LSTM models.

## Dataset

The dataset consists of daily stock prices for NASDAQ (`^IXIC`) and S&P 500 (`^GSPC`) from January 1, 2007, to December 31, 2010. The data is fetched using the `yfinance` library.

## Methodology

### 1. Data Loading and Visualization

The historical stock data is loaded into a pandas DataFrame. The closing prices of both indices are visualized to observe their trends over time.

### 2. Stationarity Check

The Augmented Dickey-Fuller (ADF) test is used to check if the time series data is stationary. The initial test shows that the data is non-stationary. To address this, first-order differencing is applied to the data, which then becomes stationary.

### 3. ARIMA Model

An Autoregressive Integrated Moving Average (ARIMA) model is used for time series forecasting.

*   **Parameter Selection:** The `p`, `d`, and `q` parameters for the ARIMA model are determined by analyzing the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots. For this analysis, `p=1`, `d=2`, `q=1` are used.
*   **Training and Prediction:** The data is split into training and testing sets. The ARIMA model is trained on the training data and then used to make predictions on the test data.
*   **Evaluation:** The model's performance is evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

### 4. LSTM Model

A Long Short-Term Memory (LSTM) neural network, a type of recurrent neural network (RNN), is also implemented for stock prediction.

*   **Data Preparation:** The data is scaled using `MinMaxScaler` and then split into sequences for training the LSTM model.
*   **Model Architecture:** A sequential LSTM model is built with multiple LSTM layers and Dense layers.
*   **Training and Prediction:** The model is trained for a number of epochs, and then it predicts the stock prices.
*   **Evaluation:** The performance of the LSTM model is also evaluated using MAE, MSE, and RMSE.

## Results

The project provides a comparison of the predictive performance of ARIMA and LSTM models on the given stock market data. The visualizations of the predictions from both models against the actual test data are provided in the notebook.

## Dependencies

The following Python libraries are required to run the analysis:

*   pandas
*   numpy
*   matplotlib
*   plotly
*   yfinance
*   statsmodels
*   scikit-learn
*   tensorflow

You can install them using pip:
```bash
pip install pandas numpy matplotlib plotly yfinance statsmodels scikit-learn tensorflow
```

## Usage

1.  Clone the repository.
2.  Install the required dependencies.
3.  Open and run the `StockAnalysis.ipynb` notebook in a Jupyter environment.
