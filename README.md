
# Stock Price Prediction Using LSTM

This project implements a **Stock Price Prediction Model** using **Long Short-Term Memory (LSTM)** networks to forecast future stock prices based on historical stock data. The model utilizes **machine learning** techniques to predict stock prices and evaluate the performance of the model using metrics like Mean Squared Error (MSE).

## Project Overview

The goal of this project is to predict stock prices using past data, leveraging the power of LSTM, which is well-suited for time series forecasting. The project follows the steps below:

1. **Data Collection**: Historical stock price data is collected using the `yfinance` library.
2. **Data Preprocessing**: Data is scaled using `MinMaxScaler` for normalization, and the dataset is split into training and test sets.
3. **Model Building**: An LSTM neural network is built with multiple layers and dropout regularization.
4. **Model Training and Evaluation**: The model is trained on 80% of the data, and performance is evaluated on the remaining 20%.
5. **Prediction**: Future stock prices are predicted and visualized, comparing the predicted values to actual values.

## Technologies Used

- **Python**: Programming language used to implement the project.
- **TensorFlow/Keras**: For building and training the LSTM model.
- **yfinance**: To fetch historical stock price data.
- **Scikit-Learn**: For data preprocessing, such as scaling the data.
- **Matplotlib**: To plot and visualize stock price data and predictions.
- **Streamlit**: To create a simple web app for predicting stock prices.
