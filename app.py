import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


model = load_model(r'C:\Users\ADMIN\Stock Price Prediction\Stock Predictions Model.keras')


st.header('Stock market prediction')
stock=st.text_input('Enter Stock Symbol','GOOG')
start='2012-01-01'
end='2022-12-31'

data=yf.download(stock,start,end)

st.subheader('Stock Data')
st.write(data)


data_train=pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test=pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

past_100_days=data_train.tail(100)
data_test=pd.concat([past_100_days,data_test],ignore_index=True)
data_test_scale=scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

x=[]
y=[]
for i in range(100,data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])
x,y=np.array(x),np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predited PRice')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'b', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)

# Calculate performance metrics
mse = mean_squared_error(y, predict)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, predict)
r2 = r2_score(y, predict)

# Display metrics
st.subheader('Model Performance Metrics')
st.write(f"Mean Squared Error (MSE): {mse:.4f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
st.write(f"R-squared (R²): {r2:.4f}")

