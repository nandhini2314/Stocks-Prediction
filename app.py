import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st

start= '2010-01-01'
end = '2022-12-31'
st.title('Stock Trend Prediction')
user_input =st.text_input ('Enter stock ticker','AAPL')
df=yf.download('AAPL',start=start,end= end)

st.subheader('Data from 2010 to 2022')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close, label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 Moving Averages')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, label='100MA')
plt.plot(df.Close, label='Original Price' )
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 and 200 Moving Averages')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))

plt.plot(ma100,'g', label='100MA')
plt.plot(ma200, label='200MA')
plt.plot(df.Close, label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)
#dont need x_train and y_train because model is already trained
model=load_model('keras_model.keras')

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days , data_testing],ignore_index=True)
input_data = scaler.fit_transform(final_df) 

x_test=[]
y_test = []

for i in range(100 ,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test,y_test =np.array(x_test), np.array(y_test)

y_predicted= model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Predicted value vs Original')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test , 'b' , label='Original Price')
plt.plot(y_predicted , 'r' , label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)