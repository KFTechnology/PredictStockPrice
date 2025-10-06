import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st 
from datetime import datetime
import yfinance as yf 
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
import numpy as np 
import time



st.set_page_config(page_title="Prediction Stock", page_icon="📈", layout="wide", )


st.markdown(
    """
    <style>
    .toolbar {
        background-color: #2C3E50;   /* dark blue-gray */
        padding: 10px;
        border-radius: 8px;
    }
    .toolbar h2 {
        color: white;
        text-align: center;
        margin: 0;
        line-height: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


with st.container():
    st.markdown('<div class="toolbar">', unsafe_allow_html=True)

    col1, col2 = st.columns([1,3])

    with col1:
        st.image("Image/logo.png", width=120)

    with col2:
        st.markdown("<h1>Predicting stock prices using artificial intelligence</h1>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
     st.markdown('<div class="toolbar">', unsafe_allow_html=True)
     
     
     StockInput = st.text_input("Enter the stock symbol for example TSLA")
     start_date = st.date_input("Select Start Date", value=datetime(2025, 1, 1))
     
    if StockInput: 
      Today = datetime.today().strftime('%Y-%m-%d')
      data = yf.download(StockInput, start=start_date, end=Today) 

      if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0) 

      df = data[["Close"]]

      st.subheader("Historical Prediction")
      if df.empty: 
        st.write("No data found for this stock symbol. Try another one.")
      else: 
        for i in range(100):
            
            x = st.line_chart(df)
            fig, ax = plt.subplots()
            ax.plot(x, y)
            x.pyplot(fig)
            time.sleep(0.05)
            
        #st.line_chart(df)
        #st.dataframe(df)


      Scalar = MinMaxScaler(feature_range=(0,1))
      Scaled = Scalar.fit_transform(df)
      sequence_length = 60
      
    
      train_size = int(len(Scaled) * 0.8)
      train_data = Scaled[:train_size]
      test_data = Scaled[train_size:]
      
      sequence_length = 60
      if len(Scaled) <= sequence_length:
       st.write("Not enough data to build sequences. Need more historical data.")
      else:
       train_size = int(len(Scaled) * 0.8)
       train_data = Scaled[:train_size]
       test_data = Scaled[train_size:]

       X_train = []
       Y_train = []  

  
       for i in range(sequence_length, len(train_data)):
        X_train.append(train_data[i-sequence_length:i, 0])
        Y_train.append(train_data[i, 0])
        
       X_train, Y_train = np.array(X_train), np.array(Y_train)
       X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))


       X_test, Y_test = [], []
       for i in range(sequence_length, len(test_data)):
        X_test.append(test_data[i-sequence_length:i, 0])
        Y_test.append(test_data[i, 0])
        
       X_test = np.array(X_test)
       
       if len(X_test.shape) == 2:
           X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
       elif len(X_test.shape) == 1: 
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
       X_test, Y_test = np.array(X_test), np.array(Y_test)
       X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
       
      model = Sequential()
      model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
      model.add(Dropout(0.2))
      model.add(LSTM(units=50))
      model.add(Dropout(0.2))
      model.add(Dense(1))
      model.compile(optimizer="adam", loss="mean_squared_error")
      model.fit(X_train, Y_train, epochs=5, batch_size=32, verbose=0)

   
      predictions = model.predict(X_test)
      predictions = Scalar.inverse_transform(predictions)
      real_test = Scalar.inverse_transform(np.array(Y_test).reshape(-1,1))

      
      last_days = Scaled[-100:]
      last_days = last_days.reshape((1, last_days.shape[0],1))
      next_day_pred = model.predict(last_days)
      next_day_pred = Scalar.inverse_transform(next_day_pred)
      st.subheader("Next Day Prediction")
      st.write(f"{next_day_pred[0][0]:.2f} Dollar")

    
      fig, ax = plt.subplots(figsize=(10,5))
      ax.plot(real_test, color='blue', label='Actual Price')
      ax.plot(predictions, color='red', label='Predicted Price')
      ax.set_title(f'Price Forecast {StockInput} Using LSTM')
      ax.set_xlabel('Time')
      ax.set_ylabel('Price')
      ax.legend()
      st.pyplot(fig)