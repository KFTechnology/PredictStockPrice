import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st 
from datetime import datetime
import yfinance as yf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import backend as Backend
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
import numpy as np

st.set_page_config(page_title="Prediction Stock", page_icon="📈", layout="wide")

# ===== Custom Styling =====
st.markdown("""
<style>
.toolbar {
    background-color: #2C3E50;
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
""", unsafe_allow_html=True)


with st.container():
    st.markdown('<div class="toolbar">', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("Image/logo.png", width=120)
    with col2:
        st.markdown("<h1>Predicting stock prices using artificial intelligence</h1>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    
with st.container():
     st.markdown('<div class="toolbar">', unsafe_allow_html=True)

StockInput = st.selectbox(
    "Select a stock symbol:", 
    ["TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA"]
)
start_date = st.date_input("Select Start Date", value=datetime(2025, 1, 1))

if StockInput:
    Today = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(StockInput, start=start_date, end=Today)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    df = data[["Close"]]

    st.subheader("📊 Historical Data")
    if df.empty:
        st.warning("No data found for this stock symbol. Try another one.")
    else:
        st.line_chart(df)
        st.dataframe(df.tail(10))

 
    Scalar = MinMaxScaler(feature_range=(0, 1))
    Scaled = Scalar.fit_transform(df)
    sequence_length = 30

    X, Y = [], []
    for i in range(sequence_length, len(Scaled)):
        X.append(Scaled[i-sequence_length:i, 0])
        Y.append(Scaled[i, 0])
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=30, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=30))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, Y, epochs=2, batch_size=32, verbose=0)

 
    last_days = Scaled[-100:]
    last_days = np.reshape(last_days, (1, last_days.shape[0], 1))
    predict_data = model.predict(last_days)
    predict_data = Scalar.inverse_transform(predict_data)

    st.subheader(f"📈 Live Update for {StockInput} (Today)")
    table_placeholder = st.empty()

  
    latest_price = data["Close"].iloc[-1]
    new_row = {"Time": pd.Timestamp.now(), "Value": latest_price}
    today_data = pd.DataFrame([new_row])
    table_placeholder.dataframe(today_data)


    st.subheader("💡 Predicted Stock Price")
    st.table({
        f"Predicted Price for {StockInput} (USD)": [f"{predict_data[0][0]:.2f} $"]
    })


    st.write("⬇️ Press the button below to show the graph")

    if st.button("Show Forecast Graph"):
        real_data = Scalar.inverse_transform(Y.reshape(-1, 1))

        fig = plt.figure(figsize=(10, 5))
        plt.plot(real_data, color='blue', label='Actual Price')
        plt.plot([None]*(len(real_data)-1) + [predict_data[0][0]], 
                 color='red', marker='o', label='Predicted Price')
        plt.title(f'Price Forecast for {StockInput} Using LSTM')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig)
        
        Backend.clear_session()
        

        
        
                
        
    
        
        
    