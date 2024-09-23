import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import numpy as np
import tensorflow as tf
import os

# Define the start date and today's date
START = "2014-03-27"
TODAY = date.today().strftime("%Y-%m-%d")

def display_demo_page():
    #st.title('Stock Forecast Demo')

    # Define stock options for the user to select
    stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
    selected_stock = st.selectbox('Select dataset for prediction', stocks)

    # Allow the user to select the number of years for prediction
    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    # Load data from Yahoo Finance
    @st.cache_data
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    # Display a loading message while the data is being loaded
    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')

    # Display raw data
    st.subheader('Raw data')
    st.write(data.tail())

    # Plot raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Opening Price"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Closing Price"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    # Load the saved model
    model_path = 'Modelsaved/'
    model_filename = os.path.join(model_path, 'combined_trained_model.h5')

    model = tf.keras.models.load_model(model_filename)

    # Prepare input data for the model
    recent_data = data['Close'].values[-91:].astype(np.float32)  # Ensure float32 data type
    input_data = np.array([recent_data])  # Input shape (1, 91)

    # Predict for each forecast period
    forecast_dates = pd.date_range(start=data['Date'].max(), periods=period+1, freq='D')[1:]  # Generate forecast dates
    predictions = []

    for _ in range(len(forecast_dates)):
        # Generate input data for the model
        input_data = recent_data[-91:].reshape(1, -1)  # Ensure input has the correct shape (1, 91)
        prediction = model.predict(input_data)

        # Ensure prediction is a scalar value
        prediction_value = prediction[0][0] if prediction.ndim == 2 else prediction[0]

        predictions.append(prediction_value)

        # Shift the recent_data array and update with the new prediction
        recent_data = np.roll(recent_data, -1)
        recent_data[-1] = prediction_value  # Update the last value with the new prediction

    predictions = np.array(predictions)

    # Generate actions based on the model's predictions
    threshold = 0.005  # Adjust threshold to 0.5% for more flexibility
    actions = np.zeros(len(predictions), dtype=int)

    for i in range(1, len(predictions)):
        # Calculate change from the previous prediction instead of recent_data[-1]
        change = (predictions[i] - predictions[i-1]) / predictions[i-1]  # Compare with previous prediction
        if change > threshold:
            actions[i] = 1  # Buy
        elif change < -threshold:
            actions[i] = 2  # Sell
        else:
            actions[i] = 0  # Stay

    # Forecast with Prophet
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    forecast_display = forecast.rename(columns={
        "ds": "Forecast Date",
        "yhat": "Predicted Price",
        "yhat_lower": "Predicted Price (Lower Bound)",
        "yhat_upper": "Predicted Price (Upper Bound)",
        "trend": "Trend",
        "trend_lower": "Trend (Lower Bound)",
        "trend_upper": "Trend (Upper Bound)"
    })

    # Display forecast data
    st.subheader('Forecast data')
    st.write(forecast_display.tail())

    # Plot forecast data
    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

    # Ensure the length of actions matches the length of forecast_dates
    forecast_dates = forecast['ds'].iloc[-len(actions):]

    # Create a DataFrame for actions (Buy, Sell, Stay)
    actions_df = pd.DataFrame({
        'Date': forecast_dates,  # Use forecast dates
        'Action': np.where(actions == 0, 'Stay', np.where(actions == 1, 'Buy', 'Sell'))
    })

    st.subheader('Actions')
    st.write(actions_df)

