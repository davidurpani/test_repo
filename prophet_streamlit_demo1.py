# prophet_app.py
import streamlit as st
import pandas as pd
from prophet import Prophet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("Time Series Forecasting with Prophet")
st.title("by David Urpani and ChatGPt")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file with time series data", type=["csv"])

if uploaded_file is not None:

    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)

    # Display the uploaded data
    st.subheader("Uploaded Data:")
    st.write(data)

    # Prepare data for Prophet
    data = data.rename(columns={"ds": "ds", "y": "y"})

    # Create a Prophet model
    model = Prophet()

    # Fit the model
    model.fit(data)

    # Specify number of days for forecasting
    num_days = st.slider("Select the number of days for forecasting:", 1, 365, 30)

    # Make future dataframe

    future = model.make_future_dataframe(periods=num_days)

    # Generate forecast
    forecast = model.predict(future)

    # Display forecast data
    st.subheader("Forecast Data:")
    st.write(forecast.tail())

    # Compute metrics
    actual_values = data["y"].values
    forecasted_values = forecast["yhat"].values[-len(data):]

    mae = mean_absolute_error(actual_values, forecasted_values)
    mse = mean_squared_error(actual_values, forecasted_values)
    rmse = np.sqrt(mse)

    st.subheader("Accuracy Metrics:")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Plot forecast
    fig1 = model.plot(forecast)
    st.subheader("Forecast Plot:")
    st.pyplot(fig1)

    # Plot components
    fig2 = model.plot_components(forecast)
    st.subheader("Forecast Components:")
    st.pyplot(fig2)
