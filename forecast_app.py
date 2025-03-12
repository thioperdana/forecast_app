import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from datetime import datetime, timedelta
import io
import base64

# Set page config
st.set_page_config(layout="wide", page_title="Aplikasi Forecast")

# Load dataset from seaborn
flights_data = sns.load_dataset("flights")
flights_data['date'] = pd.to_datetime(flights_data['year'].astype(str) + '-' + 
                                     flights_data['month'].astype(str) + '-01')

# Function to create forecast
def create_forecast(data, horizon_hours, interval_type, include_weather, account_holidays):
    # Prepare data for Prophet
    df = data.rename(columns={'date': 'ds', 'passengers': 'y'})
    
    # Create Prophet model
    model = Prophet()
    
    # Add holidays if selected
    if account_holidays:
        model.add_country_holidays(country_name='US')
    
    # Fit model
    model.fit(df)
    
    # Create future dataframe
    if interval_type == "Hourly Forecasts":
        # For hourly, we'll simulate by creating more granular data
        future = model.make_future_dataframe(periods=horizon_hours, freq='H')
    else:  # Daily Averages
        # For daily, we'll use days
        future = model.make_future_dataframe(periods=horizon_hours, freq='D')
    
    # Make forecast
    forecast = model.predict(future)
    
    return forecast, model

# Function to download data as CSV
def get_csv_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return href

# App layout
st.title("Aplikasi Forecast")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("## ⚡ Forecast Parameters")
    st.markdown("---")
    
    # Forecast Horizon
    st.markdown("### Forecast Horizon (Days) ℹ️")
    horizon_hours = st.slider("", min_value=1, max_value=730, value=30, 
                             step=1, format=None, 
                             key=None, help=None, 
                             on_change=None, args=None, kwargs=None)
    
    # Display markers for hours
    cols = st.columns([1, 1, 1, 1])
    
    # Text input for hours
    horizon_hours_input = st.text_input("", value=str(horizon_hours))
    try:
        horizon_hours = int(horizon_hours_input)
    except:
        st.error("Please enter a valid number")
    
    # Forecast Interval
    st.markdown("### Forecast Interval")
    interval_type = st.radio("", ["Daily Averages"])
    
    # Additional Options
    st.markdown("### Additional Options")
    include_weather = st.toggle("Include weather impact ℹ️", value=True)
    account_holidays = st.toggle("Account for holidays ℹ️", value=False)
    
    # Generate Forecast Button
    if st.button("GENERATE FORECAST", type="primary", use_container_width=True):
        st.session_state.generate_forecast = True
    else:
        if 'generate_forecast' not in st.session_state:
            st.session_state.generate_forecast = False

with col2:
    st.markdown("## Forecasting Information")
    st.markdown("---")
    
    if 'generate_forecast' in st.session_state and st.session_state.generate_forecast:
        # Create forecast
        forecast, model = create_forecast(
            flights_data, 
            horizon_hours, 
            interval_type, 
            include_weather, 
            account_holidays
        )
        
        # Visualization
        st.markdown("### Forecast Visualization")
        
        # Plot the forecast
        fig = plt.figure(figsize=(10, 6))
        plt.plot(flights_data['date'], flights_data['passengers'], 'b-', label='Historical Passengers')
        
        # Get the forecast data for plotting
        forecast_dates = forecast['ds'].iloc[-horizon_hours:]
        forecast_values = forecast['yhat'].iloc[-horizon_hours:]
        forecast_lower = forecast['yhat_lower'].iloc[-horizon_hours:]
        forecast_upper = forecast['yhat_upper'].iloc[-horizon_hours:]
        
        # Plot historical and forecast
        plt.plot(forecast_dates, forecast_values, 'r-', label='Forecasted Passengers')
        plt.fill_between(forecast_dates, forecast_lower, forecast_upper, color='r', alpha=0.2)
        
        plt.title('Airline Passengers Forecast')
        plt.xlabel('Date')
        plt.ylabel('Passengers')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)
        
        # Forecast Details
        st.markdown("### Forecast Details")

        trend_column = 'trend' if 'trend' in forecast.columns else 'yhat'
        
        # Create download button
        csv_link = get_csv_download_link(forecast)
        st.markdown(f'<a href="{csv_link}" download="forecast_data.csv"><button style="background-color: white; color: #0066cc; border: 1px solid #0066cc; padding: 8px 16px; border-radius: 4px; cursor: pointer; float: right;">⬇️ EXPORT</button></a>', unsafe_allow_html=True)
        
        # Display forecast details in a table with column names matching the dataset
        forecast_table = pd.DataFrame({
            'Timestamp': forecast['ds'].iloc[-5:].dt.strftime('%b %d, %Y'),
            'Forecast (Passengers)': forecast['yhat'].iloc[-5:].round().astype(int),
            'Range': [f"{int(lower)} - {int(upper)}"
                     for lower, upper in zip(forecast['yhat_lower'].iloc[-5:], forecast['yhat_upper'].iloc[-5:])],
            'Trend': forecast[trend_column].iloc[-5:].round().astype(int)
        })
        
        st.table(forecast_table)