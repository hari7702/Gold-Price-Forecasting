#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import joblib

# try:
#     model = joblib.load('forecasting gold prices.pkl')
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"An error occurred while loading the model: {e}")


# In[2]:


import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Custom CSS for new background and better visuals
st.markdown(
    """
    <style>
    /* Background Image */
    .stApp {
        background-image: url('https://media.istockphoto.com/id/1333437393/vector/gold-price-increasing-background-illustration.jpg?s=612x612&w=0&k=20&c=BtKRRDWCJ86HRJv-Yezeies-6UyOtmyI5-nFGSu2Gmw=');
        background-size: cover;
        background-position: center;
        filter: brightness(0.7); /* Slightly darkens the background */
    }

    /* Dark overlay on top of the background */
    .overlay {
        background: rgba(0, 0, 0, 0.5); /* Darker overlay for better contrast */
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 0;
    }

    /* Result box styling */
    .result-box {
        background-color: rgba(255, 215, 0, 0.85); /* Semi-transparent gold */
        color: #333333; /* Dark text for contrast */
        font-size: 24px; /* Increased font size */
        font-weight: bold;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2); /* Subtle shadow */
        z-index: 1; /* Above overlay */
    }

    /* Button Styling */
    .stButton button {
        background-color: #333333; /* Dark grey */
        color: #FFD700; /* Gold text */
        font-size: 20px;
        padding: 10px 20px;
        border-radius: 12px;
        border: 1px solid #FFD700;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #FFD700;
        color: #333333;
    }

    /* Styling for prediction input */
    .stTextInput {
        background: rgba(255, 255, 255, 0.85); /* White with opacity */
        border-radius: 8px;
        padding: 10px;
    }

    /* Custom Header */
    .custom-header {
        font-size: 36px;
        color: #FFD700; /* Gold header text */
        text-align: center;
        font-weight: bold;
        z-index: 1;
    }

    /* Custom Subheader */
    .custom-subheader {
        font-size: 28px;
        color: #FFFFFF; /* White text */
        text-align: center;
        z-index: 1;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# Title with custom styling
st.markdown("<div class='custom-header'>Gold Price Prediction</div>", unsafe_allow_html=True)

# Load your trained Random Forest model
model = joblib.load('forecasting gold prices.pkl')

# Generate historical data from 2016 to 2030
start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2030-12-31')

# Create a date range from 2016 to 2023
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Generate random prices between 1000 and 6000 for the period
np.random.seed(42)
prices = np.random.uniform(1000, 6000, size=len(dates))

# Create a DataFrame with the historical prices
historical_data = pd.DataFrame({'date': dates, 'price': prices})
historical_data.set_index('date', inplace=True)

# Subheader with custom styling
st.markdown("<div class='custom-subheader'>Historical Prices</div>", unsafe_allow_html=True)

# Create a semi-transparent container for the content
with st.container():
    # Show the historical data
    st.dataframe(historical_data)  

    # Input date
    input_date = st.date_input("Select a date for prediction", datetime.today())

    # Convert input_date to datetime
    input_date = pd.to_datetime(input_date)

    # Custom button for prediction
    if st.button("Predict Gold Price"):
        # Check if the selected date is available in historical data
        if input_date in historical_data.index:
            last_price = historical_data.loc[input_date].price
        else:
            # Get the closest previous date's price
            previous_dates = historical_data[historical_data.index < input_date]
            if not previous_dates.empty:
                last_price = previous_dates.iloc[-1].price
            else:
                st.error("No available data prior to this date.")
                last_price = None

        # If we have a last price, prepare the input for the model
        if last_price is not None:
            # Prepare input data for prediction
            input_data = np.array([[last_price]])
            st.write(f"Using last price: {last_price}")

            # Make prediction
            prediction = model.predict(input_data)

            # Display predicted price
            st.success(f"Predicted Gold Price for {input_date.date()}: ${prediction[0]:.2f}")

