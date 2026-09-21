

import streamlit as st
import joblib
import pandas as pd

# Load the trained model
try:
    model = joblib.load('delivery_delay.sav')
except FileNotFoundError:
    st.error("Error: 'delivery_delay.sav' not found. Make sure the model file is in the same directory.")
    st.stop()

# Define the feature columns (must match the order used during training)
feature_columns = ['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
                   'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
                   'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
                   'Warehouse_Processing_Time']

st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if a delivery will be delayed.')

# Input fields for each feature
delivery_distance = st.slider('Delivery Distance (km)', min_value=0.0, max_value=100.0, value=20.0)
traffic_congestion = st.slider('Traffic Congestion (1-5, 5 being highest)', min_value=1, max_value=5, value=3)
weather_condition = st.slider('Weather Condition (1-5, 5 being worst)', min_value=1, max_value=5, value=2)
delivery_slot = st.slider('Delivery Slot (1-3)', min_value=1, max_value=3, value=2)
driver_experience = st.slider('Driver Experience (years)', min_value=0, max_value=30, value=10)
num_stops = st.slider('Number of Stops', min_value=1, max_value=10, value=5)
vehicle_age = st.slider('Vehicle Age (years)', min_value=0, max_value=15, value=5)
road_condition_score = st.slider('Road Condition Score (1-5, 5 being best)', min_value=1, max_value=5, value=3)
package_weight = st.slider('Package Weight (kg)', min_value=0.0, max_value=50.0, value=10.0)
fuel_efficiency = st.slider('Fuel Efficiency (km/l)', min_value=5.0, max_value=30.0, value=15.0)
warehouse_processing_time = st.slider('Warehouse Processing Time (minutes)', min_value=0, max_value=120, value=60)

# Create a dictionary from the inputs
input_data = {
    'Delivery_Distance': delivery_distance,
    'Traffic_Congestion': traffic_congestion,
    'Weather_Condition': weather_condition,
    'Delivery_Slot': delivery_slot,
    'Driver_Experience': driver_experience,
    'Num_Stops': num_stops,
    'Vehicle_Age': vehicle_age,
    'Road_Condition_Score': road_condition_score,
    'Package_Weight': package_weight,
    'Fuel_Efficiency': fuel_efficiency,
    'Warehouse_Processing_Time': warehouse_processing_time
}

# Convert input data to a Pandas DataFrame
input_df = pd.DataFrame([input_data])

if st.button('Predict Delay'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1]

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f'The delivery is predicted to be DELAYED with a probability of {prediction_proba[0]:.2f}')
    else:
        st.success(f'The delivery is predicted to be ON TIME with a probability of {1 - prediction_proba[0]:.2f}')

    st.write('---')
    st.write('### Input Features:')
    st.table(input_df.transpose())
