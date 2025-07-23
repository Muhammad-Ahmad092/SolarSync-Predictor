import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

   # Load data and model
@st.cache_data
def load_data_and_model(cleaned_path, tabular_x_path, model_path, scaler_path):
       try:
           cleaned_df = pd.read_csv(cleaned_path, parse_dates=['DATE_TIME'])
           X_tabular = pd.read_csv(tabular_x_path, parse_dates=['DATE_TIME'], index_col='DATE_TIME')
           lr_model = joblib.load(model_path)
           scaler = joblib.load(scaler_path)
           return cleaned_df, X_tabular, lr_model, scaler
       except Exception as e:
           st.error(f"Error loading files: {str(e)}")
           return None, None, None, None

   # Predict using local model
def predict_local(features, lr_model, scaler):
       try:
           features_array = np.array([list(features.values())])
           features_scaled = scaler.transform(features_array)
           prediction = lr_model.predict(features_scaled)[0]
           return prediction
       except Exception as e:
           st.error(f"Prediction failed: {str(e)}")
           return None

   # Streamlit app
def main():
       st.title("SolarSync Predictor")
       st.write("Predict hourly solar energy output (AC Power) using weather and temporal features.")

       # Load data and model
       cleaned_path = 'cleaned_solar_data.csv'
       tabular_x_path = 'X_tabular.csv'
       model_path = 'lr_model.pkl'
       scaler_path = 'scaler.pkl'
       cleaned_df, X_tabular, lr_model, scaler = load_data_and_model(cleaned_path, tabular_x_path, model_path, scaler_path)

       if cleaned_df is None:
           return

       # Input features (raw values)
       st.subheader("Input Features")
       irradiation = st.number_input("Irradiation (W/m²)", min_value=0.0, max_value=1000.0, value=500.0)
       ambient_temp = st.number_input("Ambient Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
       module_temp = st.number_input("Module Temperature (°C)", min_value=-10.0, max_value=70.0, value=30.0)
       hour = st.slider("Hour of Day", 0, 23, 12)
       day = st.slider("Day of Month", 1, 31, 15)
       month = st.slider("Month", 1, 12, 5)
       day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
       ac_power_lag1 = st.number_input("Previous Hour AC Power (kW)", min_value=0.0, max_value=1000.0, value=400.0)
       irradiation_lag1 = st.number_input("Previous Hour Irradiation (W/m²)", min_value=0.0, max_value=1000.0, value=450.0)
       ac_power_rolling_mean = st.number_input("3-Hour AC Power Rolling Mean (kW)", min_value=0.0, max_value=1000.0, value=420.0)
       irradiation_rolling_mean = st.number_input("3-Hour Irradiation Rolling Mean (W/m²)", min_value=0.0, max_value=1000.0, value=470.0)

       # Normalize inputs
       features = {
           "IRRADIATION": irradiation,
           "AMBIENT_TEMPERATURE": ambient_temp,
           "MODULE_TEMPERATURE": module_temp,
           "hour": hour / 23.0,  # Normalize to 0-1
           "day": (day - 1) / 30.0,  # Normalize to 0-1
           "month": (month - 1) / 11.0,  # Normalize to 0-1
           "day_of_week": day_of_week / 6.0,  # Normalize to 0-1
           "ac_power_lag1": ac_power_lag1,
           "irradiation_lag1": irradiation_lag1,
           "ac_power_rolling_mean": ac_power_rolling_mean,
           "irradiation_rolling_mean": irradiation_rolling_mean
       }
       features_array = np.array([list(features.values())])
       features_scaled = scaler.transform(features_array)
       features_scaled_dict = {
           "IRRADIATION": features_scaled[0][0],
           "AMBIENT_TEMPERATURE": features_scaled[0][1],
           "MODULE_TEMPERATURE": features_scaled[0][2],
           "hour": features_scaled[0][3],
           "day": features_scaled[0][4],
           "month": features_scaled[0][5],
           "day_of_week": features_scaled[0][6],
           "ac_power_lag1": features_scaled[0][7],
           "irradiation_lag1": features_scaled[0][8],
           "ac_power_rolling_mean": features_scaled[0][9],
           "irradiation_rolling_mean": features_scaled[0][10]
       }

       # Predict button
       if st.button("Predict AC Power"):
           prediction = predict_local(features_scaled_dict, lr_model, scaler)
           if prediction is not None:
               st.success(f"Predicted AC Power: {prediction:.2f} kW")
           else:
               st.error("Prediction failed.")

       # Historical data visualization
       st.subheader("Historical AC Power and Irradiation Trends")
       fig = px.line(cleaned_df, x='DATE_TIME', y=['AC_POWER', 'IRRADIATION'], 
                     labels={'value': 'Value', 'variable': 'Metric'},
                     title='Historical AC Power and Irradiation')
       fig.update_yaxes(title_text="AC Power (kW) / Irradiation (W/m²)")
       st.plotly_chart(fig)

if __name__ == '__main__':
       main()