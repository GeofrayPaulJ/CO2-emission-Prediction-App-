import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt 


@st.cache_data
def load_data():
    path = "Dataset.csv"
    df = pd.read_csv(path)
    return df

df = load_data()


df = df[['Engine size (L)', 'Combined (L/100 km)', 'CO2 emissions (g/km)']]


X = df[['Engine size (L)', 'Combined (L/100 km)']]
y = df['CO2 emissions (g/km)']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


def predict_co2_emission(engine_size, fuel_consumption):
    input_data = pd.DataFrame({
        'Engine size (L)': [engine_size],
        'Combined (L/100 km)': [fuel_consumption]
    })
    predicted_co2 = model.predict(input_data)
    return predicted_co2[0]


st.title("CO2 Emission Prediction")
st.write("Enter the engine size and fuel consumption to predict CO2 emissions (g/km).")


engine_size = st.number_input("Enter the Engine size (L)", min_value=0.0, step=0.1)
fuel_consumption = st.number_input("Enter the fuel consumption (L/100 km)", min_value=0.0, step=0.1)


if st.button("Predict CO2 Emissions"):
    
    if engine_size > 0 and fuel_consumption > 0:
        predicted_emission = predict_co2_emission(engine_size, fuel_consumption)
        st.write(f"Predicted CO2 Emissions: {predicted_emission:.2f} g/km")
    else:
        st.write("Please enter valid values for both Engine size and Fuel consumption.")


if st.checkbox("Show Actual vs Predicted CO2 Emissions"):
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual CO2 Emissions (g/km)')
    plt.ylabel('Predicted CO2 Emissions (g/km)')
    plt.title('Actual vs Predicted CO2 Emissions')
    st.pyplot(plt)


mse = mean_squared_error(y_test, model.predict(X_test))
r2 = r2_score(y_test, model.predict(X_test))
st.write(f"Model Performance:")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")
