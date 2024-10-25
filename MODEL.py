# imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load Dataset
path = "Dataset.csv"
df=pd.read_csv(path)

# Select Relevant information    
df = df[['Engine size (L)', 'Combined (L/100 km)', 'CO2 emissions (g/km)']]
df

# Define the independent variables (features) and the dependent variable (target)
X = df[['Engine size (L)', 'Combined (L/100 km)']]
y = df['CO2 emissions (g/km)']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the multiple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Coefficients and intercept of the model
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Visualizing the relationship between predicted and actual CO2 emissions
plt.scatter(y_test, y_pred)
plt.xlabel('Actual CO2 Emissions (g/km)')
plt.ylabel('Predicted CO2 Emissions (g/km)')
plt.title('Actual vs Predicted CO2 Emissions')
plt.show()

# Evaluate the model performance on the test set
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Print the coefficients and intercept
print(f"Coefficients (Engine size, Combined fuel consumption): {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Visualise the relationship between actual and predicted CO2 emissions
plt.scatter(y_test, y_pred)
plt.xlabel('Actual CO2 Emissions (g/km)')
plt.ylabel('Predicted CO2 Emissions (g/km)')
plt.title('Actual vs Predicted CO2 Emissions')
plt.show()

from sklearn.model_selection import cross_val_score

# Perform cross-validation and compute average R² score
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"Cross-Validation R² Scores: {scores}")
print(f"Mean R² Score: {scores.mean()}")

# Compute the R² score
r2 = r2_score(y_test, y_pred)

# Accuracy Percentage
accuracy_percentage = r2 * 100

print(f"Model Accuracy (based on R² score): {accuracy_percentage:.2f}%")

def predict_co2_emission(engine_size, fuel_consumption):
    
    input_data = pd.DataFrame({
        'Engine size (L)': [engine_size],
        'Combined (L/100 km)': [fuel_consumption]
    })
    
    
    predicted_co2 = model.predict(input_data)
    
    return predicted_co2[0]

engine_size = float(input("Enter the Engine size (L): "))
fuel_consumption = float(input("Enter the Combined fuel consumption (L/100 km): "))

predicted_emission = predict_co2_emission(engine_size, fuel_consumption)
print(f"Predicted CO2 Emissions (g/km): {predicted_emission:.2f}")