import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Sample dataset: House Price vs Square Footage
data = {
    'SquareFootage': [850, 900, 1000, 1200, 1400, 1500, 1700, 1850, 2000, 2100, 2300, 2500],
    'HousePrice': [100000, 120000, 130000, 150000, 170000, 180000, 200000, 220000, 230000, 250000, 270000, 300000]
}

df = pd.DataFrame(data)

# Independent (X) and Dependent (Y) Variables
X = df[['SquareFootage']]
y = df['HousePrice']

# Split the data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Linear Regression model and train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)
print(f"Regression Equation: Price = {model.coef_[0]:.2f} * SquareFootage + {model.intercept_:.2f}")
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# Plotting the Regression Line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel("Square Footage")
plt.ylabel("House Price ($)")
plt.title("House Price Prediction using Linear Regression")
plt.legend()
plt.show()
