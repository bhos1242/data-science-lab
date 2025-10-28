# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample Dataset (Years Experience vs Salary)
data = {
    'Experience': [1.1, 2.0, 3.2, 4.0, 5.5, 6.1, 7.9, 8.3, 9.5, 10.2],
    'Salary': [35000, 45000, 50000, 60000, 62000, 70000, 85000, 90000, 95000, 105000]
}

df = pd.DataFrame(data)

# Split data
X = df[['Experience']]  # independent variable
y = df['Salary']        # dependent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Model coefficients
print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)
print("Equation: Salary = {:.2f} * Experience + {:.2f}".format(model.coef_[0], model.intercept_))

# Visualize
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Best Fit Line')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Experience (Linear Regression)")
plt.legend()
plt.show()
