## Implementation of SGD Regressor for Multivariate Linear Regression

### AIM

To write and verify a Python program to implement a multivariate linear regression model using Stochastic Gradient Descent (`SGDRegressor`) to predict house prices and the number of occupants in a house.

### Equipments Required

* **Hardware:** PC / Laptop
* **Software:** Anaconda / Python 3.7+ Installation, Jupyter Notebook (with `numpy`, `matplotlib`, and `scikit-learn`)

---

### Algorithm

1. Prepare a multivariate feature matrix $X$ (for example: area in sq. ft., number of bedrooms, and age of the house) and multi-target matrix $Y$ containing two targets: House Price ($y_1$) and Number of Occupants ($y_2$).
2. Split the dataset into training and testing sets.
3. Standardize the features using `StandardScaler` so all variables have zero mean and unit variance—this is critical for stable gradient descent convergence.
4. Initialize an `SGDRegressor` model (wrapped with `MultiOutputRegressor` because scikit-learn's `SGDRegressor` fits a single target natively, allowing simultaneous prediction of both house price and number of occupants).
5. Train the model on the scaled training features and target values using stochastic gradient descent updates.
6. Predict both target variables on the test set and for a new house sample, evaluate model performance (MSE and $R^2$ score), and plot the predicted vs. actual values.

---

### Program

```python
"""
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Darshan J
RegisterNumber: 212224020008
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Features X: [Area_sqft, Bedrooms, House_Age_years]
X = np.array([
    [900, 2, 15],
    [1100, 2, 10],
    [1300, 3, 12],
    [1500, 3, 8],
    [1750, 3, 5],
    [2000, 4, 6],
    [2200, 4, 4],
    [2500, 4, 3],
    [2750, 5, 2],
    [3000, 5, 1],
    [1200, 2, 18],
    [1600, 3, 9],
    [1900, 4, 7],
    [2350, 4, 5],
    [2850, 5, 2]
], dtype=float)

# Targets Y: [House_Price_in_Lakhs, Number_of_Occupants]
Y = np.array([
    [45.0, 3.0],
    [55.0, 3.0],
    [68.0, 4.0],
    [78.0, 4.0],
    [92.0, 4.0],
    [110.0, 5.0],
    [122.0, 5.0],
    [140.0, 6.0],
    [158.0, 6.0],
    [175.0, 7.0],
    [58.0, 3.0],
    [83.0, 4.0],
    [104.0, 5.0],
    [132.0, 6.0],
    [164.0, 6.0]
], dtype=float)

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 3. Feature scaling (important for SGD convergence)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Initialize and train Multi-Output SGD Regressor
base_sgd = SGDRegressor(max_iter=2000, tol=1e-4, eta0=0.01, learning_rate="invscaling", random_state=42)
model = MultiOutputRegressor(base_sgd)
model.fit(X_train_scaled, y_train)

# 5. Predict on test set and for a new house (1850 sq.ft, 3 bedrooms, 6 years old)
y_pred = model.predict(X_test_scaled)
new_house = scaler.transform(np.array([[1850.0, 3.0, 6.0]]))
new_pred = model.predict(new_house)[0]

# 6. Evaluate performance
mse_price = mean_squared_error(y_test[:, 0], y_pred[:, 0])
r2_price = r2_score(y_test[:, 0], y_pred[:, 0])

mse_occ = mean_squared_error(y_test[:, 1], y_pred[:, 1])
r2_occ = r2_score(y_test[:, 1], y_pred[:, 1])

print(f"New House Prediction -> Price: {new_pred[0]:.2f} Lakhs | Occupants: {new_pred[1]:.2f}")
print(f"House Price -> MSE: {mse_price:.4f}, R²: {r2_price:.4f}")
print(f"Occupants   -> MSE: {mse_occ:.4f}, R²: {r2_occ:.4f}")

# 7. Plot Actual vs. Predicted for both targets
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.scatter(y_test[:, 0], y_pred[:, 0], color="blue", label="Test Samples")
ax1.plot([y_test[:, 0].min(), y_test[:, 0].max()],
         [y_test[:, 0].min(), y_test[:, 0].max()], "r--", label="Ideal Fit")
ax1.set_title("House Price: Actual vs. Predicted")
ax1.set_xlabel("Actual Price (Lakhs)")
ax1.set_ylabel("Predicted Price (Lakhs)")
ax1.legend()
ax1.grid(True)

ax2.scatter(y_test[:, 1], y_pred[:, 1], color="green", label="Test Samples")
ax2.plot([y_test[:, 1].min(), y_test[:, 1].max()],
         [y_test[:, 1].min(), y_test[:, 1].max()], "r--", label="Ideal Fit")
ax2.set_title("Number of Occupants: Actual vs. Predicted")
ax2.set_xlabel("Actual Occupants")
ax2.set_ylabel("Predicted Occupants")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

```

---

### Output

```text
New House Prediction -> Price: 98.14 Lakhs | Occupants: 4.53
House Price -> MSE: 4.6218, R²: 0.9964
Occupants   -> MSE: 0.1429, R²: 0.9143

```

---

### Result

Thus, the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor was written and verified using Python programming.
