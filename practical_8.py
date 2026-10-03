# AIML Practical 8
# Apply Regularization to Avoid Overfitting

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Create dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [35, 40, 45, 50, 55, 60, 68, 75, 82, 90]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# Independent and dependent variables
X = df[["Study_Hours"]]
y = df["Marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Ridge Regression model
ridge_model = make_pipeline(
    PolynomialFeatures(degree=4),
    StandardScaler(),
    Ridge(alpha=1.0)
)

# Train the model
ridge_model.fit(X_train, y_train)

# Predictions
y_pred = ridge_model.predict(X_test)

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred.round(2))

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", round(mse, 2))
print("R² Score:", round(r2, 2))

# Predict for a new value
new_data = [[7]]
prediction = ridge_model.predict(new_data)

print("\nPredicted marks for 7 study hours:",
      round(prediction[0], 2))

# Visualization
X_range = pd.DataFrame({
    "Study_Hours": [
        i / 10 for i in range(10, 101)
    ]
})

y_range = ridge_model.predict(X_range)

plt.figure(figsize=(8, 5))

plt.scatter(X, y, label="Actual Data")
plt.plot(
    X_range,
    y_range,
    label="Ridge Regression"
)

plt.title("Regularization using Ridge Regression")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.legend()
plt.grid(True)

plt.show()