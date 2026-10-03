# AIML Practical 6
# Implement Linear Regression Algorithm

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [35, 40, 45, 50, 55, 60, 68, 75, 82, 90]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# Independent variable
X = df[["Study_Hours"]]

# Dependent variable
y = df["Marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

print("\nActual Marks:")
print(y_test.values)

print("\nPredicted Marks:")
print(y_pred.round(2))

# Model parameters
print("\nSlope:", round(model.coef_[0], 2))
print("Intercept:", round(model.intercept_, 2))

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", round(mse, 2))
print("R² Score:", round(r2, 2))

# Predict marks for a new student
study_hours = [[7]]
predicted_marks = model.predict(study_hours)

print("\nPredicted marks for 7 study hours:",
      round(predicted_marks[0], 2))

# Plot regression line
plt.figure(figsize=(8, 5))

plt.scatter(X, y, label="Actual Data")
plt.plot(X, model.predict(X), label="Regression Line")

plt.title("Linear Regression: Study Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.legend()
plt.grid(True)

plt.show()