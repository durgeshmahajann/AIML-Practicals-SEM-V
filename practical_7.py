# AIML Practical 7
# Implement SVM Algorithm for Regression

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
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

# Standardize the features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Create SVR model
model = SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

print("\nActual Marks:")
print(y_test.values)

print("\nPredicted Marks:")
print(y_pred.round(2))

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", round(mse, 2))
print("R² Score:", round(r2, 2))

# Predict marks for 7 study hours
new_data = scaler_X.transform([[7]])
predicted_marks = model.predict(new_data)

print("\nPredicted marks for 7 study hours:",
      round(predicted_marks[0], 2))

# Plot SVR regression
X_plot = scaler_X.transform(X)
y_plot = model.predict(X_plot)

plt.figure(figsize=(8, 5))

plt.scatter(X["Study_Hours"], y, label="Actual Data")
plt.plot(X["Study_Hours"], y_plot, label="SVR Regression Line")

plt.title("Support Vector Regression")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.legend()
plt.grid(True)

plt.show()