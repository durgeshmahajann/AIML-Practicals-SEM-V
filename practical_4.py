# AIML Practical 4
# Apply Naïve Bayes Classification Algorithm

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "Marks": [35, 40, 45, 50, 55, 60, 68, 75, 82, 90],
    "Result": ["Fail", "Fail", "Fail", "Fail", "Pass",
               "Pass", "Pass", "Pass", "Pass", "Pass"]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# Independent variables
X = df[["Study_Hours", "Attendance", "Marks"]]

# Dependent variable
y = df["Result"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create Naïve Bayes model
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Display predictions
print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))