# AIML Practical 10
# Fake News Detection from Multimodal Data
# Text + Image Features

import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. Create Sample Multimodal Dataset
# --------------------------------------------------

data = {
    "News_Text": [
        "Government announces new education policy",
        "Scientists discover new method for clean energy",
        "Breaking news celebrity wins national award",
        "New technology improves healthcare services",
        "Scientists confirm impossible miracle cure",
        "Celebrity claims drinking water can cure all diseases",
        "Government secretly gives free cars to everyone",
        "Aliens reportedly take over major city"
    ],

    # Image brightness values as sample image features
    "Image_Brightness": [
        120, 130, 115, 125,
        220, 210, 230, 225
    ],

    # Image contrast values as sample image features
    "Image_Contrast": [
        40, 45, 38, 42,
        90, 85, 95, 88
    ],

    # Target variable
    "Label": [
        "Real", "Real", "Real", "Real",
        "Fake", "Fake", "Fake", "Fake"
    ]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)


# --------------------------------------------------
# 2. Extract Text Features using TF-IDF
# --------------------------------------------------

vectorizer = TfidfVectorizer()

text_features = vectorizer.fit_transform(
    df["News_Text"]
).toarray()


# --------------------------------------------------
# 3. Extract Image Features
# --------------------------------------------------

image_features = df[
    ["Image_Brightness", "Image_Contrast"]
].values


# --------------------------------------------------
# 4. Scale Image Features
# --------------------------------------------------

scaler = StandardScaler()

image_features = scaler.fit_transform(
    image_features
)


# --------------------------------------------------
# 5. Combine Text + Image Features
# --------------------------------------------------

multimodal_features = np.hstack(
    (text_features, image_features)
)


# --------------------------------------------------
# 6. Target Variable
# --------------------------------------------------

y = df["Label"]


# --------------------------------------------------
# 7. Split Dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    multimodal_features,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 8. Train Classification Model
# --------------------------------------------------

model = LogisticRegression()

model.fit(X_train, y_train)


# --------------------------------------------------
# 9. Make Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)


# --------------------------------------------------
# 10. Evaluate Model
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:",
      round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# --------------------------------------------------
# 11. Test New News
# --------------------------------------------------

new_text = [
    "Scientists announce a new breakthrough in medical research"
]

new_text_features = vectorizer.transform(
    new_text
).toarray()

# Sample image features
new_image_features = scaler.transform(
    [[125, 42]]
)

# Combine both modalities
new_multimodal_features = np.hstack(
    (new_text_features, new_image_features)
)

prediction = model.predict(
    new_multimodal_features
)

print("\nNew News Prediction:",
      prediction[0])