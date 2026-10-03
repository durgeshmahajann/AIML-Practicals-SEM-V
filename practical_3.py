# AIML Practical 3
# Apply Dimensionality Reduction Using PCA

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Create dataset
data = {
    "Age": [20, 25, 30, 35, 40, 45, 50, 55],
    "Income": [25, 30, 35, 40, 50, 60, 70, 80],
    "Spending_Score": [80, 75, 70, 65, 55, 50, 40, 35],
    "Experience": [1, 3, 5, 7, 10, 12, 15, 18]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Select numerical features
X = df[["Age", "Income", "Spending_Score", "Experience"]]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA and reduce to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create PCA dataset
pca_df = pd.DataFrame(
    X_pca,
    columns=["Principal Component 1", "Principal Component 2"]
)

print("\nReduced Dataset using PCA:")
print(pca_df)

# Display explained variance
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal Variance Retained:",
      round(sum(pca.explained_variance_ratio_) * 100, 2), "%")

# Visualize PCA components
plt.figure(figsize=(8, 5))

plt.scatter(
    pca_df["Principal Component 1"],
    pca_df["Principal Component 2"],
    s=100
)

plt.title("Dimensionality Reduction using PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)

plt.show()