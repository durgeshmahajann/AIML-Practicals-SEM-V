# AIML Practical 2
# Apply Clustering Techniques Using Unsupervised Learning

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create dataset
data = {
    "Annual_Income": [15, 16, 17, 18, 20, 21, 22, 60, 62, 65, 67, 70],
    "Spending_Score": [80, 85, 78, 82, 75, 72, 70, 25, 20, 30, 28, 22]
}

df = pd.DataFrame(data)

# Display dataset
print("Dataset:")
print(df)

# Select features
X = df[["Annual_Income", "Spending_Score"]]

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

# Display clustered data
print("\nClustered Data:")
print(df)

# Display cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Visualize clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    df["Annual_Income"],
    df["Spending_Score"],
    c=df["Cluster"],
    s=100
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=200,
    label="Centroids"
)

plt.title("Customer Clustering using K-Means")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.grid(True)

plt.show()