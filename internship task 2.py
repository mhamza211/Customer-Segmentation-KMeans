import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load dataset containing customer information
data = pd.read_csv('Mall_Customers.csv')

# Display first few records to understand the data
print(data.head())

# Display dataset shape to see the number of entries and features
print("Dataset shape: ", data.shape)

# Display a summary of the dataset, including datatypes and missing values
print("Dataset Info: ", data.info())

# Check for any missing values in the dataset
print("Missing values check: \n", data.isnull().sum())

# Select relevant features: Annual Income and Spending Score
features = data.iloc[:, [3, 4]].values
print("Selected Features:\n", features)

# Determine the optimal number of clusters using the Elbow method (WCSS)
inertia_values = []
for n in range(1, 11):
    kmeans_model = KMeans(n_clusters=n, init='k-means++', random_state=42)
    kmeans_model.fit(features)
    inertia_values.append(kmeans_model.inertia_)

# Plot the WCSS to visualize the Elbow Point for optimal clusters
sns.set()
plt.plot(range(1, 11), inertia_values)
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()

# Set optimal number of clusters to 5 and train the KMeans model
kmeans_model = KMeans(n_clusters=5, init='k-means++', random_state=0)
cluster_labels = kmeans_model.fit_predict(features)
print("Cluster Labels:\n", cluster_labels)

# Visualize the identified clusters and their centroids
plt.figure(figsize=(8,8))
plt.scatter(features[cluster_labels == 0, 0], features[cluster_labels == 0, 1], s=50, c='green', label='Cluster 1')
plt.scatter(features[cluster_labels == 1, 0], features[cluster_labels == 1, 1], s=50, c='red', label='Cluster 2')
plt.scatter(features[cluster_labels == 2, 0], features[cluster_labels == 2, 1], s=50, c='black', label='Cluster 3')
plt.scatter(features[cluster_labels == 3, 0], features[cluster_labels == 3, 1], s=50, c='purple', label='Cluster 4')
plt.scatter(features[cluster_labels == 4, 0], features[cluster_labels == 4, 1], s=50, c='blue', label='Cluster 5')

# Mark the centroids of each cluster
plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')

plt.title('Customer Segmentation Based on Income and Spending')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
