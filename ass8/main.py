import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset from CSV file
df = pd.read_csv('tissue_gene_expression.csv')

# Extract data
X = df.drop('TissueType', axis=1)  # Gene expression features
y = df['TissueType']  # Tissue type labels

print("Shape of data:", X.shape)
print("Tissue types:", y.unique())
print("\nFirst few rows of data:")
print(df.head())

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-means clustering
n_clusters = len(y.unique())
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster assignments to dataframe
df['Cluster'] = clusters

print(f"\n--- K-Means Clustering Results (k={n_clusters}) ---")
print("\nCluster assignments:")
for i in range(n_clusters):
    tissues_in_cluster = df[df['Cluster'] == i]['TissueType'].value_counts()
    print(f"\nCluster {i}:")
    print(tissues_in_cluster)

# Calculate clustering accuracy (percentage of correctly grouped tissues)
from collections import Counter
correct = 0
total = len(df)
for i in range(n_clusters):
    cluster_tissues = df[df['Cluster'] == i]['TissueType']
    if len(cluster_tissues) > 0:
        most_common = Counter(cluster_tissues).most_common(1)[0][1]
        correct += most_common

accuracy = (correct / total) * 100
print(f"\nClustering Purity: {accuracy:.2f}%")

# Visualize clusters using first two principal components
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 5))

# Plot 1: Actual tissue types
plt.subplot(1, 2, 1)
for tissue in y.unique():
    mask = y == tissue
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=tissue, alpha=0.6, s=100)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('Actual Tissue Types')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: K-means clusters
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6, s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, edgecolors='black', label='Centroids')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('K-Means Clusters')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n--- Analysis Complete ---")
