
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Create dataset (offline)
X, y = make_blobs(n_samples=100, centers=3, random_state=42)

# Apply K-Means for different K values
for k in [2, 3, 4, 5]:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    print("K =", k)
    print("Inertia:", model.inertia_)
    print("Silhouette Score:", silhouette_score(X, labels))

# Plot clusters for K=3
model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="rainbow")
plt.scatter(model.cluster_centers_[:, 0],
            model.cluster_centers_[:, 1],
            color="black", marker="X", s=150)
plt.title("K-Means Clustering (K=3)")
plt.show()
