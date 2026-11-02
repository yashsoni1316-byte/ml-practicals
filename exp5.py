
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score

# 1. Create dataset with clusters
X, y = make_blobs(
    n_samples=90,
    centers=[[0, 0], [4, 4], [0, 5]],
    cluster_std=0.45,
    random_state=42
)

# Add noise points
noise = np.random.RandomState(42).uniform(-1, 6, (10, 2))
X = np.vstack([X, noise])
y = np.append(y, [-1] * 10)

# 2. Apply DBSCAN
model = DBSCAN(eps=0.8, min_samples=5)
labels = model.fit_predict(X)

# 3. Evaluate clustering
clusters = set(labels) - {-1}
print("Clusters:", len(clusters))
print("Noise points:", sum(labels == -1))

mask = labels != -1
if len(clusters) >= 2 and len(set(labels[mask])) >= 2:
    print("Silhouette Score:",
          silhouette_score(X[mask], labels[mask]))

print("Adjusted Rand Index:", adjusted_rand_score(y, labels))

# 4. Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="rainbow")
plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
