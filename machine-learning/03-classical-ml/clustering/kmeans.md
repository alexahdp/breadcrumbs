# K-Means Clustering

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Algorithm](#algorithm)
4. [Distance Metrics](#distance-metrics)
5. [Choosing K](#choosing-k)
6. [Convergence and Complexity](#convergence-and-complexity)
7. [Variants and Extensions](#variants-and-extensions)
8. [Practical Considerations](#practical-considerations)
9. [Implementation](#implementation)
10. [References](#references)

## Introduction

**K-Means** is a centroid-based unsupervised learning algorithm that partitions data into K distinct, non-overlapping clusters. Each data point belongs to the cluster with the nearest centroid.

### Key Characteristics

- **Centroid-based** - clusters represented by their center points
- **Hard assignment** - each point belongs to exactly one cluster
- **Spherical clusters** - assumes clusters are roughly spherical in shape
- **Fast and scalable** - efficient for large datasets
- **Simple to implement** - straightforward algorithm with few hyperparameters

### When to Use

**Best suited for:**
- Well-separated, spherical clusters
- Large datasets requiring fast clustering
- Known or estimable number of clusters
- Evenly sized clusters
- When cluster centroids are meaningful

**Not recommended for:**
- Non-convex cluster shapes (use DBSCAN)
- Clusters of varying densities or sizes
- High-dimensional data (curse of dimensionality)
- Presence of outliers (sensitive to extreme values)
- Unknown number of clusters without exploration

## Mathematical Foundation

### Objective Function

K-Means minimizes the **within-cluster sum of squares (WCSS)**, also called **inertia**:

$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2$$

where:
- $K$ is the number of clusters
- $C_k$ is the set of points in cluster $k$
- $\mu_k$ is the centroid of cluster $k$
- $x_i$ is a data point
- $||\cdot||$ is the Euclidean distance (by default)

The centroid of cluster $k$:

$$\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$

### Optimization Problem

K-Means solves:

$$\min_{C_1, \ldots, C_K} \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2$$

This is an NP-hard problem, so the algorithm uses iterative optimization to find a local minimum.

## Algorithm

### Lloyd's Algorithm (Standard K-Means)

The classic K-Means algorithm alternates between two steps:

**Input:**
- Dataset $X = \{x_1, x_2, \ldots, x_n\}$
- Number of clusters $K$

**Steps:**

1. **Initialization**: Select $K$ initial centroids $\mu_1, \mu_2, \ldots, \mu_K$
   - Random selection from data points
   - K-Means++ initialization (recommended)
   - Custom initialization based on domain knowledge

2. **Assignment Step**: Assign each point to nearest centroid
   $$C_k = \{x_i : ||x_i - \mu_k|| \leq ||x_i - \mu_j|| \text{ for all } j \neq k\}$$

3. **Update Step**: Recompute centroids as mean of assigned points
   $$\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$

4. **Repeat** steps 2-3 until convergence:
   - Centroids stop changing
   - Assignments stop changing
   - Maximum iterations reached
   - Change in WCSS below threshold

### Convergence Criteria

The algorithm stops when one of these conditions is met:
- **Centroid convergence**: $||\mu_k^{(t)} - \mu_k^{(t-1)}|| < \epsilon$ for all $k$
- **Assignment stability**: No points change clusters
- **Maximum iterations**: Prevent infinite loops
- **Tolerance**: Relative change in WCSS < threshold

## Distance Metrics

### Euclidean Distance (Default)

$$d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

**Properties:**
- Assumes isotropic (spherical) clusters
- Sensitive to scale - **requires feature normalization**
- Works well for compact, well-separated clusters

### Manhattan Distance (L1)

$$d(x, y) = \sum_{i=1}^{n}|x_i - y_i|$$

**Use when:**
- Features are on different scales
- Grid-like data (e.g., city blocks)
- Presence of outliers

### Cosine Distance

$$d(x, y) = 1 - \frac{x \cdot y}{||x|| \cdot ||y||}$$

**Use when:**
- Direction matters more than magnitude
- Text clustering (document vectors)
- High-dimensional sparse data

## Choosing K

### Elbow Method

Plot WCSS vs. number of clusters $K$:

$$WCSS(K) = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2$$

Look for the "elbow" - the point where adding more clusters doesn't significantly reduce WCSS.

**Limitations:**
- Subjective identification of elbow
- May not have clear elbow
- WCSS always decreases with more clusters

### Silhouette Score

Measures how similar a point is to its own cluster compared to other clusters:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

where:
- $a(i)$ = average distance to points in same cluster
- $b(i)$ = average distance to points in nearest neighboring cluster

Silhouette score ranges from -1 to 1:
- **1**: Point is well-matched to its cluster
- **0**: Point is on the boundary between clusters
- **-1**: Point might be assigned to wrong cluster

Average silhouette score for all points:

$$\bar{s} = \frac{1}{n} \sum_{i=1}^{n} s(i)$$

Choose $K$ that maximizes average silhouette score.

### Gap Statistic

Compares WCSS to expected WCSS under null reference distribution:

$$Gap(K) = E[\log(WCSS_{ref}(K))] - \log(WCSS(K))$$

Choose smallest $K$ such that:

$$Gap(K) \geq Gap(K+1) - s_{K+1}$$

where $s_{K+1}$ is the standard error.

### Other Methods

**Davies-Bouldin Index** - ratio of within-cluster to between-cluster distances (lower is better):

$$DB = \frac{1}{K} \sum_{k=1}^{K} \max_{j \neq k} \left( \frac{\sigma_k + \sigma_j}{d(c_k, c_j)} \right)$$

**Calinski-Harabasz Index** - ratio of between-cluster to within-cluster dispersion (higher is better)

## Convergence and Complexity

### Time Complexity

- **Per iteration**: $O(nKd)$
  - $n$ = number of points
  - $K$ = number of clusters
  - $d$ = number of dimensions

- **Total**: $O(nKdi)$
  - $i$ = number of iterations (typically small, e.g., 10-100)

### Space Complexity

- $O(nKd)$ - storing cluster assignments and centroids

### Convergence Properties

- **Guaranteed convergence** to local minimum (WCSS never increases)
- **No guarantee** of global optimum
- Convergence typically fast (few iterations)
- Result depends on initialization

## Variants and Extensions

### K-Means++

**Improved initialization** that spreads out initial centroids:

1. Choose first centroid uniformly at random from data points
2. For each subsequent centroid:
   - Compute $D(x)$ = distance from each point to nearest chosen centroid
   - Choose next centroid with probability proportional to $D(x)^2$
3. Proceed with standard K-Means

**Benefits:**
- Better final clustering (provably $O(\log K)$ competitive)
- Faster convergence
- More stable results
- Default in most implementations

### Mini-Batch K-Means

Uses random subsets (mini-batches) of data for faster training:

1. Sample mini-batch of data
2. Assign samples to nearest centroid
3. Update centroids using samples in mini-batch
4. Repeat until convergence

**Trade-offs:**
- **Faster**: 3-10x speedup on large datasets
- **Scalable**: Handles datasets that don't fit in memory
- **Slightly worse quality**: Approximate solution
- **Use when**: Dataset is very large (millions of points)

### K-Medoids (PAM)

Uses actual data points as cluster centers instead of centroids:

$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} d(x_i, m_k)$$

where $m_k$ is the medoid (most centrally located point) of cluster $k$.

**Advantages:**
- More robust to outliers
- Works with arbitrary distance metrics
- Interpretable cluster representatives

**Disadvantages:**
- Higher computational cost: $O(K(n-K)^2)$
- Slower than K-Means

### Fuzzy C-Means

Soft clustering where each point has membership degree to all clusters:

$$u_{ik} = \frac{1}{\sum_{j=1}^{K} \left(\frac{||x_i - c_k||}{||x_i - c_j||}\right)^{2/(m-1)}}$$

where:
- $u_{ik}$ = membership of point $i$ to cluster $k$
- $m$ = fuzziness parameter (typically $m = 2$)
- $\sum_{k=1}^{K} u_{ik} = 1$ for all $i$

## Practical Considerations

### Feature Scaling

**Critical requirement**: Features must be scaled before applying K-Means.

**Why?** Euclidean distance is sensitive to feature magnitude.

**Example:**
- Age: 20-80 (range ≈ 60)
- Income: 20,000-200,000 (range ≈ 180,000)
- Income dominates distance calculation!

**Solutions:**

**Standardization (Z-score normalization):**

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

**Min-Max scaling:**

$$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

### Handling Outliers

K-Means is **sensitive to outliers**:
- Outliers pull centroids away from true cluster centers
- Can create spurious clusters

**Solutions:**
1. **Remove outliers** before clustering
2. **Use robust variants**: K-Medoids, DBSCAN
3. **Clip extreme values** to reasonable range
4. **Transform data** (log transformation for skewed distributions)

### Empty Clusters

If a cluster becomes empty during iteration:

**Strategies:**
1. **Reinitialize** empty centroid to random point
2. **Split largest cluster** - choose point farthest from its centroid
3. **Choose point** with highest error (farthest from any centroid)

### Multiple Runs

Due to initialization sensitivity:
- **Run K-Means multiple times** (10-100 runs) with different initializations
- **Select best result** (lowest WCSS)
- K-Means++ reduces need but multiple runs still recommended

### High-Dimensional Data

**Curse of dimensionality** affects K-Means:
- Distances become less meaningful in high dimensions
- All points appear roughly equidistant

**Solutions:**
1. **Dimensionality reduction** first (PCA, t-SNE)
2. **Feature selection** - remove irrelevant features
3. **Use alternative methods** designed for high dimensions

## Implementation

### Scikit-learn Basic Example

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4, n_features=2,
                       cluster_std=0.60, random_state=42)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
kmeans = KMeans(
    n_clusters=4,           # Number of clusters
    init='k-means++',       # Initialization method
    n_init=10,              # Number of runs with different initializations
    max_iter=300,           # Maximum iterations
    random_state=42         # Reproducibility
)

# Fit and predict
labels = kmeans.fit_predict(X_scaled)

# Get cluster centers and inertia
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_

print(f"Inertia (WCSS): {inertia:.2f}")
print(f"Number of iterations: {kmeans.n_iter_}")
print(f"Cluster sizes: {np.bincount(labels)}")
```

### Elbow Method

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Calculate WCSS for different K
wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, wcss, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('WCSS (Inertia)', fontsize=12)
plt.title('Elbow Method for Optimal K', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()
```

### Silhouette Analysis

```python
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import numpy as np

# Calculate silhouette scores for different K
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette Score = {score:.3f}")

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Silhouette Score vs Number of Clusters', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()

# Detailed silhouette plot for specific K
from matplotlib import cm

def plot_silhouette(X, n_clusters):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)

    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)

    y_lower = 10
    for i in range(n_clusters):
        cluster_silhouette_values = sample_silhouette_values[labels == i]
        cluster_silhouette_values.sort()

        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster Label")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--",
               label=f'Average: {silhouette_avg:.3f}')
    ax.set_yticks([])
    ax.legend()
    plt.title(f'Silhouette Plot for K={n_clusters}')
    plt.show()

plot_silhouette(X_scaled, n_clusters=4)
```

### Mini-Batch K-Means

```python
from sklearn.cluster import MiniBatchKMeans
import time

# Standard K-Means
start = time.time()
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
kmeans.fit(X_scaled)
kmeans_time = time.time() - start

# Mini-Batch K-Means
start = time.time()
mb_kmeans = MiniBatchKMeans(
    n_clusters=4,
    init='k-means++',
    batch_size=100,      # Size of mini-batches
    n_init=10,
    max_iter=300,
    random_state=42
)
mb_kmeans.fit(X_scaled)
mb_time = time.time() - start

print(f"Standard K-Means: {kmeans_time:.4f}s, Inertia: {kmeans.inertia_:.2f}")
print(f"Mini-Batch K-Means: {mb_time:.4f}s, Inertia: {mb_kmeans.inertia_:.2f}")
print(f"Speedup: {kmeans_time/mb_time:.2f}x")
```

### Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Fit K-Means
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_

# Plot results
plt.figure(figsize=(12, 5))

# Original data
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.6)
plt.title('True Labels', fontsize=14)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# K-Means clustering
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=300, marker='X',
            edgecolors='black', linewidths=2, label='Centroids')
plt.title('K-Means Clustering', fontsize=14)
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.legend()

plt.tight_layout()
plt.show()
```

### Practical Workflow

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

def find_optimal_clusters(X, max_k=10):
    """Find optimal number of clusters using multiple methods"""

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    wcss = []
    silhouette_scores = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_scaled)

        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Elbow plot
    ax1.plot(range(2, max_k + 1), wcss, 'bo-')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('WCSS')
    ax1.set_title('Elbow Method')
    ax1.grid(True, alpha=0.3)

    # Silhouette plot
    ax2.plot(range(2, max_k + 1), silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Recommend K
    best_k = np.argmax(silhouette_scores) + 2
    print(f"Recommended K based on silhouette score: {best_k}")

    return X_scaled, best_k

# Usage
X_scaled, optimal_k = find_optimal_clusters(X, max_k=10)

# Apply clustering with optimal K
final_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
final_labels = final_kmeans.fit_predict(X_scaled)
```

## References

- [Scikit-learn K-Means Documentation](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- Arthur, D., & Vassilvitskii, S. (2007). "k-means++: The advantages of careful seeding"
- MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations"
- Lloyd, S. (1982). "Least squares quantization in PCM"
- The Elements of Statistical Learning (Hastie, Tibshirani, Friedman)
- Pattern Recognition and Machine Learning (Christopher Bishop)
