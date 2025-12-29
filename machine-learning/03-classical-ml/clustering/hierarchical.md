# Hierarchical Clustering

## Table of Contents

1. [Introduction](#introduction)
2. [Types of Hierarchical Clustering](#types-of-hierarchical-clustering)
3. [Distance Metrics](#distance-metrics)
4. [Linkage Criteria](#linkage-criteria)
5. [Dendrograms](#dendrograms)
6. [Determining Number of Clusters](#determining-number-of-clusters)
7. [Complexity Analysis](#complexity-analysis)
8. [Practical Considerations](#practical-considerations)
9. [Implementation](#implementation)
10. [References](#references)

## Introduction

**Hierarchical Clustering** builds a hierarchy of clusters by successively merging or splitting clusters. Unlike K-Means, it doesn't require specifying the number of clusters beforehand and produces a tree-like structure (dendrogram) showing relationships between clusters at all levels.

### Key Characteristics

- **Deterministic** - produces same result every run (no random initialization)
- **Hierarchical structure** - reveals cluster relationships at multiple scales
- **No K required upfront** - number of clusters chosen after viewing dendrogram
- **Works with any distance metric** - flexible distance/similarity measures
- **Interpretable** - dendrogram provides visual insight into data structure

### When to Use

**Best suited for:**
- Small to medium datasets (< 10,000 points)
- When cluster hierarchy is meaningful
- Exploratory data analysis
- Unknown number of clusters
- Need for visual representation of clustering
- Taxonomies and phylogenetic trees

**Not recommended for:**
- Very large datasets (computational cost)
- When only final clusters matter (no need for hierarchy)
- High-dimensional data without preprocessing
- Real-time applications (slow compared to K-Means)

## Types of Hierarchical Clustering

### Agglomerative (Bottom-Up)

**Most common approach** - starts with each point as a cluster, then merges:

1. Start: $n$ clusters (each point is a cluster)
2. Repeat: Merge two closest clusters
3. End: Single cluster containing all points

**Algorithm:**
```
1. Initialize: Each point is a cluster
2. While more than one cluster remains:
   a. Find pair of clusters with minimum distance
   b. Merge them into single cluster
   c. Update distance matrix
3. Return hierarchy of merges
```

**Advantages:**
- More commonly used
- Better performance in practice
- Easier to interpret

### Divisive (Top-Down)

**Less common** - starts with all points in one cluster, then splits:

1. Start: 1 cluster (all points)
2. Repeat: Split cluster into two sub-clusters
3. End: $n$ clusters (each point is a cluster)

**Algorithm:**
```
1. Initialize: All points in one cluster
2. While not all points are separate clusters:
   a. Choose cluster to split
   b. Split into two sub-clusters
   c. Update cluster structure
3. Return hierarchy of splits
```

**Advantages:**
- Can identify large-scale structure first
- Potentially more accurate for some datasets

**Disadvantages:**
- Computationally expensive
- Harder to implement
- Decisions at top affect entire hierarchy

## Distance Metrics

### Between Points

**Euclidean Distance:**

$$d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

**Manhattan Distance:**

$$d(x, y) = \sum_{i=1}^{n}|x_i - y_i|$$

**Cosine Distance:**

$$d(x, y) = 1 - \frac{x \cdot y}{||x|| \cdot ||y||}$$

**Mahalanobis Distance** (accounts for correlations):

$$d(x, y) = \sqrt{(x-y)^T \Sigma^{-1} (x-y)}$$

where $\Sigma$ is the covariance matrix.

### Between Clusters (Linkage Criteria)

The choice of linkage criterion dramatically affects results.

## Linkage Criteria

### Single Linkage (Minimum Linkage)

Distance between **nearest** points in two clusters:

$$d(C_i, C_j) = \min_{x \in C_i, y \in C_j} d(x, y)$$

**Characteristics:**
- Creates "chaining" effect - long, elongated clusters
- Can find non-spherical clusters
- Sensitive to noise and outliers
- Good for: Finding arbitrarily shaped clusters

**Use when:**
- Clusters are elongated or non-convex
- Need to detect chain-like structures

**Avoid when:**
- Presence of noise/outliers
- Want compact, well-separated clusters

### Complete Linkage (Maximum Linkage)

Distance between **farthest** points in two clusters:

$$d(C_i, C_j) = \max_{x \in C_i, y \in C_j} d(x, y)$$

**Characteristics:**
- Produces compact, tight clusters
- Avoids chaining
- More robust to outliers than single linkage
- Tends to break large clusters

**Use when:**
- Want compact, spherical clusters
- Presence of outliers
- Need well-separated clusters

**Avoid when:**
- Natural clusters are elongated
- Clusters have different sizes

### Average Linkage (UPGMA)

Average distance between **all pairs** of points:

$$d(C_i, C_j) = \frac{1}{|C_i| |C_j|} \sum_{x \in C_i} \sum_{y \in C_j} d(x, y)$$

**Characteristics:**
- Balance between single and complete linkage
- Less sensitive to outliers
- Produces relatively compact clusters
- Generally good default choice

**Use when:**
- Balanced approach needed
- Moderate noise in data
- General-purpose clustering

### Ward's Linkage (Minimum Variance)

Minimizes within-cluster variance when merging. Merge clusters that minimize increase in total within-cluster variance:

$$d(C_i, C_j) = \frac{|C_i| |C_j|}{|C_i| + |C_j|} ||\mu_i - \mu_j||^2$$

where $\mu_i$ and $\mu_j$ are cluster centroids.

Equivalent to minimizing the increase in sum of squared errors (SSE):

$$\Delta SSE = \sum_{x \in C_i \cup C_j} ||x - \mu_{ij}||^2 - \sum_{x \in C_i} ||x - \mu_i||^2 - \sum_{x \in C_j} ||x - \mu_j||^2$$

**Characteristics:**
- Produces compact, evenly-sized clusters
- Similar to K-Means objective
- Most popular linkage method
- Only works with Euclidean distance

**Use when:**
- Want spherical, similarly-sized clusters
- Similar goals to K-Means but want hierarchy
- Need to minimize variance

**Avoid when:**
- Clusters have very different sizes
- Non-Euclidean distance metrics

### Centroid Linkage

Distance between cluster centroids:

$$d(C_i, C_j) = ||\mu_i - \mu_j||$$

where $\mu_i = \frac{1}{|C_i|} \sum_{x \in C_i} x$.

**Characteristics:**
- Simple and intuitive
- Can produce inversions in dendrogram
- Less commonly used

## Dendrograms

### Structure

A **dendrogram** is a tree diagram showing the hierarchical relationship between clusters.

**Components:**
- **Leaves** - individual data points at bottom
- **Branches** - connections showing merges
- **Height** - represents distance/dissimilarity at which clusters merge
- **Root** - top of tree (all points in one cluster)

### Interpretation

**Reading a dendrogram:**
1. **Height of merge** - indicates dissimilarity between merged clusters
   - Low height = similar clusters
   - High height = dissimilar clusters

2. **Horizontal cuts** - each cut across dendrogram produces a clustering
   - Cut low = many small clusters
   - Cut high = few large clusters

3. **Branch lengths** - longer vertical lines indicate larger distances between clusters

### Cophenetic Distance

**Cophenetic distance** between two points = height at which they first merge in dendrogram.

**Cophenetic correlation coefficient** measures how faithfully dendrogram preserves pairwise distances:

$$c = \frac{\sum_{i<j}(d_{ij} - \bar{d})(t_{ij} - \bar{t})}{\sqrt{\sum_{i<j}(d_{ij} - \bar{d})^2 \sum_{i<j}(t_{ij} - \bar{t})^2}}$$

where:
- $d_{ij}$ = original distance between points $i$ and $j$
- $t_{ij}$ = cophenetic distance between points $i$ and $j$

**Interpretation:**
- $c \approx 1$ - dendrogram well represents original distances
- $c < 0.8$ - poor fit, consider different linkage method

## Determining Number of Clusters

### Visual Inspection of Dendrogram

Look for:
1. **Large gaps** in merge heights - suggests natural number of clusters
2. **Horizontal cuts** that separate distinct branches
3. **Consistent cluster sizes** at a given height

### Distance Threshold

Cut dendrogram at height $h$ - all merges above height $h$ are separated:

$$K = \text{number of branches at height } h$$

**Strategy:**
- Look for large jump in merge distances
- Cut before the jump to get meaningful clusters

### Inconsistency Method

Measures how inconsistent a merge is compared to nearby merges:

$$I_k = \frac{|d_k - \bar{d}_{k}|}{s_k}$$

where:
- $d_k$ = height of merge $k$
- $\bar{d}_k$ = mean height of nearby merges
- $s_k$ = standard deviation of nearby merges

High inconsistency indicates a significant merge - good place to cut.

### Elbow Method

Plot number of clusters vs. within-cluster variance:

1. Cut dendrogram at different heights
2. Calculate variance for each clustering
3. Look for elbow in plot

### Silhouette Score

Compute silhouette score for different numbers of clusters (same as K-Means):

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Choose number of clusters that maximizes average silhouette score.

## Complexity Analysis

### Time Complexity

**Naive implementation:**
- $O(n^3)$ - compute all pairwise distances, merge closest at each step

**Optimized implementations:**
- **Single, Complete, Average linkage**: $O(n^2 \log n)$ using efficient data structures
- **Ward's linkage**: $O(n^2 \log n)$

### Space Complexity

- $O(n^2)$ - store distance matrix
- Can be reduced with careful implementation

### Comparison with K-Means

| Aspect | Hierarchical | K-Means |
|--------|--------------|---------|
| Time Complexity | $O(n^2 \log n)$ | $O(nKdi)$ |
| Scalability | Small-medium data | Large datasets |
| Deterministic | Yes | No (random init) |
| K specified | After analysis | Before running |

**Bottom line:** Hierarchical is slower but more informative for smaller datasets.

## Practical Considerations

### Feature Scaling

**Required** - distance-based method sensitive to scale:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Why?** Features with larger scales dominate distance calculations.

### Handling Large Datasets

**Strategies:**

1. **Sampling** - cluster on subset, assign rest:
   ```python
   from sklearn.cluster import AgglomerativeClustering

   # Cluster sample
   sample = X[::10]  # Every 10th point
   clusterer = AgglomerativeClustering(n_clusters=5)
   sample_labels = clusterer.fit_predict(sample)

   # Assign rest using nearest centroid
   ```

2. **Mini-batch approach** - cluster batches, then merge

3. **BIRCH** - hierarchical method designed for large data:
   ```python
   from sklearn.cluster import Birch

   birch = Birch(n_clusters=5, threshold=0.5)
   labels = birch.fit_predict(X)
   ```

### Choosing Linkage Method

**Guidelines:**

| Goal | Recommended Linkage |
|------|---------------------|
| Compact clusters | Ward or Complete |
| General purpose | Ward or Average |
| Non-spherical clusters | Single |
| Avoid chaining | Complete or Ward |
| Balanced approach | Average |

**Default recommendation:** Ward's linkage with Euclidean distance.

### Outlier Sensitivity

**Single linkage** - very sensitive (causes chaining)
**Complete linkage** - moderately robust
**Ward's linkage** - moderately robust

**Solution:** Remove outliers before clustering or use robust linkage methods.

### Dealing with Non-Euclidean Data

For categorical data or custom similarity measures:
- Use **single, complete, or average linkage** with appropriate distance metric
- **Ward's linkage requires Euclidean distance**

## Implementation

### Basic Example with Scipy

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=150, centers=3, n_features=2,
                       cluster_std=0.5, random_state=42)

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute linkage matrix
# method: 'single', 'complete', 'average', 'ward'
# metric: 'euclidean', 'manhattan', 'cosine', etc.
Z = linkage(X_scaled, method='ward', metric='euclidean')

# Z is a linkage matrix with shape (n-1, 4):
# [idx1, idx2, distance, sample_count]
print(f"Linkage matrix shape: {Z.shape}")
print(f"Last 5 merges:\n{Z[-5:]}")
```

### Plotting Dendrogram

```python
# Create dendrogram
plt.figure(figsize=(12, 8))
dendrogram(Z,
           truncate_mode='lastp',    # Show only last p merged clusters
           p=12,                      # Number of clusters to show
           leaf_rotation=90,
           leaf_font_size=10,
           show_contracted=True)      # Show height of contracted nodes

plt.title('Hierarchical Clustering Dendrogram (Ward)', fontsize=16)
plt.xlabel('Cluster Size', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.axhline(y=10, color='r', linestyle='--', label='Cut height')
plt.legend()
plt.tight_layout()
plt.show()
```

### Full Dendrogram with Labels

```python
# For small datasets, show all points
plt.figure(figsize=(15, 8))
dendrogram(Z,
           labels=np.arange(len(X)),  # Point indices
           leaf_rotation=90,
           leaf_font_size=8)

plt.title('Complete Dendrogram', fontsize=16)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.tight_layout()
plt.show()
```

### Cutting Dendrogram to Get Clusters

```python
# Method 1: Specify number of clusters
from scipy.cluster.hierarchy import fcluster

n_clusters = 3
labels_k = fcluster(Z, n_clusters, criterion='maxclust')

print(f"Cluster labels: {labels_k}")
print(f"Cluster sizes: {np.bincount(labels_k)}")

# Method 2: Specify distance threshold
threshold = 10
labels_t = fcluster(Z, threshold, criterion='distance')

print(f"\nWith threshold {threshold}:")
print(f"Number of clusters: {labels_t.max()}")
print(f"Cluster sizes: {np.bincount(labels_t)}")
```

### Using Scikit-learn

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import numpy as np

# Prepare data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply agglomerative clustering
agg_clustering = AgglomerativeClustering(
    n_clusters=3,              # Number of clusters
    linkage='ward',            # Linkage method: 'ward', 'complete', 'average', 'single'
    metric='euclidean'         # Distance metric (ignored for ward)
)

labels = agg_clustering.fit_predict(X_scaled)

print(f"Cluster labels: {labels}")
print(f"Number of clusters: {agg_clustering.n_clusters_}")
print(f"Number of leaves: {agg_clustering.n_leaves_}")
print(f"Cluster sizes: {np.bincount(labels)}")
```

### Comparing Linkage Methods

```python
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

linkage_methods = ['single', 'complete', 'average', 'ward']
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, method in enumerate(linkage_methods):
    # Fit clustering
    if method == 'ward':
        metric = 'euclidean'
    else:
        metric = 'euclidean'  # Can change to 'manhattan', 'cosine', etc.

    clustering = AgglomerativeClustering(n_clusters=3, linkage=method, metric=metric)
    labels = clustering.fit_predict(X_scaled)

    # Plot
    axes[idx].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
    axes[idx].set_title(f'{method.capitalize()} Linkage', fontsize=14)
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

### Cophenetic Correlation

```python
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist

# Original distances
distances = pdist(X_scaled, metric='euclidean')

# Try different linkage methods
linkage_methods = ['single', 'complete', 'average', 'ward']

for method in linkage_methods:
    Z = linkage(X_scaled, method=method)
    c, coph_dists = cophenet(Z, distances)
    print(f"{method.capitalize()} linkage - Cophenetic correlation: {c:.4f}")
```

### Finding Optimal Number of Clusters

```python
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Test different numbers of clusters
K_range = range(2, 11)
silhouette_scores = []
wcss = []

for k in K_range:
    clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = clustering.fit_predict(X_scaled)

    # Silhouette score
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

    # Within-cluster sum of squares
    cluster_centers = np.array([X_scaled[labels == i].mean(axis=0) for i in range(k)])
    wcss_k = sum(np.sum((X_scaled[labels == i] - cluster_centers[i])**2)
                 for i in range(k))
    wcss.append(wcss_k)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
ax1.set_ylabel('Silhouette Score', fontsize=12)
ax1.set_title('Silhouette Analysis', fontsize=14)
ax1.grid(True, alpha=0.3)

ax2.plot(K_range, wcss, 'bo-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
ax2.set_ylabel('Within-Cluster Sum of Squares', fontsize=12)
ax2.set_title('Elbow Method', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

best_k = K_range[np.argmax(silhouette_scores)]
print(f"Optimal K based on silhouette score: {best_k}")
```

### Complete Workflow

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

def hierarchical_clustering_analysis(X, max_clusters=10, linkage_method='ward'):
    """Complete hierarchical clustering workflow"""

    # 1. Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Compute linkage
    Z = linkage(X_scaled, method=linkage_method, metric='euclidean')

    # 3. Plot dendrogram
    plt.figure(figsize=(15, 8))
    dendrogram(Z, truncate_mode='lastp', p=20, leaf_rotation=90, leaf_font_size=10)
    plt.title(f'Dendrogram ({linkage_method} linkage)', fontsize=16)
    plt.xlabel('Cluster Size', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.tight_layout()
    plt.show()

    # 4. Find optimal K using silhouette
    silhouette_scores = []
    K_range = range(2, max_clusters + 1)

    for k in K_range:
        labels = fcluster(Z, k, criterion='maxclust')
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        print(f"K={k}: Silhouette Score = {score:.3f}, Sizes = {np.bincount(labels)[1:]}")

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Optimal K Selection', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 5. Get final clustering
    optimal_k = K_range[np.argmax(silhouette_scores)]
    final_labels = fcluster(Z, optimal_k, criterion='maxclust')

    print(f"\nOptimal number of clusters: {optimal_k}")
    print(f"Final cluster sizes: {np.bincount(final_labels)[1:]}")

    return Z, final_labels, X_scaled

# Usage
Z, labels, X_scaled = hierarchical_clustering_analysis(X, max_clusters=10, linkage_method='ward')
```

## References

- [Scipy Hierarchical Clustering Documentation](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
- [Scikit-learn AgglomerativeClustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
- Müllner, D. (2011). "Modern hierarchical, agglomerative clustering algorithms"
- Ward, J. H. (1963). "Hierarchical grouping to optimize an objective function"
- The Elements of Statistical Learning (Hastie, Tibshirani, Friedman)
- Pattern Recognition and Machine Learning (Christopher Bishop)
