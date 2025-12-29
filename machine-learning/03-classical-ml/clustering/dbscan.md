# DBSCAN (Density-Based Spatial Clustering)

## Table of Contents

1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Algorithm](#algorithm)
4. [Parameters](#parameters)
5. [Distance Metrics](#distance-metrics)
6. [Advantages and Limitations](#advantages-and-limitations)
7. [Variants and Extensions](#variants-and-extensions)
8. [Practical Considerations](#practical-considerations)
9. [Implementation](#implementation)
10. [References](#references)

## Introduction

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** is a density-based clustering algorithm that groups together points that are closely packed while marking points in low-density regions as outliers.

Unlike K-Means and hierarchical clustering, DBSCAN:
- **Does not require** specifying the number of clusters beforehand
- **Can find** arbitrarily shaped clusters
- **Identifies** outliers as noise points
- **Is robust** to outliers

### Key Characteristics

- **Density-based** - clusters are dense regions separated by low-density regions
- **Arbitrary shapes** - can find non-spherical clusters
- **Automatic outlier detection** - marks noise points
- **No K required** - number of clusters emerges from data
- **Deterministic** - produces same result every run (for fixed order)

### When to Use

**Best suited for:**
- Spatial data with varying cluster shapes
- Presence of noise and outliers
- Unknown number of clusters
- Clusters with arbitrary shapes (crescents, spirals, etc.)
- Datasets where density-based grouping makes sense

**Not recommended for:**
- Clusters with varying densities
- High-dimensional data (distance becomes less meaningful)
- Very large datasets without spatial indexing
- When all points must be assigned to clusters

## Core Concepts

### Point Types

DBSCAN classifies points into three categories:

**1. Core Point**
- Has at least `min_samples` points within distance `eps` (including itself)
- Interior point of a dense region
- Can form a cluster

**2. Border Point**
- Has fewer than `min_samples` points within `eps`
- Is within `eps` distance of a core point
- On the edge of a cluster
- Belongs to cluster but doesn't extend it

**3. Noise Point (Outlier)**
- Not a core point
- Not within `eps` of any core point
- Labeled as -1
- May later be reclassified if more data is added

### Epsilon Neighborhood

The **ε-neighborhood** of a point $p$:

$$N_\varepsilon(p) = \{q \in D : dist(p, q) \leq \varepsilon\}$$

where:
- $D$ is the dataset
- $dist(p, q)$ is the distance between points $p$ and $q$
- $\varepsilon$ (eps) is the radius parameter

### Density-Reachability

**Directly density-reachable:**
Point $q$ is directly density-reachable from point $p$ if:
1. $q \in N_\varepsilon(p)$ (q is in p's neighborhood)
2. $p$ is a core point (has at least `min_samples` neighbors)

**Density-reachable:**
Point $q$ is density-reachable from $p$ if there exists a chain of points $p_1, \ldots, p_n$ where:
- $p_1 = p$ and $p_n = q$
- $p_{i+1}$ is directly density-reachable from $p_i$

**Density-connected:**
Points $p$ and $q$ are density-connected if there exists a point $o$ such that both $p$ and $q$ are density-reachable from $o$.

### Cluster Definition

A **cluster** $C$ is a maximal set of density-connected points:

1. **Maximality**: If point $p$ is in cluster $C$ and point $q$ is density-reachable from $p$, then $q$ is also in $C$
2. **Connectivity**: All points in $C$ are density-connected

## Algorithm

### DBSCAN Pseudocode

```
DBSCAN(D, eps, min_samples):
    C = 0                          # Cluster counter
    labels = [-1] * |D|            # Initialize all as unvisited

    for each point P in D:
        if P is already labeled:   # Skip if visited
            continue

        neighbors = find_neighbors(P, eps)

        if |neighbors| < min_samples:
            labels[P] = -1         # Mark as noise
            continue

        C = C + 1                  # New cluster
        labels[P] = C

        seed_set = neighbors - {P}

        for each point Q in seed_set:
            if labels[Q] == -1:    # Was noise, now border
                labels[Q] = C

            if labels[Q] != -1:    # Already processed
                continue

            labels[Q] = C
            Q_neighbors = find_neighbors(Q, eps)

            if |Q_neighbors| >= min_samples:  # Q is core point
                seed_set = seed_set ∪ Q_neighbors

    return labels
```

### Step-by-Step Process

1. **Initialize**: Mark all points as unvisited

2. **For each unvisited point P**:
   - Find all neighbors within distance `eps`

3. **If P has fewer than `min_samples` neighbors**:
   - Mark P as noise (label -1)
   - Continue to next point

4. **If P has at least `min_samples` neighbors**:
   - Start new cluster with P as core point
   - Add all neighbors to seed set

5. **Expand cluster**:
   - For each point Q in seed set:
     - If Q is noise, change to border point
     - If Q is unvisited:
       - Add Q to cluster
       - If Q is core point, add Q's neighbors to seed set

6. **Repeat** until all points visited

### Time Complexity

**Without spatial indexing:**
- $O(n^2)$ - need to compute distances to all points

**With spatial indexing (KD-tree, Ball-tree):**
- $O(n \log n)$ - efficient nearest neighbor search
- Practical performance much better for low-medium dimensions

**Space Complexity:**
- $O(n)$ - store labels and seed set

## Parameters

### Epsilon (eps)

**Definition:** Maximum distance between two points to be considered neighbors.

**Effect:**
- **Too small**: Many points classified as noise, many small clusters
- **Too large**: Clusters merge together, few large clusters
- **Optimal**: Reflects natural scale of data

**How to choose:**

**1. K-distance graph:**
```python
from sklearn.neighbors import NearestNeighbors

# Plot k-distance graph
k = min_samples
neighbors = NearestNeighbors(n_neighbors=k)
neighbors.fit(X)
distances, indices = neighbors.kneighbors(X)

distances = np.sort(distances[:, k-1], axis=0)

plt.plot(distances)
plt.ylabel('k-NN distance')
plt.xlabel('Points sorted by distance')
plt.title(f'{k}-distance Graph')
plt.show()

# Look for "elbow" - sharp increase indicates good eps
```

**2. Domain knowledge:**
- Use meaningful distance for your data
- E.g., geographic data: 100 meters, 1 kilometer

**3. Trial and error:**
- Start with distance at which most points have k neighbors
- Adjust based on clustering results

### MinPts (min_samples)

**Definition:** Minimum number of points required to form a dense region (core point).

**Effect:**
- **Too small**: Too many small clusters, noise treated as clusters
- **Too large**: Large clusters merge, more points classified as noise
- **Optimal**: Captures minimum cluster size

**How to choose:**

**Rule of thumb:**
$$\text{min\_samples} \geq D + 1$$

where $D$ is the number of dimensions.

**Common values:**
- **2D data**: min_samples = 4
- **Higher dimensions**: min_samples = 2 × dimensions
- **Noisy data**: Higher min_samples (e.g., 10-20)

**Domain considerations:**
- Minimum meaningful cluster size
- Expected density of clusters
- Noise level in data

## Distance Metrics

### Euclidean Distance (Default)

$$d(p, q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}$$

**Best for:**
- General-purpose clustering
- Continuous numerical features
- Low-medium dimensions

**Requires:** Feature scaling

### Manhattan Distance

$$d(p, q) = \sum_{i=1}^{n}|p_i - q_i|$$

**Best for:**
- Grid-like data
- Features with different units
- Presence of outliers

### Haversine Distance

For geographic coordinates (latitude, longitude):

$$d = 2r \arcsin\left(\sqrt{\sin^2\left(\frac{\phi_2-\phi_1}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\lambda_2-\lambda_1}{2}\right)}\right)$$

where $r$ is Earth's radius.

**Use for:** Geospatial clustering

### Custom Metrics

DBSCAN works with any distance metric that satisfies metric properties:
- Non-negativity
- Identity
- Symmetry
- Triangle inequality

## Advantages and Limitations

### Advantages

**1. No need to specify K**
- Number of clusters emerges from data
- Good for exploratory analysis

**2. Arbitrary cluster shapes**
- Can find non-convex, complex shapes
- Not limited to spherical clusters
- Examples: crescents, spirals, interleaved shapes

**3. Robust to outliers**
- Explicitly identifies noise points
- Outliers don't affect cluster formation

**4. Only two parameters**
- `eps` and `min_samples`
- Intuitive interpretation

**5. Deterministic**
- Same result every run (for fixed point order)
- No random initialization

### Limitations

**1. Sensitive to parameters**
- Different `eps` values produce very different results
- Hard to choose parameters for varying densities

**2. Varying densities**
- Struggles with clusters of different densities
- Single `eps` can't capture all densities
- Solution: Use HDBSCAN or OPTICS

**3. High dimensions**
- Distance becomes less meaningful (curse of dimensionality)
- All points appear equidistant
- Solution: Dimensionality reduction first

**4. Computational cost**
- $O(n^2)$ without indexing
- Indexing structures degrade in high dimensions

**5. Border points**
- Border points between clusters assigned arbitrarily
- Can belong to different clusters in different runs

## Variants and Extensions

### HDBSCAN (Hierarchical DBSCAN)

**Key innovation:** Builds hierarchy over varying density levels.

**Advantages:**
- Handles varying density clusters
- More robust parameter selection
- Provides stability scores for clusters
- Only one parameter: `min_cluster_size`

**How it works:**
1. Build mutual reachability graph
2. Construct minimum spanning tree
3. Build cluster hierarchy
4. Extract stable clusters

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
labels = clusterer.fit_predict(X)
```

### OPTICS (Ordering Points To Identify Clustering Structure)

**Key innovation:** Creates reachability plot showing cluster structure at all distance scales.

**Advantages:**
- Visualizes clustering structure
- Works with varying densities
- No need to specify `eps` upfront

**Parameters:**
- `min_samples`: Same as DBSCAN
- `max_eps`: Upper bound on `eps` (optional)

```python
from sklearn.cluster import OPTICS

optics = OPTICS(min_samples=5, max_eps=2.0, metric='euclidean')
labels = optics.fit_predict(X)
```

### DBSCAN with Variable Eps

Use different `eps` for different regions:

**Local Outlier Factor (LOF)** as preprocessing:
1. Compute local density for each point
2. Use density-based `eps` values
3. Apply DBSCAN variant

### ST-DBSCAN (Spatial-Temporal)

Extension for spatial-temporal data:

$$dist((x_1, t_1), (x_2, t_2)) = \sqrt{\alpha \cdot d_{spatial}(x_1, x_2)^2 + (1-\alpha) \cdot d_{temporal}(t_1, t_2)^2}$$

**Use for:** GPS trajectories, event clustering

## Practical Considerations

### Feature Scaling

**Critical:** DBSCAN is distance-based and sensitive to feature scales.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Why?**
- Unscaled features: larger-scale features dominate
- Example: Age (0-100) vs. Income (0-100,000)

### Parameter Selection Strategy

**Step-by-step approach:**

1. **Start with k-distance graph**
   ```python
   from sklearn.neighbors import NearestNeighbors
   import matplotlib.pyplot as plt

   k = 4  # min_samples candidate
   neighbors = NearestNeighbors(n_neighbors=k)
   neighbors.fit(X_scaled)
   distances, _ = neighbors.kneighbors(X_scaled)
   distances = np.sort(distances[:, k-1], axis=0)

   plt.plot(distances)
   plt.ylabel(f'{k}-NN Distance')
   plt.xlabel('Points (sorted)')
   plt.grid(True)
   plt.show()
   ```

2. **Identify elbow in k-distance graph**
   - Sharp increase indicates good `eps` value
   - Points before elbow: core/border points
   - Points after elbow: noise

3. **Set min_samples based on dimensionality**
   - Rule of thumb: `min_samples >= dimensions + 1`
   - Higher for noisy data

4. **Test and iterate**
   - Try different parameter combinations
   - Evaluate using domain knowledge and metrics

### Evaluating Results

**Metrics for DBSCAN:**

**1. Number of clusters:**
```python
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
```

**2. Number of noise points:**
```python
n_noise = list(labels).count(-1)
```

**3. Silhouette score** (excluding noise):
```python
from sklearn.metrics import silhouette_score

mask = labels != -1
if len(set(labels[mask])) > 1:
    score = silhouette_score(X_scaled[mask], labels[mask])
```

**4. Visual inspection:**
- Plot clusters with different colors
- Noise points in separate color
- Assess if results make sense

### Handling High-Dimensional Data

**Strategies:**

**1. Dimensionality reduction first:**
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X_scaled)
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_reduced)
```

**2. Feature selection:**
- Remove irrelevant features
- Keep features with meaningful distances

**3. Use alternative metrics:**
- Cosine distance for high-dimensional data
- Custom metrics for specific domains

### Border Point Assignment

Border points can belong to multiple clusters:

**Issue:** Order of processing affects border point assignment.

**Solutions:**
1. **Accept variability** - border points are inherently ambiguous
2. **Post-processing** - assign border points to nearest cluster centroid
3. **Use HDBSCAN** - provides outlier scores for better assignment

## Implementation

### Basic Example

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt

# Generate non-linear separable data
X, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(
    eps=0.3,              # Maximum distance between neighbors
    min_samples=5,        # Minimum points to form dense region
    metric='euclidean'    # Distance metric
)

labels = dbscan.fit_predict(X_scaled)

# Results
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print(f"Cluster sizes: {np.bincount(labels[labels >= 0])}")

# Core samples (points that are core points)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True

print(f"Number of core points: {sum(core_samples_mask)}")
print(f"Number of border points: {sum(labels >= 0) - sum(core_samples_mask)}")
```

### Visualization

```python
import matplotlib.pyplot as plt

# Unique labels
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

plt.figure(figsize=(12, 5))

# Plot with noise
plt.subplot(1, 2, 1)
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Noise points - black
        col = 'black'
        marker = 'x'
        size = 20
        label = 'Noise'
    else:
        marker = 'o'
        size = 50
        label = f'Cluster {k}'

    class_member_mask = (labels == k)
    xy = X[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, s=size,
                alpha=0.6, label=label, edgecolors='black', linewidth=0.5)

plt.title('DBSCAN Clustering', fontsize=14)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Highlight core vs border points
plt.subplot(1, 2, 2)
for k, col in zip(unique_labels, colors):
    if k == -1:
        continue

    class_member_mask = (labels == k)
    xy = X[class_member_mask & core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker='o', s=100,
                alpha=0.8, edgecolors='black', linewidth=1, label=f'Core {k}')

    xy = X[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker='o', s=30,
                alpha=0.4, edgecolors='black', linewidth=0.5)

# Noise
xy = X[labels == -1]
plt.scatter(xy[:, 0], xy[:, 1], c='black', marker='x', s=20, alpha=0.6)

plt.title('Core vs Border Points', fontsize=14)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()
```

### K-distance Graph for Parameter Selection

```python
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

def plot_kdistance_graph(X, k_values=[3, 4, 5, 10]):
    """Plot k-distance graphs for different k values"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for idx, k in enumerate(k_values):
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(X)
        distances, indices = neighbors.kneighbors(X)

        # Sort distances
        distances = np.sort(distances[:, k-1], axis=0)

        # Plot
        axes[idx].plot(distances, linewidth=2)
        axes[idx].set_ylabel(f'{k}-NN Distance', fontsize=12)
        axes[idx].set_xlabel('Points (sorted by distance)', fontsize=12)
        axes[idx].set_title(f'k={k} Distance Graph', fontsize=14)
        axes[idx].grid(True, alpha=0.3)

        # Suggest eps based on elbow (90th percentile as heuristic)
        suggested_eps = np.percentile(distances, 90)
        axes[idx].axhline(y=suggested_eps, color='r', linestyle='--',
                          label=f'Suggested eps: {suggested_eps:.3f}')
        axes[idx].legend()

    plt.tight_layout()
    plt.show()

# Usage
plot_kdistance_graph(X_scaled, k_values=[3, 4, 5, 10])
```

### Parameter Grid Search

```python
from sklearn.metrics import silhouette_score
import numpy as np

def dbscan_grid_search(X, eps_values, min_samples_values):
    """Find best DBSCAN parameters using silhouette score"""

    results = []

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            # Calculate silhouette (only if we have clusters and non-noise points)
            if n_clusters > 1 and n_noise < len(X) - 1:
                mask = labels != -1
                silhouette = silhouette_score(X[mask], labels[mask])
            else:
                silhouette = -1

            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette': silhouette
            })

            print(f"eps={eps:.2f}, min_samples={min_samples}: "
                  f"clusters={n_clusters}, noise={n_noise}, "
                  f"silhouette={silhouette:.3f}")

    # Find best parameters
    results_sorted = sorted(results, key=lambda x: x['silhouette'], reverse=True)
    best = results_sorted[0]

    print(f"\nBest parameters:")
    print(f"eps={best['eps']}, min_samples={best['min_samples']}")
    print(f"Silhouette score: {best['silhouette']:.3f}")

    return results, best

# Usage
eps_values = np.arange(0.1, 1.0, 0.1)
min_samples_values = [3, 4, 5, 6, 7, 8, 10]

results, best_params = dbscan_grid_search(X_scaled, eps_values, min_samples_values)
```

### Comparing with K-Means

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_moons, make_circles
import matplotlib.pyplot as plt

# Create different dataset shapes
datasets = [
    make_moons(n_samples=300, noise=0.05, random_state=42),
    make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42),
    make_blobs(n_samples=300, centers=3, random_state=42)
]

dataset_names = ['Moons', 'Circles', 'Blobs']

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for i, (X, y) in enumerate(datasets):
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Original
    axes[i, 0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.6)
    axes[i, 0].set_title(f'{dataset_names[i]} - Original', fontsize=12)

    # K-Means
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    axes[i, 1].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.6)
    axes[i, 1].set_title(f'{dataset_names[i]} - K-Means', fontsize=12)

    # DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    # Plot DBSCAN
    unique_labels = set(dbscan_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'black'
            marker = 'x'
        else:
            marker = 'o'

        class_member_mask = (dbscan_labels == k)
        xy = X[class_member_mask]
        axes[i, 2].scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker,
                           s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

    axes[i, 2].set_title(f'{dataset_names[i]} - DBSCAN', fontsize=12)

plt.tight_layout()
plt.show()
```

### Geographic Clustering Example

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Example: Clustering GPS coordinates
# Format: [latitude, longitude]
gps_data = np.array([
    [40.7128, -74.0060],  # New York
    [40.7589, -73.9851],  # Times Square
    [40.7484, -73.9857],  # Empire State
    [34.0522, -118.2437], # Los Angeles
    [34.0689, -118.4452], # Santa Monica
    [41.8781, -87.6298],  # Chicago
    [41.8819, -87.6278],  # Chicago Loop
])

# Use haversine distance for geographic coordinates
# Convert to radians
gps_radians = np.radians(gps_data)

# DBSCAN with haversine metric
# eps in radians (0.01 radians ≈ 111 km * 0.01 ≈ 1.1 km)
dbscan = DBSCAN(eps=0.01, min_samples=2, metric='haversine')
labels = dbscan.fit_predict(gps_radians)

print(f"Cluster labels: {labels}")
print(f"Number of clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")

# Plot on map
plt.figure(figsize=(10, 6))
for k in set(labels):
    if k == -1:
        col = 'black'
        marker = 'x'
        label = 'Noise'
    else:
        col = plt.cm.Spectral(k / len(set(labels)))
        marker = 'o'
        label = f'Cluster {k}'

    class_member_mask = (labels == k)
    xy = gps_data[class_member_mask]
    plt.scatter(xy[:, 1], xy[:, 0], c=[col], marker=marker,
                s=200, alpha=0.7, label=label, edgecolors='black', linewidth=2)

plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.title('Geographic Clustering with DBSCAN', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## References

- [Scikit-learn DBSCAN Documentation](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
- Ester, M., et al. (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise"
- [HDBSCAN Documentation](https://hdbscan.readthedocs.io/)
- Campello, R. J., et al. (2013). "Density-based clustering based on hierarchical density estimates"
- Ankerst, M., et al. (1999). "OPTICS: Ordering points to identify the clustering structure"
- Pattern Recognition and Machine Learning (Christopher Bishop)
