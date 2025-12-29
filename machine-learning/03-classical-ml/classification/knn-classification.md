# K-Nearest Neighbors Classification

## Table of Contents

1. [Introduction](#introduction)
2. [Algorithm Fundamentals](#algorithm-fundamentals)
3. [Distance Metrics](#distance-metrics)
4. [Choosing K](#choosing-k)
5. [Weighted KNN](#weighted-knn)
6. [Computational Complexity](#computational-complexity)
7. [Practical Considerations](#practical-considerations)
8. [Implementation](#implementation)
9. [References](#references)

## Introduction

**K-Nearest Neighbors (KNN)** is a simple, non-parametric, instance-based learning algorithm that classifies new instances based on the majority class of their k nearest neighbors in the training data.

### Key Characteristics

- **Non-parametric** - makes no assumptions about data distribution
- **Instance-based** - stores all training data (lazy learning)
- **No training phase** - all computation happens during prediction
- **Intuitive** - easy to understand and implement
- **Versatile** - works for both classification and regression

### When to Use

**Best suited for:**
- Small to medium-sized datasets
- Low-dimensional feature spaces (curse of dimensionality)
- Non-linear decision boundaries
- When interpretability is important
- Baseline model for comparison

**Not recommended for:**
- Very large datasets (slow prediction)
- High-dimensional data (unless using dimensionality reduction)
- Imbalanced datasets (without modifications)
- Real-time applications (slow inference)

### Advantages and Disadvantages

**Advantages:**
- No training time (lazy learning)
- Naturally handles multi-class problems
- Simple and intuitive
- No assumptions about data distribution
- Adapts to complex decision boundaries
- Easy to update with new data

**Disadvantages:**
- Slow prediction (must compute distance to all training points)
- Memory intensive (stores entire dataset)
- Sensitive to feature scaling
- Curse of dimensionality in high dimensions
- Sensitive to irrelevant features and noise
- Requires choosing appropriate k and distance metric

## Algorithm Fundamentals

### Classification Process

**Given:** Training set $\mathcal{D} = \{(x_1, y_1), ..., (x_n, y_n)\}$ and query point $x_q$

**Step 1:** Compute distance from $x_q$ to all training points:

$$d(x_q, x_i) = distance(x_q, x_i)$$

**Step 2:** Find k nearest neighbors $\mathcal{N}_k(x_q)$

**Step 3:** Predict class by majority vote:

$$\hat{y} = \arg\max_{c} \sum_{(x_i, y_i) \in \mathcal{N}_k(x_q)} \mathbb{1}(y_i = c)$$

where $\mathbb{1}$ is the indicator function (1 if condition true, 0 otherwise).

### Probability Estimates

KNN can provide class probability estimates:

$$P(y = c | x_q) = \frac{1}{k} \sum_{(x_i, y_i) \in \mathcal{N}_k(x_q)} \mathbb{1}(y_i = c)$$

This is simply the fraction of k neighbors belonging to class $c$.

### Decision Boundary

KNN creates **piecewise linear** (for k=1) or **smooth non-linear** (for k>1) decision boundaries.

- **k=1:** Voronoi diagram decision boundary (highly flexible, prone to overfitting)
- **Larger k:** Smoother decision boundaries (more regularized)

The decision boundary is **data-dependent** and changes with training set composition.

## Distance Metrics

### Euclidean Distance

**L2 norm** - straight-line distance between two points:

$$d(x, y) = \sqrt{\sum_{i=1}^{d}(x_i - y_i)^2} = ||x - y||_2$$

**Properties:**
- Most commonly used
- Assumes all features equally important
- Sensitive to feature scales
- Works well when features represent same units

**When to use:**
- Features on similar scales
- Physical measurements
- Continuous numerical features

### Manhattan Distance

**L1 norm** - sum of absolute differences:

$$d(x, y) = \sum_{i=1}^{d}|x_i - y_i| = ||x - y||_1$$

**Properties:**
- Less sensitive to outliers than Euclidean
- Represents grid-like paths
- Faster to compute (no squares/square root)

**When to use:**
- Grid-based problems
- High-dimensional spaces
- Presence of outliers

### Minkowski Distance

**Generalized distance metric**:

$$d(x, y) = \left(\sum_{i=1}^{d}|x_i - y_i|^p\right)^{1/p}$$

Special cases:
- $p = 1$: Manhattan distance
- $p = 2$: Euclidean distance
- $p \to \infty$: Chebyshev distance (max of absolute differences)

### Cosine Similarity

Measures the cosine of the angle between vectors:

$$similarity(x, y) = \frac{x \cdot y}{||x|| \cdot ||y||} = \frac{\sum_{i=1}^{d} x_i y_i}{\sqrt{\sum_{i=1}^{d} x_i^2} \sqrt{\sum_{i=1}^{d} y_i^2}}$$

**Cosine distance:**

$$d(x, y) = 1 - similarity(x, y)$$

**Properties:**
- Measures orientation, not magnitude
- Range: [0, 2] for distance, [-1, 1] for similarity
- Invariant to vector length

**When to use:**
- Text data (document similarity)
- High-dimensional sparse data
- When magnitude is not important

### Hamming Distance

For categorical/binary features:

$$d(x, y) = \sum_{i=1}^{d} \mathbb{1}(x_i \neq y_i)$$

Counts the number of positions where features differ.

**When to use:**
- Categorical features
- Binary data
- DNA sequences, error detection

### Mahalanobis Distance

Accounts for correlation between features:

$$d(x, y) = \sqrt{(x-y)^T \Sigma^{-1} (x-y)}$$

where $\Sigma$ is the covariance matrix.

**Properties:**
- Scale-invariant
- Accounts for feature correlations
- Computationally expensive

**When to use:**
- Correlated features
- Features with different variances
- When covariance structure is known

## Choosing K

### Impact of K

**Small k (k=1,3,5):**
- **Pros:** Flexible, captures local patterns, low bias
- **Cons:** Sensitive to noise, high variance, overfitting
- Decision boundary: Complex and jagged

**Large k (k=20,50,100):**
- **Pros:** Robust to noise, smooth decision boundary, low variance
- **Cons:** May miss local patterns, high bias, underfitting
- Decision boundary: Smooth and simple

**Rule of thumb:** $k = \sqrt{n}$ where n is the number of training samples.

### Finding Optimal K

**Cross-validation approach:**

```python
from sklearn.model_selection import cross_val_score

k_values = range(1, 31, 2)  # Try odd values
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k = k_values[np.argmax(cv_scores)]
```

**Considerations:**
- Use **odd k** for binary classification (avoids ties)
- Perform **cross-validation** to find best k
- Plot **error rate vs k** to visualize trade-off
- Consider **computational cost** (larger k is slower)

### Bias-Variance Tradeoff

$$Error = Bias^2 + Variance + Irreducible Error$$

- **k=1:** High variance (overfitting), low bias
- **k=n:** High bias (underfitting), low variance
- **Optimal k:** Balances bias and variance

## Weighted KNN

### Distance-Weighted Voting

Instead of uniform voting, weight neighbors by inverse distance:

$$w_i = \frac{1}{d(x_q, x_i)}$$

**Classification:**

$$\hat{y} = \arg\max_{c} \sum_{(x_i, y_i) \in \mathcal{N}_k(x_q)} w_i \cdot \mathbb{1}(y_i = c)$$

**Advantages:**
- Closer neighbors have more influence
- Reduces impact of distant neighbors
- More robust to choice of k

**Common weighting schemes:**

1. **Inverse distance:**
   $$w_i = \frac{1}{d(x_q, x_i) + \epsilon}$$
   where $\epsilon$ prevents division by zero

2. **Gaussian:**
   $$w_i = \exp\left(-\frac{d(x_q, x_i)^2}{2\sigma^2}\right)$$

3. **Rank-based:**
   $$w_i = \frac{1}{rank_i}$$

### Probability Estimates with Weights

$$P(y = c | x_q) = \frac{\sum_{(x_i, y_i) \in \mathcal{N}_k(x_q)} w_i \cdot \mathbb{1}(y_i = c)}{\sum_{(x_i, y_i) \in \mathcal{N}_k(x_q)} w_i}$$

## Computational Complexity

### Time Complexity

**Training:**
- $O(1)$ - no training, just store data

**Prediction (naive):**
- $O(nd)$ - compute distance to all n training points with d dimensions
- For m predictions: $O(mnd)$

**With optimizations (KD-tree, Ball tree):**
- Build tree: $O(dn \log n)$
- Query: $O(d \log n)$ average case
- Degrades to $O(dn)$ in high dimensions (d > 20)

### Space Complexity

- $O(nd)$ - must store all training data

### Optimization Techniques

**1. KD-Tree (K-Dimensional Tree):**
- Binary tree that partitions space
- Fast for low dimensions (d < 20)
- Build: $O(dn \log n)$, Query: $O(\log n)$ average
- Degrades in high dimensions

**2. Ball Tree:**
- Tree structure with hypersphere nodes
- Better than KD-tree for d > 20
- More robust to curse of dimensionality
- Build: $O(dn \log n)$, Query: $O(\log n)$ average

**3. Locality-Sensitive Hashing (LSH):**
- Approximate nearest neighbors
- Very fast, but approximate
- Excellent for very high dimensions

**4. Brute Force:**
- Compute all distances
- Always works, no approximation
- Best for small datasets or very high dimensions

## Practical Considerations

### Feature Scaling

**Critical for KNN** - features on different scales dominate distance calculations.

**Example:** Age (0-100) vs Income (0-1,000,000)
- Without scaling: income dominates
- With scaling: both contribute equally

**Standardization (Z-score normalization):**

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

**Min-Max scaling:**

$$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Curse of Dimensionality

**Problem:** In high dimensions:
- All points become approximately equidistant
- Nearest neighbors are not "near"
- Volume of hypersphere concentrates in outer shell

**Effects:**
- KNN performance degrades
- Need exponentially more data
- Distance metrics become meaningless

**Solutions:**
1. **Dimensionality reduction:**
   - PCA (Principal Component Analysis)
   - t-SNE, UMAP
   - Feature selection

2. **Feature engineering:**
   - Select relevant features
   - Domain knowledge

3. **Different distance metrics:**
   - Manhattan distance (more robust)
   - Cosine similarity

**Rule of thumb:** KNN works well when $d < 20$.

### Handling Imbalanced Data

**Problem:** Majority class dominates k-nearest neighbors.

**Solutions:**

1. **Class-weighted voting:**
   ```python
   knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')  # or 'distance'
   ```

2. **Resampling:**
   - SMOTE (Synthetic Minority Over-sampling)
   - Random undersampling of majority class

3. **Adjust k:**
   - Smaller k gives minority class better chance
   - Use odd k to avoid ties

4. **Distance threshold:**
   - Only consider neighbors within threshold
   - Adaptive k based on local density

### Handling Ties

**Problem:** Equal votes from multiple classes (only with even k).

**Solutions:**
1. **Use odd k** (simplest)
2. **Reduce k by 1** until tie breaks
3. **Distance-based tiebreaker** - choose class with closer neighbors
4. **Random selection** - choose randomly among tied classes

### Outliers and Noise

**Impact:**
- k=1 is very sensitive to outliers
- Larger k provides robustness

**Strategies:**
1. **Use larger k** - reduces outlier influence
2. **Outlier removal** - clean data before training
3. **Weighted voting** - reduces impact of distant outliers
4. **Robust distance metrics** - Manhattan instead of Euclidean

### Comparison with Other Algorithms

| Aspect | KNN | Decision Trees | SVM | Naive Bayes |
|--------|-----|----------------|-----|-------------|
| Training speed | Instant | Fast | Slow | Fast |
| Prediction speed | Slow | Fast | Fast | Fast |
| Memory usage | High | Low | Medium | Low |
| Interpretability | Medium | High | Low | High |
| High dimensions | Poor | Medium | Good | Good |
| Non-linearity | Yes | Yes | Yes (kernel) | No |
| Probabilistic | Yes | Yes | No (native) | Yes |
| Overfitting | Prone (small k) | Prone | Resistant | Resistant |

## Implementation

### Basic KNN Classification

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (critical!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN
knn = KNeighborsClassifier(
    n_neighbors=5,        # Number of neighbors
    weights='uniform',    # 'uniform' or 'distance'
    algorithm='auto',     # 'auto', 'ball_tree', 'kd_tree', 'brute'
    metric='euclidean',   # Distance metric
    p=2                   # Power parameter for Minkowski (2 = Euclidean)
)

knn.fit(X_train_scaled, y_train)

# Predictions
y_pred = knn.predict(X_test_scaled)
y_proba = knn.predict_proba(X_test_scaled)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

### Finding Optimal K

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Try different k values
k_range = range(1, 31, 2)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

# Find optimal k
optimal_k = k_range[np.argmax(k_scores)]
print(f"Optimal k: {optimal_k}")
print(f"Best accuracy: {max(k_scores):.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(k_range, k_scores, marker='o')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('KNN: Accuracy vs K')
plt.grid(True)
plt.show()
```

### Distance-Weighted KNN

```python
# Use distance-weighted voting
knn_weighted = KNeighborsClassifier(
    n_neighbors=10,
    weights='distance',  # Weight by inverse distance
    metric='euclidean'
)

knn_weighted.fit(X_train_scaled, y_train)
y_pred_weighted = knn_weighted.predict(X_test_scaled)

print(f"Weighted KNN Accuracy: {accuracy_score(y_test, y_pred_weighted):.4f}")
```

### Different Distance Metrics

```python
# Manhattan distance
knn_manhattan = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn_manhattan.fit(X_train_scaled, y_train)

# Cosine distance
knn_cosine = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn_cosine.fit(X_train_scaled, y_train)

# Minkowski with p=3
knn_minkowski = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=3)
knn_minkowski.fit(X_train_scaled, y_train)

# Compare
print(f"Euclidean: {knn.score(X_test_scaled, y_test):.4f}")
print(f"Manhattan: {knn_manhattan.score(X_test_scaled, y_test):.4f}")
print(f"Cosine: {knn_cosine.score(X_test_scaled, y_test):.4f}")
print(f"Minkowski (p=3): {knn_minkowski.score(X_test_scaled, y_test):.4f}")
```

### Grid Search for Best Parameters

```python
from sklearn.model_selection import GridSearchCV

# Parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'cosine'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Use best model
best_knn = grid_search.best_estimator_
y_pred_best = best_knn.predict(X_test_scaled)
print(f"Test accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
```

### Algorithm Comparison

```python
# Compare different algorithms for neighbor search
algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
results = {}

for algo in algorithms:
    knn = KNeighborsClassifier(n_neighbors=5, algorithm=algo)

    # Time fitting
    import time
    start = time.time()
    knn.fit(X_train_scaled, y_train)
    fit_time = time.time() - start

    # Time prediction
    start = time.time()
    y_pred = knn.predict(X_test_scaled)
    pred_time = time.time() - start

    results[algo] = {
        'fit_time': fit_time,
        'pred_time': pred_time,
        'accuracy': accuracy_score(y_test, y_pred)
    }

for algo, metrics in results.items():
    print(f"{algo:10s} - Fit: {metrics['fit_time']:.4f}s, "
          f"Pred: {metrics['pred_time']:.4f}s, "
          f"Acc: {metrics['accuracy']:.4f}")
```

### Custom Distance Function

```python
from sklearn.metrics import pairwise_distances

# Define custom distance
def custom_distance(x, y):
    """Example: weighted Euclidean distance"""
    weights = np.array([2.0, 1.0, 0.5])  # Different weights for features
    return np.sqrt(np.sum(weights * (x - y) ** 2))

# Use custom metric
knn_custom = KNeighborsClassifier(
    n_neighbors=5,
    metric=custom_distance
)

knn_custom.fit(X_train_scaled, y_train)
y_pred_custom = knn_custom.predict(X_test_scaled)
```

### Find K-Nearest Neighbors

```python
# Get indices and distances of k nearest neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# For a single query point
distances, indices = knn.kneighbors(X_test_scaled[[0]])

print(f"Distances to 5 nearest neighbors: {distances[0]}")
print(f"Indices of 5 nearest neighbors: {indices[0]}")
print(f"Labels of nearest neighbors: {y_train[indices[0]]}")
```

### Radius-Based Neighbors

```python
from sklearn.neighbors import RadiusNeighborsClassifier

# Use all neighbors within a fixed radius
rnn = RadiusNeighborsClassifier(
    radius=1.0,          # Search radius
    weights='distance',  # Weight by distance
    outlier_label='most_frequent'  # How to handle points with no neighbors
)

rnn.fit(X_train_scaled, y_train)
y_pred_radius = rnn.predict(X_test_scaled)

print(f"Radius NN Accuracy: {accuracy_score(y_test, y_pred_radius):.4f}")
```

### Handling Imbalanced Data

```python
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

# SMOTE oversampling + KNN
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
])

pipeline.fit(X_train_scaled, y_train)
y_pred_smote = pipeline.predict(X_test_scaled)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_smote))
```

## References

- [Scikit-learn KNN Documentation](https://scikit-learn.org/stable/modules/neighbors.html)
- [KNN Algorithm Explanation](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- The Elements of Statistical Learning (Hastie, Tibshirani, Friedman)
- Pattern Recognition and Machine Learning (Christopher Bishop)
- Introduction to Statistical Learning (James, Witten, Hastie, Tibshirani)
- [Curse of Dimensionality in KNN](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html)
