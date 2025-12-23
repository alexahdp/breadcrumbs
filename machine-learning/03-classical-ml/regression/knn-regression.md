# K-Nearest Neighbors Regression

## Table of Contents

1. [Introduction](#introduction)
2. [Algorithm Fundamentals](#algorithm-fundamentals)
3. [Distance Metrics](#distance-metrics)
4. [Choosing K](#choosing-k)
5. [Weighted KNN](#weighted-knn)
6. [Practical Considerations](#practical-considerations)

## Introduction

**K-Nearest Neighbors Regression (KNN Regression)** - a non-parametric, instance-based learning algorithm that predicts target values based on the average of the K nearest training examples.

**Key characteristics:**
- **Non-parametric:** Makes no assumptions about data distribution
- **Lazy learning:** No training phase (stores all training data)
- **Instance-based:** Makes predictions using stored training instances
- **Non-linear:** Can capture complex, non-linear relationships

**Use cases:**
- Non-linear regression problems
- Small to medium-sized datasets
- When interpretability of individual predictions matters
- Recommendation systems
- Pattern recognition

## Algorithm Fundamentals

### How KNN Regression Works

**Prediction process:**

1. **Given:** New point $\mathbf{x}_{query}$ to predict
2. **Find:** K nearest neighbors from training set based on distance metric
3. **Compute:** Average of target values from K neighbors
4. **Return:** Averaged value as prediction

### Mathematical Formulation

**Prediction formula:**

$$\hat{y}_{query} = \frac{1}{K}\sum_{i=1}^{K}y_i$$

where $y_i$ are the target values of the K nearest neighbors.

**Alternative notation:**

$$\hat{y}(\mathbf{x}) = \frac{1}{K}\sum_{\mathbf{x}_i \in \mathcal{N}_K(\mathbf{x})}y_i$$

where $\mathcal{N}_K(\mathbf{x})$ denotes the set of K nearest neighbors of $\mathbf{x}$.

### No Training Phase

Unlike parametric models (linear regression, neural networks), KNN has no training phase:

**Training:** Simply store all training examples $(\mathbf{X}_{train}, \mathbf{y}_{train})$

**Prediction:** Search through stored examples at query time

**Implication:**
- Fast "training" (instant)
- Slow prediction (must compute distances to all training points)

### Non-Parametric Nature

**Parametric models** (e.g., linear regression):
- Learn fixed set of parameters
- Assume functional form
- Discard training data after learning

**Non-parametric models** (e.g., KNN):
- Keep all training data
- No assumptions about underlying function
- Complexity grows with data size

## Distance Metrics

The choice of distance metric is crucial for KNN performance.

### Euclidean Distance (L2)

**Most common metric** - straight-line distance in feature space:

$$d(\mathbf{x}, \mathbf{x}') = \sqrt{\sum_{j=1}^{p}(x_j - x_j')^2} = ||\mathbf{x} - \mathbf{x}'||_2$$

**Properties:**
- Intuitive geometric interpretation
- Sensitive to feature scales
- Works well for continuous features

**When to use:**
- Features on similar scales (after normalization)
- Low-dimensional spaces
- Continuous numerical data

### Manhattan Distance (L1)

**City-block distance** - sum of absolute differences:

$$d(\mathbf{x}, \mathbf{x}') = \sum_{j=1}^{p}|x_j - x_j'| = ||\mathbf{x} - \mathbf{x}'||_1$$

**Properties:**
- Less sensitive to outliers than Euclidean
- Useful in high-dimensional spaces
- Computationally efficient

**When to use:**
- High-dimensional data
- Presence of outliers
- Grid-like feature spaces

### Minkowski Distance

**Generalization** of Euclidean and Manhattan distances:

$$d(\mathbf{x}, \mathbf{x}') = \left(\sum_{j=1}^{p}|x_j - x_j'|^q\right)^{1/q}$$

where:
- $q = 1$: Manhattan distance
- $q = 2$: Euclidean distance
- $q \to \infty$: Chebyshev distance (maximum coordinate difference)

### Cosine Distance

**Angle-based metric** - measures angular similarity:

$$d(\mathbf{x}, \mathbf{x}') = 1 - \frac{\mathbf{x} \cdot \mathbf{x}'}{||\mathbf{x}|| \cdot ||\mathbf{x}'||}$$

**When to use:**
- Text data (TF-IDF vectors)
- Direction matters more than magnitude
- High-dimensional sparse data

### Mahalanobis Distance

**Accounts for correlation** between features:

$$d(\mathbf{x}, \mathbf{x}') = \sqrt{(\mathbf{x} - \mathbf{x}')^T \mathbf{S}^{-1} (\mathbf{x} - \mathbf{x}')}$$

where $\mathbf{S}$ is the covariance matrix.

**When to use:**
- Features are correlated
- Different features have different variances
- Need to account for data distribution

**Note:** Computationally expensive for large datasets.

### Custom Distance Functions

KNN can use domain-specific distance metrics:
- **Hamming distance:** For categorical features
- **Edit distance:** For strings
- **Dynamic Time Warping:** For time series

## Choosing K

The hyperparameter K critically affects model performance and behavior.

### Effect of K Value

**K = 1 (Nearest Neighbor):**
- Uses only the closest training point
- High variance, low bias
- Overfitting risk (learns noise)
- Decision boundary: highly irregular, complex

**K small (e.g., 3-5):**
- More sensitive to noise
- Complex decision boundaries
- Lower bias, higher variance
- Better for capturing local patterns

**K large (e.g., 50-100):**
- Smoother predictions
- Simpler decision boundaries
- Higher bias, lower variance
- More robust to outliers
- Risk of underfitting

**K = N (all training points):**
- Predicts mean of all training targets
- Maximum bias, minimum variance
- Underfitting

### Mathematical Perspective

**Bias-Variance Tradeoff:**

$$\text{Expected MSE} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

As K increases:
- **Bias increases:** Predictions become smoother, miss local patterns
- **Variance decreases:** Predictions more stable across different training sets

### Selecting Optimal K

**1. Cross-Validation**

Most reliable method:

```python
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

k_values = range(1, 51)
cv_scores = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5,
                            scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())

optimal_k = k_values[np.argmin(cv_scores)]
print(f"Optimal K: {optimal_k}")
```

**2. Rule of Thumb**

Quick approximation: $K \approx \sqrt{N}$ where N is the number of training samples.

**Note:** This is a starting point, not a substitute for cross-validation.

**3. Grid Search**

Systematic search over K values:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': range(1, 51)}
grid = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5,
                   scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

print(f"Best K: {grid.best_params_['n_neighbors']}")
```

### Odd vs Even K

**Recommendation:** Use odd K values when possible

**Reason:** For classification, odd K avoids ties. For regression, less critical but still preferred by convention.

## Weighted KNN

**Problem:** Standard KNN treats all K neighbors equally, regardless of their distance to query point.

**Solution:** Weight neighbors by their distance - closer neighbors have more influence.

### Weight Functions

**1. Uniform Weights (Standard KNN)**

$$\hat{y}_{query} = \frac{1}{K}\sum_{i=1}^{K}y_i$$

All neighbors contribute equally.

**2. Distance-Based Weights**

$$\hat{y}_{query} = \frac{\sum_{i=1}^{K}w_i \cdot y_i}{\sum_{i=1}^{K}w_i}$$

**Common weighting schemes:**

**Inverse distance:**

$$w_i = \frac{1}{d(\mathbf{x}_{query}, \mathbf{x}_i)}$$

**Inverse squared distance:**

$$w_i = \frac{1}{d(\mathbf{x}_{query}, \mathbf{x}_i)^2}$$

**Gaussian kernel:**

$$w_i = \exp\left(-\frac{d(\mathbf{x}_{query}, \mathbf{x}_i)^2}{2\sigma^2}\right)$$

**Handling zero distance:** When $d = 0$ (query point in training set), set $\hat{y} = y_{exact}$ directly.

### Advantages of Weighted KNN

- **More accurate:** Closer neighbors more influential
- **Smoother predictions:** Gradual transition between regions
- **Handles varying density:** Works better when data density varies
- **Less sensitive to K:** Optimal K range broader

### Implementation

```python
from sklearn.neighbors import KNeighborsRegressor

# Uniform weights (standard KNN)
knn_uniform = KNeighborsRegressor(n_neighbors=5, weights='uniform')

# Distance-based weights
knn_weighted = KNeighborsRegressor(n_neighbors=5, weights='distance')

# Custom weight function
def custom_weights(distances):
    return np.exp(-distances**2 / (2 * 0.5**2))  # Gaussian with σ=0.5

knn_custom = KNeighborsRegressor(n_neighbors=5, weights=custom_weights)
```

## Practical Considerations

### Feature Scaling

**Critical requirement:** Always scale features before using KNN!

**Why?** Distance metrics are sensitive to feature scales.

**Example:**
- Feature 1: Age [0-100]
- Feature 2: Income [0-1,000,000]

Without scaling, income dominates distance calculations.

**Scaling methods:**

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization (zero mean, unit variance)
scaler_std = StandardScaler()
X_scaled = scaler_std.fit_transform(X)

# Min-Max normalization [0, 1]
scaler_minmax = MinMaxScaler()
X_scaled = scaler_minmax.fit_transform(X)
```

### Curse of Dimensionality

**Problem:** KNN performance degrades in high-dimensional spaces.

**Reasons:**
- All points become approximately equidistant
- "Nearest" neighbors no longer meaningfully close
- Volume of space increases exponentially

**Typical threshold:** Problems appear when $p > 10$ features (depends on sample size).

**Solutions:**
1. **Dimensionality reduction:** PCA, t-SNE, feature selection
2. **Feature engineering:** Create more informative features
3. **Use different algorithm:** Tree-based models, neural networks

### Computational Complexity

**Training:** $O(1)$ - just store data

**Prediction for one point:** $O(Np)$ where:
- $N$ = number of training samples
- $p$ = number of features

**Total prediction:** $O(NMp)$ for $M$ test points

**Optimization strategies:**

**1. KD-Trees / Ball Trees**

Spatial data structures for faster neighbor search:

```python
knn = KNeighborsRegressor(n_neighbors=5, algorithm='kd_tree')  # or 'ball_tree'
```

- **KD-Tree:** Best for low dimensions ($p < 20$)
- **Ball-Tree:** Better for higher dimensions
- **Brute force:** Simple distance calculation (best for small N or high p)

**2. Approximate Nearest Neighbors**

Trade accuracy for speed (LSH, ANNOY, FAISS).

### Handling Outliers

**Issue:** KNN sensitive to outliers in both features and targets.

**Solutions:**

**1. Outlier removal:**
```python
from scipy import stats
import numpy as np

# Remove outliers using Z-score
mask = (np.abs(stats.zscore(X)) < 3).all(axis=1)
X_clean = X[mask]
y_clean = y[mask]
```

**2. Robust distance metrics:**
- Use Manhattan instead of Euclidean
- Median instead of mean for aggregation

**3. Trim extreme neighbors:**
```python
# Custom KNN that excludes extreme target values among neighbors
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=10)
nn.fit(X_train)
distances, indices = nn.kneighbors(X_test)

# Use middle 5 neighbors (exclude extremes)
y_pred = np.median(y_train[indices[:, 2:7]], axis=1)
```

### When to Use KNN Regression

**Advantages:**
- No assumptions about data distribution
- Naturally handles non-linear relationships
- Simple to understand and implement
- No training required
- Can update with new data easily (add to training set)

**Disadvantages:**
- Computationally expensive at prediction time
- Requires large memory (stores all training data)
- Sensitive to feature scales
- Poor performance in high dimensions
- Sensitive to irrelevant features

**Good fit when:**
- Small to medium dataset (N < 100,000)
- Low to moderate dimensions (p < 20)
- Non-linear relationships
- Local patterns important
- Need simple baseline model

**Poor fit when:**
- Large dataset (slow predictions)
- High-dimensional data
- Real-time predictions required
- Memory constrained
- Linear relationships (use linear regression instead)

### Complete Implementation Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Generate non-linear data
np.random.seed(42)
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(200) * 0.1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Find optimal K using cross-validation
param_grid = {
    'n_neighbors': range(1, 31),
    'weights': ['uniform', 'distance']
}

grid = GridSearchCV(
    KNeighborsRegressor(),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid.best_params_}")

# Train final model
knn = grid.best_estimator_
y_pred = knn.predict(X_test_scaled)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# Visualize
plt.figure(figsize=(12, 5))

# Predictions
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.5, label='Train')
plt.scatter(X_test, y_test, alpha=0.5, label='Test')

X_line = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
X_line_scaled = scaler.transform(X_line)
y_line = knn.predict(X_line_scaled)

plt.plot(X_line, y_line, 'r-', linewidth=2, label=f'KNN (K={grid.best_params_["n_neighbors"]})')
plt.legend()
plt.title('KNN Regression Predictions')

# K vs CV Score
plt.subplot(1, 2, 2)
results = grid.cv_results_
for weights in ['uniform', 'distance']:
    mask = [params['weights'] == weights for params in results['params']]
    k_values = [params['n_neighbors'] for params, m in zip(results['params'], mask) if m]
    scores = [-score for score, m in zip(results['mean_test_score'], mask) if m]
    plt.plot(k_values, scores, marker='o', label=f'weights={weights}')

plt.xlabel('K (number of neighbors)')
plt.ylabel('Mean Squared Error')
plt.title('Cross-Validation Scores')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Comparison with Parametric Methods

```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

models = {
    'Linear Regression': LinearRegression(),
    'KNN (K=5)': KNeighborsRegressor(n_neighbors=5, weights='distance'),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
```

## References

- An Introduction to Statistical Learning (James, Witten, Hastie, Tibshirani)
- Pattern Recognition and Machine Learning (Christopher Bishop)
- Scikit-learn Documentation: [K-Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) (Hastie, Tibshirani, Friedman)
