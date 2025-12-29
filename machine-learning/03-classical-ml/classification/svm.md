# Support Vector Machines (SVM)

## Table of Contents

1. [Introduction](#introduction)
2. [Linear SVM](#linear-svm)
3. [Kernel Trick](#kernel-trick)
4. [SVM for Classification](#svm-for-classification)
5. [SVM for Regression](#svm-for-regression)
6. [Optimization and Training](#optimization-and-training)
7. [Practical Considerations](#practical-considerations)
8. [Implementation](#implementation)
9. [References](#references)

## Introduction

**Support Vector Machine (SVM)** is a powerful supervised learning algorithm that finds the optimal hyperplane to separate classes in feature space. It's particularly effective for high-dimensional data and non-linear classification problems.

### Key Characteristics

- **Maximum margin classifier** - maximizes distance between classes
- **Kernel methods** - handles non-linear decision boundaries
- **Robust to overfitting** - especially in high-dimensional spaces
- **Memory efficient** - only uses support vectors for prediction
- **Versatile** - works for classification, regression, and outlier detection

### When to Use

**Best suited for:**
- High-dimensional datasets (text classification, genomics)
- Clear margin of separation between classes
- Small to medium-sized datasets
- Non-linear classification with kernel trick

**Not recommended for:**
- Very large datasets (slow training time)
- Data with lots of noise (use probabilistic models)
- When probability estimates are critical
- Multi-class problems with many classes

## Linear SVM

### Maximum Margin Classifier

The goal is to find a hyperplane that maximally separates two classes:

$$w^T x + b = 0$$

where:
- $w$ is the normal vector to the hyperplane
- $b$ is the bias term
- $x$ is a data point

**Margin** - the distance from the hyperplane to the nearest data point from either class.

**Support Vectors** - the data points that lie on the margin boundaries. They are the critical elements that define the hyperplane.

### Hard Margin SVM

For linearly separable data, find $w$ and $b$ that maximize the margin:

$$\max_{w,b} \frac{2}{||w||}$$

subject to:

$$y_i(w^T x_i + b) \geq 1, \quad \forall i$$

where $y_i \in \{-1, +1\}$ is the class label.

Equivalent to minimizing:

$$\min_{w,b} \frac{1}{2}||w||^2$$

**Geometric margin** for point $x_i$:

$$\gamma_i = y_i\frac{w^T x_i + b}{||w||}$$

The margin is $\frac{1}{||w||}$, so minimizing $||w||^2$ maximizes the margin.

### Soft Margin SVM

For non-separable data, introduce **slack variables** $\xi_i$ to allow some misclassification:

$$\min_{w,b,\xi} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n}\xi_i$$

subject to:

$$y_i(w^T x_i + b) \geq 1 - \xi_i$$
$$\xi_i \geq 0, \quad \forall i$$

where:
- $\xi_i$ is the distance from the margin for misclassified points
- $C$ is the regularization parameter (penalty for violations)

**C parameter** controls the trade-off:
- **Large C**: hard margin, less tolerance for errors (may overfit)
- **Small C**: soft margin, more tolerance for errors (may underfit)

### Hinge Loss

Soft margin SVM can be rewritten using **hinge loss**:

$$L_{hinge}(y, \hat{y}) = \max(0, 1 - y\hat{y})$$

Complete objective:

$$\min_{w,b} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n}\max(0, 1 - y_i(w^T x_i + b))$$

Properties:
- Loss is 0 if prediction is correct with margin ≥ 1
- Linear penalty for violations
- Not differentiable at $y\hat{y} = 1$

## Kernel Trick

### Non-Linear Classification

For non-linearly separable data, map to higher-dimensional space where data becomes linearly separable:

$$\phi: \mathbb{R}^d \rightarrow \mathbb{R}^D, \quad D >> d$$

Decision function in feature space:

$$f(x) = w^T\phi(x) + b$$

### Kernel Function

**Kernel trick** - compute inner products in high-dimensional space without explicit mapping:

$$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

The decision function becomes:

$$f(x) = \sum_{i=1}^{n}\alpha_i y_i K(x_i, x) + b$$

where $\alpha_i$ are Lagrange multipliers (non-zero only for support vectors).

### Common Kernels

**Linear Kernel:**

$$K(x_i, x_j) = x_i^T x_j$$

Use when: data is linearly separable, high-dimensional data.

**Polynomial Kernel:**

$$K(x_i, x_j) = (\gamma x_i^T x_j + r)^d$$

Parameters:
- $d$ - polynomial degree
- $\gamma$ - kernel coefficient
- $r$ - independent term

Use when: features interact in polynomial fashion.

**Radial Basis Function (RBF) / Gaussian Kernel:**

$$K(x_i, x_j) = \exp\left(-\gamma ||x_i - x_j||^2\right)$$

where $\gamma = \frac{1}{2\sigma^2}$

Properties:
- Maps to infinite-dimensional space
- $K(x, x) = 1$
- Decreases as points get farther apart

Use when: no prior knowledge about data, general-purpose kernel.

**Sigmoid Kernel:**

$$K(x_i, x_j) = \tanh(\gamma x_i^T x_j + r)$$

Use when: mimicking neural networks (rarely used in practice).

### Kernel Parameters

**RBF kernel parameter $\gamma$:**
- **Large $\gamma$**: small radius of influence → high variance, may overfit
- **Small $\gamma$**: large radius of influence → high bias, may underfit

**Polynomial degree $d$:**
- Higher degree allows more complex boundaries
- Computational cost increases with degree
- More prone to overfitting with high degrees

## SVM for Classification

### Binary Classification

Decision function:

$$f(x) = \text{sign}\left(\sum_{i \in SV}\alpha_i y_i K(x_i, x) + b\right)$$

where $SV$ is the set of support vectors.

Prediction:
- $f(x) > 0$: class +1
- $f(x) < 0$: class -1
- $f(x) = 0$: on the decision boundary

**Distance from hyperplane** (unnormalized):

$$d(x) = \sum_{i \in SV}\alpha_i y_i K(x_i, x) + b$$

Larger absolute value indicates higher confidence.

### Multiclass Classification

SVM is inherently binary. Two main strategies for multiclass:

**One-vs-Rest (OvR):**
- Train $K$ binary classifiers (one per class)
- Classifier $k$: class $k$ vs all others
- Prediction: class with largest decision function value

**One-vs-One (OvO):**
- Train $\frac{K(K-1)}{2}$ binary classifiers
- One classifier for each pair of classes
- Prediction: majority voting

Scikit-learn uses OvO by default for SVC.

### Probability Estimates

SVM doesn't naturally output probabilities. Platt scaling fits a logistic regression on SVM outputs:

$$P(y=1|x) = \frac{1}{1 + e^{Af(x) + B}}$$

where $A$ and $B$ are fitted using cross-validation.

Enable with `probability=True` in scikit-learn (increases training time).

## SVM for Regression

### Support Vector Regression (SVR)

Instead of finding a hyperplane that separates classes, SVR finds a function that deviates from targets by at most $\epsilon$ (epsilon-tube).

**Objective:**

$$\min_{w,b,\xi,\xi^*} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n}(\xi_i + \xi_i^*)$$

subject to:

$$y_i - (w^T x_i + b) \leq \epsilon + \xi_i$$
$$(w^T x_i + b) - y_i \leq \epsilon + \xi_i^*$$
$$\xi_i, \xi_i^* \geq 0$$

**Epsilon-insensitive loss:**

$$L_\epsilon(y, \hat{y}) = \max(0, |y - \hat{y}| - \epsilon)$$

- Errors within $\epsilon$-tube have zero loss
- Errors outside are penalized linearly

Parameters:
- $\epsilon$ - width of epsilon-tube
- $C$ - penalty for errors outside tube

## Optimization and Training

### Dual Formulation

The primal optimization problem can be converted to its dual form using Lagrange multipliers:

$$\max_{\alpha} \sum_{i=1}^{n}\alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i\alpha_j y_i y_j K(x_i, x_j)$$

subject to:

$$0 \leq \alpha_i \leq C, \quad \sum_{i=1}^{n}\alpha_i y_i = 0$$

**Karush-Kuhn-Tucker (KKT) conditions:**
- $\alpha_i = 0$ → point is correctly classified beyond margin
- $0 < \alpha_i < C$ → point is on the margin (support vector)
- $\alpha_i = C$ → point violates margin (support vector)

Only support vectors (where $\alpha_i > 0$) contribute to the decision function.

### Sequential Minimal Optimization (SMO)

Standard algorithm for training SVMs:

1. Select two Lagrange multipliers $\alpha_i$ and $\alpha_j$
2. Optimize them while keeping others fixed
3. Repeat until convergence

Advantages:
- Doesn't require large matrix operations
- Fast for most problems
- Memory efficient

### Computational Complexity

**Training:**
- Best case: $O(n^2)$
- Worst case: $O(n^3)$
- $n$ is the number of training samples

**Prediction:**
- $O(n_{sv} \cdot d)$ where $n_{sv}$ is number of support vectors and $d$ is feature dimension
- For kernel SVM: $O(n_{sv})$ kernel evaluations

## Practical Considerations

### Feature Scaling

**Critical for SVM** - features on different scales can dominate the kernel function.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Hyperparameter Tuning

**Key parameters:**

1. **C (regularization):**
   - Controls trade-off between margin maximization and training error
   - Try: `[0.1, 1, 10, 100]`

2. **gamma (for RBF kernel):**
   - Controls influence radius of support vectors
   - Try: `[0.001, 0.01, 0.1, 1]` or `['scale', 'auto']`

3. **kernel:**
   - Try: `['linear', 'rbf', 'poly']`

**Grid search strategy:**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_scaled, y_train)
```

### Handling Imbalanced Data

**Class weights:**

```python
# Automatic weight balancing
svm = SVC(class_weight='balanced')

# Manual weights
svm = SVC(class_weight={0: 1, 1: 10})  # Penalize class 1 errors 10x more
```

### Choosing the Right Kernel

**Decision guide:**

1. **Start with linear kernel** if:
   - High-dimensional data ($d > n$)
   - Text classification
   - Sparse features

2. **Try RBF kernel** if:
   - General-purpose, no prior knowledge
   - Non-linear patterns expected
   - Medium-sized dataset

3. **Polynomial kernel** if:
   - Specific domain knowledge suggests polynomial relationships
   - Image processing (low-degree polynomials)

4. **Custom kernel** if:
   - Domain-specific similarity measure
   - Structured data (graphs, strings)

### Comparison with Other Algorithms

| Aspect | SVM | Logistic Regression | Decision Trees |
|--------|-----|---------------------|----------------|
| Decision boundary | Linear/Non-linear | Linear | Non-linear |
| High dimensions | Excellent | Good | Poor |
| Training time | Slow ($O(n^2)$-$O(n^3)$) | Fast ($O(n)$) | Fast ($O(n\log n)$) |
| Interpretability | Low | High | High |
| Probability output | Not native | Native | Native |
| Overfitting | Resistant | Moderate | Prone |

## Implementation

### Binary Classification with Linear SVM

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (critical!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train linear SVM
svm_linear = SVC(
    kernel='linear',
    C=1.0,
    random_state=42
)

svm_linear.fit(X_train_scaled, y_train)

# Predictions
y_pred = svm_linear.predict(X_test_scaled)

# Evaluation
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Support vectors
print(f"Number of support vectors: {svm_linear.n_support_}")
print(f"Support vector indices: {svm_linear.support_}")
```

### RBF Kernel SVM

```python
# Train RBF SVM
svm_rbf = SVC(
    kernel='rbf',
    C=10.0,
    gamma='scale',  # 1 / (n_features * X.var())
    random_state=42
)

svm_rbf.fit(X_train_scaled, y_train)
y_pred_rbf = svm_rbf.predict(X_test_scaled)

# Get decision function values (distance from hyperplane)
decision_values = svm_rbf.decision_function(X_test_scaled)
print(f"Decision function range: [{decision_values.min():.2f}, {decision_values.max():.2f}]")
```

### Probability Estimates

```python
# Enable probability estimates
svm_proba = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,  # Enables Platt scaling
    random_state=42
)

svm_proba.fit(X_train_scaled, y_train)

# Get probabilities
y_proba = svm_proba.predict_proba(X_test_scaled)
print(f"Probability for first sample: {y_proba[0]}")
```

### Grid Search for Hyperparameters

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear', 'poly']
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Use best model
best_svm = grid_search.best_estimator_
y_pred_best = best_svm.predict(X_test_scaled)
```

### Multiclass Classification

```python
# SVM automatically handles multiclass
svm_multiclass = SVC(
    kernel='rbf',
    decision_function_shape='ovr',  # 'ovr' or 'ovo'
    random_state=42
)

svm_multiclass.fit(X_train_scaled, y_train)

# Number of classifiers
print(f"Number of classes: {len(svm_multiclass.classes_)}")
print(f"Classes: {svm_multiclass.classes_}")
```

### Support Vector Regression

```python
from sklearn.svm import SVR

# Train SVR
svr = SVR(
    kernel='rbf',
    C=100,
    gamma='scale',
    epsilon=0.1  # Width of epsilon-tube
)

svr.fit(X_train_scaled, y_train)
y_pred_svr = svr.predict(X_test_scaled)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
print(f"MSE: {mean_squared_error(y_test, y_pred_svr):.4f}")
print(f"R²: {r2_score(y_test, y_pred_svr):.4f}")
```

### LinearSVC for Large Datasets

```python
from sklearn.svm import LinearSVC

# Faster alternative for linear kernel
linear_svc = LinearSVC(
    C=1.0,
    max_iter=10000,
    random_state=42
)

linear_svc.fit(X_train_scaled, y_train)
y_pred_linear = linear_svc.predict(X_test_scaled)

# Note: LinearSVC uses different library (liblinear) than SVC
# Results may differ slightly from SVC with linear kernel
```

### Custom Kernel

```python
import numpy as np

# Define custom kernel (example: exponential kernel)
def exponential_kernel(X, Y, sigma=1.0):
    """Exponential kernel: K(x,y) = exp(-||x-y|| / (2*sigma^2))"""
    from sklearn.metrics.pairwise import euclidean_distances
    dists = euclidean_distances(X, Y)
    return np.exp(-dists / (2 * sigma**2))

# Use custom kernel
svm_custom = SVC(kernel=exponential_kernel)
# Note: precomputed kernel matrix required
```

## References

- [Scikit-learn SVM Documentation](https://scikit-learn.org/stable/modules/svm.html)
- A Tutorial on Support Vector Machines for Pattern Recognition (Christopher Burges)
- The Elements of Statistical Learning (Hastie, Tibshirani, Friedman)
- Pattern Recognition and Machine Learning (Christopher Bishop)
- [SMO Algorithm Paper](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/) (John Platt, 1998)
