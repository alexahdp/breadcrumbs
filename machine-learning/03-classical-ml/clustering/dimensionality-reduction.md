# Dimensionality Reduction

## Table of Contents

1. [Introduction](#introduction)
2. [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
3. [t-SNE](#t-sne)
4. [UMAP](#umap)
5. [Linear Discriminant Analysis (LDA)](#linear-discriminant-analysis-lda)
6. [Other Methods](#other-methods)
7. [Choosing a Method](#choosing-a-method)
8. [Practical Considerations](#practical-considerations)
9. [Implementation](#implementation)
10. [References](#references)

## Introduction

**Dimensionality Reduction** is the process of reducing the number of features (dimensions) in a dataset while preserving as much information as possible. It's essential for visualization, computational efficiency, and addressing the curse of dimensionality.

### Why Dimensionality Reduction?

**1. Curse of Dimensionality**
- Distance metrics become less meaningful in high dimensions
- Data becomes increasingly sparse
- Models require exponentially more data
- Overfitting risk increases

**2. Visualization**
- Humans can visualize 2D and 3D data
- Understand patterns and clusters
- Exploratory data analysis

**3. Computational Efficiency**
- Faster training and inference
- Reduced memory requirements
- Lower computational cost

**4. Noise Reduction**
- Remove redundant or irrelevant features
- Improve model performance
- Better generalization

**5. Feature Engineering**
- Create meaningful combinations of features
- Capture underlying structure
- Improve interpretability

### Types of Dimensionality Reduction

**Linear Methods:**
- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)
- Factor Analysis
- Independent Component Analysis (ICA)

**Non-linear Methods:**
- t-SNE (t-distributed Stochastic Neighbor Embedding)
- UMAP (Uniform Manifold Approximation and Projection)
- Isomap
- Locally Linear Embedding (LLE)
- Autoencoders

## Principal Component Analysis (PCA)

### Overview

**PCA** finds orthogonal axes (principal components) that capture maximum variance in the data. It's a linear transformation that projects data onto a lower-dimensional space.

**Key idea:** Transform correlated features into uncorrelated principal components, ordered by explained variance.

### Mathematical Foundation

**Objective:** Find directions (principal components) that maximize variance.

Given data matrix $X \in \mathbb{R}^{n \times d}$ (n samples, d features):

1. **Center the data:**
   $$X_c = X - \bar{X}$$
   where $\bar{X}$ is the mean of each feature.

2. **Compute covariance matrix:**
   $$C = \frac{1}{n-1}X_c^T X_c$$

3. **Eigenvalue decomposition:**
   $$C = V \Lambda V^T$$
   where:
   - $V$ = eigenvectors (principal components)
   - $\Lambda$ = diagonal matrix of eigenvalues (variances)

4. **Select top k components:**
   Keep eigenvectors with largest eigenvalues.

5. **Transform data:**
   $$X_{reduced} = X_c \cdot V_k$$
   where $V_k$ contains first $k$ eigenvectors.

### Principal Components

**First principal component (PC1):**
$$PC_1 = \arg\max_{||w||=1} Var(Xw)$$

Direction of maximum variance.

**Second principal component (PC2):**
$$PC_2 = \arg\max_{\substack{||w||=1 \\ w \perp PC_1}} Var(Xw)$$

Direction of maximum variance orthogonal to PC1.

**General k-th component:** Maximum variance orthogonal to all previous components.

### Explained Variance

**Variance explained by component i:**
$$\frac{\lambda_i}{\sum_{j=1}^{d}\lambda_j}$$

**Cumulative explained variance:**
$$\frac{\sum_{i=1}^{k}\lambda_i}{\sum_{j=1}^{d}\lambda_j}$$

Common threshold: Keep components that explain 95% of variance.

### Properties

**Advantages:**
- **Fast and scalable** - $O(d^2n + d^3)$ complexity
- **Interpretable** - components show feature importance
- **Removes correlation** - principal components are orthogonal
- **Optimal for reconstruction** - minimizes reconstruction error
- **Deterministic** - same result every run

**Limitations:**
- **Linear only** - cannot capture non-linear relationships
- **Assumes high variance = high importance** - may not be true
- **Sensitive to scaling** - requires feature normalization
- **Hard to interpret** - components are linear combinations
- **Outlier sensitive** - variance-based method

### When to Use PCA

**Best for:**
- Linear relationships in data
- Need for fast computation
- When variance captures information
- Feature decorrelation
- Preprocessing for other algorithms

**Not recommended for:**
- Non-linear manifolds (use t-SNE, UMAP)
- Visualization of complex structures
- Categorical data
- When interpretability is critical

## t-SNE

### Overview

**t-SNE (t-distributed Stochastic Neighbor Embedding)** is a non-linear technique primarily used for visualization. It excels at preserving local structure and revealing clusters.

**Key idea:** Convert high-dimensional Euclidean distances into conditional probabilities representing similarities, then find low-dimensional embedding that preserves these probabilities.

### Mathematical Foundation

**High-dimensional similarities:**

Probability that point $j$ is neighbor of point $i$:

$$p_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i}\exp(-||x_i - x_k||^2 / 2\sigma_i^2)}$$

Symmetrized joint probability:

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

**Low-dimensional similarities:**

Using Student t-distribution (heavy tails):

$$q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l}(1 + ||y_k - y_l||^2)^{-1}}$$

**Objective:**

Minimize Kullback-Leibler divergence between $P$ and $Q$:

$$KL(P||Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

### Perplexity Parameter

**Perplexity** controls the effective number of neighbors:

$$Perplexity = 2^{H(P_i)}$$

where $H(P_i) = -\sum_j p_{j|i} \log_2 p_{j|i}$ is entropy.

**Effect:**
- **Low perplexity (5-20)**: Focus on very local structure
- **Medium perplexity (30-50)**: Balanced view (default: 30)
- **High perplexity (>50)**: Focus on global structure

**Rule of thumb:**
$$5 \leq \text{perplexity} \leq \frac{n}{2}$$

Typical: 30-50 for most datasets.

### Properties

**Advantages:**
- **Excellent visualization** - reveals clusters beautifully
- **Preserves local structure** - neighbors stay together
- **Non-linear** - captures complex relationships
- **Flexible** - works with various distance metrics

**Limitations:**
- **Slow** - $O(n^2)$ complexity, impractical for >10k points
- **Stochastic** - different results each run
- **No out-of-sample** - can't transform new points
- **Distances not meaningful** - cluster sizes/distances don't reflect reality
- **Global structure not preserved** - relative positions of clusters unreliable
- **Computationally expensive** - requires many iterations

### When to Use t-SNE

**Best for:**
- Visualizing high-dimensional data
- Discovering clusters
- Exploratory analysis
- Small to medium datasets (<10k points)
- When local structure matters

**Not recommended for:**
- Large datasets (use UMAP instead)
- Feature extraction for other algorithms
- When global structure matters
- Interpreting distances between clusters
- Real-time applications

## UMAP

### Overview

**UMAP (Uniform Manifold Approximation and Projection)** is a modern non-linear technique that preserves both local and global structure better than t-SNE, while being much faster.

**Key idea:** Build topological representation of data using manifold learning, then optimize low-dimensional layout.

### Mathematical Foundation

Based on Riemannian geometry and topological data analysis:

1. **Build fuzzy topological representation** of high-dimensional data
2. **Optimize low-dimensional embedding** to have similar topological structure

**High-dimensional graph:**

$$w(x_i, x_j) = \exp\left(-\frac{\max(0, ||x_i - x_j|| - \rho_i)}{\sigma_i}\right)$$

where:
- $\rho_i$ = distance to nearest neighbor
- $\sigma_i$ = normalization factor

**Optimization:**

Uses cross-entropy between high and low-dimensional fuzzy sets:

$$CE = \sum_{i,j} w_{ij} \log\frac{w_{ij}}{v_{ij}} + (1-w_{ij})\log\frac{1-w_{ij}}{1-v_{ij}}$$

### Key Parameters

**n_neighbors (default: 15)**
- Controls local vs global structure balance
- Low values (5-10): Focus on local structure
- High values (50-100): Preserve more global structure

**min_dist (default: 0.1)**
- Minimum distance between points in embedding
- Low values (0.0-0.1): Tight clusters
- High values (0.5-1.0): Looser, more spread out

**n_components (default: 2)**
- Number of dimensions for embedding
- Typically 2 or 3 for visualization

### Properties

**Advantages:**
- **Fast** - much faster than t-SNE, scales to millions of points
- **Preserves local and global structure** - better than t-SNE
- **Theoretical foundation** - based on manifold learning
- **Out-of-sample projection** - can transform new points
- **Flexible** - many distance metrics supported
- **Stable** - more consistent results across runs

**Limitations:**
- **Stochastic** - still has randomness
- **Hyperparameter sensitive** - results vary with parameters
- **Black box** - harder to interpret than PCA
- **Not deterministic** - initialization affects results

### When to Use UMAP

**Best for:**
- Large-scale visualization (>10k points)
- Preserving both local and global structure
- Feature extraction for downstream tasks
- When speed matters
- Out-of-sample projection needed

**Not recommended for:**
- When deterministic results required
- When full interpretability needed
- Very small datasets (<100 points)

## Linear Discriminant Analysis (LDA)

### Overview

**LDA** is a supervised dimensionality reduction technique that finds directions maximizing class separation. Unlike PCA (unsupervised), LDA uses class labels.

**Key idea:** Project data to maximize between-class variance while minimizing within-class variance.

### Mathematical Foundation

**Within-class scatter matrix:**

$$S_W = \sum_{i=1}^{C} \sum_{x \in C_i} (x - \mu_i)(x - \mu_i)^T$$

**Between-class scatter matrix:**

$$S_B = \sum_{i=1}^{C} n_i (\mu_i - \mu)(\mu_i - \mu)^T$$

where:
- $C$ = number of classes
- $\mu_i$ = mean of class $i$
- $\mu$ = overall mean
- $n_i$ = number of samples in class $i$

**Objective:**

Maximize Fisher criterion:

$$J(w) = \frac{w^T S_B w}{w^T S_W w}$$

**Solution:**

Solve generalized eigenvalue problem:

$$S_B w = \lambda S_W w$$

Select eigenvectors with largest eigenvalues.

**Maximum components:** $C - 1$ (one less than number of classes)

### Properties

**Advantages:**
- **Supervised** - uses class information
- **Maximizes class separation** - good for classification
- **Interpretable** - shows discriminative features
- **Fewer components** - typically need fewer than PCA

**Limitations:**
- **Requires labels** - supervised method
- **Linear only** - assumes linear decision boundaries
- **Limited components** - max $C-1$ components
- **Assumes Gaussian distributions** - may not hold
- **Sensitive to outliers** - uses mean and covariance

### When to Use LDA

**Best for:**
- Classification problems
- When class labels available
- Feature extraction before classifier
- When want interpretable discriminative features

**Not recommended for:**
- Unsupervised learning
- Non-linear class boundaries
- Visualization (limited to C-1 dimensions)
- Small number of classes

## Other Methods

### Truncated SVD (LSA)

**Singular Value Decomposition** without centering data:

$$X = U \Sigma V^T$$

Keep top $k$ singular values:

$$X_{reduced} = U_k \Sigma_k$$

**Use for:**
- Sparse matrices (text data)
- Non-negative data
- Latent Semantic Analysis (LSA)

```python
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=50)
X_reduced = svd.fit_transform(X_sparse)
```

### Kernel PCA

**Non-linear PCA** using kernel trick:

1. Map data to high-dimensional space via kernel
2. Perform PCA in that space
3. Never explicitly compute mapping

**Common kernels:**
- RBF: $k(x,y) = \exp(-\gamma||x-y||^2)$
- Polynomial: $k(x,y) = (x^Ty + c)^d$
- Sigmoid: $k(x,y) = \tanh(\gamma x^Ty + c)$

```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_reduced = kpca.fit_transform(X)
```

### Independent Component Analysis (ICA)

Separates multivariate signal into **independent** components:

$$X = AS$$

where:
- $A$ = mixing matrix
- $S$ = independent sources

**Use for:**
- Blind source separation
- Signal processing
- Finding independent features

```python
from sklearn.decomposition import FastICA

ica = FastICA(n_components=10)
X_reduced = ica.fit_transform(X)
```

### Isomap

**Isometric Mapping** - preserves geodesic distances on manifold:

1. Build k-nearest neighbor graph
2. Compute shortest paths (geodesic distances)
3. Apply classical MDS on geodesic distances

**Use for:**
- Non-linear manifolds
- When geodesic distance matters
- Swiss roll, S-curve type data

```python
from sklearn.manifold import Isomap

isomap = Isomap(n_neighbors=5, n_components=2)
X_reduced = isomap.fit_transform(X)
```

### Autoencoders

**Neural network** approach to dimensionality reduction:

```
Encoder: X → Z (bottleneck)
Decoder: Z → X'
```

Minimize reconstruction error: $||X - X'||^2$

**Use for:**
- Complex non-linear relationships
- Large datasets
- When neural networks beneficial

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Encoder
input_dim = X.shape[1]
encoding_dim = 10

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)

# Decoder
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Autoencoder
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(X, X, epochs=50, batch_size=256, shuffle=True)

# Get encoder
encoder = Model(input_layer, encoded)
X_reduced = encoder.predict(X)
```

## Choosing a Method

### Decision Tree

```
Need labels?
├─ Yes → LDA (if linear) or supervised UMAP
└─ No
   └─ What's the goal?
      ├─ Visualization
      │  ├─ Small data (<5k) → t-SNE or UMAP
      │  └─ Large data (>5k) → UMAP
      ├─ Feature extraction for ML
      │  ├─ Linear relationships → PCA
      │  ├─ Non-linear → Kernel PCA, UMAP, or Autoencoder
      │  └─ Sparse data → Truncated SVD
      └─ Interpretation
         └─ PCA or LDA (if supervised)
```

### Comparison Table

| Method | Type | Speed | Preserves | Best For | Max Dims |
|--------|------|-------|-----------|----------|----------|
| PCA | Linear | Fast | Global variance | Feature extraction, preprocessing | All |
| t-SNE | Non-linear | Slow | Local structure | Visualization (<10k points) | 2-3 |
| UMAP | Non-linear | Fast | Local + global | Visualization, features (large data) | Any |
| LDA | Linear | Fast | Class separation | Classification tasks | C-1 |
| Kernel PCA | Non-linear | Medium | Non-linear variance | Non-linear features | All |
| Isomap | Non-linear | Slow | Geodesic distances | Manifold learning | Any |
| Autoencoder | Non-linear | Medium | Learned features | Complex patterns, large data | Any |

### Guidelines

**Use PCA when:**
- Need fast, deterministic results
- Linear relationships sufficient
- Preprocessing for other algorithms
- Want interpretable components
- Need to transform new points easily

**Use t-SNE when:**
- Visualization is primary goal
- Data is relatively small (<10k)
- Local structure is critical
- Don't need to transform new points

**Use UMAP when:**
- Need both local and global structure
- Data is large (>10k)
- Need to transform new points
- Speed is important
- Want general-purpose method

**Use LDA when:**
- Have labeled data
- Goal is classification
- Want maximally discriminative features
- Need interpretability

## Practical Considerations

### Feature Scaling

**Critical for most methods** (except tree-based):

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Why?**
- Features with large scales dominate
- Distance-based methods affected
- PCA: High-variance features dominate
- t-SNE/UMAP: Scale affects neighborhoods

### Choosing Number of Components

**PCA:**

```python
# Method 1: Explained variance threshold (e.g., 95%)
pca = PCA()
pca.fit(X_scaled)
cumsum = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cumsum >= 0.95) + 1

# Method 2: Scree plot
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.show()
```

**Visualization methods (t-SNE, UMAP):**
- Almost always use 2 or 3 dimensions
- Higher dimensions hard to visualize

**Feature extraction:**
- Use cross-validation
- Try different numbers and compare model performance

### Handling Missing Values

**Before dimensionality reduction:**

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
```

**Why?** Most methods require complete data.

### Computational Considerations

**Large datasets:**

1. **PCA:**
   ```python
   from sklearn.decomposition import IncrementalPCA

   # Process in batches
   ipca = IncrementalPCA(n_components=50, batch_size=1000)
   ipca.fit(X)
   X_reduced = ipca.transform(X)
   ```

2. **t-SNE:**
   ```python
   # Use PCA preprocessing
   pca = PCA(n_components=50)
   X_pca = pca.fit_transform(X)
   tsne = TSNE(n_components=2)
   X_tsne = tsne.fit_transform(X_pca)
   ```

3. **UMAP:** Handles large data naturally

### Interpreting Results

**PCA:**
```python
# Feature importance in each component
components = pca.components_
feature_names = ['feature_1', 'feature_2', ...]

for i, comp in enumerate(components):
    print(f"\nPC{i+1} top features:")
    top_features = sorted(zip(feature_names, comp), key=lambda x: abs(x[1]), reverse=True)[:5]
    for feat, weight in top_features:
        print(f"  {feat}: {weight:.3f}")
```

**t-SNE/UMAP:**
- Don't interpret distances between clusters
- Focus on local neighborhoods
- Cluster presence, not size/position

**LDA:**
```python
# Feature importance for discrimination
for i, weights in enumerate(lda.coef_):
    print(f"\nDiscriminant {i+1} top features:")
    top_features = sorted(zip(feature_names, weights), key=lambda x: abs(x[1]), reverse=True)[:5]
    for feat, weight in top_features:
        print(f"  {feat}: {weight:.3f}")
```

## Implementation

### PCA Complete Example

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load data
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA
pca = PCA()
pca.fit(X_scaled)

# Plot explained variance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(cumsum) + 1), cumsum, 'ro-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.95, color='green', linestyle='--', label='95% threshold')
plt.title('Cumulative Explained Variance')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Choose number of components (95% variance)
n_components = np.argmax(cumsum >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components}")

# Transform to 2D for visualization
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)

# Visualize
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA Visualization')
plt.colorbar(scatter, label='Digit')
plt.grid(True, alpha=0.3)
plt.show()

# Reconstruction
pca_k = PCA(n_components=n_components)
X_reduced = pca_k.fit_transform(X_scaled)
X_reconstructed = pca_k.inverse_transform(X_reduced)

reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"Reconstruction error: {reconstruction_error:.6f}")
```

### t-SNE Example

```python
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Prepare data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA preprocessing (recommended for high-dim data)
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

# Apply t-SNE with different perplexities
perplexities = [5, 30, 50, 100]
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
axes = axes.ravel()

for idx, perplexity in enumerate(perplexities):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=1000,
        random_state=42,
        verbose=0
    )
    X_tsne = tsne.fit_transform(X_pca)

    axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10',
                      alpha=0.6, s=20)
    axes[idx].set_title(f't-SNE (perplexity={perplexity})', fontsize=14)
    axes[idx].set_xlabel('Dimension 1')
    axes[idx].set_ylabel('Dimension 2')

plt.tight_layout()
plt.show()
```

### UMAP Example

```python
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Prepare data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply UMAP with different parameters
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
params = [
    {'n_neighbors': 5, 'min_dist': 0.1},
    {'n_neighbors': 15, 'min_dist': 0.1},
    {'n_neighbors': 50, 'min_dist': 0.1},
    {'n_neighbors': 15, 'min_dist': 0.5},
]

for idx, param in enumerate(params):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=param['n_neighbors'],
        min_dist=param['min_dist'],
        random_state=42
    )
    X_umap = reducer.fit_transform(X_scaled)

    ax = axes[idx // 2, idx % 2]
    ax.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10',
               alpha=0.6, s=20)
    ax.set_title(f'UMAP (neighbors={param["n_neighbors"]}, min_dist={param["min_dist"]})',
                 fontsize=12)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

plt.tight_layout()
plt.show()
```

### LDA Example

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Prepare data (requires labels)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)  # Max: n_classes - 1
X_lda = lda.fit_transform(X_scaled, y)

# Visualize
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.xlabel(f'LD1 ({lda.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'LD2 ({lda.explained_variance_ratio_[1]:.2%} variance)')
plt.title('LDA Visualization')
plt.colorbar(scatter, label='Class')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Explained variance ratio: {lda.explained_variance_ratio_}")
```

### Comparison Visualization

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Prepare data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply methods
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_50 = PCA(n_components=50)
X_pca50 = pca_50.fit_transform(X_scaled)

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_pca50)  # PCA preprocessing

umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X_scaled)

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# Visualize all
fig, axes = plt.subplots(2, 2, figsize=(15, 15))

methods = [
    ('PCA', X_pca),
    ('t-SNE', X_tsne),
    ('UMAP', X_umap),
    ('LDA', X_lda)
]

for idx, (name, X_transformed) in enumerate(methods):
    ax = axes[idx // 2, idx % 2]
    scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1],
                         c=y, cmap='tab10', alpha=0.6, s=20)
    ax.set_title(f'{name}', fontsize=16)
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Feature Extraction Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Create pipeline with PCA
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),  # Keep 95% variance
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Evaluate
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f"Accuracy with PCA: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Compare without PCA
pipeline_no_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

scores_no_pca = cross_val_score(pipeline_no_pca, X, y, cv=5, scoring='accuracy')
print(f"Accuracy without PCA: {scores_no_pca.mean():.4f} (+/- {scores_no_pca.std():.4f})")
```

## References

- [Scikit-learn Dimensionality Reduction](https://scikit-learn.org/stable/modules/decomposition.html)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- Jolliffe, I. T. (2002). "Principal Component Analysis"
- van der Maaten, L., & Hinton, G. (2008). "Visualizing Data using t-SNE"
- McInnes, L., et al. (2018). "UMAP: Uniform Manifold Approximation and Projection"
- Fisher, R. A. (1936). "The use of multiple measurements in taxonomic problems"
- The Elements of Statistical Learning (Hastie, Tibshirani, Friedman)
- Pattern Recognition and Machine Learning (Christopher Bishop)
