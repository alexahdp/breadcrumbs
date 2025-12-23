# Ridge and Lasso Regression

## Table of Contents

1. [Introduction to Regularization](#introduction-to-regularization)
2. [Ridge Regression (L2)](#ridge-regression-l2)
3. [Lasso Regression (L1)](#lasso-regression-l1)
4. [Elastic Net](#elastic-net)
5. [Comparison and Selection](#comparison-and-selection)
6. [Hyperparameter Tuning](#hyperparameter-tuning)

## Introduction to Regularization

**Regularization** - a technique to prevent overfitting by adding a penalty term to the loss function, discouraging overly complex models.

### The Problem: Overfitting

**Overfitting** occurs when a model learns noise in the training data, leading to:
- Perfect fit on training data
- Poor generalization to new data
- Large coefficient values (high sensitivity to input changes)

**Common causes:**
- Too many features relative to observations
- High multicollinearity
- High-degree polynomial features
- Complex models with insufficient data

### Regularization Solution

**Idea:** Constrain the magnitude of coefficients by adding a penalty to the loss function.

**General form:**

$$J(\boldsymbol{\beta}) = \text{Loss}(\boldsymbol{\beta}) + \lambda \cdot \text{Penalty}(\boldsymbol{\beta})$$

where:
- $\text{Loss}(\boldsymbol{\beta})$ is the standard loss (e.g., MSE)
- $\lambda \geq 0$ is the regularization strength
- $\text{Penalty}(\boldsymbol{\beta})$ discourages large coefficients

**Benefits:**
- Reduces overfitting
- Improves generalization
- Handles multicollinearity
- Can perform feature selection

## Ridge Regression (L2)

**Ridge Regression** (also called **L2 Regularization** or **Tikhonov Regularization**) - adds the sum of squared coefficients as penalty.

### Mathematical Formulation

**Cost function:**

$$J(\boldsymbol{\beta}) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda\sum_{j=1}^{p}\beta_j^2$$

Or equivalently:

$$J(\boldsymbol{\beta}) = ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \lambda||\boldsymbol{\beta}||^2$$

where:
- First term: Mean Squared Error (MSE)
- Second term: L2 penalty (squared magnitude of coefficients)
- $\lambda$: regularization parameter ($\lambda \geq 0$)

**Note:** The intercept $\beta_0$ is typically not penalized.

### Closed-Form Solution

Unlike standard linear regression, Ridge has a unique solution even when $\mathbf{X}^T\mathbf{X}$ is singular:

$$\boldsymbol{\beta}_{ridge} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

where $\mathbf{I}$ is the identity matrix.

**Key insight:** Adding $\lambda\mathbf{I}$ ensures $(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})$ is always invertible for $\lambda > 0$.

### Properties of Ridge

**1. Coefficient Shrinkage**

Ridge shrinks coefficients toward zero but never exactly to zero:

$$\beta_j^{ridge} \rightarrow 0 \text{ as } \lambda \rightarrow \infty$$

**2. Handles Multicollinearity**

When features are highly correlated:
- OLS coefficients become unstable (large variance)
- Ridge stabilizes estimates by shrinking correlated coefficients

**3. Bias-Variance Tradeoff**

- $\lambda = 0$: No regularization (standard OLS)
- $\lambda$ small: Slight bias, reduced variance
- $\lambda$ large: High bias, low variance
- $\lambda \rightarrow \infty$: All coefficients → 0 (intercept-only model)

**4. All Features Retained**

Ridge keeps all features in the model (no feature selection).

### When to Use Ridge

**Good fit when:**
- Many features with small to medium effects
- Multicollinearity present
- Want to keep all features
- Prediction accuracy more important than interpretability

**Poor fit when:**
- Need sparse model (feature selection required)
- Many irrelevant features
- Interpretability critical

## Lasso Regression (L1)

**Lasso** (**L**east **A**bsolute **S**hrinkage and **S**election **O**perator) - uses absolute value of coefficients as penalty.

### Mathematical Formulation

**Cost function:**

$$J(\boldsymbol{\beta}) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda\sum_{j=1}^{p}|\beta_j|$$

Or equivalently:

$$J(\boldsymbol{\beta}) = ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \lambda||\boldsymbol{\beta}||_1$$

where $||\boldsymbol{\beta}||_1 = \sum_{j=1}^{p}|\beta_j|$ is the L1 norm.

### No Closed-Form Solution

Unlike Ridge, Lasso has no analytical solution due to the absolute value function. Must use numerical optimization:
- **Coordinate Descent**
- **Least Angle Regression (LARS)**
- **Proximal Gradient Descent**

### Properties of Lasso

**1. Automatic Feature Selection**

Lasso can shrink coefficients exactly to zero:

$$\beta_j^{lasso} = 0 \text{ for some } j$$

This performs automatic feature selection, creating sparse models.

**2. Sparse Solutions**

As $\lambda$ increases:
- More coefficients become exactly zero
- Model becomes simpler
- Only most important features retained

**3. Handles High-Dimensional Data**

Works well when:
- Number of features $p$ > number of samples $n$
- Many irrelevant features present

**4. Limitation with Correlated Features**

When features are highly correlated:
- Lasso tends to arbitrarily select one and zero out others
- Selection can be unstable across different samples

**5. Geometric Interpretation**

L1 penalty creates a diamond-shaped constraint region in coefficient space. The optimal solution often occurs at corners where some coefficients are exactly zero.

### When to Use Lasso

**Good fit when:**
- Feature selection needed
- Many irrelevant/redundant features
- Sparse model desired
- Interpretability important

**Poor fit when:**
- All features are relevant
- Features highly correlated (consider Elastic Net)
- Number of important features > number of samples

## Elastic Net

**Elastic Net** - combines L1 and L2 penalties to get benefits of both Ridge and Lasso.

### Mathematical Formulation

**Cost function:**

$$J(\boldsymbol{\beta}) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda_1\sum_{j=1}^{p}|\beta_j| + \lambda_2\sum_{j=1}^{p}\beta_j^2$$

**Alternative formulation:**

$$J(\boldsymbol{\beta}) = ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \lambda[(1-\alpha)||\boldsymbol{\beta}||_2^2 + \alpha||\boldsymbol{\beta}||_1]$$

where:
- $\lambda$: overall regularization strength
- $\alpha \in [0, 1]$: mixing parameter
  - $\alpha = 0$: Pure Ridge
  - $\alpha = 1$: Pure Lasso
  - $0 < \alpha < 1$: Elastic Net

### Properties of Elastic Net

**1. Grouped Selection**

When features are correlated, Elastic Net tends to select groups together (unlike Lasso).

**2. Stability**

More stable than Lasso when features are highly correlated.

**3. Feature Selection + Shrinkage**

Combines Lasso's feature selection with Ridge's coefficient shrinkage.

**4. Flexibility**

Two hyperparameters ($\lambda$ and $\alpha$) provide more flexibility but require more tuning.

### When to Use Elastic Net

**Good fit when:**
- Features highly correlated
- Want feature selection but more stable than Lasso
- Number of features >> number of samples
- Unsure whether to use Ridge or Lasso

**Poor fit when:**
- Simple model sufficient (try Ridge or Lasso first)
- Computational cost is concern (more hyperparameters to tune)

## Comparison and Selection

### Visual Comparison

| Aspect | Ridge (L2) | Lasso (L1) | Elastic Net |
|--------|-----------|-----------|-------------|
| **Penalty** | $\sum \beta_j^2$ | $\sum |\beta_j|$ | $\alpha\sum|\beta_j| + (1-\alpha)\sum\beta_j^2$ |
| **Feature Selection** | No | Yes | Yes |
| **Coefficients** | Shrink toward 0 | Some exactly 0 | Some exactly 0 |
| **Multicollinearity** | Handles well | Arbitrary selection | Handles well |
| **Interpretability** | Moderate | High | High |
| **Computational Cost** | Low | Moderate | Moderate-High |

### Effect of Lambda (λ)

**$\lambda = 0$:**
- No regularization (standard OLS)
- High variance, low bias
- Overfitting risk

**$\lambda$ small:**
- Slight regularization
- Coefficients close to OLS
- Some variance reduction

**$\lambda$ moderate:**
- Balanced bias-variance
- Significant coefficient shrinkage
- Best generalization (typically)

**$\lambda$ large:**
- Heavy regularization
- Most/all coefficients near zero
- High bias, low variance
- Underfitting risk

**$\lambda \rightarrow \infty$:**
- All coefficients → 0
- Intercept-only model

### Decision Guide

```
Start
  ↓
Many irrelevant features?
  ├─ Yes → Features correlated?
  │         ├─ Yes → Elastic Net
  │         └─ No → Lasso
  └─ No → All features relevant?
            ├─ Yes → Ridge
            └─ No → Lasso or Elastic Net
```

## Hyperparameter Tuning

### Grid Search with Cross-Validation

**Approach:** Try different values of $\lambda$ (and $\alpha$ for Elastic Net) and select the one with best cross-validation performance.

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV

# Ridge
ridge = Ridge()
params_ridge = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_ridge = GridSearchCV(ridge, params_ridge, cv=5, scoring='neg_mean_squared_error')
grid_ridge.fit(X_train, y_train)

print(f"Best alpha: {grid_ridge.best_params_['alpha']}")
print(f"Best score: {-grid_ridge.best_score_:.4f}")

# Lasso
lasso = Lasso(max_iter=10000)
params_lasso = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_lasso = GridSearchCV(lasso, params_lasso, cv=5, scoring='neg_mean_squared_error')
grid_lasso.fit(X_train, y_train)

# Elastic Net
elastic = ElasticNet(max_iter=10000)
params_elastic = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # l1_ratio = α in formulation
}
grid_elastic = GridSearchCV(elastic, params_elastic, cv=5, scoring='neg_mean_squared_error')
grid_elastic.fit(X_train, y_train)
```

### Regularization Path

**Regularization Path** - plot of coefficient values vs. regularization strength.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path, ridge_path

# Lasso path
alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_train, alphas=np.logspace(-4, 1, 100))

plt.figure(figsize=(12, 5))

# Lasso regularization path
plt.subplot(1, 2, 1)
for coef in coefs_lasso:
    plt.plot(alphas_lasso, coef)
plt.xscale('log')
plt.xlabel('Alpha (λ)')
plt.ylabel('Coefficients')
plt.title('Lasso Regularization Path')
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

# Ridge regularization path
alphas_ridge = np.logspace(-4, 4, 100)
coefs_ridge = []
for alpha in alphas_ridge:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    coefs_ridge.append(ridge.coef_)
coefs_ridge = np.array(coefs_ridge).T

plt.subplot(1, 2, 2)
for coef in coefs_ridge:
    plt.plot(alphas_ridge, coef)
plt.xscale('log')
plt.xlabel('Alpha (λ)')
plt.ylabel('Coefficients')
plt.title('Ridge Regularization Path')
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
```

**Interpretation:**
- **Lasso:** Coefficients hit zero at certain $\lambda$ values
- **Ridge:** Coefficients asymptotically approach zero but never reach it

### Cross-Validation Curves

```python
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

# Automatic CV with built-in methods
alphas = np.logspace(-4, 4, 100)

ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train, y_train)
print(f"Optimal Ridge alpha: {ridge_cv.alpha_}")

lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000)
lasso_cv.fit(X_train, y_train)
print(f"Optimal Lasso alpha: {lasso_cv.alpha_}")

elastic_cv = ElasticNetCV(alphas=alphas, l1_ratio=[0.1, 0.5, 0.7, 0.9], cv=5, max_iter=10000)
elastic_cv.fit(X_train, y_train)
print(f"Optimal Elastic Net alpha: {elastic_cv.alpha_}")
print(f"Optimal Elastic Net l1_ratio: {elastic_cv.l1_ratio_}")
```

### Feature Importance from Lasso

```python
import pandas as pd

# Train Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Get non-zero coefficients
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lasso.coef_
})

# Filter non-zero
feature_importance = feature_importance[feature_importance['Coefficient'] != 0]
feature_importance = feature_importance.sort_values('Coefficient', key=abs, ascending=False)

print(f"Number of selected features: {len(feature_importance)}")
print("\nTop features:")
print(feature_importance)
```

### Practical Implementation Example

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Generate synthetic data with many irrelevant features
X, y = make_regression(n_samples=200, n_features=100, n_informative=10,
                       noise=10, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# IMPORTANT: Scale features before regularization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare models
models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1, max_iter=10000),
    'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    n_nonzero = np.sum(model.coef_ != 0)

    print(f"\n{name}:")
    print(f"  MSE: {mse:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Non-zero coefficients: {n_nonzero}/{len(model.coef_)}")
```

### Feature Scaling Importance

**Critical:** Always scale features before applying regularization!

**Why?** Regularization penalizes coefficient magnitude. Features on different scales will have different magnitude coefficients even if equally important.

**Example:**
- Feature 1: range [0, 1] → coefficient might be 100
- Feature 2: range [0, 1000] → coefficient might be 0.1
- Without scaling, Feature 1 gets penalized 1000× more!

```python
from sklearn.pipeline import make_pipeline

# Correct approach: scaling in pipeline
model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)

model.fit(X_train, y_train)
```

## References

- An Introduction to Statistical Learning (James, Witten, Hastie, Tibshirani)
- The Elements of Statistical Learning (Hastie, Tibshirani, Friedman)
- Scikit-learn Documentation: [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [Ridge Regression and Other Kernels for Genomic Selection (Endelman, 2011)](https://doi.org/10.1186/1471-2156-12-70)
