# Polynomial Regression

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Feature Engineering](#feature-engineering)
4. [Model Complexity](#model-complexity)
5. [Choosing the Right Degree](#choosing-the-right-degree)
6. [Practical Considerations](#practical-considerations)

## Introduction

**Polynomial Regression** - extends linear regression by adding polynomial terms of the features, allowing the model to capture non-linear relationships while still using linear regression methods.

**Key insight:** Although the relationship with features is non-linear, the model remains linear in its parameters (coefficients), making it a special case of multiple linear regression.

**Use cases:**
- Modeling curved relationships
- Capturing growth patterns (exponential, logarithmic)
- Approximating complex non-linear functions
- Time series with trend components

## Mathematical Foundation

### From Linear to Polynomial

**Standard linear regression:**

$$y = \beta_0 + \beta_1 x + \epsilon$$

**Polynomial regression (degree $d$):**

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + ... + \beta_d x^d + \epsilon$$

### Multivariate Polynomial Regression

For multiple features, polynomial regression can include:
- **Power terms:** $x_1^2, x_1^3, x_2^2$, etc.
- **Interaction terms:** $x_1 x_2, x_1 x_2^2$, etc.

**Example with 2 features (degree 2):**

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1^2 + \beta_4 x_2^2 + \beta_5 x_1 x_2 + \epsilon$$

### Matrix Representation

Polynomial regression is still a linear model in the transformed feature space:

$$\mathbf{y} = \mathbf{X}_{poly}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

where $\mathbf{X}_{poly}$ contains the original features and their polynomial transformations.

**Solved using standard OLS:**

$$\boldsymbol{\beta} = (\mathbf{X}_{poly}^T\mathbf{X}_{poly})^{-1}\mathbf{X}_{poly}^T\mathbf{y}$$

## Feature Engineering

**Feature Engineering** - the process of creating new features from existing ones to improve model performance.

### Polynomial Features

**Single variable transformation:**

For feature $x$ with degree $d=3$:

$$[x] \rightarrow [1, x, x^2, x^3]$$

**Multiple variables with interactions:**

For features $[x_1, x_2]$ with degree $d=2$:

$$[x_1, x_2] \rightarrow [1, x_1, x_2, x_1^2, x_1 x_2, x_2^2]$$

### Number of Features

The number of polynomial features grows rapidly with degree and number of original features.

**Formula for $n$ features and degree $d$:**

$$\text{Number of terms} = \binom{n + d}{d} = \frac{(n + d)!}{n! \cdot d!}$$

**Example:**
- 2 features, degree 2: $\binom{2+2}{2} = 6$ terms
- 5 features, degree 3: $\binom{5+3}{3} = 56$ terms
- 10 features, degree 3: $\binom{10+3}{3} = 286$ terms

**Curse of Dimensionality:** High polynomial degrees dramatically increase feature count, leading to:
- Increased computational cost
- Risk of overfitting
- Need for more training data

### Feature Scaling

**Critical importance:** Polynomial features create values with vastly different scales.

**Example:**
- If $x \in [1, 100]$, then $x^2 \in [1, 10000]$ and $x^3 \in [1, 1000000]$

**Solution:** Always scale features before polynomial transformation:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=3)),
    ('model', LinearRegression())
])
```

## Model Complexity

### Bias-Variance Tradeoff

**Underfitting (High Bias):**
- Polynomial degree too low
- Model too simple to capture underlying pattern
- High training and test error

**Overfitting (High Variance):**
- Polynomial degree too high
- Model fits training noise
- Low training error, high test error

**Optimal complexity:**
- Balances bias and variance
- Generalizes well to unseen data

### Overfitting Example

**Degree 1 (Underfitting):** Straight line, misses curve pattern

**Degree 3 (Good fit):** Captures pattern without overfitting

**Degree 15 (Overfitting):** Passes through all training points but oscillates wildly

### Mathematical Perspective

As polynomial degree increases:
- **Model capacity increases:** Can represent more complex functions
- **Training error decreases:** Better fit to training data
- **Test error may increase:** Poor generalization due to overfitting

## Choosing the Right Degree

### Cross-Validation

**K-Fold Cross-Validation** - the most reliable method for selecting polynomial degree:

1. Split data into K folds
2. For each degree $d$:
   - Train on K-1 folds
   - Validate on remaining fold
   - Repeat K times
3. Choose degree with lowest average validation error

**Typical approach:**

```python
from sklearn.model_selection import cross_val_score
import numpy as np

degrees = range(1, 10)
cv_scores = []

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    scores = cross_val_score(model, X_poly, y, cv=5,
                            scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())

# Select degree with minimum CV error
optimal_degree = degrees[np.argmin(cv_scores)]
```

### Learning Curves

**Learning Curves** - plot training and validation error vs. training set size or model complexity.

**What to look for:**
- **Converging errors:** Good generalization
- **Large gap:** Overfitting (reduce complexity)
- **Both errors high:** Underfitting (increase complexity)

### Regularization

When using high-degree polynomials, apply regularization to prevent overfitting:

- **Ridge Regression (L2):** Shrinks coefficients
- **Lasso Regression (L1):** Can zero out coefficients
- **Elastic Net:** Combination of L1 and L2

This allows higher degree polynomials while controlling overfitting (see `ridge-lasso.md`).

### Domain Knowledge

Consider the underlying phenomenon:
- **Physical laws:** May suggest polynomial degree (e.g., $E = mc^2$)
- **Growth patterns:** Exponential, quadratic, logarithmic
- **Periodic behavior:** May need trigonometric transforms instead

## Practical Considerations

### When to Use Polynomial Regression

**Good fit when:**
- Clear non-linear relationship in data
- Limited number of features (avoid dimensionality explosion)
- Smooth, continuous curves needed
- Interpretability still important

**Poor fit when:**
- Relationship is highly complex or discontinuous
- Too many features (use splines or neural networks instead)
- Need automatic feature engineering
- Extrapolation beyond training range required

### Limitations

**1. Extrapolation Issues**

Polynomials behave poorly outside the training range:
- High-degree polynomials tend to infinity at extremes
- Predictions can be wildly inaccurate

**2. Oscillation (Runge's Phenomenon)**

High-degree polynomials oscillate between data points:
- Even with perfect fit at training points
- Creates unrealistic behavior

**Solution:** Use piecewise polynomials (splines) instead.

**3. Interpretability**

Higher degrees reduce interpretability:
- $x^7$ term has no intuitive meaning
- Interaction terms become complex

### Alternatives to Consider

**Splines:**
- Piecewise polynomial functions
- Better extrapolation
- Less oscillation

**Generalized Additive Models (GAM):**
- Smooth functions of individual features
- More flexible than polynomials

**Neural Networks:**
- Universal function approximators
- Handle complex non-linearities
- Require more data

### Python Implementation

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data with non-linear relationship
np.random.seed(42)
X = np.sort(np.random.rand(100, 1) * 10, axis=0)
y = 2 + 3*X + 0.5*X**2 - 0.1*X**3 + np.random.randn(100, 1) * 5

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compare different polynomial degrees
degrees = [1, 2, 3, 5, 10]
plt.figure(figsize=(15, 3))

for i, degree in enumerate(degrees):
    # Create polynomial regression model
    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Plot
    plt.subplot(1, len(degrees), i + 1)
    plt.scatter(X_train, y_train, alpha=0.5, label='Train')
    plt.scatter(X_test, y_test, alpha=0.5, label='Test')

    X_line = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, 'r-', label=f'Degree {degree}')

    plt.title(f'Degree {degree}\nMSE: {mse:.2f}, R²: {r2:.3f}')
    plt.legend()

plt.tight_layout()
plt.show()
```

### Feature Interaction Example

```python
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

# Create sample dataset
X = pd.DataFrame({
    'x1': [1, 2, 3, 4, 5],
    'x2': [2, 4, 6, 8, 10]
})

# Generate polynomial features with interactions
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X)

# Show feature names
feature_names = poly.get_feature_names_out(['x1', 'x2'])
X_poly_df = pd.DataFrame(X_poly, columns=feature_names)

print(X_poly_df)
# Output:
#    1  x1  x2  x1^2  x1 x2  x2^2
# 0  1   1   2     1      2     4
# 1  1   2   4     4      8    16
# ...
```

### Regularization with Polynomial Features

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

# High degree polynomial with Ridge regularization
model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=10),
    Ridge(alpha=1.0)  # Regularization prevents overfitting
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## References

- An Introduction to Statistical Learning (James, Witten, Hastie, Tibshirani)
- Pattern Recognition and Machine Learning (Christopher Bishop)
- Scikit-learn Documentation: [Polynomial Features](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
