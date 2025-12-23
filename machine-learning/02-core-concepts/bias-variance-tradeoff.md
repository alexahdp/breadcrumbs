# Bias-Variance Tradeoff

## Table of Contents

1. [Introduction](#introduction)
2. [Understanding Bias](#understanding-bias)
3. [Understanding Variance](#understanding-variance)
4. [The Tradeoff](#the-tradeoff)
5. [Overfitting and Underfitting](#overfitting-and-underfitting)
6. [Model Complexity](#model-complexity)
7. [Diagnosing and Addressing Issues](#diagnosing-and-addressing-issues)

## Introduction

The **bias-variance tradeoff** is one of the most fundamental concepts in machine learning. It describes the relationship between model complexity, training error, and generalization error.

Understanding this tradeoff is essential for:
- Choosing appropriate model complexity
- Diagnosing model performance issues
- Making informed decisions about data collection
- Selecting regularization strategies

## Understanding Bias

**Bias** measures how far off the model's predictions are from the true values on average. It represents the error introduced by approximating a real-world problem with a simplified model.

### Mathematical Definition

For a model $\hat{f}(x)$ trained on different datasets, bias at point $x$ is:

$$Bias[\hat{f}(x)] = E[\hat{f}(x)] - f(x)$$

where:
- $E[\hat{f}(x)]$ is the expected prediction across different training sets
- $f(x)$ is the true function value

### Characteristics of High Bias

**Symptoms:**
- Model is too simple
- Poor performance on training data
- Systematic errors in predictions
- Model makes strong assumptions about data

**Examples of high-bias models:**
- Linear regression for non-linear relationships
- Shallow decision trees for complex patterns
- Small neural networks for complex tasks

**Real-world example:**

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Generate non-linear data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, 100)

# High-bias model: linear fit to non-linear data
model_high_bias = LinearRegression()
model_high_bias.fit(X, y)
# This will have high training error - underfitting
```

### Causes of High Bias

1. **Insufficient model capacity**: Model lacks parameters to capture patterns
2. **Wrong model family**: Linear model for non-linear data
3. **Missing features**: Important predictors not included
4. **Over-regularization**: Too much constraint on model complexity

## Understanding Variance

**Variance** measures how much the model's predictions vary when trained on different datasets. It represents the model's sensitivity to fluctuations in the training data.

### Mathematical Definition

Variance at point $x$ is:

$$Var[\hat{f}(x)] = E[(\hat{f}(x) - E[\hat{f}(x)])^2]$$

This measures how much $\hat{f}(x)$ varies around its expected value across different training sets.

### Characteristics of High Variance

**Symptoms:**
- Model is too complex
- Excellent performance on training data
- Poor performance on test data
- Model memorizes training data including noise

**Examples of high-variance models:**
- High-degree polynomial regression
- Deep decision trees without pruning
- K-NN with very small K
- Large neural networks without regularization

**Real-world example:**

```python
from sklearn.tree import DecisionTreeRegressor

# High-variance model: very deep tree
model_high_variance = DecisionTreeRegressor(max_depth=20)
model_high_variance.fit(X_train, y_train)

print(f"Training R²: {model_high_variance.score(X_train, y_train)}")  # ~1.0
print(f"Test R²: {model_high_variance.score(X_test, y_test)}")  # Much lower
# This shows overfitting - high variance
```

### Causes of High Variance

1. **Excessive model capacity**: Too many parameters
2. **Small training dataset**: Not enough data to learn stable patterns
3. **Noisy features**: Including irrelevant or noisy predictors
4. **Insufficient regularization**: Model not constrained enough

## The Tradeoff

### Bias-Variance Decomposition

The expected prediction error at a point $x$ can be decomposed as:

$$E[(y - \hat{f}(x))^2] = \underbrace{(Bias[\hat{f}(x)])^2}_{\text{Bias}^2} + \underbrace{Var[\hat{f}(x)]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible Error}}$$

where:
- **Bias²**: Error from incorrect assumptions
- **Variance**: Error from sensitivity to training data
- **Irreducible Error** ($\sigma^2$): Noise inherent in the data (cannot be reduced)

### Key Insight

**You cannot simultaneously minimize both bias and variance.**

- Decreasing bias typically increases variance
- Decreasing variance typically increases bias
- Optimal model balances both sources of error

### Visual Understanding

```
Error
  ↑
  │     Total Error
  │        ╱╲
  │       ╱  ╲
  │      ╱    ╲  ← Variance
  │     ╱      ╲
  │____╱________╲____________
  │              ╲          ╱
  │               ╲  Bias² ╱
  │                ╲______╱
  │
  └────────────────────────→ Model Complexity
   Simple              Complex
```

### The Tradeoff in Action

| Model Complexity | Bias | Variance | Total Error |
|-----------------|------|----------|-------------|
| Very Low | High | Low | High (underfitting) |
| Optimal | Medium | Medium | **Minimum** |
| Very High | Low | High | High (overfitting) |

## Overfitting and Underfitting

### Underfitting

**Underfitting** occurs when a model is too simple to capture the underlying structure of the data.

**Characteristics:**
- High training error
- High test error
- Both errors are similar
- Model has high bias

**Mathematical indicator:**

$$Train\_Error > Acceptable\_Threshold$$
$$|Train\_Error - Test\_Error| \approx 0$$

**How to detect:**

```python
from sklearn.metrics import mean_squared_error

train_error = mean_squared_error(y_train, model.predict(X_train))
test_error = mean_squared_error(y_test, model.predict(X_test))

if train_error > threshold and abs(train_error - test_error) < 0.1 * train_error:
    print("Model is underfitting")
```

**Solutions:**
- Increase model complexity
- Add more features
- Reduce regularization
- Train longer
- Use more complex model family

### Overfitting

**Overfitting** occurs when a model learns the training data too well, including its noise and peculiarities.

**Characteristics:**
- Very low training error
- High test error
- Large gap between training and test error
- Model has high variance

**Mathematical indicator:**

$$Train\_Error < Acceptable\_Threshold$$
$$Test\_Error - Train\_Error \gg 0$$

**How to detect:**

```python
train_error = mean_squared_error(y_train, model.predict(X_train))
test_error = mean_squared_error(y_test, model.predict(X_test))

if train_error < 0.05 and (test_error - train_error) / train_error > 2:
    print("Model is overfitting")
```

**Solutions:**
- Collect more training data
- Reduce model complexity
- Add regularization
- Use dropout (for neural networks)
- Perform feature selection
- Use cross-validation
- Apply early stopping
- Data augmentation

### The "Sweet Spot"

The optimal model achieves balance:

$$\text{Optimal Complexity} = \arg\min_{complexity} [Bias^2 + Variance + \sigma^2]$$

**Indicators of good fit:**
- Training error is acceptably low
- Test error is close to training error
- Both errors are acceptable for the task
- Model generalizes to new data

```python
# Good model characteristics
train_error = 0.15
test_error = 0.18
gap = test_error - train_error  # 0.03 (small gap)

# Both errors are low and close together
```

## Model Complexity

### Factors Affecting Complexity

**For different model types:**

| Model Type | Complexity Factors |
|-----------|-------------------|
| Linear Models | Number of features, polynomial degree |
| Decision Trees | Maximum depth, minimum samples per leaf |
| Neural Networks | Number of layers, neurons per layer |
| K-NN | Value of K (inversely related) |
| SVM | Kernel choice, regularization parameter C |
| Ensemble Methods | Number of estimators, individual model complexity |

### Controlling Complexity

**Polynomial Regression Example:**

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

complexities = range(1, 15)
train_errors = []
test_errors = []

for degree in complexities:
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Fit model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Calculate errors
    train_errors.append(mean_squared_error(y_train, model.predict(X_train_poly)))
    test_errors.append(mean_squared_error(y_test, model.predict(X_test_poly)))

# Plot learning curves to find optimal complexity
```

**Decision Tree Example:**

```python
from sklearn.tree import DecisionTreeClassifier

depths = range(1, 20)
train_scores = []
test_scores = []

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)

    train_scores.append(tree.score(X_train, y_train))
    test_scores.append(tree.score(X_test, y_test))

optimal_depth = depths[np.argmax(test_scores)]
```

## Diagnosing and Addressing Issues

### Diagnostic Process

**Step 1: Calculate learning curves**

Plot training and validation error as a function of:
- Training set size
- Model complexity
- Training iterations

**Step 2: Analyze the gap**

```python
def diagnose_model(train_error, val_error, threshold=0.05):
    """
    Diagnose if model is overfitting, underfitting, or well-fit.
    """
    if train_error > threshold:
        if abs(train_error - val_error) < 0.1 * train_error:
            return "Underfitting (high bias)"
        else:
            return "High bias and variance"
    else:
        if val_error - train_error > 2 * train_error:
            return "Overfitting (high variance)"
        else:
            return "Good fit"

diagnosis = diagnose_model(0.15, 0.17)
print(diagnosis)  # "Good fit"
```

### Solution Strategy

**If underfitting (high bias):**

1. **Increase model capacity:**
   ```python
   # More complex model
   from sklearn.ensemble import RandomForestRegressor
   model = RandomForestRegressor(n_estimators=100, max_depth=10)
   ```

2. **Feature engineering:**
   ```python
   # Add polynomial features
   from sklearn.preprocessing import PolynomialFeatures
   poly = PolynomialFeatures(degree=2, include_bias=False)
   X_enhanced = poly.fit_transform(X)
   ```

3. **Reduce regularization:**
   ```python
   # Less regularization
   from sklearn.linear_model import Ridge
   model = Ridge(alpha=0.01)  # Smaller alpha = less regularization
   ```

**If overfitting (high variance):**

1. **Get more data:**
   ```python
   # If possible, collect more training samples
   # Or use data augmentation
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
   ```

2. **Regularization:**
   ```python
   # Add regularization
   from sklearn.linear_model import Ridge
   model = Ridge(alpha=10.0)  # Larger alpha = more regularization
   ```

3. **Reduce complexity:**
   ```python
   # Simpler model
   tree = DecisionTreeClassifier(
       max_depth=5,           # Limit depth
       min_samples_leaf=10,   # Require more samples per leaf
       max_features='sqrt'    # Use subset of features
   )
   ```

4. **Feature selection:**
   ```python
   from sklearn.feature_selection import SelectKBest, f_classif
   selector = SelectKBest(f_classif, k=10)
   X_selected = selector.fit_transform(X_train, y_train)
   ```

5. **Cross-validation:**
   ```python
   from sklearn.model_selection import cross_val_score
   scores = cross_val_score(model, X_train, y_train, cv=5)
   print(f"CV Score: {scores.mean():.3f} (+/- {scores.std():.3f})")
   ```

### Learning Curves

**Learning curves** show how error changes with training set size:

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='neg_mean_squared_error'
)

train_mean = -train_scores.mean(axis=1)
val_mean = -val_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, label='Training error')
plt.plot(train_sizes, val_mean, label='Validation error')
plt.xlabel('Training set size')
plt.ylabel('Error')
plt.legend()
plt.title('Learning Curves')
```

**Interpretation:**

- **Both curves high and close**: Underfitting - need more complex model
- **Large gap between curves**: Overfitting - need more data or regularization
- **Training error decreasing, validation plateauing**: More data might help
- **Both curves low and close**: Good fit - model is appropriate

### Validation Curves

**Validation curves** show how error changes with model complexity:

```python
from sklearn.model_selection import validation_curve

param_range = range(1, 20)
train_scores, val_scores = validation_curve(
    DecisionTreeClassifier(), X, y,
    param_name="max_depth", param_range=param_range,
    cv=5, scoring="accuracy"
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.plot(param_range, train_mean, label='Training score')
plt.plot(param_range, val_mean, label='Validation score')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.legend()
```

## Practical Guidelines

### General Rules

1. **Start simple**: Begin with a simple model and increase complexity as needed
2. **Always use validation set**: Never tune based on test set performance
3. **Monitor both errors**: Track training and validation error during development
4. **Use cross-validation**: For robust performance estimates
5. **Regularize by default**: Start with some regularization and adjust

### Data Considerations

**How much data do you need?**

Rule of thumb for supervised learning:
- Linear models: $n > 10 \times d$ (10 samples per feature)
- Neural networks: $n > 1000 \times d$ (much more data needed)
- Tree-based: Less sensitive to sample size, but more data always helps

**When to collect more data:**
- Large gap between training and validation error
- Learning curves show validation error still decreasing
- You have high variance (overfitting)

**When not to collect more data:**
- High training error (high bias) - won't help much
- Learning curves have plateaued
- Constraint is model capacity, not data

### Model Selection Strategy

```python
def iterative_model_improvement(X_train, y_train, X_val, y_val):
    """
    Systematic approach to finding good model.
    """
    # Start simple
    model = LinearRegression()
    model.fit(X_train, y_train)

    train_error = mean_squared_error(y_train, model.predict(X_train))
    val_error = mean_squared_error(y_val, model.predict(X_val))

    print(f"Simple model - Train: {train_error:.3f}, Val: {val_error:.3f}")

    # If underfitting, increase complexity
    if train_error > 0.1:
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=2)
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)

        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        train_error = mean_squared_error(y_train, model.predict(X_train_poly))
        val_error = mean_squared_error(y_val, model.predict(X_val_poly))

        print(f"Complex model - Train: {train_error:.3f}, Val: {val_error:.3f}")

    # If overfitting, add regularization
    if val_error - train_error > 0.05:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        model.fit(X_train_poly, y_train)

        train_error = mean_squared_error(y_train, model.predict(X_train_poly))
        val_error = mean_squared_error(y_val, model.predict(X_val_poly))

        print(f"Regularized model - Train: {train_error:.3f}, Val: {val_error:.3f}")

    return model
```

## References

- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning, Chapter 5
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning
