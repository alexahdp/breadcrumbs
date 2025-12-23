# Cross-Validation

## Table of Contents

1. [Introduction](#introduction)
2. [The Hold-Out Method](#the-hold-out-method)
3. [K-Fold Cross-Validation](#k-fold-cross-validation)
4. [Stratified K-Fold](#stratified-k-fold)
5. [Leave-One-Out (LOO)](#leave-one-out-loo)
6. [Time Series Cross-Validation](#time-series-cross-validation)
7. [Other Cross-Validation Methods](#other-cross-validation-methods)
8. [Practical Guidelines](#practical-guidelines)

## Introduction

**Cross-Validation (CV)** is a statistical method for evaluating and comparing learning algorithms by dividing data into subsets for training and testing.

### Why Cross-Validation?

**Problems with single train-test split:**
- Performance estimate depends on random split
- May not use all data efficiently
- High variance in performance estimates
- Can be unlucky with split (e.g., all hard examples in test set)

**Benefits of cross-validation:**
- More robust performance estimates
- Uses all data for both training and validation
- Reduces variance in evaluation
- Better assessment of generalization
- Helps detect overfitting

### Key Principle

The fundamental idea is to evaluate model performance on data not seen during training, repeating this process multiple times with different data splits.

## The Hold-Out Method

The simplest validation approach: split data once into training and test sets.

### Methodology

$$D = D_{train} \cup D_{test}$$

Typically:
- 70-80% for training
- 20-30% for testing

### Implementation

```python
from sklearn.model_selection import train_test_split

# Simple train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42,    # For reproducibility
    stratify=y          # Maintain class distribution
)

# Train model
model.fit(X_train, y_train)

# Evaluate
test_score = model.score(X_test, y_test)
```

### Three-Way Split

For hyperparameter tuning, use three sets:

```python
# Split into train (60%), validation (20%), test (20%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Train on train set
model.fit(X_train, y_train)

# Tune hyperparameters using validation set
val_score = model.score(X_val, y_val)

# Final evaluation on test set (only once!)
test_score = model.score(X_test, y_test)
```

### Limitations

- Wastes data (test set never used for training)
- Results depend on random split
- High variance in estimates
- Not suitable for small datasets

## K-Fold Cross-Validation

**K-Fold CV** divides data into $K$ equal-sized folds, using each fold as test set exactly once.

### Methodology

1. Split data into $K$ folds: $D = D_1 \cup D_2 \cup ... \cup D_K$
2. For $i = 1$ to $K$:
   - Use fold $i$ as test set
   - Use remaining $K-1$ folds as training set
   - Train model and evaluate
3. Average performance across all folds

### Mathematical Formulation

Performance estimate:

$$CV_K = \frac{1}{K}\sum_{i=1}^{K} Error_i$$

Standard error:

$$SE = \frac{\sigma}{\sqrt{K}}$$

where $\sigma$ is the standard deviation of fold errors.

### Implementation

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)

# Perform 5-fold cross-validation
scores = cross_val_score(
    model, X, y,
    cv=5,                    # Number of folds
    scoring='accuracy'       # Metric to use
)

print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
print(f"Individual fold scores: {scores}")
```

### Detailed Cross-Validation

```python
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)

    print(f"Fold {fold + 1}: {score:.3f}")

print(f"\nMean: {np.mean(scores):.3f}")
print(f"Std: {np.std(scores):.3f}")
```

### Choosing K

**Common values:**
- $K = 5$ or $K = 10$ (most common)
- $K = n$ (Leave-One-Out, for small datasets)

**Tradeoffs:**

| K Value | Training Set Size | Computational Cost | Bias | Variance |
|---------|------------------|-------------------|------|----------|
| Small (2-3) | Smaller | Low | Higher | Lower |
| Medium (5-10) | Medium | Medium | Medium | Medium |
| Large (n) | Largest | High | Lower | Higher |

**Bias-Variance of CV estimate:**
- Smaller $K$: Higher bias (training set smaller), lower variance
- Larger $K$: Lower bias (training set larger), higher variance

### Multiple Metrics

```python
from sklearn.model_selection import cross_validate

# Evaluate multiple metrics
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

scores = cross_validate(
    model, X, y,
    cv=5,
    scoring=scoring,
    return_train_score=True
)

# Access different metrics
print(f"Test Accuracy: {scores['test_accuracy'].mean():.3f}")
print(f"Test F1: {scores['test_f1_macro'].mean():.3f}")
print(f"Train Accuracy: {scores['train_accuracy'].mean():.3f}")
```

## Stratified K-Fold

**Stratified K-Fold** preserves the percentage of samples for each class in each fold.

### Why Stratification?

**Problem with regular K-Fold:**
- Random splits may create imbalanced folds
- Some folds might have very few examples of rare classes
- Performance estimates become unreliable

**Solution:**
- Ensure each fold has approximately the same class distribution as the whole dataset

### When to Use

- Classification with imbalanced classes
- Small datasets
- Multiclass problems
- When class distribution matters

### Implementation

```python
from sklearn.model_selection import StratifiedKFold

# Stratified K-Fold
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=skfold, scoring='f1_weighted')
print(f"Stratified CV Score: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Comparison: Regular vs Stratified

```python
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np

# Imbalanced dataset
y_imbalanced = np.array([0]*90 + [1]*10)  # 90% class 0, 10% class 1
X_imbalanced = np.random.randn(100, 10)

# Regular K-Fold
kfold = KFold(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(kfold.split(X_imbalanced)):
    y_train, y_test = y_imbalanced[train_idx], y_imbalanced[test_idx]
    print(f"Fold {fold + 1}: Train class 1: {y_train.sum()/len(y_train):.1%}, "
          f"Test class 1: {y_test.sum()/len(y_test):.1%}")

print("\n" + "="*50 + "\n")

# Stratified K-Fold
skfold = StratifiedKFold(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(skfold.split(X_imbalanced, y_imbalanced)):
    y_train, y_test = y_imbalanced[train_idx], y_imbalanced[test_idx]
    print(f"Fold {fold + 1}: Train class 1: {y_train.sum()/len(y_train):.1%}, "
          f"Test class 1: {y_test.sum()/len(y_test):.1%}")
```

### Stratified for Regression

For regression, stratify based on binned target values:

```python
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Bin continuous target into categories
y_binned = np.digitize(y, bins=np.percentile(y, [25, 50, 75]))

# Use binned version for stratification
skfold = StratifiedKFold(n_splits=5)
for train_idx, test_idx in skfold.split(X, y_binned):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train and evaluate
```

## Leave-One-Out (LOO)

**Leave-One-Out Cross-Validation (LOOCV)** is K-Fold CV with $K = n$ (number of samples).

### Methodology

- For each sample $i$:
  - Train on all samples except $i$
  - Test on sample $i$
- Average performance across all $n$ iterations

$$CV_{LOO} = \frac{1}{n}\sum_{i=1}^{n} Error_i$$

### Implementation

```python
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error

loo = LeaveOneOut()
errors = []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    error = (y_test[0] - y_pred[0])**2
    errors.append(error)

mse = np.mean(errors)
print(f"LOOCV MSE: {mse:.3f}")
```

### Using cross_val_score

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')
mse = -scores.mean()
```

### Advantages and Disadvantages

**Advantages:**
- Maximum use of data (trains on n-1 samples)
- Deterministic (no randomness)
- Nearly unbiased estimate

**Disadvantages:**
- Computationally expensive ($n$ model fits)
- High variance in estimate
- Not suitable for large datasets
- Models are very similar (differ by only one sample)

**When to use:**
- Very small datasets (n < 100)
- Computational cost is acceptable
- Need unbiased estimate

### Leave-P-Out

Generalization: leave $p$ samples out instead of 1.

```python
from sklearn.model_selection import LeavePOut

lpo = LeavePOut(p=2)  # Leave 2 samples out
scores = cross_val_score(model, X, y, cv=lpo)

# Warning: number of iterations = C(n, p) = n!/(p!(n-p)!)
# For n=50, p=2: 1,225 iterations!
```

## Time Series Cross-Validation

For time series data, we must respect temporal order - cannot use future data to predict the past.

### TimeSeriesSplit

**Forward chaining** approach:
- Fold 1: train [1], test [2]
- Fold 2: train [1, 2], test [3]
- Fold 3: train [1, 2, 3], test [4]
- ...

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold + 1}:")
    print(f"  Train: {train_idx[0]:3d} to {train_idx[-1]:3d}")
    print(f"  Test:  {test_idx[0]:3d} to {test_idx[-1]:3d}")
```

### Visualization

```
Fold 1:  train [████████] test [██]
Fold 2:  train [████████████] test [██]
Fold 3:  train [████████████████] test [██]
Fold 4:  train [████████████████████] test [██]
Fold 5:  train [████████████████████████] test [██]
```

### Implementation

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)

print(f"Time Series CV Score: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")
```

### Blocked Time Series CV

For data with gaps or specific blocking:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, gap=10)
# gap: number of samples to exclude from end of train and beginning of test
```

### Walk-Forward Validation

```python
def walk_forward_validation(X, y, model, n_test=30):
    """
    Walk-forward validation for time series.

    Args:
        X: Features
        y: Target
        model: Model to evaluate
        n_test: Size of test set for each fold
    """
    predictions = []
    actuals = []

    # Start with minimum training size
    train_size = len(X) - len(X) // 3

    for i in range(train_size, len(X) - n_test):
        # Train on data up to i
        X_train, y_train = X[:i], y[:i]

        # Test on next n_test samples
        X_test, y_test = X[i:i+n_test], y[i:i+n_test]

        # Fit and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        predictions.extend(y_pred)
        actuals.extend(y_test)

    return np.array(actuals), np.array(predictions)
```

## Other Cross-Validation Methods

### Repeated K-Fold

Repeat K-Fold multiple times with different random splits:

```python
from sklearn.model_selection import RepeatedKFold

# 5-fold CV repeated 10 times = 50 model fits
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

scores = cross_val_score(model, X, y, cv=rkf)
print(f"Repeated CV: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

**Advantages:**
- More robust estimate
- Lower variance
- Better confidence intervals

**Disadvantages:**
- More computationally expensive

### Repeated Stratified K-Fold

```python
from sklearn.model_selection import RepeatedStratifiedKFold

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
scores = cross_val_score(model, X, y, cv=rskf, scoring='f1_weighted')
```

### Group K-Fold

When samples come from groups and we want to avoid data leakage:

```python
from sklearn.model_selection import GroupKFold

# Example: medical data from different patients
groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]  # Patient IDs

# Ensure samples from same patient are in same fold
gkf = GroupKFold(n_splits=4)

for train_idx, test_idx in gkf.split(X, y, groups):
    # Samples from same group never in both train and test
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

**Use cases:**
- Medical data (patient groups)
- Text data (document groups)
- Image data (scene groups)
- Any data with natural groupings

### Stratified Group K-Fold

Combines stratification and grouping:

```python
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5)
for train_idx, test_idx in sgkf.split(X, y, groups):
    # Both stratified by class and grouped
    pass
```

### Shuffle Split

Random permutation CV - alternative to K-Fold:

```python
from sklearn.model_selection import ShuffleSplit

# 10 iterations with 80/20 split
ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

scores = cross_val_score(model, X, y, cv=ss)
```

**Advantages:**
- Flexible train/test sizes
- Control over number of iterations
- Can create larger training sets

### Nested Cross-Validation

For unbiased hyperparameter tuning:

```python
from sklearn.model_selection import GridSearchCV, cross_val_score

# Outer CV for model evaluation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Inner CV for hyperparameter tuning
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Grid search with inner CV
param_grid = {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]}
grid_search = GridSearchCV(
    DecisionTreeClassifier(),
    param_grid,
    cv=inner_cv,
    scoring='accuracy'
)

# Outer CV evaluates the entire grid search process
nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv)
print(f"Nested CV Score: {nested_scores.mean():.3f} (+/- {nested_scores.std():.3f})")
```

## Practical Guidelines

### Choosing Cross-Validation Strategy

| Data Type | Recommended CV | Notes |
|-----------|---------------|-------|
| General classification | Stratified K-Fold (k=5 or 10) | Preserves class distribution |
| General regression | K-Fold (k=5 or 10) | Standard approach |
| Time series | TimeSeriesSplit | Respects temporal order |
| Small dataset (n < 100) | Leave-One-Out or k=n | Maximum data usage |
| Large dataset (n > 10,000) | Hold-out or k=3 | Computational efficiency |
| Imbalanced classes | Stratified K-Fold | Maintains class ratios |
| Grouped data | Group K-Fold | Prevents data leakage |
| Hyperparameter tuning | Nested CV | Unbiased evaluation |

### Common Pitfalls

**1. Data leakage:**

```python
# WRONG: Scale before split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses info from test set!
scores = cross_val_score(model, X_scaled, y, cv=5)

# CORRECT: Scale within each fold
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
scores = cross_val_score(pipeline, X, y, cv=5)
```

**2. Using test set for tuning:**

```python
# WRONG: Tune on test set
scores = []
for alpha in [0.1, 1.0, 10.0]:
    model = Ridge(alpha=alpha)
    score = cross_val_score(model, X_test, y_test, cv=5).mean()  # Using test set!
    scores.append(score)
best_alpha = [0.1, 1.0, 10.0][np.argmax(scores)]

# CORRECT: Use nested CV or separate validation set
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(Ridge(), {'alpha': [0.1, 1.0, 10.0]}, cv=5)
grid_search.fit(X_train, y_train)
best_alpha = grid_search.best_params_['alpha']
```

**3. Not shuffling when needed:**

```python
# If data is ordered (e.g., sorted by class)
# WRONG: Don't shuffle
kfold = KFold(n_splits=5, shuffle=False)  # May create imbalanced folds

# CORRECT: Shuffle
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
```

**4. Ignoring temporal order:**

```python
# For time series data
# WRONG: Use regular K-Fold
kfold = KFold(n_splits=5, shuffle=True)  # Uses future to predict past!

# CORRECT: Use TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

### Best Practices

**1. Use pipelines to prevent leakage:**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(k=10)),
    ('classifier', RandomForestClassifier())
])

scores = cross_val_score(pipeline, X, y, cv=5)
```

**2. Set random_state for reproducibility:**

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**3. Report mean and standard deviation:**

```python
scores = cross_val_score(model, X, y, cv=5)
print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
# ±2 std gives approximate 95% confidence interval
```

**4. Use appropriate metrics:**

```python
# For imbalanced classification
scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')

# For regression
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
```

**5. Save predictions from CV:**

```python
from sklearn.model_selection import cross_val_predict

# Get predictions for entire dataset
y_pred = cross_val_predict(model, X, y, cv=5)

# Now can compute any metric
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))
```

### Computational Considerations

**Parallelization:**

```python
# Use all CPU cores
scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)

# Or in GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
```

**Sample size per fold:**

Ensure each fold has enough samples:
- Minimum ~30 samples per fold
- For classification: at least 5-10 samples per class per fold

**Total number of model fits:**

- K-Fold: $K$ fits
- Repeated K-Fold: $K \times \text{n_repeats}$ fits
- Nested CV with Grid Search: $K_{outer} \times K_{inner} \times \text{n_params}$ fits

### Reporting Results

**Standard format:**

```python
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"Cross-Validation Results:")
print(f"  Metric: Accuracy")
print(f"  Folds: 5")
print(f"  Mean: {scores.mean():.4f}")
print(f"  Std: {scores.std():.4f}")
print(f"  95% CI: [{scores.mean() - 2*scores.std():.4f}, "
      f"{scores.mean() + 2*scores.std():.4f}]")
print(f"  Individual scores: {scores}")
```

## References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning
- Kohavi, R. (1995). A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection
- Bergmeir, C., & Benítez, J. M. (2012). On the Use of Cross-validation for Time Series Predictor Evaluation
