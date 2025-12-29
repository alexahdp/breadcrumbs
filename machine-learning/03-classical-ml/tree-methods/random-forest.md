# Random Forest

## Table of Contents

1. [Introduction](#introduction)
2. [Ensemble Learning](#ensemble-learning)
3. [Bagging (Bootstrap Aggregating)](#bagging-bootstrap-aggregating)
4. [Random Forest Algorithm](#random-forest-algorithm)
5. [Feature Importance](#feature-importance)
6. [Advantages and Disadvantages](#advantages-and-disadvantages)
7. [Practical Implementation](#practical-implementation)
8. [Hyperparameter Tuning](#hyperparameter-tuning)

## Introduction

**Random Forest** - an ensemble learning method that constructs multiple decision trees during training and outputs the mode (classification) or mean (regression) of the predictions from individual trees. It's one of the most powerful and widely used machine learning algorithms.

The "random" in random forest comes from two sources of randomness:
1. **Bootstrap sampling** - each tree is trained on a random subset of data
2. **Random feature selection** - at each split, only a random subset of features is considered

This randomness reduces overfitting and makes the model more robust compared to individual decision trees.

### Why Random Forest Works

Individual decision trees are prone to overfitting and high variance - small changes in training data can lead to completely different trees. Random forest addresses this by:
- Creating many diverse trees
- Averaging their predictions
- Reducing overall variance while maintaining low bias

## Ensemble Learning

**Ensemble Learning** - a machine learning paradigm where multiple models (often called "weak learners") are trained and combined to solve the same problem, typically producing better results than individual models.

### Types of Ensemble Methods

**Bagging (Bootstrap Aggregating)** - trains multiple models independently on different random subsets of data and aggregates their predictions:
- Reduces variance
- Works well with high-variance models (like decision trees)
- Examples: Random Forest, Bagged Decision Trees

**Boosting** - trains models sequentially, where each model tries to correct the errors of previous models:
- Reduces bias
- Works well with high-bias models
- Examples: AdaBoost, Gradient Boosting, XGBoost

**Stacking** - combines predictions from multiple models using another model (meta-learner):
- Can combine different types of models
- More complex to implement and tune

### Bias-Variance Trade-off in Ensembles

**Total Error** can be decomposed as:

$$Error = Bias^2 + Variance + Irreducible\ Error$$

- **Bagging** (Random Forest) primarily reduces **variance**
- **Boosting** primarily reduces **bias**
- Both can improve overall prediction accuracy

## Bagging (Bootstrap Aggregating)

**Bagging** - a technique that creates multiple versions of a predictor and uses them to get an aggregated predictor.

### Bootstrap Sampling

**Bootstrap** - a resampling technique where samples are drawn with replacement from the original dataset:

1. Given dataset $D$ with $n$ samples
2. Create bootstrap sample $D_i$ by randomly sampling $n$ samples from $D$ **with replacement**
3. Some original samples may appear multiple times, others may not appear at all
4. Each bootstrap sample is used to train one model

**Out-of-Bag (OOB) samples** - samples not selected in a bootstrap sample (approximately 37% of data):

$$P(\text{sample not selected}) = \left(1 - \frac{1}{n}\right)^n \approx \frac{1}{e} \approx 0.368$$

OOB samples can be used for validation without needing a separate validation set.

### Aggregation Methods

**Classification** - majority voting:

$$\hat{y} = \text{mode}\{h_1(x), h_2(x), ..., h_T(x)\}$$

Or using soft voting (class probabilities):

$$\hat{y} = \arg\max_c \sum_{t=1}^{T} P_t(y=c|x)$$

**Regression** - averaging:

$$\hat{y} = \frac{1}{T}\sum_{t=1}^{T} h_t(x)$$

where $T$ is the number of trees and $h_t(x)$ is the prediction of tree $t$.

## Random Forest Algorithm

### Algorithm Steps

**Training:**
1. Select $T$ (number of trees to build)
2. For each tree $t = 1, ..., T$:
   - Draw a bootstrap sample $D_t$ from training data $D$
   - Build a decision tree from $D_t$:
     - At each node, randomly select $m$ features from total $p$ features
     - Choose the best split among these $m$ features
     - Split the node into child nodes
     - Repeat until stopping criteria met (max depth, min samples, etc.)
   - Trees are grown without pruning (fully grown)

**Prediction:**
- **Classification**: Each tree votes for a class, final prediction is the majority vote
- **Regression**: Each tree predicts a value, final prediction is the average

### Key Parameters

**Number of trees ($T$ or `n_estimators`):**
- More trees → better performance (up to a point)
- More trees → longer training time
- Typically: 100-1000 trees
- No risk of overfitting with more trees

**Number of features per split ($m$ or `max_features`):**
- Default for classification: $m = \sqrt{p}$
- Default for regression: $m = \frac{p}{3}$
- Smaller $m$ → more randomness, more diversity, less correlation between trees
- Larger $m$ → stronger individual trees, but more correlation

**Tree depth (`max_depth`):**
- Deeper trees → more complex, prone to overfitting
- Typically left unrestricted in random forests (trees fully grown)
- Can limit for computational efficiency

**Minimum samples per leaf (`min_samples_leaf`):**
- Controls the minimum size of leaf nodes
- Larger values → more regularization
- Typical values: 1-10

### Out-of-Bag Error Estimation

Since each tree uses only ~63% of the data, the remaining ~37% (OOB samples) can be used for validation:

**OOB Error Algorithm:**
1. For each sample $x_i$, find all trees where $x_i$ was OOB
2. Make prediction using only those trees
3. Calculate error by comparing OOB predictions with true labels

**Benefits:**
- No need for separate validation set
- Provides unbiased estimate of test error
- Computed during training at no extra cost

**OOB Score:**

$$OOB\ Score = 1 - \frac{\sum_{i=1}^{n} I(y_i \neq \hat{y}_i^{OOB})}{n}$$

where $\hat{y}_i^{OOB}$ is the OOB prediction for sample $i$.

## Feature Importance

Random forests provide two main methods for measuring feature importance:

### Mean Decrease in Impurity (MDI)

**Gini Importance** (for classification) or **MSE Importance** (for regression):

$$Importance(f) = \frac{1}{T}\sum_{t=1}^{T}\sum_{node\ splits\ on\ f} \Delta impurity$$

For each tree:
- Track the impurity decrease when splitting on each feature
- Sum over all nodes where feature $f$ was used
- Average across all trees

**Characteristics:**
- Fast to compute (calculated during training)
- Biased toward high-cardinality features
- Can be misleading with correlated features

### Permutation Importance

**Permutation Feature Importance** - more robust method:

1. Train the model and record baseline performance
2. For each feature $f$:
   - Randomly shuffle values of feature $f$ in validation set
   - Measure performance degradation
   - Importance = baseline score - shuffled score
3. Optionally repeat multiple times and average

**Characteristics:**
- More reliable than MDI
- Can be computed on any model
- Computationally more expensive
- Works with validation/test data

Formula:

$$Importance(f) = score_{baseline} - score_{f\ permuted}$$

## Advantages and Disadvantages

### Advantages

✓ **Highly accurate** - one of the best performing algorithms
✓ **Reduces overfitting** - averages multiple trees
✓ **Handles large datasets** - efficient with high-dimensional data
✓ **No feature scaling required** - tree-based, not distance-based
✓ **Handles missing values** - can use surrogate splits
✓ **Works with mixed data types** - numerical and categorical
✓ **Provides feature importance** - helps with feature selection
✓ **OOB error estimation** - built-in cross-validation
✓ **Robust to outliers** - uses bootstrap sampling
✓ **Parallelizable** - trees can be trained independently
✓ **Few hyperparameters** - works well with default settings

### Disadvantages

✗ **Black box model** - less interpretable than single decision tree
✗ **Memory intensive** - stores multiple trees
✗ **Slower prediction** - must aggregate predictions from all trees
✗ **Poor extrapolation** - cannot predict beyond range of training data
✗ **Biased toward majority class** - in imbalanced datasets
✗ **Large model size** - can be problematic for deployment
✗ **No online learning** - cannot update model incrementally

## Practical Implementation

### Scikit-learn Classification Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Random Forest classifier
rf_clf = RandomForestClassifier(
    n_estimators=100,           # Number of trees
    criterion='gini',           # 'gini' or 'entropy'
    max_depth=None,             # Unlimited depth
    min_samples_split=2,        # Minimum samples to split
    min_samples_leaf=1,         # Minimum samples in leaf
    max_features='sqrt',        # Number of features per split: 'sqrt', 'log2', or int
    bootstrap=True,             # Use bootstrap sampling
    oob_score=True,             # Calculate OOB score
    n_jobs=-1,                  # Use all processors
    random_state=42,
    verbose=0
)

# Train the model
rf_clf.fit(X_train, y_train)

# OOB Score
print(f"OOB Score: {rf_clf.oob_score_:.4f}")

# Make predictions
y_pred = rf_clf.predict(X_test)
y_pred_proba = rf_clf.predict_proba(X_test)

# Evaluate
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

### Scikit-learn Regression Example

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Create Random Forest regressor
rf_reg = RandomForestRegressor(
    n_estimators=100,
    criterion='squared_error',  # 'squared_error', 'absolute_error', 'friedman_mse'
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=1.0,           # For regression: typically 1.0 or 0.33
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)

# Train
rf_reg.fit(X_train, y_train)

# OOB Score (R² on OOB samples)
print(f"OOB R² Score: {rf_reg.oob_score_:.4f}")

# Predict
y_pred = rf_reg.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
```

### Feature Importance Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Get feature importances (MDI)
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]

# Print feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. {feature_names[indices[f]]}: {importances[indices[f]]:.4f}")

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances (MDI)")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# Permutation Importance (more reliable)
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(
    rf_clf, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# Plot permutation importance
sorted_idx = perm_importance.importances_mean.argsort()[::-1]

plt.figure(figsize=(10, 6))
plt.boxplot(perm_importance.importances[sorted_idx].T,
            labels=[feature_names[i] for i in sorted_idx])
plt.xticks(rotation=90)
plt.title("Permutation Importance")
plt.tight_layout()
plt.show()
```

### Handling Imbalanced Data

```python
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter

# Check class distribution
print(f"Original class distribution: {Counter(y_train)}")

# Option 1: Use class_weight parameter
rf_balanced = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',    # Automatically adjust weights
    random_state=42
)

# Option 2: Use SMOTE for oversampling
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"Balanced class distribution: {Counter(y_train_balanced)}")

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_balanced, y_train_balanced)

# Option 3: Manual class weights
from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))

rf_weighted = RandomForestClassifier(
    n_estimators=100,
    class_weight=class_weight_dict,
    random_state=42
)
```

## Hyperparameter Tuning

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Use best model
best_rf = grid_search.best_estimator_
```

### Randomized Search (More Efficient)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define parameter distributions
param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None, 0.5, 0.7]
}

# Randomized search
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=100,              # Number of parameter combinations to try
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=2
)

random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.4f}")
```

### Learning Curves

```python
import matplotlib.pyplot as plt

# Vary number of trees
n_estimators_range = range(10, 201, 10)
oob_scores = []
test_scores = []

for n in n_estimators_range:
    rf = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    oob_scores.append(rf.oob_score_)
    test_scores.append(rf.score(X_test, y_test))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, oob_scores, label='OOB Score')
plt.plot(n_estimators_range, test_scores, label='Test Score')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Random Forest: Score vs Number of Trees')
plt.legend()
plt.grid(True)
plt.show()
```

## Comparison with Other Methods

### Random Forest vs Single Decision Tree

| Aspect | Decision Tree | Random Forest |
|--------|--------------|---------------|
| Variance | High | Low (averaged) |
| Overfitting | Prone | Resistant |
| Interpretability | High | Low |
| Training time | Fast | Slower |
| Prediction time | Very fast | Slower |
| Accuracy | Lower | Higher |

### Random Forest vs Gradient Boosting

| Aspect | Random Forest | Gradient Boosting |
|--------|---------------|-------------------|
| Training | Parallel | Sequential |
| Speed | Faster training | Slower training |
| Overfitting | More resistant | Requires careful tuning |
| Hyperparameters | Fewer, easier | More, harder to tune |
| Performance | Very good | Often slightly better |

## References

- Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.
- Breiman, L. (1996). Bagging predictors. Machine learning, 24(2), 123-140.
- [Scikit-learn Random Forest Documentation](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.
- Louppe, G. (2014). Understanding random forests: From theory to practice. arXiv preprint arXiv:1407.7502.
