# Boosting Algorithms

## Table of Contents

1. [Introduction to Boosting](#introduction-to-boosting)
2. [AdaBoost](#adaboost)
3. [Gradient Boosting](#gradient-boosting)
4. [XGBoost](#xgboost)
5. [LightGBM](#lightgbm)
6. [CatBoost](#catboost)
7. [Comparison of Boosting Methods](#comparison-of-boosting-methods)
8. [Practical Implementation](#practical-implementation)
9. [Hyperparameter Tuning](#hyperparameter-tuning)

## Introduction to Boosting

**Boosting** - an ensemble technique that sequentially trains weak learners (typically shallow decision trees), where each new model attempts to correct the errors made by previous models. The final prediction is a weighted combination of all models.

### Key Concepts

**Weak Learner** - a model that performs slightly better than random guessing. In boosting, typically shallow decision trees (depth 1-6) are used as weak learners.

**Sequential Learning** - unlike bagging (Random Forest) where models are trained independently in parallel, boosting trains models sequentially, with each model learning from the mistakes of previous models.

**Weighted Combination** - the final prediction is a weighted sum of predictions from all weak learners:

$$F(x) = \sum_{m=1}^{M} \alpha_m h_m(x)$$

where:
- $M$ is the number of weak learners
- $h_m(x)$ is the prediction of the $m$-th weak learner
- $\alpha_m$ is the weight assigned to the $m$-th learner

### Boosting vs Bagging

| Aspect | Bagging (Random Forest) | Boosting |
|--------|------------------------|----------|
| Training | Parallel | Sequential |
| Focus | Reduce variance | Reduce bias |
| Base learners | Deep trees | Shallow trees (stumps) |
| Independence | Independent | Dependent |
| Overfitting risk | Lower | Higher (if not regularized) |
| Speed | Faster (parallelizable) | Slower (sequential) |

## AdaBoost

**AdaBoost (Adaptive Boosting)** - one of the first and most influential boosting algorithms, developed by Freund and Schapire (1996).

### Algorithm

**Key idea:** Iteratively train weak learners on weighted versions of the data, giving more weight to misclassified examples.

**Steps:**

1. Initialize sample weights: $w_i^{(1)} = \frac{1}{n}$ for all $i = 1, ..., n$

2. For each iteration $m = 1, ..., M$:

   a. Train weak learner $h_m$ on weighted data

   b. Calculate weighted error:

   $$\epsilon_m = \frac{\sum_{i=1}^{n} w_i^{(m)} \cdot I(y_i \neq h_m(x_i))}{\sum_{i=1}^{n} w_i^{(m)}}$$

   c. Calculate learner weight:

   $$\alpha_m = \frac{1}{2}\ln\left(\frac{1-\epsilon_m}{\epsilon_m}\right)$$

   d. Update sample weights:

   $$w_i^{(m+1)} = w_i^{(m)} \cdot e^{-\alpha_m y_i h_m(x_i)}$$

   e. Normalize weights: $w_i^{(m+1)} = \frac{w_i^{(m+1)}}{\sum_{j=1}^{n} w_j^{(m+1)}}$

3. Final prediction:

$$H(x) = \text{sign}\left(\sum_{m=1}^{M} \alpha_m h_m(x)\right)$$

### Characteristics

- **Adaptively** increases weights on misclassified examples
- Works well with weak learners (even decision stumps)
- Can achieve high accuracy with simple base learners
- Sensitive to noisy data and outliers
- Less prone to overfitting than other boosting methods (but still possible)

## Gradient Boosting

**Gradient Boosting** - a generalization of boosting that frames the problem as gradient descent in function space.

### Core Idea

Instead of adjusting sample weights, gradient boosting fits new models to the **residuals** (errors) of the previous models.

**Goal:** Minimize a loss function $L(y, F(x))$ by iteratively adding models:

$$F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)$$

where:
- $F_m(x)$ is the ensemble prediction after $m$ iterations
- $\nu$ is the learning rate (shrinkage parameter)
- $h_m(x)$ is the new weak learner

### Algorithm

**Steps:**

1. Initialize model with a constant value:

$$F_0(x) = \arg\min_{\gamma} \sum_{i=1}^{n} L(y_i, \gamma)$$

2. For each iteration $m = 1, ..., M$:

   a. Compute pseudo-residuals (negative gradient):

   $$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

   b. Fit a weak learner $h_m(x)$ to residuals $r_{im}$

   c. Compute step size $\gamma_m$:

   $$\gamma_m = \arg\min_{\gamma} \sum_{i=1}^{n} L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i))$$

   d. Update model:

   $$F_m(x) = F_{m-1}(x) + \nu \cdot \gamma_m h_m(x)$$

3. Final prediction: $F_M(x)$

### Loss Functions

**Regression:**
- **Squared Error (L2)**: $L(y, F) = \frac{1}{2}(y - F)^2$
  - Gradient: $-r = F - y$ (residual)
- **Absolute Error (L1)**: $L(y, F) = |y - F|$
  - More robust to outliers
- **Huber Loss**: Combination of L2 and L1
  - Smooth near zero, linear for large errors

**Classification:**
- **Log Loss (Binary)**: $L(y, F) = \log(1 + e^{-yF})$
- **Exponential Loss**: $L(y, F) = e^{-yF}$ (used in AdaBoost)
- **Multinomial Deviance**: For multiclass classification

### Regularization in Gradient Boosting

**Learning Rate (Shrinkage)** - $\nu \in (0, 1]$:
- Smaller learning rate requires more trees but often gives better performance
- Typical values: 0.01 - 0.3

**Subsampling (Stochastic Gradient Boosting)**:
- Train each tree on a random subset of data
- Reduces overfitting and speeds up training
- Typical values: 0.5 - 0.8

**Tree constraints:**
- **Max depth**: typically 3-10
- **Min samples per leaf**: prevents overfitting
- **Max features**: random feature selection

## XGBoost

**XGBoost (Extreme Gradient Boosting)** - an optimized and highly efficient implementation of gradient boosting, developed by Tianqi Chen (2014).

### Key Innovations

**Regularized Objective Function:**

$$Obj^{(t)} = \sum_{i=1}^{n} L(y_i, \hat{y}_i^{(t)}) + \sum_{k=1}^{t} \Omega(f_k)$$

where the regularization term is:

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

Parameters:
- $T$ is the number of leaves
- $w_j$ is the score in leaf $j$
- $\gamma$ controls minimum loss reduction for split
- $\lambda$ is L2 regularization on leaf weights

**Second-Order Taylor Approximation:**

Uses both first and second derivatives of the loss function:

$$Obj^{(t)} \approx \sum_{i=1}^{n} [g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \Omega(f_t)$$

where:
- $g_i = \frac{\partial L(y_i, \hat{y}^{(t-1)})}{\partial \hat{y}^{(t-1)}}$ (first derivative/gradient)
- $h_i = \frac{\partial^2 L(y_i, \hat{y}^{(t-1)})}{\partial (\hat{y}^{(t-1)})^2}$ (second derivative/Hessian)

This leads to faster and more accurate optimization.

### Features

✓ **Regularization** - L1 and L2 regularization on weights
✓ **Sparsity awareness** - handles missing values automatically
✓ **Weighted quantile sketch** - efficient approximate tree learning
✓ **Column block** - parallel tree construction
✓ **Cache awareness** - optimized data structure for speed
✓ **Out-of-core computing** - handles data larger than memory
✓ **Built-in cross-validation** - easy model tuning
✓ **Early stopping** - stops when validation score doesn't improve
✓ **Tree pruning** - uses max_depth and prunes backwards
✓ **Parallel processing** - multi-threading support

### Important Parameters

**Tree parameters:**
- `max_depth` (default=6): Maximum tree depth
- `min_child_weight` (default=1): Minimum sum of instance weight in child
- `gamma` (default=0): Minimum loss reduction for split
- `subsample` (default=1): Fraction of samples for each tree
- `colsample_bytree` (default=1): Fraction of features for each tree

**Regularization:**
- `lambda` (default=1): L2 regularization
- `alpha` (default=0): L1 regularization

**Learning:**
- `learning_rate` (eta, default=0.3): Step size shrinkage
- `n_estimators` (default=100): Number of boosting rounds

## LightGBM

**LightGBM (Light Gradient Boosting Machine)** - a fast, distributed, high-performance gradient boosting framework by Microsoft (2017).

### Key Innovations

**Gradient-based One-Side Sampling (GOSS):**

Traditional gradient boosting uses all data instances. GOSS keeps all instances with large gradients and performs random sampling on instances with small gradients.

**Idea:** Instances with small gradients are already well-fitted and contribute less to information gain.

**Algorithm:**
1. Sort instances by absolute gradient values
2. Keep top $a \times 100\%$ instances with largest gradients
3. Randomly sample $b \times 100\%$ from remaining instances
4. Amplify sampled instances by factor $\frac{1-a}{b}$ when computing information gain

**Exclusive Feature Bundling (EFB):**

Bundles mutually exclusive features (features that rarely take non-zero values simultaneously) to reduce number of features.

**Example:** In sparse features (like one-hot encoded categorical variables), many features are mutually exclusive.

**Benefits:**
- Reduces feature dimension
- Speeds up training without sacrificing accuracy

**Leaf-wise Tree Growth:**

Unlike level-wise (breadth-first) growth in most boosting algorithms, LightGBM grows trees leaf-wise (best-first).

**Strategy:**
- Split the leaf with maximum delta loss
- Can achieve lower loss with same number of splits
- Risks overfitting with deep trees (use `max_depth` to control)

**Comparison:**

```
Level-wise (XGBoost):          Leaf-wise (LightGBM):
        [1]                            [1]
       /   \                          /   \
     [2]   [3]                      [2]   [5]
    /  \   /  \                    /  \
  [4] [5][6] [7]                 [3]  [4]
```

### Features

✓ **Faster training** - up to 20x faster than traditional GBDT
✓ **Lower memory usage** - optimized memory consumption
✓ **Better accuracy** - leaf-wise growth can achieve better accuracy
✓ **Handles large datasets** - distributed learning support
✓ **GPU support** - can train on GPU
✓ **Categorical features** - native support without encoding
✓ **Network communication** - supports parallel learning

### Important Parameters

**Core:**
- `num_leaves` (default=31): Max number of leaves (main parameter)
- `max_depth` (default=-1): Limit depth to prevent overfitting
- `learning_rate` (default=0.1): Shrinkage rate
- `n_estimators` (default=100): Number of boosting rounds

**GOSS:**
- `top_rate` (default=0.2): Retain ratio for large gradient data
- `other_rate` (default=0.1): Retain ratio for small gradient data

**Speed:**
- `feature_fraction` (default=1.0): Randomly select fraction of features
- `bagging_fraction` (default=1.0): Randomly select fraction of data
- `bagging_freq` (default=0): Frequency for bagging

**Categorical:**
- `categorical_feature`: Specify categorical features by index or name

## CatBoost

**CatBoost (Categorical Boosting)** - gradient boosting library by Yandex (2017), designed to handle categorical features efficiently.

### Key Innovations

**Ordered Boosting:**

Traditional gradient boosting suffers from **prediction shift** - the same data used for calculating residuals is used for fitting the model, leading to overfitting.

**Solution:** Use different subsets of data for calculating residuals and fitting trees.

**Algorithm:**
- Maintains multiple models with different random permutations of data
- For each sample, gradient is computed using only previous samples in permutation
- Reduces target leakage and overfitting

**Ordered Target Statistics:**

For categorical features, instead of standard target encoding (which causes target leakage):

$$\hat{x}_k^i = \frac{\sum_{j=1}^{n} [x_j^k = x_i^k] \cdot y_j}{\sum_{j=1}^{n} [x_j^k = x_i^k]}$$

CatBoost uses **ordered target statistics**:

$$\hat{x}_k^i = \frac{\sum_{j=1}^{p-1} [x_{\sigma(j)}^k = x_{\sigma(p)}^k] \cdot y_{\sigma(j)} + a \cdot P}{\sum_{j=1}^{p-1} [x_{\sigma(j)}^k = x_{\sigma(p)}^k] + a}$$

where:
- $\sigma$ is a random permutation
- $p$ is the position of current sample in permutation
- $a$ is prior weight
- $P$ is prior value (usually average target)

This uses only previous examples in the permutation, preventing target leakage.

**Oblivious Trees (Symmetric Trees):**

CatBoost builds oblivious decision trees where the same splitting criterion is used across an entire level.

**Structure:**
- All nodes at the same depth use the same feature and threshold
- Creates balanced trees
- Faster prediction (can use lookup tables)
- Better generalization
- Less prone to overfitting

### Features

✓ **Categorical features** - automatic handling without preprocessing
✓ **Robust to overfitting** - ordered boosting reduces target leakage
✓ **Fast prediction** - oblivious trees enable efficient prediction
✓ **GPU support** - fast training on GPU
✓ **No preprocessing needed** - handles missing values and categorical features
✓ **Great with defaults** - often works well with default parameters
✓ **Text features** - can handle text directly
✓ **Visualization tools** - built-in plotting and model analysis

### Important Parameters

**Tree:**
- `depth` (default=6): Depth of trees
- `l2_leaf_reg` (default=3.0): L2 regularization
- `model_size_reg` (default=0.5): Model size regularization

**Learning:**
- `learning_rate` (default=auto): Shrinkage rate
- `iterations` (default=1000): Number of trees
- `early_stopping_rounds`: Stop if no improvement

**Categorical:**
- `cat_features`: List of categorical feature indices
- `one_hot_max_size` (default=2): Use one-hot encoding for small cardinality

**Sampling:**
- `subsample` (default=0.66): Sample rate for bagging
- `rsm` (default=1): Random subspace method (feature sampling)

## Comparison of Boosting Methods

### Performance Comparison

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| **Speed** | Fast | Very Fast | Moderate |
| **Accuracy** | High | High | Very High |
| **Categorical handling** | Manual encoding | Basic support | Excellent (native) |
| **Memory usage** | Moderate | Low | Moderate |
| **Overfitting resistance** | Good | Moderate | Excellent |
| **Default parameters** | Needs tuning | Needs tuning | Good defaults |
| **GPU support** | Yes | Yes | Yes |
| **Large datasets** | Good | Excellent | Good |
| **Small datasets** | Good | May overfit | Excellent |

### When to Use Each

**XGBoost:**
- Industry standard, well-tested
- Good balance of speed and accuracy
- Extensive documentation and community
- When you need proven reliability

**LightGBM:**
- Very large datasets (millions of samples)
- When speed is critical
- High-dimensional data
- When you have computational constraints

**CatBoost:**
- Many categorical features
- Small to medium datasets
- When you want good results with minimal tuning
- When overfitting is a concern
- Time-series data with categorical features

## Practical Implementation

### XGBoost Example

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classification
xgb_clf = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    objective='binary:logistic',    # 'multi:softmax' for multiclass
    booster='gbtree',                # 'gbtree', 'gblinear', 'dart'
    gamma=0,                         # Min loss reduction for split
    min_child_weight=1,
    subsample=0.8,                   # Row sampling
    colsample_bytree=0.8,            # Column sampling
    reg_alpha=0,                     # L1 regularization
    reg_lambda=1,                    # L2 regularization
    random_state=42,
    n_jobs=-1
)

# Train with evaluation set
eval_set = [(X_train, y_train), (X_test, y_test)]
xgb_clf.fit(
    X_train, y_train,
    eval_set=eval_set,
    eval_metric='logloss',           # 'auc', 'error', 'logloss'
    early_stopping_rounds=10,
    verbose=True
)

# Predict
y_pred = xgb_clf.predict(X_test)
y_pred_proba = xgb_clf.predict_proba(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Regression
xgb_reg = xgb.XGBRegressor(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    objective='reg:squarederror',
    random_state=42
)

xgb_reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10)
y_pred = xgb_reg.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# Feature importance
import matplotlib.pyplot as plt

xgb.plot_importance(xgb_clf, max_num_features=10)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()
```

### LightGBM Example

```python
import lightgbm as lgb
from sklearn.metrics import accuracy_score

# Classification
lgb_clf = lgb.LGBMClassifier(
    num_leaves=31,
    max_depth=-1,
    learning_rate=0.1,
    n_estimators=100,
    objective='binary',              # 'multiclass' for multiclass
    boosting_type='gbdt',            # 'gbdt', 'dart', 'goss', 'rf'
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=1,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1
)

# Train with evaluation
lgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='logloss',
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

y_pred = lgb_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Using native API for more control
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

gbm = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[test_data],
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

# Regression
lgb_reg = lgb.LGBMRegressor(
    num_leaves=31,
    learning_rate=0.1,
    n_estimators=100,
    objective='regression',
    random_state=42
)

lgb_reg.fit(X_train, y_train, eval_set=[(X_test, y_test)])

# Feature importance
lgb.plot_importance(lgb_clf, max_num_features=10)
plt.title("LightGBM Feature Importance")
plt.show()
```

### CatBoost Example

```python
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import accuracy_score

# Identify categorical features
cat_features = ['category_col1', 'category_col2']  # Column names or indices

# Classification
cat_clf = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',         # 'MultiClass' for multiclass
    eval_metric='Accuracy',
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=50,                       # Print every 50 iterations
    early_stopping_rounds=10,
    use_best_model=True
)

# Fit with categorical features (no encoding needed!)
cat_clf.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    plot=True                         # Show training progress plot
)

y_pred = cat_clf.predict(X_test)
y_pred_proba = cat_clf.predict_proba(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Using Pool for better performance
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

cat_clf.fit(train_pool, eval_set=test_pool)

# Regression
cat_reg = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    loss_function='RMSE',
    random_seed=42,
    verbose=False
)

cat_reg.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test)
)

# Feature importance
feature_importance = cat_clf.get_feature_importance(train_pool)
feature_names = X_train.columns if hasattr(X_train, 'columns') else range(X_train.shape[1])

import pandas as pd
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df.head(10))

# Visualize
cat_clf.plot_tree(tree_idx=0, pool=train_pool)
```

## Hyperparameter Tuning

### XGBoost Tuning Strategy

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Step 1: Fix learning rate and estimate n_estimators
xgb_clf = xgb.XGBClassifier(learning_rate=0.1, random_state=42)
xgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50)
optimal_trees = xgb_clf.best_iteration

# Step 2: Tune tree-specific parameters
param_grid_1 = {
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5]
}
grid_search = GridSearchCV(
    xgb.XGBClassifier(n_estimators=optimal_trees, learning_rate=0.1, random_state=42),
    param_grid_1,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

# Step 3: Tune regularization
param_grid_2 = {
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

# Step 4: Tune sampling parameters
param_grid_3 = {
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
}

# Step 5: Lower learning rate and increase trees
final_model = xgb.XGBClassifier(
    **best_params,
    learning_rate=0.01,
    n_estimators=optimal_trees * 10,
    random_state=42
)
```

### Cross-Validation with Early Stopping

```python
# XGBoost CV
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {'max_depth': 6, 'eta': 0.1, 'objective': 'binary:logistic'}

cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=1000,
    nfold=5,
    metrics='auc',
    early_stopping_rounds=10,
    seed=42
)

print(f"Best iteration: {cv_results.shape[0]}")
print(f"Best score: {cv_results['test-auc-mean'].max():.4f}")

# LightGBM CV
train_data = lgb.Dataset(X_train, label=y_train)
params = {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt'}

cv_results = lgb.cv(
    params,
    train_data,
    num_boost_round=1000,
    nfold=5,
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

# CatBoost CV
cv_params = cat_clf.get_params()
cv_data = Pool(X_train, y_train, cat_features=cat_features)

cv_results = cat_clf.cv(
    cv_data,
    cv_params,
    fold_count=5,
    early_stopping_rounds=10,
    plot=True
)
```

### Optuna for Advanced Hyperparameter Tuning

```python
import optuna

def objective(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'random_state': 42
    }

    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best parameters: {study.best_params}")
print(f"Best score: {study.best_value:.4f}")
```

## Best Practices

### Preventing Overfitting

1. **Use cross-validation** for hyperparameter tuning
2. **Early stopping** on validation set
3. **Regularization**: L1/L2, min_child_weight, gamma
4. **Reduce model complexity**: lower max_depth, increase min_child_weight
5. **Sampling**: subsample rows and columns
6. **Lower learning rate** with more iterations

### Feature Engineering for Boosting

```python
# Boosting benefits from good features
import pandas as pd
import numpy as np

# Interaction features
X['feature1_x_feature2'] = X['feature1'] * X['feature2']
X['feature1_div_feature2'] = X['feature1'] / (X['feature2'] + 1e-5)

# Binning continuous features
X['age_binned'] = pd.cut(X['age'], bins=[0, 18, 35, 50, 100], labels=['young', 'adult', 'middle', 'senior'])

# Target encoding for categorical (with cross-validation to prevent leakage)
from category_encoders import TargetEncoder

encoder = TargetEncoder()
X_encoded = encoder.fit_transform(X['category'], y)

# Aggregations for grouped data
grouped_stats = df.groupby('category')['value'].agg(['mean', 'std', 'min', 'max'])
X = X.merge(grouped_stats, left_on='category', right_index=True)
```

### Handling Imbalanced Data

```python
# Option 1: Class weights
xgb_clf = xgb.XGBClassifier(scale_pos_weight=len(y[y==0]) / len(y[y==1]))

# Option 2: Focal loss (custom objective)
def focal_loss(y_pred, dtrain):
    y_true = dtrain.get_label()
    gamma = 2.0
    alpha = 0.25
    p = 1 / (1 + np.exp(-y_pred))
    grad = -(y_true - p) * (alpha * (1 - p) ** gamma - (1 - alpha) * p ** gamma)
    hess = p * (1 - p) * ((y_true - p) * gamma * (alpha * (1 - p) ** (gamma - 1) + (1 - alpha) * p ** (gamma - 1)) +
                          alpha * (1 - p) ** gamma + (1 - alpha) * p ** gamma)
    return grad, hess

# Option 3: SMOTE + Boosting
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_train, y_train)

# Option 4: Adjust decision threshold
y_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]
optimal_threshold = 0.3  # Find using ROC curve
y_pred = (y_pred_proba >= optimal_threshold).astype(int)
```

## References

- Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. Journal of computer and system sciences, 55(1), 119-139.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of statistics, 1189-1232.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD.
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. NeurIPS.
- Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. NeurIPS.
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)
