# Decision Trees

## Table of Contents

1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Splitting Criteria](#splitting-criteria)
4. [Decision Tree Algorithms](#decision-tree-algorithms)
5. [Pruning Techniques](#pruning-techniques)
6. [Advantages and Disadvantages](#advantages-and-disadvantages)
7. [Practical Implementation](#practical-implementation)

## Introduction

**Decision Tree** - a non-parametric supervised learning algorithm used for both classification and regression tasks. It creates a tree-like model of decisions based on feature values, where each internal node represents a test on a feature, each branch represents the outcome of the test, and each leaf node represents a class label (classification) or continuous value (regression).

Decision trees mimic human decision-making processes and are highly interpretable, making them one of the most popular machine learning algorithms.

### How Decision Trees Work

The algorithm recursively partitions the feature space into regions, making splits that maximize the separation between classes (classification) or minimize prediction error (regression). The process continues until a stopping criterion is met.

**Key terminology:**
- **Root node** - the topmost node representing the entire dataset
- **Internal node** - a node with at least one child, representing a decision rule
- **Leaf node (terminal node)** - a node with no children, representing the final prediction
- **Depth** - the length of the longest path from root to leaf
- **Branch** - a subsection of the entire tree

## Core Concepts

### Information Gain

Decision trees aim to create the most homogeneous subsets possible. To measure homogeneity, various impurity measures are used:

**Entropy** - a measure of randomness or uncertainty in a dataset:

$$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

where:
- $S$ is the dataset
- $c$ is the number of classes
- $p_i$ is the proportion of samples belonging to class $i$

**Information Gain** - the reduction in entropy achieved by splitting on a particular attribute:

$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

where:
- $A$ is the attribute being considered
- $Values(A)$ is the set of all possible values for attribute $A$
- $S_v$ is the subset of $S$ where attribute $A$ has value $v$

The attribute with the highest information gain is selected for splitting.

### Gini Impurity

**Gini Index** - measures the probability of incorrectly classifying a randomly chosen element:

$$Gini(S) = 1 - \sum_{i=1}^{c} p_i^2$$

where $p_i$ is the probability of class $i$.

Properties:
- $Gini = 0$: perfect purity (all samples belong to one class)
- $Gini = 0.5$: maximum impurity for binary classification (equal distribution)
- Generally: lower Gini index indicates better split

**Gini Gain** for a split:

$$Gini_{split}(S, A) = \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Gini(S_v)$$

The attribute with the lowest Gini impurity after splitting is selected.

### Regression Trees

For regression tasks, different criteria are used:

**Mean Squared Error (MSE)** - measures prediction error:

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \bar{y})^2$$

where $\bar{y}$ is the mean of target values in the node.

**Variance Reduction** - the split that achieves the maximum reduction in variance:

$$Var_{reduction} = Var(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Var(S_v)$$

## Splitting Criteria

### Continuous Variables

For continuous features, decision trees consider all possible threshold values and select the one that maximizes information gain or minimizes impurity.

Algorithm:
1. Sort all unique values of the feature
2. Consider midpoints between consecutive values as potential thresholds
3. For each threshold, calculate the splitting criterion
4. Select the threshold with the best criterion value

**Example:** For feature values [1.2, 2.5, 3.1, 4.0], test thresholds: 1.85, 2.8, 3.55

### Categorical Variables

For categorical features, the algorithm considers different groupings of categories:

- **Binary trees**: Split into two groups
- **Multi-way split**: Create one branch for each category value

### Missing Values

Common strategies for handling missing values:
- **Surrogate splits**: Use alternative features that correlate with the primary split
- **Probabilistic splitting**: Send instances down multiple branches with weights
- **Separate branch**: Create a dedicated branch for missing values

## Decision Tree Algorithms

### ID3 (Iterative Dichotomiser 3)

**ID3** - one of the earliest decision tree algorithms, developed by Ross Quinlan in 1986.

**Characteristics:**
- Uses **entropy** and **information gain** for splitting
- Only handles categorical features
- Creates multi-way splits (one branch per category value)
- No pruning mechanism (prone to overfitting)
- Cannot handle missing values

**Algorithm steps:**
1. Calculate entropy of the target variable
2. For each feature, calculate information gain
3. Select feature with highest information gain
4. Create a branch for each value of the selected feature
5. Recursively repeat for each branch until:
   - All samples in a node belong to the same class
   - No features remain for splitting
   - No samples remain

### C4.5

**C4.5** - an improved version of ID3, also developed by Ross Quinlan (1993).

**Improvements over ID3:**
- Handles both continuous and categorical features
- Uses **gain ratio** instead of information gain to avoid bias toward features with many values
- Implements tree pruning (post-pruning with error-based pruning)
- Handles missing values
- Can handle attributes with different costs

**Gain Ratio** - normalized information gain:

$$GainRatio(S, A) = \frac{IG(S, A)}{SplitInfo(S, A)}$$

where:

$$SplitInfo(S, A) = -\sum_{v \in Values(A)} \frac{|S_v|}{|S|} \log_2\left(\frac{|S_v|}{|S|}\right)$$

The split information penalizes attributes that split data into many small partitions.

### C5.0

**C5.0** - further improvement of C4.5 with better performance:
- Faster training
- More memory efficient
- Smaller trees with comparable or better accuracy
- Support for boosting
- Weighted attributes

### CART (Classification and Regression Trees)

**CART** - developed by Breiman et al. (1984), the most widely used decision tree algorithm today.

**Characteristics:**
- Handles both classification and regression
- Uses **Gini index** for classification
- Uses **variance reduction** for regression
- Creates **binary splits** only (easier to interpret)
- Includes cost-complexity pruning
- Handles missing values with surrogate splits

**Binary splitting:** For categorical variables with $k$ categories, CART considers $2^{k-1} - 1$ possible binary partitions.

**Cost-complexity pruning parameter ($\alpha$):**

$$R_\alpha(T) = R(T) + \alpha |T|$$

where:
- $R(T)$ is the total misclassification error
- $|T|$ is the number of terminal nodes
- $\alpha$ controls the trade-off between tree size and accuracy

## Pruning Techniques

**Pruning** - the process of removing branches from a decision tree to reduce overfitting and improve generalization.

### Pre-pruning (Early Stopping)

Stop growing the tree early based on conditions:
- **Maximum depth**: limit tree depth to a fixed value
- **Minimum samples per split**: require minimum number of samples to allow splitting
- **Minimum samples per leaf**: require minimum number of samples in each leaf
- **Maximum number of leaf nodes**: limit total number of leaves
- **Minimum impurity decrease**: only split if impurity reduction exceeds threshold

**Advantages:** Computationally efficient, prevents unnecessary splits

**Disadvantages:** May stop too early and miss important patterns

### Post-pruning (Reduced Error Pruning)

Grow the full tree, then remove branches that don't improve validation accuracy:

1. Build the complete tree
2. Evaluate each internal node:
   - Replace subtree with a leaf
   - Check if validation error decreases
3. Keep the pruned version if it performs better
4. Repeat until no improvement is possible

**Cost-Complexity Pruning (used in CART):**
- Create a sequence of trees by progressively increasing $\alpha$
- Select the tree with lowest cross-validation error

## Advantages and Disadvantages

### Advantages

✓ **Easy to interpret and visualize** - tree structure mirrors human decision-making
✓ **Requires little data preparation** - no need for normalization or scaling
✓ **Handles both numerical and categorical data**
✓ **Non-parametric** - no assumptions about data distribution
✓ **Captures non-linear relationships** automatically
✓ **Feature importance** - can identify most important features
✓ **Robust to outliers** - splitting based on thresholds, not distances
✓ **Fast prediction** - O(log n) complexity for balanced trees

### Disadvantages

✗ **Prone to overfitting** - especially with deep trees
✗ **Unstable** - small changes in data can drastically change tree structure
✗ **Biased toward features with more levels** - especially in ID3
✗ **Not optimal** - greedy algorithm doesn't guarantee globally optimal tree
✗ **Poor extrapolation** - cannot predict beyond the range of training data
✗ **Axis-parallel splits only** - cannot capture diagonal decision boundaries naturally
✗ **Imbalanced datasets** - biased toward majority class

## Practical Implementation

### Scikit-learn Example (Classification)

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

# Load and split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create decision tree classifier
clf = DecisionTreeClassifier(
    criterion='gini',           # 'gini' or 'entropy'
    max_depth=5,                # Maximum depth of the tree
    min_samples_split=20,       # Minimum samples required to split
    min_samples_leaf=10,        # Minimum samples required in a leaf
    max_features=None,          # Number of features to consider for best split
    random_state=42
)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Visualize the tree
plt.figure(figsize=(20, 10))
tree.plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names)
plt.show()

# Feature importance
importances = clf.feature_importances_
for i, imp in enumerate(importances):
    print(f"{feature_names[i]}: {imp:.4f}")
```

### Scikit-learn Example (Regression)

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create decision tree regressor
reg = DecisionTreeRegressor(
    criterion='squared_error',  # or 'friedman_mse', 'absolute_error'
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

# Train the model
reg.fit(X_train, y_train)

# Make predictions
y_pred = reg.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy']
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Use best model
best_clf = grid_search.best_estimator_
```

### Dealing with Overfitting

```python
# Compare different max_depth values
depths = range(1, 20)
train_scores = []
test_scores = []

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)

    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test, y_test))

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, label='Training accuracy')
plt.plot(depths, test_scores, label='Testing accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Decision Tree: Training vs Testing Accuracy')
plt.show()
```

## References

- Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). Classification and regression trees. CRC press.
- Quinlan, J. R. (1986). Induction of decision trees. Machine learning, 1(1), 81-106.
- Quinlan, J. R. (1993). C4.5: programs for machine learning. Morgan Kaufmann.
- [Scikit-learn Decision Trees Documentation](https://scikit-learn.org/stable/modules/tree.html)
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.
