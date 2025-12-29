# Logistic Regression

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Binary Logistic Regression](#binary-logistic-regression)
4. [Multiclass Classification](#multiclass-classification)
5. [Training and Optimization](#training-and-optimization)
6. [Model Evaluation](#model-evaluation)
7. [Practical Considerations](#practical-considerations)
8. [Implementation](#implementation)
9. [References](#references)

## Introduction

**Logistic Regression** is a statistical method for binary classification that models the probability of an instance belonging to a particular class. Despite its name containing "regression," it's a classification algorithm.

### Key Characteristics

- **Linear decision boundary** - separates classes with a linear function
- **Probabilistic output** - returns probability of class membership (0 to 1)
- **Fast training** - efficient for large datasets
- **Interpretable** - coefficients show feature importance and direction of influence

### When to Use

**Best suited for:**
- Binary classification problems
- Linearly separable data
- When probability estimates are needed
- Baseline model for classification tasks

**Not recommended for:**
- Complex non-linear relationships (use kernel methods or tree-based models)
- Multi-modal class distributions
- Very high-dimensional data with complex interactions

## Mathematical Foundation

### Sigmoid Function (Logistic Function)

The **sigmoid function** maps any real-valued number to the (0, 1) range:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Properties:
- Domain: $(-\infty, +\infty)$
- Range: $(0, 1)$
- $\sigma(0) = 0.5$
- $\sigma(-z) = 1 - \sigma(z)$
- Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

### Odds and Log-Odds

**Odds** - the ratio of the probability of success to the probability of failure:

$$odds = \frac{p}{1-p}$$

**Log-Odds (Logit)** - the natural logarithm of odds:

$$logit(p) = \ln\left(\frac{p}{1-p}\right)$$

Logistic regression models the log-odds as a linear function of features.

## Binary Logistic Regression

### Model Formulation

The probability that instance $x$ belongs to the positive class:

$$P(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$

where:
- $x$ is the feature vector
- $w$ is the weight vector
- $b$ is the bias term
- $\sigma$ is the sigmoid function

The linear combination $z = w^T x + b$ is called the **logit** or **log-odds**.

### Decision Boundary

The decision boundary is where $P(y=1|x) = 0.5$:

$$w^T x + b = 0$$

This defines a hyperplane in feature space. Classification rule:
- If $w^T x + b > 0$: predict class 1
- If $w^T x + b < 0$: predict class 0
- If $w^T x + b = 0$: on the decision boundary

### Loss Function (Log Loss)

**Binary Cross-Entropy Loss** (Log Loss):

$$L(w, b) = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

where:
- $y_i$ is the true label (0 or 1)
- $\hat{y}_i = \sigma(w^T x_i + b)$ is the predicted probability
- $n$ is the number of training examples

For a single example:
- If $y = 1$: loss = $-\log(\hat{y})$
- If $y = 0$: loss = $-\log(1 - \hat{y})$

Properties:
- Convex function (has global minimum)
- Heavily penalizes confident wrong predictions
- Approaches infinity as predictions approach wrong class

## Multiclass Classification

### One-vs-Rest (OvR)

Train $K$ binary classifiers, one for each class:
- Classifier $k$ distinguishes class $k$ from all other classes
- Prediction: choose class with highest probability

$$\hat{y} = \arg\max_{k} P(y=k|x)$$

### Multinomial Logistic Regression (Softmax Regression)

Generalizes logistic regression to multiple classes using the **softmax function**:

$$P(y=k|x) = \frac{e^{w_k^T x + b_k}}{\sum_{j=1}^{K} e^{w_j^T x + b_j}}$$

where:
- $K$ is the number of classes
- $w_k, b_k$ are weights and bias for class $k$

Properties:
- Outputs sum to 1
- Each output is between 0 and 1
- Generalizes sigmoid (when $K=2$)

**Cross-Entropy Loss** for multiclass:

$$L = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})$$

where $y_{ik}$ is 1 if sample $i$ belongs to class $k$, else 0.

## Training and Optimization

### Gradient Descent

The gradient of the log loss with respect to weights:

$$\frac{\partial L}{\partial w_j} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)x_{ij}$$

Update rule:

$$w_j := w_j - \alpha \frac{\partial L}{\partial w_j}$$

where $\alpha$ is the learning rate.

### Optimization Algorithms

**Gradient Descent Variants:**
- **Batch Gradient Descent** - uses entire dataset
- **Stochastic Gradient Descent (SGD)** - uses one sample at a time
- **Mini-batch Gradient Descent** - uses small batches

**Advanced Optimizers:**
- **LBFGS** - quasi-Newton method, good for small datasets
- **Newton-CG** - conjugate gradient method
- **SAG/SAGA** - stochastic average gradient methods

### Regularization

**L2 Regularization (Ridge):**

$$L_{L2} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right] + \lambda \sum_{j=1}^{p} w_j^2$$

**L1 Regularization (Lasso):**

$$L_{L1} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right] + \lambda \sum_{j=1}^{p} |w_j|$$

**Elastic Net:**

$$L_{elastic} = L + \lambda_1 \sum_{j=1}^{p} |w_j| + \lambda_2 \sum_{j=1}^{p} w_j^2$$

where $\lambda$ (or $C = 1/\lambda$) controls regularization strength.

Effects:
- **L2**: shrinks all coefficients, handles multicollinearity
- **L1**: performs feature selection, creates sparse models
- **Elastic Net**: combines both benefits

## Model Evaluation

### Predicted Probabilities

Logistic regression outputs probabilities:

```python
# Get probability estimates
proba = model.predict_proba(X)[:, 1]  # Probability of positive class
```

Default threshold is 0.5, but can be adjusted based on:
- Cost of false positives vs false negatives
- Class imbalance
- Business requirements

### Threshold Tuning

Choose threshold $t$ to optimize for specific metric:

$$\hat{y} = \begin{cases} 1 & \text{if } P(y=1|x) \geq t \\ 0 & \text{otherwise} \end{cases}$$

**Common approaches:**
- ROC curve analysis
- Precision-recall trade-off
- Cost-sensitive threshold selection

### Interpretation of Coefficients

For feature $x_j$ with coefficient $w_j$:

**Odds Ratio:**

$$OR = e^{w_j}$$

Interpretation:
- $OR > 1$: feature increases odds of positive class
- $OR < 1$: feature decreases odds of positive class
- $OR = 1$: no effect

A one-unit increase in $x_j$ multiplies the odds by $e^{w_j}$.

## Practical Considerations

### Feature Scaling

**Required** for logistic regression:
- Features on different scales affect convergence speed
- Regularization penalizes large-scale features more

Common methods:
- **Standardization**: $z = \frac{x - \mu}{\sigma}$
- **Min-Max scaling**: $x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$

### Handling Class Imbalance

**Techniques:**
1. **Class weights** - penalize misclassification of minority class more:
   ```python
   model = LogisticRegression(class_weight='balanced')
   ```

2. **Resampling**:
   - Oversample minority class (SMOTE)
   - Undersample majority class

3. **Threshold adjustment** - lower threshold for minority class

### Multicollinearity

**Problem:** Highly correlated features lead to:
- Unstable coefficient estimates
- High variance in predictions
- Difficult interpretation

**Solutions:**
- Remove correlated features (VIF > 10)
- Use L2 regularization
- Apply PCA for dimensionality reduction

### Assumptions

Logistic regression assumes:
1. **Binary outcome** (for binary classification)
2. **Independence** of observations
3. **Linear relationship** between log-odds and features
4. **No perfect multicollinearity**
5. **Large sample size** (rule of thumb: 10-20 events per predictor)

## Implementation

### Scikit-learn Example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(
    penalty='l2',           # L2 regularization
    C=1.0,                  # Inverse regularization strength
    solver='lbfgs',         # Optimization algorithm
    max_iter=1000,          # Maximum iterations
    class_weight='balanced' # Handle imbalance
)

model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
print(classification_report(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Coefficients
print(f"Coefficients: {model.coef_[0]}")
print(f"Intercept: {model.intercept_[0]}")
```

### Custom Threshold

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

# Find optimal threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

# Apply custom threshold
y_pred_custom = (y_proba >= optimal_threshold).astype(int)
```

### Multiclass Example

```python
# Multinomial logistic regression
model_multi = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000
)

model_multi.fit(X_train_scaled, y_train)

# One-vs-Rest
model_ovr = LogisticRegression(
    multi_class='ovr',
    solver='lbfgs',
    max_iter=1000
)

model_ovr.fit(X_train_scaled, y_train)
```

### Regularization Comparison

```python
from sklearn.model_selection import GridSearchCV

# Grid search for best regularization
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']  # Supports both L1 and L2
}

grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    cv=5,
    scoring='roc_auc'
)

grid_search.fit(X_train_scaled, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

## References

- [Scikit-learn Logistic Regression Documentation](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- An Introduction to Statistical Learning (James, Witten, Hastie, Tibshirani)
- The Elements of Statistical Learning (Hastie, Tibshirani, Friedman)
- Pattern Recognition and Machine Learning (Christopher Bishop)
