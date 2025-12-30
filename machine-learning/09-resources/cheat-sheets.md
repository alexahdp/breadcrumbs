# Machine Learning Cheat Sheets

Quick reference guides for common machine learning concepts, formulas, and workflows.

## Table of Contents

1. [Evaluation Metrics](#evaluation-metrics)
2. [Loss Functions](#loss-functions)
3. [Activation Functions](#activation-functions)
4. [Optimization Algorithms](#optimization-algorithms)
5. [Regularization Techniques](#regularization-techniques)
6. [Distance Metrics](#distance-metrics)
7. [Statistical Tests](#statistical-tests)
8. [Data Preprocessing](#data-preprocessing)
9. [Model Selection Guide](#model-selection-guide)
10. [Hyperparameter Ranges](#hyperparameter-ranges)

## Evaluation Metrics

### Regression Metrics

| Metric | Formula | Range | Best Value | Notes |
|--------|---------|-------|------------|-------|
| MSE | $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ | $[0, \infty)$ | 0 | Sensitive to outliers |
| RMSE | $\sqrt{MSE}$ | $[0, \infty)$ | 0 | Same units as target |
| MAE | $\frac{1}{n}\sum_{i=1}^{n}\|y_i - \hat{y}_i\|$ | $[0, \infty)$ | 0 | Robust to outliers |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | $(-\infty, 1]$ | 1 | Can be negative |
| MAPE | $\frac{100\%}{n}\sum_{i=1}^{n}\|\frac{y_i - \hat{y}_i}{y_i}\|$ | $[0, \infty)$ | 0 | Percentage error |

### Classification Metrics

| Metric | Formula | Range | Best Value | Use Case |
|--------|---------|-------|------------|----------|
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | $[0, 1]$ | 1 | Balanced classes |
| Precision | $\frac{TP}{TP + FP}$ | $[0, 1]$ | 1 | Minimize false positives |
| Recall | $\frac{TP}{TP + FN}$ | $[0, 1]$ | 1 | Minimize false negatives |
| F1 Score | $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$ | $[0, 1]$ | 1 | Balanced measure |
| Specificity | $\frac{TN}{TN + FP}$ | $[0, 1]$ | 1 | True negative rate |
| AUC-ROC | Area under ROC curve | $[0, 1]$ | 1 | Overall performance |

### Confusion Matrix Reference

|  | Predicted Positive | Predicted Negative |
|---|-------------------|-------------------|
| **Actually Positive** | True Positive (TP) | False Negative (FN) |
| **Actually Negative** | False Positive (FP) | True Negative (TN) |

## Loss Functions

### Regression Loss Functions

**Mean Squared Error (MSE)**
$$L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$
- Use when: Large errors should be heavily penalized
- Properties: Differentiable, convex

**Mean Absolute Error (MAE)**
$$L = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$
- Use when: Robust to outliers needed
- Properties: Less sensitive to outliers

**Huber Loss**
$$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \leq \delta \\ \delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$
- Use when: Balance between MSE and MAE
- Properties: Combines benefits of both

### Classification Loss Functions

**Binary Cross-Entropy**
$$L = -\frac{1}{n}\sum_{i=1}^{n}[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$
- Use when: Binary classification
- Output: Sigmoid activation

**Categorical Cross-Entropy**
$$L = -\sum_{i=1}^{n}\sum_{j=1}^{C} y_{ij} \log(\hat{y}_{ij})$$
- Use when: Multi-class classification (one-hot encoded)
- Output: Softmax activation

**Sparse Categorical Cross-Entropy**
- Same as categorical but accepts integer labels instead of one-hot
- Use when: Many classes, memory efficiency needed

**Hinge Loss (SVM)**
$$L = \sum_{i=1}^{n} \max(0, 1 - y_i \cdot \hat{y}_i)$$
- Use when: Support Vector Machines
- Properties: Margin-based loss

## Activation Functions

| Function | Formula | Range | Derivative | Use Case |
|----------|---------|-------|------------|----------|
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | $(0, 1)$ | $\sigma(x)(1-\sigma(x))$ | Binary output |
| **Tanh** | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1, 1)$ | $1 - tanh^2(x)$ | Hidden layers (RNN) |
| **ReLU** | $\max(0, x)$ | $[0, \infty)$ | $\begin{cases}1 & x>0\\0 & x\leq0\end{cases}$ | Default for hidden layers |
| **Leaky ReLU** | $\max(0.01x, x)$ | $(-\infty, \infty)$ | $\begin{cases}1 & x>0\\0.01 & x\leq0\end{cases}$ | Avoid dying ReLU |
| **ELU** | $\begin{cases}x & x>0\\\alpha(e^x-1) & x\leq0\end{cases}$ | $(-\alpha, \infty)$ | $\begin{cases}1 & x>0\\ELU(x)+\alpha & x\leq0\end{cases}$ | Better than ReLU |
| **Softmax** | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | $(0, 1)$ | Complex | Multi-class output |
| **Swish** | $x \cdot \sigma(x)$ | $(-\infty, \infty)$ | $\sigma(x) + x\sigma(x)(1-\sigma(x))$ | Recent alternative |

### Choosing Activation Functions

- **Output layer (regression)**: Linear
- **Output layer (binary classification)**: Sigmoid
- **Output layer (multi-class)**: Softmax
- **Hidden layers (default)**: ReLU
- **Hidden layers (better performance)**: Leaky ReLU, ELU, Swish
- **RNN/LSTM**: Tanh (hidden state), Sigmoid (gates)

## Optimization Algorithms

### Comparison Table

| Optimizer | Update Rule | Learning Rate | Memory | Convergence | Use Case |
|-----------|-------------|---------------|---------|-------------|----------|
| **SGD** | $\theta = \theta - \alpha \nabla L$ | Required | Low | Slow | Simple problems |
| **SGD + Momentum** | $v = \beta v - \alpha \nabla L$<br>$\theta = \theta + v$ | Required | Low | Faster | General use |
| **RMSprop** | $s = \beta s + (1-\beta)(\nabla L)^2$<br>$\theta = \theta - \frac{\alpha}{\sqrt{s+\epsilon}}\nabla L$ | Adaptive | Medium | Good | RNNs |
| **Adam** | Combines momentum + RMSprop | Adaptive | Medium | Fast | Default choice |
| **AdaGrad** | Accumulates squared gradients | Adaptive | Medium | Can slow down | Sparse data |
| **AdaDelta** | Extension of AdaGrad | No LR needed | Medium | Good | Alternative to Adam |

### Hyperparameter Recommendations

**SGD with Momentum**
- Learning rate: 0.01 - 0.1
- Momentum: 0.9 - 0.99

**Adam (most common)**
- Learning rate: 0.001 (default)
- β₁: 0.9
- β₂: 0.999
- ε: 1e-8

**RMSprop**
- Learning rate: 0.001
- ρ: 0.9

## Regularization Techniques

### L1 vs L2 Regularization

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|------------|------------|
| **Formula** | $L + \lambda \sum_i \|\theta_i\|$ | $L + \lambda \sum_i \theta_i^2$ |
| **Effect** | Sparse weights (feature selection) | Small distributed weights |
| **Derivative** | Non-differentiable at 0 | Always differentiable |
| **Use when** | Many irrelevant features | All features potentially relevant |

### Dropout

**Training time:**
- Randomly set neurons to 0 with probability $p$ (typically 0.2-0.5)
- Scale remaining neurons by $\frac{1}{1-p}$

**Inference time:**
- Use all neurons without dropout

**Typical dropout rates:**
- Input layer: 0.1 - 0.2
- Hidden layers: 0.3 - 0.5
- Larger networks can handle higher dropout

### Early Stopping

Monitor validation loss and stop when:
- Validation loss stops decreasing
- Patience parameter reached (e.g., 10 epochs)
- Save best model based on validation performance

### Batch Normalization

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

Benefits:
- Faster training
- Allows higher learning rates
- Reduces internal covariate shift
- Acts as regularization

## Distance Metrics

| Metric | Formula | Use Case | Properties |
|--------|---------|----------|------------|
| **Euclidean** | $\sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$ | General purpose | Sensitive to scale |
| **Manhattan** | $\sum_{i=1}^{n}\|x_i - y_i\|$ | Grid-like paths | Less sensitive to outliers |
| **Cosine** | $1 - \frac{x \cdot y}{\\|x\\| \\|y\\|}$ | Text, high-dim | Scale-invariant |
| **Minkowski** | $(\sum_{i=1}^{n}\|x_i - y_i\|^p)^{1/p}$ | Generalization | p=1: Manhattan, p=2: Euclidean |
| **Hamming** | Count of differing positions | Categorical data | Binary/categorical |
| **Jaccard** | $1 - \frac{\|A \cap B\|}{\|A \cup B\|}$ | Sets, binary features | Similarity measure |

## Statistical Tests

### Choosing the Right Test

| Data Type | Purpose | Test |
|-----------|---------|------|
| Continuous vs Continuous | Correlation | Pearson / Spearman |
| Categorical vs Categorical | Independence | Chi-square |
| Two groups (continuous) | Mean difference | t-test |
| Multiple groups (continuous) | Mean differences | ANOVA |
| Non-normal distribution | Compare medians | Mann-Whitney U |
| Paired samples | Mean difference | Paired t-test |
| Normality check | Distribution | Shapiro-Wilk |

### Common Significance Levels

- **α = 0.05**: Standard (95% confidence)
- **α = 0.01**: Strict (99% confidence)
- **α = 0.10**: Exploratory (90% confidence)

### Multiple Testing Correction

**Bonferroni**: $\alpha_{new} = \frac{\alpha}{m}$ where $m$ is number of tests
- Very conservative
- Use when tests are independent

**FDR (False Discovery Rate)**: Controls proportion of false positives
- Less conservative than Bonferroni
- Better for many tests

## Data Preprocessing

### Scaling Methods

| Method | Formula | Range | Use When |
|--------|---------|-------|----------|
| **Normalization (Min-Max)** | $\frac{x - x_{min}}{x_{max} - x_{min}}$ | $[0, 1]$ | Bounded distributions, neural networks |
| **Standardization (Z-score)** | $\frac{x - \mu}{\sigma}$ | Unbounded | Normal distribution, many algorithms |
| **Robust Scaling** | $\frac{x - Q_2}{Q_3 - Q_1}$ | Unbounded | Presence of outliers |
| **Max Abs Scaling** | $\frac{x}{\|x\|_{max}}$ | $[-1, 1]$ | Sparse data |

### Handling Missing Data

| Method | Description | When to Use |
|--------|-------------|-------------|
| **Drop rows** | Remove samples with missing values | Few missing values, large dataset |
| **Drop columns** | Remove features with missing values | High % missing (>50%) |
| **Mean/Median imputation** | Replace with mean or median | MCAR, numerical data |
| **Mode imputation** | Replace with most frequent value | Categorical data |
| **Forward/Backward fill** | Use previous/next value | Time series data |
| **KNN imputation** | Use k-nearest neighbors | Structured relationships |
| **Model-based** | Predict missing values | Complex patterns |

### Handling Categorical Data

**Label Encoding**: Map categories to integers (0, 1, 2, ...)
- Use for: Ordinal data, tree-based models

**One-Hot Encoding**: Create binary column for each category
- Use for: Nominal data, linear models, neural networks
- Warning: High cardinality creates many features

**Target Encoding**: Replace category with target mean
- Use for: High cardinality features
- Warning: Risk of overfitting (use cross-validation)

**Frequency Encoding**: Replace with frequency of occurrence
- Use for: High cardinality, when frequency matters

## Model Selection Guide

### Regression

| Problem | Model | Pros | Cons |
|---------|-------|------|------|
| Linear relationship | Linear Regression | Simple, interpretable | Assumes linearity |
| Non-linear, small data | Polynomial Regression | Flexible | Overfitting risk |
| High-dimensional | Ridge/Lasso | Regularization | Feature engineering needed |
| Complex patterns | Random Forest | Robust, non-linear | Less interpretable |
| Best performance | XGBoost/LightGBM | State-of-art | Hyperparameter tuning |
| Very complex | Neural Network | Universal approximator | Needs large data |

### Classification

| Problem | Model | Pros | Cons |
|---------|-------|------|------|
| Linear separable | Logistic Regression | Simple, probabilistic | Linear decision boundary |
| Small data | Naive Bayes | Fast, works with little data | Independence assumption |
| Clear margin | SVM | Effective in high-dim | Slow on large data |
| Non-linear | Decision Tree | Interpretable | Overfitting |
| Robust performance | Random Forest | Accurate, robust | Black box |
| Best performance | XGBoost/LightGBM | State-of-art | Tuning required |
| Images, text | Neural Network | Best for complex data | Needs large data |

### Clustering

| Problem | Algorithm | Pros | Cons |
|---------|-----------|------|------|
| Spherical clusters | K-Means | Fast, simple | Needs K, spherical only |
| Hierarchical structure | Hierarchical | No K needed, dendrogram | Slow on large data |
| Arbitrary shapes | DBSCAN | Finds any shape, outliers | Sensitive to parameters |
| High-dimensional | Spectral Clustering | Works with complex structures | Computationally expensive |

## Hyperparameter Ranges

### Common Hyperparameters

**Random Forest**
- `n_estimators`: [100, 500, 1000]
- `max_depth`: [10, 20, 30, None]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]
- `max_features`: ['sqrt', 'log2', None]

**XGBoost**
- `n_estimators`: [100, 500, 1000]
- `learning_rate`: [0.01, 0.05, 0.1, 0.3]
- `max_depth`: [3, 5, 7, 9]
- `subsample`: [0.6, 0.8, 1.0]
- `colsample_bytree`: [0.6, 0.8, 1.0]

**Neural Networks**
- `learning_rate`: [1e-4, 1e-3, 1e-2]
- `batch_size`: [16, 32, 64, 128]
- `hidden_units`: [64, 128, 256, 512]
- `dropout_rate`: [0.2, 0.3, 0.5]
- `weight_decay`: [1e-5, 1e-4, 1e-3]

**SVM**
- `C`: [0.1, 1, 10, 100]
- `kernel`: ['linear', 'rbf', 'poly']
- `gamma`: ['scale', 'auto', 0.001, 0.01, 0.1]

**K-Means**
- `n_clusters`: [2, 5, 10, 20] (depends on data)
- `init`: ['k-means++', 'random']
- `n_init`: [10, 20, 50]

## Python Quick Reference

### Scikit-learn Pipeline Template

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', YourModel())
])

# Hyperparameter tuning
param_grid = {
    'model__param1': [value1, value2],
    'model__param2': [value3, value4]
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='accuracy'
)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
```

### Cross-Validation Template

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X, y, cv=5, scoring='accuracy'
)
print(f"Mean: {scores.mean():.3f}, Std: {scores.std():.3f}")
```

### Feature Importance

```python
# Tree-based models
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(X.shape[1]):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.3f}")
```

## References

- Scikit-learn Documentation: https://scikit-learn.org/
- XGBoost Documentation: https://xgboost.readthedocs.io/
- Deep Learning Book (Goodfellow et al.)
- Hands-On Machine Learning (Géron)
