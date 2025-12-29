# ROC Curve and AUC

## Table of Contents

1. [Overview](#overview)
2. [ROC Curve](#roc-curve)
3. [AUC Score](#auc-score)
4. [Precision-Recall Curve](#precision-recall-curve)
5. [Comparing Models](#comparing-models)
6. [Threshold Selection](#threshold-selection)
7. [Multi-Class ROC](#multi-class-roc)

## Overview

**ROC (Receiver Operating Characteristic) Curve** and **AUC (Area Under the Curve)** are threshold-independent metrics for evaluating binary classifiers. Unlike accuracy, precision, or recall which depend on a specific classification threshold, ROC-AUC evaluates performance across all possible thresholds.

**Key advantages:**
- Threshold-independent evaluation
- Robust to class imbalance
- Allows comparison of different models
- Provides insight into trade-offs between true and false positive rates

## ROC Curve

### Definition

**ROC Curve** - a graph showing the diagnostic ability of a binary classifier as its discrimination threshold is varied. It plots:

- **Y-axis**: True Positive Rate (TPR) = Recall = Sensitivity
- **X-axis**: False Positive Rate (FPR) = 1 - Specificity

$$TPR = \frac{TP}{TP + FN} = \frac{\text{True Positives}}{\text{Actual Positives}}$$

$$FPR = \frac{FP}{FP + TN} = \frac{\text{False Positives}}{\text{Actual Negatives}}$$

### How It Works

Most classifiers output probabilities rather than direct classifications:

```python
# Classifier outputs probabilities
probabilities = [0.9, 0.8, 0.6, 0.4, 0.3, 0.1]

# Different thresholds give different classifications
threshold_0.5 = [1, 1, 1, 0, 0, 0]  # Standard threshold
threshold_0.7 = [1, 1, 0, 0, 0, 0]  # Higher threshold (fewer positives)
threshold_0.3 = [1, 1, 1, 1, 1, 0]  # Lower threshold (more positives)
```

For each threshold, we can compute TPR and FPR, creating one point on the ROC curve.

### Creating a ROC Curve

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Example data
y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.7, 0.3, 0.85, 0.15])

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Compute AUC
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()
```

### Interpreting the ROC Curve

**Key points on the curve:**

- **(0, 0)**: Predict everything as negative (threshold = ∞)
  - TPR = 0 (no true positives detected)
  - FPR = 0 (no false positives)

- **(1, 1)**: Predict everything as positive (threshold = -∞)
  - TPR = 1 (all positives detected)
  - FPR = 1 (all negatives misclassified)

- **Diagonal line (0,0) to (1,1)**: Random classifier
  - No discrimination ability
  - 50% chance of correct classification

- **Top-left corner (0, 1)**: Perfect classifier
  - TPR = 1 (all positives detected)
  - FPR = 0 (no false positives)

**Curve positions:**
- **Above diagonal**: Better than random
- **On diagonal**: Random guessing
- **Below diagonal**: Worse than random (invert predictions!)
- **Closer to top-left**: Better classifier

### Understanding Trade-offs

Moving along the ROC curve represents changing the classification threshold:

```python
# Show threshold trade-offs
for i in range(0, len(thresholds), max(1, len(thresholds)//5)):
    print(f"Threshold: {thresholds[i]:.2f}")
    print(f"  TPR: {tpr[i]:.2f} (sensitivity)")
    print(f"  FPR: {fpr[i]:.2f} (1 - specificity)")
    print()
```

**Practical meaning:**
- **Lower threshold**: More predictions as positive → Higher TPR, Higher FPR
- **Higher threshold**: Fewer predictions as positive → Lower TPR, Lower FPR

## AUC Score

### Definition

**AUC (Area Under the ROC Curve)** - a single scalar value summarizing the ROC curve performance:

$$AUC = \int_0^1 TPR(FPR) \, d(FPR)$$

**Properties:**
- Ranges from 0 to 1
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random classifier
- AUC < 0.5: Worse than random (flip predictions)
- AUC = 0.0: Perfectly wrong classifier

### Probabilistic Interpretation

**AUC represents the probability that the classifier will rank a randomly chosen positive instance higher than a randomly chosen negative instance.**

```python
from sklearn.metrics import roc_auc_score

y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.7, 0.3, 0.85, 0.15])

auc_score = roc_auc_score(y_true, y_scores)
print(f"AUC: {auc_score:.3f}")
```

### AUC Interpretation Guidelines

| AUC Range | Interpretation |
|-----------|----------------|
| 0.90 - 1.00 | Excellent discrimination |
| 0.80 - 0.90 | Good discrimination |
| 0.70 - 0.80 | Fair discrimination |
| 0.60 - 0.70 | Poor discrimination |
| 0.50 - 0.60 | Fail (barely better than random) |
| < 0.50 | Worse than random |

### When to Use AUC

**Use AUC when:**
- You need a single metric for model comparison
- Class distribution is imbalanced
- You want threshold-independent evaluation
- Cost of FP and FN are unknown or variable
- Evaluating multiple models

**Don't use AUC when:**
- You need to select a specific operating threshold
- You care about performance at a specific point
- Precision and recall are more important (use PR-AUC instead)
- You have highly imbalanced data (PR-AUC is better)

### Advantages of AUC

1. **Threshold-independent**: Evaluates all possible thresholds
2. **Scale-invariant**: Measures rank predictions, not absolute values
3. **Classification-threshold-invariant**: Measures quality of predictions regardless of threshold
4. **Robust to imbalance**: Works well with imbalanced datasets

### Disadvantages of AUC

1. **Not always desirable**: Sometimes false positives and false negatives have very different costs
2. **Obscures performance details**: Single number hides threshold-specific trade-offs
3. **Misleading for imbalanced data**: Can be overly optimistic (use PR-AUC instead)
4. **Doesn't match deployment metric**: You still need to choose a threshold

## Precision-Recall Curve

### Definition

**Precision-Recall (PR) Curve** - alternative to ROC curve, especially useful for imbalanced datasets. It plots:

- **Y-axis**: Precision = $\frac{TP}{TP + FP}$
- **X-axis**: Recall = $\frac{TP}{TP + FN}$

### Why Use PR Curves?

For imbalanced datasets with few positives:
- ROC curves can be overly optimistic
- Large number of TN can dominate FPR
- PR curve focuses on positive class performance

```python
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.7, 0.3, 0.85, 0.15])

# Compute PR curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# Compute average precision (PR-AUC)
ap_score = average_precision_score(y_true, y_scores)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2,
         label=f'PR curve (AP = {ap_score:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.show()
```

### PR-AUC (Average Precision)

**Average Precision (AP)** - area under the PR curve:

$$AP = \sum_n (R_n - R_{n-1}) P_n$$

where $P_n$ and $R_n$ are precision and recall at threshold $n$.

```python
# Calculate PR-AUC
from sklearn.metrics import average_precision_score, auc

# Method 1: Average Precision (preferred)
ap = average_precision_score(y_true, y_scores)

# Method 2: Area under PR curve
precision, recall, _ = precision_recall_curve(y_true, y_scores)
pr_auc = auc(recall, precision)

print(f"Average Precision: {ap:.3f}")
print(f"PR-AUC: {pr_auc:.3f}")
```

### ROC-AUC vs PR-AUC

**Use ROC-AUC when:**
- Classes are relatively balanced
- You care about both classes equally
- Need standard metric for comparison

**Use PR-AUC when:**
- Highly imbalanced datasets
- Positive class is more important
- FPR is less meaningful (too many negatives)

**Example with imbalanced data:**

```python
# Imbalanced dataset: 5% positive, 95% negative
from sklearn.metrics import roc_auc_score, average_precision_score

# Create imbalanced dataset
n_samples = 1000
n_positive = 50  # 5%
n_negative = 950  # 95%

y_true = np.array([1]*n_positive + [0]*n_negative)
y_scores = np.concatenate([
    np.random.beta(8, 2, n_positive),    # Scores for positives
    np.random.beta(2, 8, n_negative)     # Scores for negatives
])

roc_auc = roc_auc_score(y_true, y_scores)
pr_auc = average_precision_score(y_true, y_scores)

print(f"ROC-AUC: {roc_auc:.3f}")  # Often high even for poor models
print(f"PR-AUC:  {pr_auc:.3f}")   # More informative for imbalanced data
```

## Comparing Models

### Comparing Multiple ROC Curves

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC(probability=True)
}

plt.figure(figsize=(10, 8))

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)

    # Get predictions
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.plot(fpr, tpr, lw=2,
             label=f'{name} (AUC = {roc_auc:.2f})')

# Add random classifier line
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Model Comparison')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()
```

### Statistical Significance

Compare AUC scores with confidence intervals:

```python
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

def bootstrap_auc(y_true, y_scores, n_bootstraps=1000):
    """Calculate AUC with confidence interval using bootstrap"""
    aucs = []

    for i in range(n_bootstraps):
        # Bootstrap sample
        indices = resample(range(len(y_true)), replace=True)
        y_true_boot = y_true[indices]
        y_scores_boot = y_scores[indices]

        # Calculate AUC
        auc = roc_auc_score(y_true_boot, y_scores_boot)
        aucs.append(auc)

    # Calculate confidence interval
    aucs = np.array(aucs)
    lower = np.percentile(aucs, 2.5)
    upper = np.percentile(aucs, 97.5)
    mean = np.mean(aucs)

    return mean, lower, upper

# Usage
mean_auc, lower_ci, upper_ci = bootstrap_auc(y_true, y_scores)
print(f"AUC: {mean_auc:.3f} (95% CI: [{lower_ci:.3f}, {upper_ci:.3f}])")
```

## Threshold Selection

### Finding Optimal Threshold

The "optimal" threshold depends on your specific use case:

#### 1. Youden's Index (Maximize TPR - FPR)

```python
def find_optimal_threshold_youden(y_true, y_scores):
    """Find threshold that maximizes Youden's index (TPR - FPR)"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Youden's index
    j_scores = tpr - fpr

    # Find optimal threshold
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold, tpr[optimal_idx], fpr[optimal_idx]

threshold, tpr, fpr = find_optimal_threshold_youden(y_true, y_scores)
print(f"Optimal threshold: {threshold:.3f}")
print(f"TPR: {tpr:.3f}, FPR: {fpr:.3f}")
```

#### 2. Closest to Top-Left Corner

```python
def find_optimal_threshold_topleft(y_true, y_scores):
    """Find threshold closest to top-left corner (0, 1)"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Distance to top-left corner
    distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)

    # Find minimum distance
    optimal_idx = np.argmin(distances)
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold, tpr[optimal_idx], fpr[optimal_idx]
```

#### 3. Cost-Based Threshold

```python
def find_cost_based_threshold(y_true, y_scores, cost_fp=1, cost_fn=1):
    """Find threshold that minimizes cost"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Calculate costs at each threshold
    n_positive = np.sum(y_true == 1)
    n_negative = np.sum(y_true == 0)

    fnr = 1 - tpr  # False Negative Rate

    costs = (fpr * n_negative * cost_fp +
             fnr * n_positive * cost_fn)

    # Find minimum cost
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold, costs[optimal_idx]

# Example: FN is 10x more costly than FP
threshold, cost = find_cost_based_threshold(y_true, y_scores,
                                           cost_fp=1, cost_fn=10)
print(f"Optimal threshold: {threshold:.3f} (total cost: {cost:.0f})")
```

#### 4. Desired Sensitivity/Specificity

```python
def find_threshold_for_sensitivity(y_true, y_scores, min_sensitivity=0.95):
    """Find threshold that achieves minimum sensitivity"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Find thresholds where TPR >= min_sensitivity
    valid_indices = np.where(tpr >= min_sensitivity)[0]

    if len(valid_indices) == 0:
        return None

    # Among valid, choose one with lowest FPR
    optimal_idx = valid_indices[np.argmin(fpr[valid_indices])]
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold, tpr[optimal_idx], fpr[optimal_idx]

# Example: Require at least 95% sensitivity (recall)
threshold, tpr, fpr = find_threshold_for_sensitivity(y_true, y_scores, 0.95)
print(f"Threshold for 95% sensitivity: {threshold:.3f}")
print(f"Actual TPR: {tpr:.3f}, FPR: {fpr:.3f}")
```

### Visualizing Threshold Effects

```python
def plot_threshold_effects(y_true, y_scores):
    """Plot how metrics change with threshold"""
    from sklearn.metrics import precision_score, recall_score, f1_score

    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1s = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall', linewidth=2)
    plt.plot(thresholds, f1s, label='F1 Score', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Threshold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

plot_threshold_effects(y_true, y_scores)
```

## Multi-Class ROC

### One-vs-Rest (OvR) Approach

For multi-class problems, compute ROC for each class:

```python
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle

# Binarize labels for multi-class
y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
n_classes = y_true_bin.shape[1]

# Get predicted probabilities
y_scores = model.predict_proba(X_test)

# Compute ROC curve for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot
plt.figure(figsize=(10, 8))
colors = cycle(['blue', 'red', 'green', 'yellow', 'purple'])

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curves (One-vs-Rest)')
plt.legend(loc="lower right")
plt.show()
```

### Micro and Macro Averaging

```python
from sklearn.metrics import roc_auc_score

# Multi-class AUC
y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
y_scores = [[0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.2, 0.1, 0.7]]

# One-vs-Rest with different averaging
auc_ovr_macro = roc_auc_score(y_true, y_scores,
                              multi_class='ovr', average='macro')
auc_ovr_weighted = roc_auc_score(y_true, y_scores,
                                 multi_class='ovr', average='weighted')

print(f"OvR Macro AUC: {auc_ovr_macro:.3f}")
print(f"OvR Weighted AUC: {auc_ovr_weighted:.3f}")
```

## Practical Guidelines

### Complete Evaluation Function

```python
def evaluate_classifier_roc(y_true, y_scores, plot=True):
    """Comprehensive ROC/PR evaluation"""
    from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                                 average_precision_score, roc_auc_score)

    # ROC
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # PR
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    # Print metrics
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"PR-AUC:  {pr_auc:.3f}")

    # Find optimal threshold (Youden's index)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = roc_thresholds[optimal_idx]
    print(f"\nOptimal threshold (Youden): {optimal_threshold:.3f}")
    print(f"  TPR: {tpr[optimal_idx]:.3f}")
    print(f"  FPR: {fpr[optimal_idx]:.3f}")

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # ROC curve
        ax1.plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax1.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
                label=f'Optimal threshold = {optimal_threshold:.2f}')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # PR curve
        ax2.plot(recall, precision, 'b-', lw=2, label=f'PR (AUC = {pr_auc:.2f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    return roc_auc, pr_auc, optimal_threshold
```

## Common Pitfalls

1. **Using predicted classes instead of scores**: ROC requires probability scores or decision values
2. **Not checking class balance**: Use PR-AUC for highly imbalanced data
3. **Comparing AUC across different datasets**: AUC is dataset-dependent
4. **Forgetting to choose threshold**: AUC doesn't give you a working classifier
5. **Ignoring confidence intervals**: Single AUC value can be misleading
6. **Wrong multi-class strategy**: Choose OvR or OvO based on problem

## References

- Fawcett, T. (2006) "An introduction to ROC analysis", Pattern Recognition Letters
- Davis, J. and Goadrich, M. (2006) "The relationship between Precision-Recall and ROC curves"
- Saito, T. and Rehmsmeier, M. (2015) "The Precision-Recall Plot Is More Informative than the ROC Plot"
- Scikit-learn ROC and PR Documentation
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman (Chapter 9)
