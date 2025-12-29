# Classification Metrics

## Table of Contents

1. [Overview](#overview)
2. [Accuracy](#accuracy)
3. [Precision](#precision)
4. [Recall (Sensitivity)](#recall-sensitivity)
5. [Specificity](#specificity)
6. [F1 Score](#f1-score)
7. [F-Beta Score](#f-beta-score)
8. [Matthews Correlation Coefficient (MCC)](#matthews-correlation-coefficient-mcc)
9. [Multi-Class Metrics](#multi-class-metrics)
10. [Choosing the Right Metric](#choosing-the-right-metric)

## Overview

Classification metrics evaluate how well a model assigns correct categories to instances. These metrics are built upon the confusion matrix, which categorizes predictions into:

- **True Positives (TP)**: Correctly predicted positive cases
- **True Negatives (TN)**: Correctly predicted negative cases
- **False Positives (FP)**: Incorrectly predicted positive cases (Type I error)
- **False Negatives (FN)**: Incorrectly predicted negative cases (Type II error)

See [Confusion Matrix](./confusion-matrix.md) for detailed explanations.

## Accuracy

**Accuracy** - the proportion of correct predictions among all predictions:

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

**Properties:**
- Ranges from 0 to 1 (or 0% to 100%)
- Accuracy = 1: perfect classification
- Accuracy = 0: all predictions wrong

**When to use:**
- Balanced datasets (similar number of samples in each class)
- When all classes are equally important
- As a quick initial assessment

**Advantages:**
- Easy to understand and interpret
- Single number summary
- Intuitive for stakeholders

**Disadvantages:**
- Misleading with imbalanced datasets
- Doesn't distinguish between error types
- Can hide poor performance on minority classes

**Python example:**

```python
from sklearn.metrics import accuracy_score
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.2%}")  # Accuracy: 75.00%

# Manual calculation
correct = np.sum(y_true == y_pred)
total = len(y_true)
accuracy_manual = correct / total
```

**The Imbalanced Data Problem:**

```python
# Example: 95% negative class, 5% positive class
# Naive model: always predict negative

y_true = np.array([0]*95 + [1]*5)  # 95 negative, 5 positive
y_pred = np.array([0]*100)         # Predict all negative

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.2%}")  # Accuracy: 95.00%

# This looks good but the model is useless!
# It never detects positive cases
```

## Precision

**Precision** (Positive Predictive Value) - the proportion of true positive predictions among all positive predictions:

$$Precision = \frac{TP}{TP + FP} = \frac{\text{True Positives}}{\text{Predicted Positives}}$$

**Interpretation**: "Of all cases predicted as positive, how many are actually positive?"

**Properties:**
- Ranges from 0 to 1
- Precision = 1: no false positives
- Focuses on minimizing false positives

**When to use:**
- When false positives are costly
- Spam detection (don't want to mark legitimate emails as spam)
- Fraud detection (don't want to block legitimate transactions)
- Medical diagnosis when treatments are invasive

**Advantages:**
- Focuses on positive predictions
- Good for imbalanced datasets
- Useful when FP are costly

**Disadvantages:**
- Ignores false negatives
- Can be high by predicting positive rarely
- Should be used with recall

**Python example:**

```python
from sklearn.metrics import precision_score, confusion_matrix

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])

precision = precision_score(y_true, y_pred)
print(f"Precision: {precision:.2%}")  # Precision: 75.00%

# Manual calculation from confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
precision_manual = tp / (tp + fp)
print(f"Manual Precision: {precision_manual:.2%}")
```

## Recall (Sensitivity)

**Recall** (Sensitivity, True Positive Rate, Hit Rate) - the proportion of actual positives correctly identified:

$$Recall = \frac{TP}{TP + FN} = \frac{\text{True Positives}}{\text{Actual Positives}}$$

**Interpretation**: "Of all actual positive cases, how many did we correctly identify?"

**Properties:**
- Ranges from 0 to 1
- Recall = 1: no false negatives
- Focuses on minimizing false negatives

**When to use:**
- When false negatives are costly
- Disease detection (don't want to miss sick patients)
- Fraud detection (don't want to miss actual fraud)
- Safety-critical applications

**Advantages:**
- Focuses on finding all positive cases
- Critical for imbalanced datasets
- Important when missing positives is costly

**Disadvantages:**
- Ignores false positives
- Can be high by predicting positive often
- Should be used with precision

**Python example:**

```python
from sklearn.metrics import recall_score

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])

recall = recall_score(y_true, y_pred)
print(f"Recall: {recall:.2%}")  # Recall: 75.00%

# Manual calculation
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
recall_manual = tp / (tp + fn)
print(f"Manual Recall: {recall_manual:.2%}")
```

**Precision-Recall Trade-off:**

```python
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# For models that output probabilities
y_scores = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.85, 0.6, 0.15])
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])

precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

# Lowering threshold: higher recall, lower precision
# Raising threshold: higher precision, lower recall
```

## Specificity

**Specificity** (True Negative Rate, Selectivity) - the proportion of actual negatives correctly identified:

$$Specificity = \frac{TN}{TN + FP} = \frac{\text{True Negatives}}{\text{Actual Negatives}}$$

**Interpretation**: "Of all actual negative cases, how many did we correctly identify?"

**Relationship to other metrics:**

$$Specificity = 1 - FPR$$

where FPR (False Positive Rate) = $\frac{FP}{TN + FP}$

**When to use:**
- When correctly identifying negatives is important
- Medical screening (correctly identifying healthy patients)
- Quality control (identifying good products)

**Python example:**

```python
from sklearn.metrics import confusion_matrix

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
specificity = tn / (tn + fp)
print(f"Specificity: {specificity:.2%}")

# False Positive Rate
fpr = fp / (tn + fp)
print(f"FPR: {fpr:.2%}")
print(f"Specificity = 1 - FPR: {1 - fpr:.2%}")
```

## F1 Score

**F1 Score** - the harmonic mean of precision and recall:

$$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall} = \frac{2TP}{2TP + FP + FN}$$

**Alternative formula:**

$$F1 = \frac{2}{\frac{1}{Precision} + \frac{1}{Recall}}$$

**Properties:**
- Ranges from 0 to 1
- F1 = 1: perfect precision and recall
- F1 = 0: either precision or recall is zero
- Gives equal weight to precision and recall
- Harmonic mean penalizes extreme values

**Why harmonic mean?**

The harmonic mean is more conservative than arithmetic mean. If either precision or recall is low, F1 will be low:

```
Arithmetic mean: (0.9 + 0.1) / 2 = 0.50
Harmonic mean (F1): 2 * (0.9 * 0.1) / (0.9 + 0.1) = 0.18
```

**When to use:**
- Imbalanced datasets
- When you need balance between precision and recall
- Single metric for model comparison
- When both FP and FN are important

**Advantages:**
- Balances precision and recall
- Good for imbalanced datasets
- Single number summary
- Widely used standard

**Disadvantages:**
- Doesn't account for true negatives
- Equal weighting may not suit all problems
- Can be misleading if precision/recall have very different importance

**Python example:**

```python
from sklearn.metrics import f1_score, precision_score, recall_score

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])

f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")

# Manual calculation
f1_manual = 2 * (precision * recall) / (precision + recall)
print(f"Manual F1: {f1_manual:.2%}")
```

## F-Beta Score

**F-Beta Score** - generalization of F1 that allows weighting precision vs recall:

$$F_\beta = (1 + \beta^2) \cdot \frac{Precision \cdot Recall}{\beta^2 \cdot Precision + Recall}$$

**Parameter β:**
- $\beta = 1$: F1 score (equal weight)
- $\beta < 1$: favor precision (e.g., β=0.5)
- $\beta > 1$: favor recall (e.g., β=2)

**Common values:**
- **F0.5**: Emphasizes precision (2x more than recall)
- **F1**: Equal balance
- **F2**: Emphasizes recall (2x more than precision)

**When to use:**
- When precision and recall have different importance
- F2 for disease detection (prioritize recall)
- F0.5 for spam detection (prioritize precision)

**Python example:**

```python
from sklearn.metrics import fbeta_score

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])

f1 = fbeta_score(y_true, y_pred, beta=1.0)
f2 = fbeta_score(y_true, y_pred, beta=2.0)   # Favor recall
f05 = fbeta_score(y_true, y_pred, beta=0.5)  # Favor precision

print(f"F1 Score: {f1:.3f}")
print(f"F2 Score (recall-focused): {f2:.3f}")
print(f"F0.5 Score (precision-focused): {f05:.3f}")
```

## Matthews Correlation Coefficient (MCC)

**Matthews Correlation Coefficient** - correlation coefficient between observed and predicted classifications:

$$MCC = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

**Properties:**
- Ranges from -1 to +1
- MCC = +1: perfect prediction
- MCC = 0: random prediction
- MCC = -1: complete disagreement
- Accounts for all four confusion matrix values
- Balanced even with imbalanced classes

**When to use:**
- Highly imbalanced datasets
- When you need a single balanced metric
- When all four confusion matrix values matter
- Comparing very different classifiers

**Advantages:**
- Most informative single metric for binary classification
- Accounts for class imbalance
- Symmetric (treats both classes equally)
- Returns meaningful values even for very imbalanced data

**Disadvantages:**
- Less intuitive than precision/recall
- Not as widely known
- Can be undefined if denominator is zero (rare edge cases)

**Python example:**

```python
from sklearn.metrics import matthews_corrcoef

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])

mcc = matthews_corrcoef(y_true, y_pred)
print(f"MCC: {mcc:.3f}")

# Manual calculation
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
numerator = (tp * tn) - (fp * fn)
denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
mcc_manual = numerator / denominator if denominator != 0 else 0
print(f"Manual MCC: {mcc_manual:.3f}")
```

**Example with imbalanced data:**

```python
# 90% negative, 10% positive
y_true = np.array([0]*90 + [1]*10)
y_pred_good = np.array([0]*88 + [1]*2 + [1]*8 + [0]*2)  # Good model
y_pred_naive = np.array([0]*100)  # Always predict negative

print(f"Naive Accuracy: {accuracy_score(y_true, y_pred_naive):.2%}")  # 90%
print(f"Naive MCC: {matthews_corrcoef(y_true, y_pred_naive):.3f}")    # 0.000

print(f"Good Accuracy: {accuracy_score(y_true, y_pred_good):.2%}")    # 96%
print(f"Good MCC: {matthews_corrcoef(y_true, y_pred_good):.3f}")      # 0.700

# MCC correctly shows naive model is useless
```

## Multi-Class Metrics

### Averaging Strategies

For multi-class classification, binary metrics are extended using averaging:

**Macro-averaging** - calculate metric for each class, then average:

$$Metric_{macro} = \frac{1}{n_{classes}}\sum_{i=1}^{n_{classes}} Metric_i$$

- Treats all classes equally
- Good when all classes are important
- Can be dominated by performance on rare classes

**Micro-averaging** - aggregate all TP, FP, FN across classes, then calculate:

$$Precision_{micro} = \frac{\sum TP_i}{\sum TP_i + \sum FP_i}$$

- Weighted by class frequency
- Good for imbalanced datasets
- Emphasizes performance on common classes

**Weighted-averaging** - weighted average by class support:

$$Metric_{weighted} = \frac{1}{n_{samples}}\sum_{i=1}^{n_{classes}} n_i \cdot Metric_i$$

- Accounts for class imbalance
- Most representative of overall performance

**Python example:**

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Multi-class example: 3 classes
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 2])
y_pred = np.array([0, 2, 2, 0, 1, 1, 0, 1, 2, 2])

# Different averaging strategies
precision_macro = precision_score(y_true, y_pred, average='macro')
precision_micro = precision_score(y_true, y_pred, average='micro')
precision_weighted = precision_score(y_true, y_pred, average='weighted')

print(f"Precision (macro): {precision_macro:.3f}")
print(f"Precision (micro): {precision_micro:.3f}")
print(f"Precision (weighted): {precision_weighted:.3f}")

# Per-class metrics
precision_per_class = precision_score(y_true, y_pred, average=None)
print(f"Precision per class: {precision_per_class}")
```

### One-vs-Rest (OvR)

Each class is treated as positive, all others as negative:

```python
from sklearn.metrics import classification_report

# Detailed multi-class report
print(classification_report(y_true, y_pred,
                          target_names=['Class 0', 'Class 1', 'Class 2']))
```

## Choosing the Right Metric

### Decision Flow

```
Is the dataset balanced?
├─ Yes → Start with Accuracy, also check Precision/Recall
└─ No  → Use F1, MCC, or Precision/Recall

What errors are most costly?
├─ False Positives → Optimize Precision
├─ False Negatives → Optimize Recall
└─ Both equally    → Optimize F1 or MCC

Need a single metric?
├─ Balanced data   → Accuracy or F1
├─ Imbalanced data → F1 or MCC
└─ Custom weights  → F-beta
```

### Common Scenarios

| Scenario | Primary Metric | Secondary Metrics |
|----------|---------------|-------------------|
| Spam detection | Precision | F0.5, Accuracy |
| Disease diagnosis | Recall | F2, Specificity |
| Fraud detection | F1 or F2 | Precision, Recall |
| Balanced classification | Accuracy | F1, MCC |
| Highly imbalanced | MCC | F1, Precision-Recall |
| Multi-class balanced | Accuracy | Macro F1 |
| Multi-class imbalanced | Weighted F1 | Macro F1, MCC |

### Practical Example

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, matthews_corrcoef, classification_report)

def comprehensive_classification_metrics(y_true, y_pred, labels=None):
    """Calculate and display all relevant classification metrics"""

    print("=== Classification Metrics ===\n")

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy:  {accuracy:.4f}")

    # For binary or macro-averaged multi-class
    avg = 'binary' if len(np.unique(y_true)) == 2 else 'macro'

    precision = precision_score(y_true, y_pred, average=avg, zero_division=0)
    recall = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"MCC:       {mcc:.4f}")

    # Detailed report
    print("\n" + classification_report(y_true, y_pred, target_names=labels))

    # Check for imbalance
    unique, counts = np.unique(y_true, return_counts=True)
    imbalance_ratio = counts.max() / counts.min()
    if imbalance_ratio > 2:
        print(f"\n⚠ Warning: Class imbalance detected (ratio: {imbalance_ratio:.1f})")
        print("  Consider using F1, MCC, or resampling techniques")

# Usage
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
comprehensive_classification_metrics(y_true, y_pred, labels=['Negative', 'Positive'])
```

## Common Pitfalls

1. **Using accuracy on imbalanced data** - can be very misleading
2. **Optimizing only one metric** - precision without recall or vice versa
3. **Ignoring class distribution** - not checking if data is balanced
4. **Wrong averaging for multi-class** - using micro when macro is appropriate
5. **Not setting a baseline** - always compare to simple baselines
6. **Threshold selection** - default 0.5 threshold may not be optimal

## References

- Scikit-learn Classification Metrics Documentation
- "Learning from Imbalanced Data Sets" by Alberto Fernández et al.
- Powers, D.M.W. (2011) "Evaluation: From Precision, Recall and F-Measure to ROC"
- Chicco, D. & Jurman, G. (2020) "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy"
