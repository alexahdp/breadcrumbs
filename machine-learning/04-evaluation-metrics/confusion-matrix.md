# Confusion Matrix

## Table of Contents

1. [Overview](#overview)
2. [Components of Confusion Matrix](#components-of-confusion-matrix)
3. [Binary Classification](#binary-classification)
4. [Multi-Class Confusion Matrix](#multi-class-confusion-matrix)
5. [Deriving Metrics from Confusion Matrix](#deriving-metrics-from-confusion-matrix)
6. [Interpreting Confusion Matrices](#interpreting-confusion-matrices)
7. [Visualization](#visualization)

## Overview

**Confusion Matrix** - a table that summarizes the performance of a classification model by comparing predicted and actual class labels. It provides a complete picture of how the model is performing and what types of errors it makes.

The confusion matrix is the foundation for most classification metrics including accuracy, precision, recall, F1 score, and many others.

## Components of Confusion Matrix

### Binary Classification

For binary classification (two classes: Positive and Negative), the confusion matrix is a 2×2 table:

|                    | **Predicted Negative** | **Predicted Positive** |
|--------------------|------------------------|------------------------|
| **Actual Negative** | True Negative (TN)     | False Positive (FP)    |
| **Actual Positive** | False Negative (FN)    | True Positive (TP)     |

### Four Key Values

**True Positive (TP)** - cases that are actually positive and correctly predicted as positive:
- Model predicted: Positive
- Actual label: Positive
- Result: Correct prediction

**True Negative (TN)** - cases that are actually negative and correctly predicted as negative:
- Model predicted: Negative
- Actual label: Negative
- Result: Correct prediction

**False Positive (FP)** - cases that are actually negative but incorrectly predicted as positive:
- Model predicted: Positive
- Actual label: Negative
- Result: **Type I Error** (false alarm)

**False Negative (FN)** - cases that are actually positive but incorrectly predicted as negative:
- Model predicted: Negative
- Actual label: Positive
- Result: **Type II Error** (miss)

### Memory Aid

Think of "True/False" as whether the prediction was correct:
- **True** = Correct prediction
- **False** = Incorrect prediction

Think of "Positive/Negative" as what the model predicted:
- **Positive** = Model said "yes"
- **Negative** = Model said "no"

## Binary Classification

### Example: Medical Diagnosis

Consider a model detecting disease (Positive = has disease, Negative = healthy):

```
Predicted:     [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
Actual:        [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
```

Confusion Matrix:

|                | **Predicted Negative** | **Predicted Positive** |
|----------------|------------------------|------------------------|
| **Actual Negative** | TN = 3                | FP = 1                 |
| **Actual Positive** | FN = 1                | TP = 5                 |

**Interpretation:**
- **TP = 5**: Correctly identified 5 sick patients
- **TN = 3**: Correctly identified 3 healthy patients
- **FP = 1**: Incorrectly diagnosed 1 healthy patient as sick (unnecessary treatment)
- **FN = 1**: Missed 1 sick patient (dangerous!)

### Python Implementation

```python
from sklearn.metrics import confusion_matrix
import numpy as np

y_true = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
y_pred = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Extract values
tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")
```

### Understanding Error Types

**Type I Error (False Positive)**
- Also called: False alarm, α error
- Example: Fire alarm goes off when there's no fire
- In medicine: Diagnosing disease when patient is healthy
- In spam: Marking legitimate email as spam

**Type II Error (False Negative)**
- Also called: Miss, β error
- Example: Fire alarm doesn't go off when there is a fire
- In medicine: Missing a disease in a sick patient
- In spam: Allowing spam email into inbox

**Which error is worse depends on the application:**

| Application | Worse Error | Reason |
|-------------|-------------|---------|
| Cancer screening | FN | Missing cancer is life-threatening |
| Spam detection | FP | Blocking important emails is costly |
| Fraud detection | Depends | FN = lost money, FP = customer frustration |
| Airport security | FN | Missing a threat is catastrophic |

## Multi-Class Confusion Matrix

For multi-class classification with $K$ classes, the confusion matrix is $K \times K$:

### Example: 3-Class Classification

```python
from sklearn.metrics import confusion_matrix
import numpy as np

# Classes: 0, 1, 2
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 2])
y_pred = np.array([0, 2, 2, 0, 1, 1, 0, 1, 2, 2])

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
```

Output:
```
           Predicted
           0   1   2
Actual 0 [[3   0   0]
       1  [0   3   1]
       2  [0   1   3]]
```

**Interpretation:**
- Diagonal elements (3, 3, 3): Correct predictions
- Off-diagonal elements: Misclassifications
- Row sum: Total actual instances of each class
- Column sum: Total predicted instances of each class

**Reading the matrix:**
- Row 1, Column 2 (value=1): Class 1 was predicted as Class 2 once
- Row 2, Column 1 (value=1): Class 2 was predicted as Class 1 once

### Per-Class Metrics

For each class in multi-class setting:

```python
def per_class_metrics(cm):
    """Calculate precision, recall for each class"""
    n_classes = cm.shape[0]

    for i in range(n_classes):
        # For class i:
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp  # Column sum minus diagonal
        fn = cm[i, :].sum() - tp  # Row sum minus diagonal
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"Class {i}:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
```

## Deriving Metrics from Confusion Matrix

All major classification metrics can be derived from the confusion matrix:

### Basic Metrics

**Accuracy** - overall correctness:

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

**Error Rate** - overall incorrectness:

$$Error\ Rate = \frac{FP + FN}{TP + TN + FP + FN} = 1 - Accuracy$$

### Positive Class Metrics

**Precision** (Positive Predictive Value):

$$Precision = \frac{TP}{TP + FP}$$

**Recall** (Sensitivity, True Positive Rate):

$$Recall = \frac{TP}{TP + FN}$$

**F1 Score**:

$$F1 = \frac{2TP}{2TP + FP + FN}$$

### Negative Class Metrics

**Specificity** (True Negative Rate):

$$Specificity = \frac{TN}{TN + FP}$$

**Negative Predictive Value (NPV)**:

$$NPV = \frac{TN}{TN + FN}$$

### Rate Metrics

**False Positive Rate (FPR)**:

$$FPR = \frac{FP}{FP + TN} = 1 - Specificity$$

**False Negative Rate (FNR)**:

$$FNR = \frac{FN}{FN + TP} = 1 - Recall$$

**False Discovery Rate (FDR)**:

$$FDR = \frac{FP}{FP + TP} = 1 - Precision$$

### Complete Calculation Example

```python
import numpy as np
from sklearn.metrics import confusion_matrix

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

# Get confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\n")

# Calculate all metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

print(f"Accuracy:    {accuracy:.3f}")
print(f"Precision:   {precision:.3f}")
print(f"Recall:      {recall:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"F1 Score:    {f1:.3f}")
print(f"FPR:         {fpr:.3f}")
print(f"FNR:         {fnr:.3f}")
```

## Interpreting Confusion Matrices

### Ideal Confusion Matrix

Perfect classifier (all predictions correct):

|                | **Predicted Negative** | **Predicted Positive** |
|----------------|------------------------|------------------------|
| **Actual Negative** | High TN               | **0 FP**               |
| **Actual Positive** | **0 FN**              | High TP                |

- Diagonal elements are high
- Off-diagonal elements are zero

### Common Patterns

**High FP (Type I errors):**
```
           Pred Neg  Pred Pos
Act Neg  [[   50       50  ]
Act Pos  [    5       95  ]]
```
- Model over-predicts positive class
- Low precision, high recall
- Solution: Increase prediction threshold

**High FN (Type II errors):**
```
           Pred Neg  Pred Pos
Act Neg  [[   95        5  ]
Act Pos  [    50       50  ]]
```
- Model under-predicts positive class
- High precision, low recall
- Solution: Decrease prediction threshold

**Balanced errors:**
```
           Pred Neg  Pred Pos
Act Neg  [[   70       30  ]
Act Pos  [    30       70  ]]
```
- Symmetrical confusion
- Balanced precision and recall
- May need better features or model

### Class Imbalance Impact

With 95% negative, 5% positive:

**Naive model (always predict negative):**
```
           Pred Neg  Pred Pos
Act Neg  [[   95        0  ]
Act Pos  [     5        0  ]]
```
- Accuracy = 95% (misleading!)
- Recall = 0% (useless for positive class)
- Confusion matrix reveals the problem

## Visualization

### Using Matplotlib

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(y_true, y_pred, class_names=None,
                         normalize=False, cmap='Blues'):
    """
    Plot confusion matrix with annotations

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list, optional
        Names of classes
    normalize : bool
        If True, normalize by row (show percentages)
    cmap : str
        Colormap name
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=class_names or 'auto',
                yticklabels=class_names or 'auto',
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()
    plt.show()

# Example usage
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

plot_confusion_matrix(y_true, y_pred,
                     class_names=['Negative', 'Positive'])
```

### Using Sklearn's ConfusionMatrixDisplay

```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Create and plot confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(
    y_true, y_pred,
    display_labels=['Negative', 'Positive'],
    cmap='Blues'
)

disp.plot()
plt.title('Confusion Matrix')
plt.show()
```

### Normalized Confusion Matrix

Show percentages instead of counts:

```python
# Normalize by true class (rows)
disp = ConfusionMatrixDisplay.from_predictions(
    y_true, y_pred,
    display_labels=['Negative', 'Positive'],
    normalize='true',  # 'true', 'pred', or 'all'
    cmap='Blues',
    values_format='.2%'
)

plt.title('Normalized Confusion Matrix (by true class)')
plt.show()
```

## Practical Tips

### 1. Always Check the Confusion Matrix

Don't rely solely on accuracy:

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Get predictions
y_pred = model.predict(X_test)

# Check multiple views
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))
```

### 2. Analyze Error Patterns

Look at what the model confuses:

```python
# For multi-class, find most confused pairs
cm = confusion_matrix(y_true, y_pred)
n_classes = cm.shape[0]

# Exclude diagonal (correct predictions)
np.fill_diagonal(cm, 0)

# Find biggest confusions
max_confusion = np.unravel_index(cm.argmax(), cm.shape)
print(f"Most confused: Class {max_confusion[0]} → Class {max_confusion[1]}")
print(f"Count: {cm[max_confusion]}")
```

### 3. Monitor During Training

Track confusion matrix changes:

```python
# During model training
for epoch in range(n_epochs):
    # Train model...

    # Evaluate
    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)

    # Track metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print(f"Epoch {epoch}: Precision={precision:.3f}, Recall={recall:.3f}")
```

### 4. Use with Cross-Validation

Aggregate confusion matrices:

```python
from sklearn.model_selection import cross_val_predict

# Get predictions from cross-validation
y_pred = cross_val_predict(model, X, y, cv=5)

# Confusion matrix on all predictions
cm = confusion_matrix(y, y_pred)
print("Cross-validated Confusion Matrix:")
print(cm)
```

## Common Pitfalls

1. **Swapping rows and columns**: Always check which axis is actual vs predicted
2. **Ignoring class imbalance**: Raw counts can be misleading, consider normalization
3. **Not examining off-diagonal elements**: They reveal error patterns
4. **Trusting accuracy alone**: Confusion matrix provides much more information
5. **Forgetting to normalize**: For imbalanced data, percentages are more informative

## References

- Scikit-learn Confusion Matrix Documentation
- Fawcett, T. (2006) "An introduction to ROC analysis"
- Powers, D.M.W. (2011) "Evaluation: From Precision, Recall and F-Measure to ROC"
- "Pattern Recognition and Machine Learning" by Christopher Bishop
