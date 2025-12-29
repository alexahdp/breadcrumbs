# Evaluation Metrics

## Overview

Evaluation metrics are essential tools for assessing machine learning model performance. Choosing the right metric depends on the problem type, data characteristics, and business objectives. This section covers the most important metrics for regression and classification tasks.

## Why Metrics Matter

A machine learning model is only as good as our ability to measure its performance. Different metrics capture different aspects of model behavior:

- **Regression metrics** measure how close predictions are to actual values
- **Classification metrics** evaluate how well a model assigns correct categories
- **Loss functions** guide the training process by quantifying errors

Understanding metrics helps you:
1. Compare different models objectively
2. Detect overfitting and underfitting
3. Communicate model performance to stakeholders
4. Make informed decisions about model deployment

## Contents

### 1. [Regression Metrics](./regression-metrics.md)
Metrics for continuous value prediction:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score (Coefficient of Determination)
- Adjusted R²
- Mean Absolute Percentage Error (MAPE)

### 2. [Classification Metrics](./classification-metrics.md)
Metrics for categorical prediction:
- Accuracy
- Precision
- Recall (Sensitivity)
- F1 Score
- Specificity
- Matthews Correlation Coefficient (MCC)

### 3. [Confusion Matrix](./confusion-matrix.md)
Foundation for classification metrics:
- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)
- Interpreting confusion matrices

### 4. [ROC and AUC](./roc-auc.md)
Threshold-independent evaluation:
- ROC Curve (Receiver Operating Characteristic)
- AUC Score (Area Under the Curve)
- Precision-Recall Curve
- Trade-offs between metrics

### 5. [Loss Functions](./loss-functions.md)
Functions that guide model training:
- Regression losses (MSE, MAE, Huber)
- Classification losses (Cross-Entropy, Hinge)
- Custom loss functions
- Relationship between losses and metrics

## Choosing the Right Metric

### For Regression Problems

| Metric | Use When | Advantages | Disadvantages |
|--------|----------|------------|---------------|
| MSE | Standard regression, penalize large errors | Differentiable, widely used | Sensitive to outliers |
| MAE | Robust to outliers needed | More robust, interpretable | Not differentiable at zero |
| R² | Need explained variance | Normalized, intuitive | Can be misleading with bad fits |
| MAPE | Relative errors matter | Scale-independent | Undefined for zero values |

### For Classification Problems

| Metric | Use When | Best For |
|--------|----------|----------|
| Accuracy | Balanced classes | General overview |
| Precision | False positives costly | Spam detection, fraud detection |
| Recall | False negatives costly | Disease detection, safety systems |
| F1 Score | Balance precision/recall | Imbalanced datasets |
| AUC-ROC | Threshold-independent eval | Model comparison |

## Imbalanced Data Considerations

When dealing with imbalanced datasets (e.g., 95% negative, 5% positive):

1. **Avoid accuracy** - a model predicting all negatives gets 95% accuracy
2. **Use precision/recall** - better reflect minority class performance
3. **Consider F1 or MCC** - balance multiple aspects
4. **Examine confusion matrix** - understand error types
5. **Use stratified sampling** - maintain class distribution in validation

## Multi-Class vs Binary Classification

**Binary classification**: Two classes (positive/negative)
- Use: accuracy, precision, recall, F1, AUC-ROC

**Multi-class classification**: Three or more classes
- **Macro-averaging**: Calculate metric for each class, then average (treats all classes equally)
- **Micro-averaging**: Aggregate contributions of all classes, then calculate metric (weighted by class frequency)
- **Weighted-averaging**: Average weighted by class support

## Practical Tips

1. **Never rely on a single metric** - use multiple metrics to get a complete picture
2. **Understand the business context** - what type of error is more costly?
3. **Establish baselines** - compare against simple models (e.g., always predicting the mean)
4. **Use cross-validation** - ensure metrics are stable across different data splits
5. **Monitor multiple metrics during training** - detect overfitting early
6. **Document metric choices** - explain why you chose specific metrics

## Common Pitfalls

### Data Leakage
Calculating metrics on training data instead of validation/test data gives overly optimistic results.

### Inappropriate Metrics
Using accuracy on highly imbalanced data or R² when prediction ranges differ significantly.

### Ignoring Uncertainty
Single metric values without confidence intervals or standard deviations can be misleading.

### Overfitting to Metrics
Optimizing too heavily for a single metric can hurt overall model performance.

## Relationship Between Metrics and Loss

**Loss functions** are used during training to optimize model parameters:
- Must be differentiable
- Guide gradient descent
- Examples: MSE loss, cross-entropy loss

**Evaluation metrics** assess final model performance:
- Don't need to be differentiable
- More interpretable for stakeholders
- Examples: Accuracy, F1 score

Sometimes they coincide (e.g., MSE can be both), but often the training objective differs from the evaluation metric.

## References

- "Machine Learning Yearning" by Andrew Ng
- "Hands-On Machine Learning" by Aurélien Géron
- Scikit-learn documentation on metrics
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
