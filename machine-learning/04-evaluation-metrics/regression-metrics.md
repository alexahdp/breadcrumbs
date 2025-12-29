# Regression Metrics

## Table of Contents

1. [Overview](#overview)
2. [Mean Squared Error (MSE)](#mean-squared-error-mse)
3. [Root Mean Squared Error (RMSE)](#root-mean-squared-error-rmse)
4. [Mean Absolute Error (MAE)](#mean-absolute-error-mae)
5. [R² Score](#r²-score-coefficient-of-determination)
6. [Adjusted R²](#adjusted-r²)
7. [Mean Absolute Percentage Error (MAPE)](#mean-absolute-percentage-error-mape)
8. [Comparing Metrics](#comparing-metrics)

## Overview

Regression metrics evaluate how well a model predicts continuous values. These metrics measure the difference between predicted values $\hat{y}$ and actual values $y$.

**Key Concepts:**
- **Residual (error)**: $e_i = y_i - \hat{y}_i$
- **Predictions**: $\hat{y}_i$
- **Actual values**: $y_i$
- **Number of samples**: $n$

All regression metrics aim to quantify prediction error, but they differ in how they treat errors and their sensitivity to outliers.

## Mean Squared Error (MSE)

**Mean Squared Error** - the average of squared differences between predicted and actual values:

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**Properties:**
- Always non-negative (MSE ≥ 0)
- Perfect predictions yield MSE = 0
- Penalizes large errors more heavily (due to squaring)
- Units are squared (e.g., if predicting price in dollars, MSE is in dollars²)
- Differentiable everywhere (good for gradient-based optimization)

**When to use:**
- Standard regression problems
- When large errors are particularly undesirable
- As a loss function during training

**Advantages:**
- Widely used and well-understood
- Smooth gradient for optimization
- Mathematically convenient

**Disadvantages:**
- Very sensitive to outliers
- Not in original units (hard to interpret)
- Can be dominated by a few large errors

**Python example:**

```python
from sklearn.metrics import mean_squared_error
import numpy as np

y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse}")  # MSE: 0.375

# Manual calculation
mse_manual = np.mean((y_true - y_pred) ** 2)
```

## Root Mean Squared Error (RMSE)

**Root Mean Squared Error** - the square root of MSE:

$$RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**Properties:**
- Always non-negative (RMSE ≥ 0)
- Same units as the target variable
- More interpretable than MSE
- Still penalizes large errors

**When to use:**
- When you need interpretable error in original units
- Comparing models on the same dataset
- Reporting to non-technical stakeholders

**Advantages:**
- Same units as predictions (more interpretable than MSE)
- Penalizes large deviations
- Standard metric in many domains

**Disadvantages:**
- Still sensitive to outliers
- Scale-dependent (can't compare across different datasets)
- Not a good choice if errors follow non-normal distributions

**Python example:**

```python
from sklearn.metrics import mean_squared_error
import numpy as np

y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE: {rmse}")  # RMSE: 0.612

# Or using squared=False parameter (scikit-learn 0.22+)
rmse = mean_squared_error(y_true, y_pred, squared=False)
```

## Mean Absolute Error (MAE)

**Mean Absolute Error** - the average of absolute differences between predictions and actual values:

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**Properties:**
- Always non-negative (MAE ≥ 0)
- Same units as the target variable
- Treats all errors equally (linear penalty)
- More robust to outliers than MSE/RMSE

**When to use:**
- When outliers shouldn't dominate the metric
- When all errors should be weighted equally
- When interpretability is important

**Advantages:**
- Robust to outliers
- Easy to understand and interpret
- Same units as predictions
- All errors contribute proportionally

**Disadvantages:**
- Not differentiable at zero (can complicate optimization)
- Doesn't distinguish between small and large errors as much
- May underestimate impact of large errors

**Python example:**

```python
from sklearn.metrics import mean_absolute_error
import numpy as np

y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae}")  # MAE: 0.5

# Manual calculation
mae_manual = np.mean(np.abs(y_true - y_pred))
```

**Comparison: MSE vs MAE with outliers:**

```python
# Example showing robustness difference
y_true = np.array([1, 2, 3, 4, 100])  # 100 is an outlier
y_pred = np.array([1.1, 2.1, 2.9, 4.2, 105])

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

print(f"MAE: {mae:.2f}")   # MAE: 3.00
print(f"MSE: {mse:.2f}")   # MSE: 5.21
print(f"RMSE: {rmse:.2f}") # RMSE: 2.28

# MSE is more affected by the outlier (100 vs 105)
```

## R² Score (Coefficient of Determination)

**R² Score** (R-squared) - represents the proportion of variance in the dependent variable explained by the model:

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

where:
- $SS_{res}$ = Sum of Squared Residuals (unexplained variance)
- $SS_{tot}$ = Total Sum of Squares (total variance)
- $\bar{y}$ = mean of actual values

**Alternative formula:**

$$R^2 = 1 - \frac{MSE(\text{model})}{Var(y)}$$

**Properties:**
- Ranges from $-\infty$ to 1
- R² = 1: perfect predictions
- R² = 0: model no better than predicting the mean
- R² < 0: model worse than predicting the mean
- Scale-independent (unitless)

**When to use:**
- Comparing models on the same dataset
- Understanding explained variance
- When you need a normalized metric

**Advantages:**
- Normalized between 0 and 1 (for reasonable models)
- Easy to interpret as "percentage of variance explained"
- Scale-independent

**Disadvantages:**
- Can be misleading with non-linear relationships
- Always increases when adding more features (even irrelevant ones)
- Can be negative for very bad models
- Not suitable for non-linear models without modification

**Python example:**

```python
from sklearn.metrics import r2_score
import numpy as np

y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

r2 = r2_score(y_true, y_pred)
print(f"R²: {r2}")  # R²: 0.948

# Manual calculation
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2_manual = 1 - (ss_res / ss_tot)
```

**Interpreting R²:**
- R² = 0.95: model explains 95% of variance
- R² = 0.50: model explains 50% of variance
- R² = 0.00: model no better than mean baseline
- R² < 0.00: model worse than mean baseline

## Adjusted R²

**Adjusted R²** - R² penalized for the number of features, preventing overfitting:

$$R^2_{adj} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$$

where:
- $n$ = number of samples
- $p$ = number of features
- $R^2$ = regular R² score

**Properties:**
- Penalizes adding features that don't improve the model
- Can decrease when adding irrelevant features
- More conservative than R²
- Useful for feature selection

**When to use:**
- Comparing models with different numbers of features
- Feature selection
- Preventing overfitting

**Advantages:**
- Accounts for model complexity
- Better for comparing models with different feature counts
- Discourages adding unnecessary features

**Disadvantages:**
- Still not perfect for model selection
- Can be negative
- Less commonly supported in libraries

**Python example:**

```python
from sklearn.metrics import r2_score
import numpy as np

def adjusted_r2(y_true, y_pred, n_features):
    """Calculate adjusted R²"""
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return adj_r2

y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])
n_features = 2

r2 = r2_score(y_true, y_pred)
adj_r2 = adjusted_r2(y_true, y_pred, n_features)

print(f"R²: {r2:.4f}")           # R²: 0.9486
print(f"Adjusted R²: {adj_r2:.4f}")  # Adjusted R²: 0.8972
```

## Mean Absolute Percentage Error (MAPE)

**Mean Absolute Percentage Error** - average of absolute percentage errors:

$$MAPE = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

**Properties:**
- Expressed as a percentage
- Scale-independent (can compare across different datasets)
- Easy to interpret
- Undefined when $y_i = 0$

**When to use:**
- Comparing models across different scales
- When relative errors matter more than absolute errors
- Business reporting (easy to understand percentages)

**Advantages:**
- Scale-independent
- Easy to interpret (percentage)
- Good for comparing different datasets

**Disadvantages:**
- Undefined for zero actual values
- Asymmetric (penalizes over-predictions more than under-predictions)
- Can be heavily skewed by small actual values
- Not suitable when actual values near zero

**Python example:**

```python
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np

y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

# Note: MAPE can give unexpected results with values near zero
mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE: {mape:.2%}")  # As percentage

# Manual calculation
mape_manual = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Safe version avoiding division by zero
def safe_mape(y_true, y_pred, epsilon=1e-10):
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
```

**Alternative: Symmetric MAPE (sMAPE)**

Addresses asymmetry issue:

$$sMAPE = \frac{100\%}{n}\sum_{i=1}^{n}\frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}$$

## Comparing Metrics

### MSE vs MAE

| Aspect | MSE | MAE |
|--------|-----|-----|
| Sensitivity to outliers | High | Low |
| Unit | Squared | Original |
| Differentiability | Everywhere | Not at zero |
| Interpretation | Harder | Easier |
| Use case | Penalize large errors | Robust estimation |

**Rule of thumb**:
- If $RMSE \approx MAE$: errors are uniform
- If $RMSE \gg MAE$: errors have outliers or high variance

### When to Use Each Metric

```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_regression(y_true, y_pred):
    """Comprehensive regression evaluation"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    # Check if outliers are present
    if rmse > 1.5 * mae:
        print("\nWarning: Large difference between RMSE and MAE suggests outliers")

    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
```

## Practical Guidelines

### Choosing a Primary Metric

1. **For most cases**: Start with RMSE (interpretable, standard)
2. **With outliers**: Use MAE (more robust)
3. **Cross-dataset comparison**: Use R² or MAPE
4. **Feature selection**: Use Adjusted R²
5. **Business reporting**: Use MAPE or RMSE

### Metric Baselines

Always compare against simple baselines:

```python
# Baseline 1: Always predict mean
y_pred_mean = np.full_like(y_true, np.mean(y_true))
baseline_mse = mean_squared_error(y_true, y_pred_mean)

# Baseline 2: Previous value (time series)
# y_pred_previous = y_true[:-1]

print(f"Model MSE: {mse:.4f}")
print(f"Baseline MSE: {baseline_mse:.4f}")
print(f"Improvement: {(1 - mse/baseline_mse)*100:.1f}%")
```

### Common Pitfalls

1. **Using only one metric**: Always evaluate with multiple metrics
2. **Ignoring the scale**: MSE on normalized vs raw data gives different insights
3. **Not checking baselines**: Your model should beat simple baselines
4. **Overfitting to metrics**: Train/validation/test split is essential
5. **Forgetting units**: Remember MSE is squared, RMSE and MAE are in original units

## References

- Scikit-learn Metrics Documentation
- "Introduction to Statistical Learning" (Chapter 2: Assessing Model Accuracy)
- "The Elements of Statistical Learning" (Chapter 7: Model Assessment)
- Willmott, C.J. and Matsuura, K. (2005) "Advantages of the mean absolute error (MAE) over the root mean square error (RMSE)"
