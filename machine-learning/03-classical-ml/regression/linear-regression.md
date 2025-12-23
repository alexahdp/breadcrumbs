# Linear Regression

## Table of Contents

1. [Introduction](#introduction)
2. [Simple Linear Regression](#simple-linear-regression)
3. [Multiple Linear Regression](#multiple-linear-regression)
4. [Ordinary Least Squares (OLS)](#ordinary-least-squares-ols)
5. [Model Assumptions](#model-assumptions)
6. [Model Evaluation](#model-evaluation)
7. [Practical Considerations](#practical-considerations)

## Introduction

**Linear Regression** - a supervised learning algorithm that models the relationship between a dependent variable and one or more independent variables using a linear function. It's one of the most fundamental and widely used techniques in machine learning.

**Use cases:**
- Predicting continuous numerical values (house prices, temperature, stock prices)
- Understanding relationships between variables
- Forecasting and trend analysis
- Feature importance analysis

## Simple Linear Regression

**Simple Linear Regression** - models the relationship between two variables using a straight line:

$$y = \beta_0 + \beta_1 x + \epsilon$$

where:
- $y$ is the dependent variable (target)
- $x$ is the independent variable (feature)
- $\beta_0$ is the intercept (bias term)
- $\beta_1$ is the slope (coefficient)
- $\epsilon$ is the error term (residual)

### Geometric Interpretation

The goal is to find the line that best fits the data by minimizing the distance between predicted and actual values.

**Prediction formula:**

$$\hat{y} = \beta_0 + \beta_1 x$$

where $\hat{y}$ is the predicted value.

## Multiple Linear Regression

**Multiple Linear Regression** - extends simple linear regression to multiple independent variables:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon$$

**Matrix notation:**

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

where:
- $\mathbf{y}$ is the $n \times 1$ target vector
- $\mathbf{X}$ is the $n \times (p+1)$ design matrix (includes intercept column)
- $\boldsymbol{\beta}$ is the $(p+1) \times 1$ coefficient vector
- $\boldsymbol{\epsilon}$ is the $n \times 1$ error vector

## Ordinary Least Squares (OLS)

**Ordinary Least Squares** - the most common method for estimating regression coefficients by minimizing the sum of squared residuals.

### Cost Function

**Mean Squared Error (MSE)** - the loss function to minimize:

$$J(\boldsymbol{\beta}) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \frac{1}{n}\sum_{i=1}^{n}(y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2$$

### Closed-Form Solution

The optimal coefficients can be calculated analytically using the **Normal Equation**:

$$\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

**Advantages:**
- Exact solution (no iterations needed)
- No hyperparameters to tune

**Disadvantages:**
- Requires matrix inversion: $O(n^3)$ computational complexity
- Fails when $\mathbf{X}^T\mathbf{X}$ is singular (not invertible)
- Inefficient for very large datasets

### Gradient Descent Solution

Alternatively, coefficients can be estimated iteratively using gradient descent:

$$\boldsymbol{\beta}_{new} = \boldsymbol{\beta}_{old} - \alpha \nabla J(\boldsymbol{\beta})$$

where $\alpha$ is the learning rate and:

$$\nabla J(\boldsymbol{\beta}) = -\frac{2}{n}\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

**Advantages:**
- Scales better to large datasets
- Works when $\mathbf{X}^T\mathbf{X}$ is not invertible

**Disadvantages:**
- Requires choosing learning rate
- Needs multiple iterations
- May converge to local minimum (though MSE is convex)

## Model Assumptions

Linear regression relies on several key assumptions. Violations can lead to biased or inefficient estimates.

### 1. Linearity

The relationship between independent and dependent variables must be linear.

**Check:** Residual plots should show no patterns

### 2. Independence

Observations must be independent of each other.

**Check:** Durbin-Watson test for autocorrelation

### 3. Homoscedasticity

Constant variance of residuals across all levels of independent variables.

$$Var(\epsilon_i) = \sigma^2 \text{ for all } i$$

**Check:** Residuals vs. fitted values plot should show constant spread

**Violation consequences:**
- Standard errors become unreliable
- Hypothesis tests invalid
- Confidence intervals inaccurate

### 4. Normality of Residuals

Residuals should follow a normal distribution:

$$\epsilon \sim \mathcal{N}(0, \sigma^2)$$

**Check:** Q-Q plot, Shapiro-Wilk test, histogram of residuals

**Note:** This assumption is less critical for large samples due to Central Limit Theorem.

### 5. No Multicollinearity

Independent variables should not be highly correlated with each other.

**Check:**
- Variance Inflation Factor (VIF): $VIF > 10$ indicates problematic multicollinearity
- Correlation matrix: $|r| > 0.8$ suggests high correlation

$$VIF_j = \frac{1}{1 - R_j^2}$$

where $R_j^2$ is the R-squared from regressing $x_j$ on all other predictors.

**Consequences:**
- Unstable coefficient estimates
- Large standard errors
- Difficulty interpreting individual effects

### 6. No Influential Outliers

Extreme values can disproportionately affect the regression line.

**Check:**
- Cook's distance: values > 1 indicate influential points
- Leverage: measures how far independent variable values are from their mean
- DFBETAS: change in coefficients when observation is removed

## Model Evaluation

### Coefficient of Determination (R²)

**R-squared** - proportion of variance in the dependent variable explained by independent variables:

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

where:
- $SS_{res}$ = sum of squared residuals
- $SS_{tot}$ = total sum of squares

**Interpretation:**
- $R^2 = 0$: model explains none of the variance
- $R^2 = 1$: model explains all variance perfectly
- Typical values: 0.3-0.7 (domain-dependent)

**Limitation:** R² always increases when adding more features, even if they're not meaningful.

### Adjusted R²

**Adjusted R-squared** - penalizes model complexity:

$$R_{adj}^2 = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

where:
- $n$ = number of observations
- $p$ = number of predictors

**Use:** Comparing models with different numbers of features.

### Statistical Significance

**t-statistic** - tests if individual coefficients are significantly different from zero:

$$t = \frac{\beta_j}{SE(\beta_j)}$$

**F-statistic** - tests if the overall model is significant:

$$F = \frac{(SS_{tot} - SS_{res})/p}{SS_{res}/(n-p-1)}$$

**p-value interpretation:**
- $p < 0.05$: coefficient/model is statistically significant
- $p \geq 0.05$: fail to reject null hypothesis (coefficient might be zero)

## Practical Considerations

### Feature Scaling

Linear regression is sensitive to feature scales, especially when using gradient descent.

**Standardization** (Z-score normalization):

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

**Min-Max normalization:**

$$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

### Handling Categorical Variables

Convert categorical variables to numerical using:
- **One-hot encoding:** create binary columns for each category
- **Label encoding:** assign integers (only for ordinal variables)

**Note:** Drop one category to avoid multicollinearity (dummy variable trap).

### When to Use Linear Regression

**Good fit when:**
- Relationship between variables is approximately linear
- Need interpretable model
- Fast training required
- Baseline model for comparison

**Poor fit when:**
- Relationship is highly non-linear
- Severe multicollinearity present
- Many outliers exist
- Need to capture complex interactions

### Python Implementation

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
```

### Diagnostics

```python
import matplotlib.pyplot as plt
from scipy import stats

# Calculate residuals
residuals = y_test - y_pred

# Residual plot
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Q-Q plot for normality
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()

# Check for multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)
```

## References

- An Introduction to Statistical Learning (James, Witten, Hastie, Tibshirani)
- The Elements of Statistical Learning (Hastie, Tibshirani, Friedman)
- Scikit-learn Documentation: [Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
