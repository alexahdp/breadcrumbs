# Data Preprocessing

## Table of Contents

1. [Why Preprocessing Matters](#why-preprocessing-matters)
2. [Feature Scaling](#feature-scaling)
3. [Handling Missing Data](#handling-missing-data)
4. [Outlier Detection and Treatment](#outlier-detection-and-treatment)
5. [Encoding Categorical Variables](#encoding-categorical-variables)
6. [Feature Engineering](#feature-engineering)
7. [Data Splitting](#data-splitting)
8. [Practical Examples](#practical-examples)

## Why Preprocessing Matters

### Common Data Issues

Real-world datasets often have:
- **Different scales** - age (0-100) vs income (0-1,000,000)
- **Missing values** - incomplete records, sensor failures
- **Outliers** - measurement errors, extreme but valid values
- **Categorical data** - text labels that need numeric encoding
- **Skewed distributions** - most values clustered, few extremes
- **Irrelevant features** - noise that doesn't help prediction

### Impact on Models

Different algorithms are affected differently:
- **Distance-based** (KNN, K-Means, SVM): sensitive to scale
- **Tree-based** (Decision Trees, Random Forest): robust to scale
- **Neural Networks**: sensitive to scale and distribution
- **Linear models**: sensitive to multicollinearity and scale

## Feature Scaling

Feature scaling transforms features to a similar range, preventing features with larger magnitudes from dominating.

### Standardization (Z-Score Normalization)

**Standardization** transforms data to have mean = 0 and standard deviation = 1:

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

where:
- $\mu$ = mean of the feature
- $\sigma$ = standard deviation

**Properties:**
- Resulting distribution has $\mu = 0$, $\sigma = 1$
- Does not bound values to specific range
- Preserves outliers (they become large |z-scores|)
- Less affected by outliers than Min-Max scaling

**When to use:**
- When features are normally distributed
- For algorithms that assume normally distributed data (logistic regression, linear regression)
- When you want to preserve outlier information
- Most neural networks

**Python example:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler!
```

**Important:** Always fit the scaler on training data only, then apply to both training and test data!

### Min-Max Normalization

**Min-Max scaling** transforms features to a fixed range, typically [0, 1]:

$$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

For custom range $[a, b]$:
$$x_{scaled} = a + \frac{(x - x_{min})(b - a)}{x_{max} - x_{min}}$$

**Properties:**
- Bounded to specific range
- Preserves zero values
- Very sensitive to outliers (they compress the rest of the data)
- All values in same range

**When to use:**
- When you need bounded values
- For neural networks with specific activation functions
- When features are uniformly distributed
- When you don't have significant outliers

**Python example:**
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_train)
```

### Robust Scaling

**Robust scaling** uses median and interquartile range, making it robust to outliers:

$$x_{scaled} = \frac{x - Q_2}{Q_3 - Q_1}$$

where:
- $Q_2$ = median
- $Q_1$ = 25th percentile
- $Q_3$ = 75th percentile

**When to use:**
- When you have many outliers
- When outliers are invalid (measurement errors)

**Python example:**
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train)
```

### Max Abs Scaling

**Max Abs scaling** scales each feature by its maximum absolute value:

$$x_{scaled} = \frac{x}{|x|_{max}}$$

**Properties:**
- Preserves zero values and sparsity
- Values in range [-1, 1]
- Good for sparse data

**When to use:**
- Sparse data (many zeros)
- When zero is meaningful

### Log Transformation

**Log transformation** reduces right skewness:

$$x_{transformed} = \log(x + 1)$$

We add 1 to handle zero values.

**When to use:**
- Highly skewed distributions
- Exponential relationships
- When values span multiple orders of magnitude

**Python example:**
```python
import numpy as np

X_log = np.log1p(X)  # log1p = log(1 + x)
```

### Power Transformation

**Box-Cox transformation** finds optimal power transformation:

$$x_{transformed} = \begin{cases}
\frac{x^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\
\log(x) & \text{if } \lambda = 0
\end{cases}$$

**Yeo-Johnson transformation** works with negative values too.

**Python example:**
```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='box-cox')  # or 'yeo-johnson'
X_transformed = pt.fit_transform(X_train)
```

## Handling Missing Data

### Types of Missing Data

**MCAR (Missing Completely At Random)** - missingness is independent of any data
- Example: sensor randomly fails

**MAR (Missing At Random)** - missingness depends on observed data
- Example: older people less likely to report income

**MNAR (Missing Not At Random)** - missingness depends on unobserved data
- Example: people with very high income don't report it

### Detection

```python
import pandas as pd

# Count missing values
print(df.isnull().sum())

# Percentage missing
print(df.isnull().mean() * 100)

# Visualize missing data
import missingno as msno
msno.matrix(df)
```

### Strategies for Handling Missing Data

#### 1. Deletion

**Listwise deletion** - remove entire row if any value is missing:

```python
df_clean = df.dropna()
```

**When to use:**
- Small percentage of missing data (< 5%)
- Data is MCAR
- Large dataset

**Drawbacks:**
- Loses information
- Can introduce bias if not MCAR
- Reduces sample size

#### 2. Imputation

**Mean/Median/Mode imputation:**

```python
from sklearn.impute import SimpleImputer

# Mean imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_train)

# Median imputation (better for outliers)
imputer = SimpleImputer(strategy='median')

# Mode imputation (for categorical)
imputer = SimpleImputer(strategy='most_frequent')
```

**When to use:**
- Mean: normally distributed, no outliers
- Median: skewed distribution, outliers present
- Mode: categorical variables

**Drawbacks:**
- Reduces variance
- Can distort relationships
- Doesn't account for uncertainty

**Forward/Backward fill** (for time series):

```python
df_filled = df.fillna(method='ffill')  # Forward fill
df_filled = df.fillna(method='bfill')  # Backward fill
```

#### 3. Advanced Imputation

**KNN Imputation** - use k-nearest neighbors to impute:

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X_train)
```

**Iterative Imputation** (MICE - Multiple Imputation by Chained Equations):

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(random_state=42)
X_imputed = imputer.fit_transform(X_train)
```

**Model-based imputation:**
- Train a model to predict missing values
- Use other features as predictors

#### 4. Create Indicator Variable

Add binary indicator for missingness:

```python
df['income_missing'] = df['income'].isnull().astype(int)
```

This captures potential information in the missingness pattern.

## Outlier Detection and Treatment

### Detection Methods

#### 1. Statistical Methods

**Z-Score method** - values beyond ±3 standard deviations:

```python
from scipy import stats
import numpy as np

z_scores = np.abs(stats.zscore(df))
outliers = (z_scores > 3).all(axis=1)
```

**IQR (Interquartile Range) method:**

```python
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Define outliers
outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
```

Values beyond $Q1 - 1.5 \times IQR$ or $Q3 + 1.5 \times IQR$ are considered outliers.

#### 2. Isolation Forest

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.1, random_state=42)
outliers = iso_forest.fit_predict(X)
# Returns -1 for outliers, 1 for inliers
```

#### 3. Local Outlier Factor (LOF)

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20)
outliers = lof.fit_predict(X)
```

### Treatment Strategies

**1. Remove outliers:**
```python
# Using Z-score
df_clean = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
```

**2. Cap outliers (Winsorization):**
```python
# Cap at 1st and 99th percentiles
lower = df.quantile(0.01)
upper = df.quantile(0.99)
df_capped = df.clip(lower=lower, upper=upper, axis=1)
```

**3. Transform data:**
- Log transformation
- Square root transformation
- Box-Cox transformation

**4. Use robust methods:**
- Robust scaling
- Tree-based models (naturally robust to outliers)

**When to remove vs keep:**
- **Remove:** measurement errors, data entry mistakes
- **Keep:** rare but valid events (fraud detection, anomaly detection)

## Encoding Categorical Variables

### Types of Categorical Data

**Nominal** - no inherent order (color, city, product type)
**Ordinal** - has meaningful order (education level, rating)

### Encoding Methods

#### 1. Label Encoding

Assigns integer to each category:

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])
# ['NYC', 'LA', 'SF'] -> [0, 1, 2]
```

**When to use:**
- Ordinal variables (when order is meaningful)
- Tree-based models (they can handle it)

**Warning:** Don't use for nominal variables with linear models - implies ordering!

#### 2. One-Hot Encoding

Creates binary column for each category:

```python
# Using pandas
df_encoded = pd.get_dummies(df, columns=['city'], drop_first=False)

# Using sklearn
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False, drop='first')
encoded = ohe.fit_transform(df[['city']])
```

**Properties:**
- No ordinal relationship implied
- Increases dimensionality
- Can create multicollinearity

**When to use:**
- Nominal variables with few categories (< 10-20)
- Linear models, neural networks

**drop_first=True:** Prevents multicollinearity by removing one category (use n-1 columns for n categories).

#### 3. Target Encoding (Mean Encoding)

Replaces category with mean of target variable:

```python
# For each category, compute mean target value
city_means = df.groupby('city')['target'].mean()
df['city_encoded'] = df['city'].map(city_means)
```

**When to use:**
- High-cardinality categorical features
- Tree-based models

**Warning:** Risk of data leakage and overfitting! Use cross-validation:

```python
from category_encoders import TargetEncoder

te = TargetEncoder()
df['city_encoded'] = te.fit_transform(df['city'], df['target'])
```

#### 4. Frequency Encoding

Replaces category with its frequency:

```python
freq = df['city'].value_counts(normalize=True)
df['city_freq'] = df['city'].map(freq)
```

#### 5. Binary Encoding

Hybrid of label and one-hot encoding - converts to binary digits:

```python
from category_encoders import BinaryEncoder

be = BinaryEncoder(cols=['city'])
df_encoded = be.fit_transform(df)
```

**When to use:**
- High-cardinality features
- Reduces dimensionality compared to one-hot

### High-Cardinality Features

Features with many unique values (e.g., zip codes, user IDs).

**Strategies:**
1. **Group rare categories** into "Other"
2. **Target encoding** with regularization
3. **Hash encoding** for extremely high cardinality
4. **Embeddings** (for deep learning)

```python
# Group rare categories
threshold = 0.01
freq = df['city'].value_counts(normalize=True)
rare_categories = freq[freq < threshold].index
df['city'] = df['city'].replace(rare_categories, 'Other')
```

## Feature Engineering

### Creating New Features

**Polynomial features:**
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
# Creates x1, x2, x1^2, x1*x2, x2^2
```

**Interaction features:**
```python
df['price_per_sqft'] = df['price'] / df['sqft']
df['age_income'] = df['age'] * df['income']
```

**Domain-specific features:**
```python
# For datetime
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# For text
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
```

### Feature Selection

**Remove low-variance features:**
```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)
```

**Remove highly correlated features:**
```python
# Compute correlation matrix
corr_matrix = df.corr().abs()

# Select upper triangle
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation > 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
df_reduced = df.drop(columns=to_drop)
```

## Data Splitting

### Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Parameters:**
- **test_size**: proportion for test set (0.2 = 20%)
- **random_state**: seed for reproducibility
- **stratify**: maintain class distribution in splits (for classification)

### Train-Validation-Test Split

```python
# First split: train+val vs test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: train vs val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
)
# Result: 60% train, 20% val, 20% test
```

### Time Series Split

**Important:** Don't shuffle time series data!

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

## Practical Examples

### Complete Preprocessing Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define column types
numeric_features = ['age', 'income', 'credit_score']
categorical_features = ['city', 'occupation']

# Numeric pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse=False))
])

# Combine pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit and transform
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
```

### Handling Categorical Variables Correctly

**Wrong approach:**
```python
# DON'T DO THIS!
df['city'] = df['city'].map({'NYC': 0, 'LA': 1, 'SF': 2})
# This implies NYC < LA < SF, which is meaningless!
```

**Correct approach:**
```python
# Create separate binary columns
df = pd.get_dummies(df, columns=['city'], drop_first=True)
```

When converting categorical text features to numeric, be careful not to create ordinal relationships where none exist. Assigning cities numbers and keeping them in one column creates an implied order and allows arithmetic operations. Instead, distribute categorical features across multiple binary columns (one-hot encoding).

## Summary

### Key Preprocessing Steps

1. **Understand your data**
   - Check distributions
   - Identify missing values
   - Find outliers
   - Understand feature types

2. **Handle missing data**
   - Choose appropriate strategy
   - Consider why data is missing
   - Document assumptions

3. **Handle outliers**
   - Decide: keep, remove, or transform
   - Consider domain knowledge
   - Use robust methods

4. **Scale features**
   - Choose method based on distribution
   - Always use same scaler for train/test
   - Fit only on training data

5. **Encode categorical variables**
   - Use one-hot for nominal
   - Use label encoding for ordinal
   - Consider cardinality

6. **Split data properly**
   - Use stratification for imbalanced data
   - Don't shuffle time series
   - Keep test set completely separate

### Best Practices

- **Fit on train, transform on train and test** - prevent data leakage
- **Use pipelines** - ensure reproducibility and prevent errors
- **Document decisions** - track what preprocessing was done and why
- **Validate assumptions** - check if normality, independence assumptions hold
- **Domain knowledge** - understand what features mean
- **Cross-validation** - use to evaluate preprocessing choices

## References

- Feature Engineering for Machine Learning (Alice Zheng, Amanda Casari)
- Hands-On Machine Learning (Aurélien Géron)
- [Scikit-learn Preprocessing Guide](https://scikit-learn.org/stable/modules/preprocessing.html)
