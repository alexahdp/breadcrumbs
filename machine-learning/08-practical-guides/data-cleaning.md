# Data Cleaning

## Table of Contents

1. [Why Data Cleaning Matters](#why-data-cleaning-matters)
2. [Initial Data Exploration](#initial-data-exploration)
3. [Handling Missing Values](#handling-missing-values)
4. [Detecting and Treating Outliers](#detecting-and-treating-outliers)
5. [Dealing with Duplicates](#dealing-with-duplicates)
6. [Data Type Issues](#data-type-issues)
7. [Inconsistent Data](#inconsistent-data)
8. [Data Validation](#data-validation)
9. [Scaling and Normalization](#scaling-and-normalization)

## Why Data Cleaning Matters

**The 80/20 rule:** Data scientists spend 60-80% of their time on data cleaning and preparation, only 20-40% on actual modeling.

**Impact on model performance:**
- Garbage in, garbage out - poor data quality leads to poor models
- Missing values can bias results or cause errors
- Outliers can skew model parameters
- Inconsistent data reduces model generalization

**Key principle:** Understand your data before cleaning. Different problems require different solutions.

## Initial Data Exploration

### Quick Overview

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data.csv')

# Basic information
print(df.info())
print(df.describe())
print(df.head())

# Check shape
print(f"Shape: {df.shape}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Check data types
print(df.dtypes)

# Memory usage
print(df.memory_usage(deep=True))
```

### Missing Values Assessment

```python
# Count missing values
missing = df.isnull().sum()
missing_pct = 100 * missing / len(df)

missing_df = pd.DataFrame({
    'column': missing.index,
    'missing_count': missing.values,
    'missing_pct': missing_pct.values
})

print(missing_df[missing_df['missing_count'] > 0].sort_values('missing_pct', ascending=False))
```

### Visualizing Missing Data

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Heatmap of missing values
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Missing values bar chart
missing_pct.sort_values(ascending=False).head(20).plot(kind='bar')
plt.title('Top 20 Columns by Missing %')
plt.ylabel('Percentage Missing')
plt.show()
```

## Handling Missing Values

### Types of Missing Data

**MCAR (Missing Completely At Random):**
- No relationship between missingness and any values
- Safe to delete or impute
- Example: sensor randomly fails

**MAR (Missing At Random):**
- Missingness related to observed data, not the missing value itself
- Can often impute using other features
- Example: older respondents skip income questions more often

**MNAR (Missing Not At Random):**
- Missingness related to the unobserved value
- Most problematic type
- Example: high earners refuse to report income

### Strategy 1: Deletion

**Remove rows (listwise deletion):**
```python
# Remove rows with any missing values
df_clean = df.dropna()

# Remove rows with missing values in specific columns
df_clean = df.dropna(subset=['important_column'])

# Remove rows with missing in all columns
df_clean = df.dropna(how='all')

# Remove rows with too many missing values
threshold = 0.5  # keep rows with at least 50% non-null values
df_clean = df.dropna(thresh=int(threshold * df.shape[1]))
```

**When to use:**
- MCAR data only
- Small percentage of missing data (< 5%)
- Large dataset where losing rows doesn't hurt
- Missing values in target variable

**Remove columns:**
```python
# Remove columns with > 50% missing
threshold = 0.5
df_clean = df.loc[:, df.isnull().mean() < threshold]

# Remove specific columns
df_clean = df.drop(columns=['col_with_many_missing'])
```

**When to use:**
- Column has > 50% missing values
- Column not important for prediction
- Too expensive to impute accurately

### Strategy 2: Imputation

**Simple imputation methods:**

```python
from sklearn.impute import SimpleImputer

# Mean imputation (for numeric features)
imputer = SimpleImputer(strategy='mean')
df['feature'] = imputer.fit_transform(df[['feature']])

# Median imputation (robust to outliers)
imputer = SimpleImputer(strategy='median')
df['feature'] = imputer.fit_transform(df[['feature']])

# Most frequent (for categorical features)
imputer = SimpleImputer(strategy='most_frequent')
df['category'] = imputer.fit_transform(df[['category']])

# Constant value
imputer = SimpleImputer(strategy='constant', fill_value=0)
df['feature'] = imputer.fit_transform(df[['feature']])
```

**Forward/backward fill (time series):**
```python
# Forward fill - propagate last valid observation
df['feature'] = df['feature'].fillna(method='ffill')

# Backward fill - use next valid observation
df['feature'] = df['feature'].fillna(method='bfill')

# Interpolation for time series
df['feature'] = df['feature'].interpolate(method='linear')
df['feature'] = df['feature'].interpolate(method='time')  # time-aware
```

**Advanced imputation - KNN:**
```python
from sklearn.impute import KNNImputer

# Impute using k-nearest neighbors
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)
```

**Advanced imputation - Iterative (MICE):**
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Multivariate imputation by chained equations
imputer = IterativeImputer(random_state=42, max_iter=10)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)
```

### Strategy 3: Missing Indicator

Create a binary indicator for missingness:

```python
# Create missing indicator
df['feature_missing'] = df['feature'].isnull().astype(int)

# Then impute the original feature
df['feature'] = df['feature'].fillna(df['feature'].median())
```

**When to use:**
- Missingness itself may be informative (MNAR)
- Example: "income_missing" might predict loan default

### Choosing the Right Strategy

| Situation | Recommended Strategy |
|-----------|---------------------|
| < 5% missing, MCAR | Delete rows |
| > 50% missing in column | Delete column or use missing indicator |
| Numeric, normally distributed | Mean imputation |
| Numeric, skewed or outliers | Median imputation |
| Categorical | Mode imputation |
| Time series | Forward/backward fill, interpolation |
| Complex patterns | KNN or iterative imputation |
| Missingness is informative | Missing indicator + imputation |

## Detecting and Treating Outliers

### What is an Outlier?

**Outlier** - a data point that significantly differs from other observations. Can be caused by:
- **Measurement errors** - sensor malfunction, data entry mistakes
- **Natural variation** - genuine extreme values
- **Data processing errors** - incorrect transformations

### Detection Methods

**1. Z-score method:**

$$Z = \frac{x - \mu}{\sigma}$$

```python
from scipy import stats

# Calculate z-scores
z_scores = np.abs(stats.zscore(df.select_dtypes(include=np.number)))

# Flag outliers (|z| > 3)
outliers = (z_scores > 3).any(axis=1)
print(f"Outliers found: {outliers.sum()}")

# Remove outliers
df_clean = df[(z_scores < 3).all(axis=1)]
```

**When to use:** Normally distributed data, univariate outliers

**2. IQR (Interquartile Range) method:**

$$IQR = Q_3 - Q_1$$

$$\text{Outliers: } x < Q_1 - 1.5 \times IQR \text{ or } x > Q_3 + 1.5 \times IQR$$

```python
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply to specific column
df_clean = remove_outliers_iqr(df, 'price')

# Apply to all numeric columns
for col in df.select_dtypes(include=np.number).columns:
    df_clean = remove_outliers_iqr(df_clean, col)
```

**When to use:** Non-normally distributed data, robust to extreme values

**3. Isolation Forest:**

```python
from sklearn.ensemble import IsolationForest

# Detect outliers using Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outliers = iso_forest.fit_predict(df.select_dtypes(include=np.number))

# -1 for outliers, 1 for inliers
df_clean = df[outliers == 1]
```

**When to use:** Multivariate outliers, complex patterns

**4. Local Outlier Factor (LOF):**

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
outliers = lof.fit_predict(df.select_dtypes(include=np.number))

df_clean = df[outliers == 1]
```

**When to use:** Density-based outlier detection, clusters with different densities

### Treatment Strategies

**1. Remove outliers:**
```python
# Only when outliers are clearly errors
df_clean = df[z_scores < 3]
```

**2. Cap/Winsorize:**
```python
# Cap at percentiles
lower, upper = df['price'].quantile([0.01, 0.99])
df['price'] = df['price'].clip(lower, upper)

# Using scipy
from scipy.stats.mstats import winsorize
df['price'] = winsorize(df['price'], limits=[0.01, 0.01])
```

**3. Transform:**
```python
# Log transformation (for right-skewed data)
df['price_log'] = np.log1p(df['price'])

# Square root transformation
df['price_sqrt'] = np.sqrt(df['price'])

# Box-Cox transformation (finds optimal transformation)
from scipy.stats import boxcox
df['price_boxcox'], lambda_param = boxcox(df['price'] + 1)
```

**4. Separate modeling:**
```python
# Build separate models for outliers and normal data
outliers_mask = z_scores > 3
df_outliers = df[outliers_mask]
df_normal = df[~outliers_mask]
```

### Visualization for Outlier Detection

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, y='price')
plt.title('Price Distribution with Outliers')
plt.show()

# Scatter plot with outliers highlighted
z_scores = np.abs(stats.zscore(df[['feature1', 'feature2']]))
outliers = (z_scores > 3).any(axis=1)

plt.figure(figsize=(10, 6))
plt.scatter(df.loc[~outliers, 'feature1'], df.loc[~outliers, 'feature2'],
            alpha=0.5, label='Normal')
plt.scatter(df.loc[outliers, 'feature1'], df.loc[outliers, 'feature2'],
            color='red', label='Outliers')
plt.legend()
plt.show()
```

## Dealing with Duplicates

### Detecting Duplicates

```python
# Check for exact duplicates (all columns)
duplicates = df.duplicated()
print(f"Duplicate rows: {duplicates.sum()}")

# Check duplicates in specific columns
duplicates = df.duplicated(subset=['id', 'timestamp'])
print(f"Duplicate IDs: {duplicates.sum()}")

# View duplicate rows
print(df[df.duplicated(keep=False)])
```

### Removing Duplicates

```python
# Remove exact duplicates (keep first occurrence)
df_clean = df.drop_duplicates()

# Remove duplicates based on specific columns
df_clean = df.drop_duplicates(subset=['user_id', 'date'], keep='first')

# Keep last occurrence instead
df_clean = df.drop_duplicates(keep='last')

# Mark all duplicates (don't keep any)
df_clean = df.drop_duplicates(keep=False)
```

### Handling Near-Duplicates

For text data:

```python
from difflib import SequenceMatcher

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Find similar strings
threshold = 0.9
for i, row1 in df.iterrows():
    for j, row2 in df.iterrows():
        if i < j:
            sim = similarity(row1['name'], row2['name'])
            if sim > threshold:
                print(f"Similar: {row1['name']} vs {row2['name']} ({sim:.2f})")
```

For numeric data:

```python
# Find rows with very similar feature values
from sklearn.metrics.pairwise import cosine_similarity

# Compute pairwise similarity
similarities = cosine_similarity(df.select_dtypes(include=np.number))

# Find pairs with high similarity
threshold = 0.99
similar_pairs = np.argwhere((similarities > threshold) & (similarities < 1.0))
```

## Data Type Issues

### Correcting Data Types

```python
# Convert to numeric (coerce errors to NaN)
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Convert to datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Convert to categorical (saves memory)
df['category'] = df['category'].astype('category')

# Convert boolean
df['is_active'] = df['is_active'].astype(bool)
```

### Handling Mixed Types

```python
# Column with mixed numeric and text
def safe_convert(x):
    try:
        return float(x)
    except:
        return np.nan

df['mixed_column'] = df['mixed_column'].apply(safe_convert)
```

### String Cleaning

```python
# Remove whitespace
df['name'] = df['name'].str.strip()

# Lowercase
df['name'] = df['name'].str.lower()

# Remove special characters
df['name'] = df['name'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

# Fix encoding issues
df['text'] = df['text'].str.encode('utf-8', errors='ignore').str.decode('utf-8')
```

## Inconsistent Data

### Standardizing Categories

```python
# Standardize categorical values
category_mapping = {
    'NY': 'New York',
    'ny': 'New York',
    'new york': 'New York',
    'CA': 'California',
    'calif': 'California',
}

df['state'] = df['state'].replace(category_mapping)

# Or use fuzzy matching
from fuzzywuzzy import process

def standardize_category(value, choices):
    if pd.isna(value):
        return value
    match, score = process.extractOne(value, choices)
    return match if score > 80 else value

standard_categories = ['New York', 'California', 'Texas']
df['state'] = df['state'].apply(lambda x: standardize_category(x, standard_categories))
```

### Standardizing Formats

```python
# Phone numbers
df['phone'] = df['phone'].str.replace(r'[\s\-\(\)]', '', regex=True)

# Email addresses
df['email'] = df['email'].str.lower().str.strip()

# URLs
df['url'] = df['url'].str.lower().str.replace('http://', 'https://')

# Currency
df['price'] = df['price'].str.replace(r'[$,]', '', regex=True).astype(float)
```

### Handling Units

```python
# Extract numeric value and unit separately
df['height_value'] = df['height'].str.extract(r'([\d.]+)').astype(float)
df['height_unit'] = df['height'].str.extract(r'([a-zA-Z]+)')

# Convert all to same unit
def convert_to_cm(row):
    if row['height_unit'] == 'm':
        return row['height_value'] * 100
    elif row['height_unit'] == 'in':
        return row['height_value'] * 2.54
    else:
        return row['height_value']

df['height_cm'] = df.apply(convert_to_cm, axis=1)
```

## Data Validation

### Range Validation

```python
# Check if values are within expected range
assert df['age'].between(0, 120).all(), "Age values out of valid range"
assert df['percentage'].between(0, 100).all(), "Percentage out of range"

# Or filter invalid values
df = df[df['age'].between(0, 120)]
```

### Relationship Validation

```python
# Check logical constraints
assert (df['end_date'] >= df['start_date']).all(), "End date before start date"
assert (df['total'] == df['subtotal'] + df['tax']).all(), "Total calculation error"

# Flag violations
df['date_error'] = df['end_date'] < df['start_date']
```

### Uniqueness Validation

```python
# Check if column should be unique
assert df['id'].is_unique, "Duplicate IDs found"

# Check for unexpected duplicates
duplicates = df[df.duplicated(subset=['email'], keep=False)]
if len(duplicates) > 0:
    print(f"Warning: {len(duplicates)} duplicate emails found")
```

### Completeness Validation

```python
# Check required fields
required_fields = ['id', 'name', 'email']
for field in required_fields:
    missing = df[field].isnull().sum()
    if missing > 0:
        print(f"Warning: {missing} missing values in required field '{field}'")
```

## Scaling and Normalization

### When to Scale

**Scale features when:**
- Using distance-based algorithms (KNN, SVM, K-Means)
- Using gradient descent (neural networks, linear regression)
- Features have different units or scales
- Using regularization (L1, L2)

**Don't scale for:**
- Tree-based methods (Decision Trees, Random Forest, XGBoost)
- Naive Bayes

### Standardization (Z-score normalization)

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit on training data only!
scaler.fit(X_train)

# Transform both train and test
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Or use fit_transform (only on training data)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Properties:**
- Mean = 0, standard deviation = 1
- Preserves outliers
- Assumes roughly normal distribution

### Min-Max Normalization

$$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Custom range
scaler = MinMaxScaler(feature_range=(-1, 1))
```

**Properties:**
- Bounded range [0, 1] (or custom range)
- Sensitive to outliers
- Preserves zero values and sparsity

### Robust Scaling

$$x_{scaled} = \frac{x - Q_{median}}{Q_{75} - Q_{25}}$$

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Properties:**
- Robust to outliers
- Uses median and IQR instead of mean and std
- Good for data with outliers

### Comparison

| Method | Range | Outliers | Use Case |
|--------|-------|----------|----------|
| StandardScaler | Unbounded | Sensitive | Normally distributed data |
| MinMaxScaler | [0, 1] | Very sensitive | Bounded ranges needed |
| RobustScaler | Unbounded | Robust | Data with outliers |

## Complete Data Cleaning Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def clean_data(df):
    """Complete data cleaning pipeline"""

    # 1. Remove duplicates
    df = df.drop_duplicates()

    # 2. Handle missing values
    # Drop columns with >50% missing
    threshold = 0.5
    df = df.loc[:, df.isnull().mean() < threshold]

    # Impute numeric columns with median
    numeric_cols = df.select_dtypes(include=np.number).columns
    num_imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    # Impute categorical columns with mode
    cat_cols = df.select_dtypes(include='object').columns
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # 3. Remove outliers (IQR method)
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    # 4. Standardize text
    for col in cat_cols:
        df[col] = df[col].str.strip().str.lower()

    # 5. Fix data types
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass

    return df

# Usage
df_clean = clean_data(df)
```

## References

- "Data Cleaning" (Ilya Rahkovsky, John Wiley & Sons)
- "Python for Data Analysis" (Wes McKinney)
- "Feature Engineering and Selection" (Kuhn & Johnson)
- scikit-learn documentation: https://scikit-learn.org/stable/modules/preprocessing.html
