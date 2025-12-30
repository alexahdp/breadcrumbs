# Feature Engineering

## Table of Contents

1. [What is Feature Engineering](#what-is-feature-engineering)
2. [Numeric Feature Transformations](#numeric-feature-transformations)
3. [Categorical Encoding](#categorical-encoding)
4. [Feature Extraction](#feature-extraction)
5. [Feature Construction](#feature-construction)
6. [Feature Selection](#feature-selection)
7. [Domain-Specific Features](#domain-specific-features)
8. [Automated Feature Engineering](#automated-feature-engineering)

## What is Feature Engineering

**Feature Engineering** - the process of creating new features or transforming existing features to improve model performance.

**Why it matters:**
- "Applied machine learning is basically feature engineering" - Andrew Ng
- Often provides bigger performance gains than algorithm selection
- Good features can make simple models outperform complex ones
- Domain knowledge becomes competitive advantage

**Types of feature engineering:**
- **Transformation** - mathematical transformations of existing features
- **Encoding** - converting categorical data to numeric
- **Extraction** - deriving features from raw data (text, images, time)
- **Construction** - combining features to create new ones
- **Selection** - choosing the most relevant features

## Numeric Feature Transformations

### Scaling Transformations

Already covered in [Data Cleaning](./data-cleaning.md#scaling-and-normalization):
- StandardScaler - zero mean, unit variance
- MinMaxScaler - bounded range
- RobustScaler - robust to outliers

### Non-Linear Transformations

**Log transformation** - for right-skewed data:

```python
import numpy as np

# Natural log (use log1p to handle zeros)
df['price_log'] = np.log1p(df['price'])

# Log base 10
df['price_log10'] = np.log10(df['price'] + 1)
```

**When to use:**
- Right-skewed distributions (long tail to the right)
- Variables spanning several orders of magnitude
- Reducing effect of outliers
- Making relationships more linear

**Square root transformation:**

```python
# Square root
df['area_sqrt'] = np.sqrt(df['area'])
```

**When to use:**
- Moderately skewed data
- Count data (Poisson-distributed)

**Power transformations:**

```python
from sklearn.preprocessing import PowerTransformer

# Box-Cox transformation (only for positive values)
pt = PowerTransformer(method='box-cox')
df['feature_boxcox'] = pt.fit_transform(df[['feature']])

# Yeo-Johnson transformation (works with negative values)
pt = PowerTransformer(method='yeo-johnson')
df['feature_yj'] = pt.fit_transform(df[['feature']])
```

**When to use:**
- Making data more Gaussian
- Stabilizing variance
- Improving model performance (especially for linear models)

### Binning (Discretization)

**Equal-width binning:**

```python
# Create bins of equal width
df['age_binned'] = pd.cut(df['age'], bins=5, labels=['Very Young', 'Young', 'Middle', 'Senior', 'Very Senior'])

# Custom bin edges
bins = [0, 18, 35, 50, 65, 100]
labels = ['Child', 'Young Adult', 'Adult', 'Senior', 'Elderly']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
```

**Equal-frequency binning (quantiles):**

```python
# Create bins with approximately equal number of samples
df['income_quantile'] = pd.qcut(df['income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Deciles
df['income_decile'] = pd.qcut(df['income'], q=10)
```

**When to use:**
- Capturing non-linear relationships
- Reducing impact of outliers
- Making numeric features more interpretable
- When you have domain knowledge about meaningful thresholds

**Scikit-learn binarization:**

```python
from sklearn.preprocessing import KBinsDiscretizer

kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
df['feature_binned'] = kbd.fit_transform(df[['feature']])

# Can also use one-hot encoding
kbd = KBinsDiscretizer(n_bins=5, encode='onehot', strategy='uniform')
```

### Binarization

```python
from sklearn.preprocessing import Binarizer

# Convert to 0/1 based on threshold
binarizer = Binarizer(threshold=100)
df['high_income'] = binarizer.fit_transform(df[['income']])

# Or simply
df['high_income'] = (df['income'] > 100).astype(int)
```

## Categorical Encoding

### Label Encoding

**Ordinal encoding** - for ordered categories:

```python
from sklearn.preprocessing import OrdinalEncoder

# Natural ordering
education_order = ['High School', 'Bachelor', 'Master', 'PhD']
encoder = OrdinalEncoder(categories=[education_order])
df['education_encoded'] = encoder.fit_transform(df[['education']])

# Or manually with mapping
education_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
df['education_encoded'] = df['education'].map(education_map)
```

**When to use:**
- Ordinal categories with clear ordering
- Tree-based models (can handle arbitrary mappings)

**When NOT to use:**
- Nominal categories (no natural ordering)
- Linear models (will assume numeric relationship)

### One-Hot Encoding

```python
# Pandas get_dummies
df_encoded = pd.get_dummies(df, columns=['color', 'size'], drop_first=False)

# Drop first to avoid multicollinearity
df_encoded = pd.get_dummies(df, columns=['color'], drop_first=True)

# Scikit-learn OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(df[['color', 'size']])

# Get feature names
feature_names = encoder.get_feature_names_out(['color', 'size'])
df_encoded = pd.DataFrame(encoded, columns=feature_names)
```

**When to use:**
- Nominal categories (no ordering)
- Linear models, neural networks
- Low cardinality (< 10-15 unique values)

**Limitations:**
- **High cardinality** - too many unique values create too many features
- **Curse of dimensionality** - sparse matrices
- **New categories** - unseen categories at test time

### Target Encoding (Mean Encoding)

**Concept:** Replace category with target mean for that category.

```python
# Calculate mean target for each category
target_means = df.groupby('category')['target'].mean()

# Map to feature
df['category_encoded'] = df['category'].map(target_means)
```

**With cross-validation to prevent leakage:**

```python
from sklearn.model_selection import KFold

def target_encode_cv(df, categorical_col, target_col, n_folds=5):
    """Target encoding with cross-validation"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    encoded = np.zeros(len(df))

    for train_idx, val_idx in kf.split(df):
        # Calculate means on training fold
        means = df.iloc[train_idx].groupby(categorical_col)[target_col].mean()

        # Apply to validation fold
        encoded[val_idx] = df.iloc[val_idx][categorical_col].map(means)

    return encoded

df['category_target_encoded'] = target_encode_cv(df, 'category', 'target')
```

**When to use:**
- High cardinality categorical features
- Tree-based models (especially gradient boosting)
- Strong relationship between category and target

**Risks:**
- **Overfitting** - especially with small categories
- **Data leakage** - must use cross-validation or separate encoding set
- **Target leakage** - never use test set for encoding

### Frequency Encoding

```python
# Replace category with its frequency
freq_map = df['category'].value_counts(normalize=True).to_dict()
df['category_freq'] = df['category'].map(freq_map)
```

**When to use:**
- High cardinality
- Frequency itself is informative
- Complement to other encodings

### Binary Encoding

```python
import category_encoders as ce

# Binary encoding - convert to binary digits
encoder = ce.BinaryEncoder(cols=['category'])
df_encoded = encoder.fit_transform(df)
```

**When to use:**
- High cardinality (reduces dimensions compared to one-hot)
- Middle ground between label and one-hot encoding

### Feature Hashing

```python
from sklearn.feature_extraction import FeatureHasher

# Hash features to fixed number of dimensions
hasher = FeatureHasher(n_features=10, input_type='string')
hashed = hasher.transform(df['category'].apply(lambda x: [x]))
```

**When to use:**
- Very high cardinality (millions of categories)
- Online learning (new categories)
- Memory constraints

**Limitation:** Hash collisions (different categories → same hash)

### Handling Rare Categories

```python
def group_rare_categories(series, threshold=0.01, replace_with='Other'):
    """Group rare categories into 'Other'"""
    freq = series.value_counts(normalize=True)
    rare_categories = freq[freq < threshold].index

    return series.replace(rare_categories, replace_with)

df['category_grouped'] = group_rare_categories(df['category'], threshold=0.05)
```

### Encoding Comparison

| Method | Cardinality | Model Type | Pros | Cons |
|--------|-------------|------------|------|------|
| Label | Low-Medium | Trees | Simple, compact | Implies ordering |
| One-Hot | Low | Linear, NN | No assumptions | High dimensionality |
| Target | High | Trees | Powerful | Overfitting risk |
| Frequency | Any | Any | Simple, informative | Limited information |
| Hashing | Very High | Any | Fixed size | Hash collisions |

## Feature Extraction

### Text Features

**Bag of Words (Count Vectorization):**

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# Get feature names (words)
print(vectorizer.get_feature_names_out())

# Limit vocabulary
vectorizer = CountVectorizer(max_features=100, min_df=2, max_df=0.8)
```

**TF-IDF (Term Frequency-Inverse Document Frequency):**

$$TF\text{-}IDF(t,d) = TF(t,d) \times IDF(t)$$

where:
$$IDF(t) = \log\frac{N}{df(t)}$$

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=1000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2),  # unigrams and bigrams
    stop_words='english'
)

X = vectorizer.fit_transform(corpus)
```

**N-grams:**

```python
# Bigrams and trigrams
vectorizer = TfidfVectorizer(ngram_range=(2, 3))

# Character n-grams (useful for misspellings)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
```

**Basic text features:**

```python
# Text length features
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['avg_word_length'] = df['text_length'] / df['word_count']

# Special characters
df['num_exclamation'] = df['text'].str.count('!')
df['num_question'] = df['text'].str.count('\?')
df['num_uppercase'] = df['text'].str.count(r'[A-Z]')

# Sentiment (using libraries)
from textblob import TextBlob
df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
```

### DateTime Features

```python
# Convert to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['hour'] = df['date'].dt.hour
df['minute'] = df['date'].dt.minute

# Is weekend
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

# Quarter
df['quarter'] = df['date'].dt.quarter

# Week of year
df['week_of_year'] = df['date'].dt.isocalendar().week

# Time since epoch
df['timestamp'] = df['date'].astype(int) / 10**9

# Cyclical encoding (for cyclical features like hour, month)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

**Why cyclical encoding?**
- Hour 23 and hour 0 are close, but numerically far (23 vs 0)
- Sin/cos encoding preserves circular relationship
- Use for: hour, day of week, month, compass direction

### Time Series Features

```python
# Lag features (previous values)
df['sales_lag1'] = df['sales'].shift(1)
df['sales_lag7'] = df['sales'].shift(7)
df['sales_lag30'] = df['sales'].shift(30)

# Rolling statistics
df['sales_rolling_mean_7'] = df['sales'].rolling(window=7).mean()
df['sales_rolling_std_7'] = df['sales'].rolling(window=7).std()
df['sales_rolling_max_7'] = df['sales'].rolling(window=7).max()

# Expanding features (all previous data)
df['sales_expanding_mean'] = df['sales'].expanding().mean()

# Difference features
df['sales_diff'] = df['sales'].diff()
df['sales_pct_change'] = df['sales'].pct_change()

# Time since last event
df['days_since_purchase'] = (df['current_date'] - df['last_purchase_date']).dt.days
```

### Geospatial Features

```python
# Distance from a point
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance between two points on Earth (in km)"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

# Distance from city center
city_center_lat, city_center_lon = 40.7128, -74.0060  # NYC
df['distance_from_center'] = df.apply(
    lambda row: haversine(row['longitude'], row['latitude'],
                         city_center_lon, city_center_lat),
    axis=1
)

# Cluster locations
from sklearn.cluster import KMeans
coords = df[['latitude', 'longitude']].values
kmeans = KMeans(n_clusters=10, random_state=42)
df['location_cluster'] = kmeans.fit_predict(coords)
```

## Feature Construction

### Interaction Features

**Multiplication:**

```python
# Simple interaction
df['price_per_sqft'] = df['price'] / df['sqft']
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
df['age_income_interaction'] = df['age'] * df['income']
```

**Polynomial features:**

```python
from sklearn.preprocessing import PolynomialFeatures

# Generate polynomial and interaction features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Get feature names
feature_names = poly.get_feature_names_out(X.columns)

# degree=2 creates: [x1, x2, x1^2, x1*x2, x2^2]
# degree=3 creates: [x1, x2, x1^2, x1*x2, x2^2, x1^3, x1^2*x2, x1*x2^2, x2^3]
```

**When to use:**
- Linear models (adds non-linearity)
- Small number of features (explodes with many features)
- Domain knowledge suggests interactions

### Ratio Features

```python
# Financial ratios
df['debt_to_income'] = df['debt'] / df['income']
df['savings_rate'] = df['savings'] / df['income']

# Performance metrics
df['click_through_rate'] = df['clicks'] / df['impressions']
df['conversion_rate'] = df['purchases'] / df['visits']

# Efficiency metrics
df['speed'] = df['distance'] / df['time']
df['productivity'] = df['output'] / df['hours_worked']
```

### Aggregation Features

**Group statistics:**

```python
# Per-group statistics
df['avg_price_by_category'] = df.groupby('category')['price'].transform('mean')
df['max_price_by_category'] = df.groupby('category')['price'].transform('max')
df['price_vs_category_avg'] = df['price'] / df['avg_price_by_category']

# Multiple aggregations
agg_features = df.groupby('user_id').agg({
    'purchase_amount': ['mean', 'sum', 'count', 'std'],
    'days_since_last': ['min', 'max']
}).reset_index()

# Flatten column names
agg_features.columns = ['_'.join(col).strip() for col in agg_features.columns.values]
```

### Domain-Specific Combinations

```python
# E-commerce: Average basket value
df['avg_basket_value'] = df['total_revenue'] / df['num_orders']

# Real estate: Price per room
df['price_per_room'] = df['price'] / (df['bedrooms'] + df['bathrooms'])

# Healthcare: BMI
df['bmi'] = df['weight_kg'] / (df['height_m'] ** 2)

# Marketing: Customer lifetime value
df['customer_lifetime_value'] = df['avg_purchase_value'] * df['purchase_frequency'] * df['customer_lifespan']
```

## Feature Selection

### Why Feature Selection?

**Benefits:**
- Reduce overfitting (fewer features = lower variance)
- Improve model interpretability
- Reduce training time
- Reduce prediction latency
- Reduce storage requirements

**When to use:**
- High-dimensional data (many features)
- Correlated features
- Computational constraints
- Need for interpretability

### Filter Methods

**Correlation with target:**

```python
# Pearson correlation for numeric target
correlations = df.corr()['target'].abs().sort_values(ascending=False)
selected_features = correlations[correlations > 0.5].index.tolist()

# Visualize
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.show()
```

**Remove highly correlated features:**

```python
def remove_correlated_features(df, threshold=0.95):
    """Remove features with correlation > threshold"""
    corr_matrix = df.corr().abs()

    # Upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation > threshold
    to_drop = [column for column in upper.columns
               if any(upper[column] > threshold)]

    return df.drop(columns=to_drop)

df_reduced = remove_correlated_features(df, threshold=0.90)
```

**Statistical tests:**

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Select top k features using F-test (for classification)
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
mask = selector.get_support()
selected_features = X.columns[mask]

# Using mutual information
selector = SelectKBest(score_func=mutual_info_classif, k=10)
X_selected = selector.fit_transform(X, y)
```

**Variance threshold:**

```python
from sklearn.feature_selection import VarianceThreshold

# Remove features with low variance
selector = VarianceThreshold(threshold=0.1)
X_selected = selector.fit_transform(X)

# For binary features, remove if >95% same value
selector = VarianceThreshold(threshold=0.05 * (1 - 0.05))
```

### Wrapper Methods

**Recursive Feature Elimination (RFE):**

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Select top 10 features
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
selector = RFE(estimator, n_features_to_select=10, step=1)
selector.fit(X, y)

# Get selected features
selected_features = X.columns[selector.support_]
feature_ranking = selector.ranking_

print(f"Selected features: {selected_features}")
```

**RFECV (with cross-validation):**

```python
from sklearn.feature_selection import RFECV

selector = RFECV(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    step=1,
    cv=5,
    scoring='f1'
)
selector.fit(X, y)

print(f"Optimal number of features: {selector.n_features_}")
selected_features = X.columns[selector.support_]
```

### Embedded Methods

**L1 Regularization (Lasso):**

```python
from sklearn.linear_model import LassoCV

# Lasso automatically performs feature selection
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X, y)

# Features with non-zero coefficients
importance = np.abs(lasso.coef_)
selected_features = X.columns[importance > 0]

print(f"Selected {len(selected_features)} features")
```

**Tree-based feature importance:**

```python
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importance
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Plot top 20
importance.head(20).plot(x='feature', y='importance', kind='barh', figsize=(10, 8))
plt.xlabel('Importance')
plt.title('Feature Importance (Random Forest)')
plt.show()

# Select top k features
k = 20
top_features = importance.head(k)['feature'].tolist()
X_selected = X[top_features]
```

**Permutation importance:**

```python
from sklearn.inspection import permutation_importance

# More reliable than built-in feature importance
result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)

importance = pd.DataFrame({
    'feature': X.columns,
    'importance': result.importances_mean,
    'std': result.importances_std
}).sort_values('importance', ascending=False)

print(importance.head(20))
```

### Feature Selection Comparison

| Method | Speed | Accuracy | Model Dependency |
|--------|-------|----------|------------------|
| Filter | Fast | Low | Independent |
| Wrapper | Slow | High | Dependent |
| Embedded | Medium | High | Dependent |

## Domain-Specific Features

### E-commerce

```python
# Customer behavior
df['days_since_last_purchase'] = (today - df['last_purchase_date']).dt.days
df['purchase_frequency'] = df['num_purchases'] / df['customer_age_days']
df['avg_basket_size'] = df['total_items'] / df['num_orders']
df['return_rate'] = df['num_returns'] / df['num_purchases']

# Product features
df['discount_pct'] = (df['original_price'] - df['sale_price']) / df['original_price']
df['price_vs_category_avg'] = df['price'] / df['category_avg_price']
```

### Finance

```python
# Technical indicators
df['moving_avg_20'] = df['close'].rolling(20).mean()
df['moving_avg_50'] = df['close'].rolling(50).mean()
df['volatility'] = df['returns'].rolling(20).std()

# Financial ratios
df['pe_ratio'] = df['market_cap'] / df['earnings']
df['debt_to_equity'] = df['total_debt'] / df['equity']
df['current_ratio'] = df['current_assets'] / df['current_liabilities']
```

### Healthcare

```python
# Vital signs
df['bmi'] = df['weight_kg'] / (df['height_m'] ** 2)
df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100],
                            labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

# Age groups
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100],
                         labels=['Child', 'Young', 'Middle', 'Senior', 'Elderly'])
```

## Automated Feature Engineering

### Featuretools

```python
import featuretools as ft

# Create entity set
es = ft.EntitySet(id='data')

# Add entities
es = es.add_dataframe(
    dataframe_name='transactions',
    dataframe=transactions_df,
    index='transaction_id',
    time_index='transaction_time'
)

es = es.add_dataframe(
    dataframe_name='customers',
    dataframe=customers_df,
    index='customer_id'
)

# Define relationships
es = es.add_relationship('customers', 'customer_id', 'transactions', 'customer_id')

# Generate features
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name='customers',
    max_depth=2,
    verbose=True
)
```

### When to Use Automated Tools

**Pros:**
- Quick generation of many features
- Can discover non-obvious patterns
- Saves time in exploration phase

**Cons:**
- Creates many irrelevant features
- Computationally expensive
- Reduces interpretability
- Still need feature selection afterward

## Best Practices

### General Guidelines

1. **Start simple** - begin with basic transformations before complex features
2. **Understand your data** - domain knowledge is crucial
3. **Iterate** - feature engineering is experimental
4. **Monitor for leakage** - never use test/future information
5. **Track everything** - document what works and what doesn't
6. **Consider interpretability** - complex features harder to explain

### Feature Engineering Workflow

```python
def feature_engineering_pipeline(df):
    """Complete feature engineering pipeline"""

    # 1. Basic transformations
    df['log_price'] = np.log1p(df['price'])
    df['price_per_sqft'] = df['price'] / df['sqft']

    # 2. Datetime features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)

    # 3. Categorical encoding
    df = pd.get_dummies(df, columns=['category'], drop_first=True)

    # 4. Aggregation features
    df['avg_price_by_location'] = df.groupby('location')['price'].transform('mean')

    # 5. Interaction features
    df['age_income'] = df['age'] * df['income']

    return df
```

### Avoiding Common Mistakes

**Don't:**
- Create features using test data
- Forget to handle missing values in new features
- Create too many features without selection
- Ignore feature scaling when needed
- Overlook feature correlations

**Do:**
- Use cross-validation when encoding
- Check for data leakage
- Monitor feature importance
- Document feature definitions
- Version control your features

## References

- "Feature Engineering for Machine Learning" (Alice Zheng, Amanda Casari)
- "Feature Engineering and Selection" (Max Kuhn, Kjell Johnson)
- scikit-learn Feature Engineering Guide
- Kaggle Feature Engineering courses
