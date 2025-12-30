# Model Selection

## Table of Contents

1. [Problem Type Identification](#problem-type-identification)
2. [The Model Selection Process](#the-model-selection-process)
3. [Algorithm Selection by Problem Type](#algorithm-selection-by-problem-type)
4. [Model Complexity Tradeoffs](#model-complexity-tradeoffs)
5. [No Free Lunch Theorem](#no-free-lunch-theorem)
6. [Decision Flowcharts](#decision-flowcharts)
7. [Ensemble Methods](#ensemble-methods)
8. [Practical Considerations](#practical-considerations)

## Problem Type Identification

### Supervised Learning

**Regression** - predicting continuous values:
- House prices, temperature, stock prices
- Output: Real number (e.g., $250,000, 72.5°F)
- Metrics: MSE, RMSE, MAE, R²

**Classification** - predicting categories:
- Binary: spam/not spam, fraud/legitimate
- Multi-class: digit recognition (0-9), image classification
- Multi-label: tag assignment (can have multiple tags)
- Output: Class label or probabilities
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC

### Unsupervised Learning

**Clustering** - grouping similar instances:
- Customer segmentation, document clustering
- No labels provided
- Metrics: Silhouette score, Davies-Bouldin index

**Dimensionality Reduction** - reducing feature space:
- Visualization, noise reduction, preprocessing
- Examples: PCA, t-SNE, UMAP

**Anomaly Detection** - finding outliers:
- Fraud detection, equipment failure prediction
- Examples: Isolation Forest, One-Class SVM

### Other Types

**Ranking** - ordering items by relevance:
- Search engines, recommendation systems

**Time Series Forecasting** - predicting future values:
- Sales forecasting, demand prediction
- Requires temporal structure

## The Model Selection Process

### Step 1: Understand the Problem

Ask these questions:
1. **What am I predicting?** (continuous, categorical, ordinal)
2. **How much data do I have?** (rows and features)
3. **Is the data labeled?** (supervised vs unsupervised)
4. **Do I need interpretability?** (can I use black-box models?)
5. **What are the computational constraints?** (training time, inference latency)
6. **What is the cost of errors?** (false positives vs false negatives)

### Step 2: Establish a Baseline

Always start with simple baselines:

```python
from sklearn.dummy import DummyClassifier, DummyRegressor

# Classification baseline
dummy_clf = DummyClassifier(strategy='most_frequent')
dummy_clf.fit(X_train, y_train)
baseline_score = dummy_clf.score(X_test, y_test)

# Regression baseline
dummy_reg = DummyRegressor(strategy='mean')
dummy_reg.fit(X_train, y_train)
baseline_score = dummy_reg.score(X_test, y_test)

print(f"Baseline score: {baseline_score:.3f}")
print(f"Any model must beat this to be useful!")
```

Simple model baselines:
- **Regression:** Linear Regression
- **Classification:** Logistic Regression
- **Trees:** Single Decision Tree

### Step 3: Try Multiple Models

Don't commit to one algorithm immediately. Try several:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    results[name] = {
        'mean': scores.mean(),
        'std': scores.std()
    }
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Select best model
best_model_name = max(results, key=lambda k: results[k]['mean'])
print(f"\nBest model: {best_model_name}")
```

### Step 4: Hyperparameter Tuning

Once you've identified promising algorithms, tune them:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# Use best model
best_model = grid_search.best_estimator_
```

### Step 5: Evaluate on Test Set

Final evaluation on held-out test set:

```python
from sklearn.metrics import classification_report, confusion_matrix

# Get predictions
y_pred = best_model.predict(X_test)

# Comprehensive evaluation
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

## Algorithm Selection by Problem Type

### Regression Problems

**Linear Regression**
```python
from sklearn.linear_model import LinearRegression
```

**When to use:**
- Linear relationships
- Need interpretability
- Baseline model
- Small to medium datasets

**Pros:** Fast, interpretable, no hyperparameters
**Cons:** Assumes linearity, sensitive to outliers

---

**Ridge/Lasso Regression**
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Ridge (L2 regularization)
ridge = Ridge(alpha=1.0)

# Lasso (L1 regularization, feature selection)
lasso = Lasso(alpha=1.0)

# ElasticNet (L1 + L2)
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
```

**When to use:**
- High-dimensional data
- Multicollinearity present
- Need regularization
- Feature selection (Lasso)

---

**Decision Trees**
```python
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(max_depth=5)
```

**When to use:**
- Non-linear relationships
- Mixed feature types
- Need interpretability
- Interactions between features

**Pros:** No feature scaling needed, handles non-linearity
**Cons:** Prone to overfitting, unstable

---

**Random Forest**
```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, max_depth=10)
```

**When to use:**
- Default choice for tabular data
- Non-linear relationships
- Large datasets
- Mixed feature types

**Pros:** Robust, handles non-linearity, less overfitting than single tree
**Cons:** Slower than linear models, less interpretable

---

**Gradient Boosting (XGBoost, LightGBM, CatBoost)**
```python
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

xgb = XGBRegressor(n_estimators=100, learning_rate=0.1)
lgbm = LGBMRegressor(n_estimators=100, learning_rate=0.1)
catboost = CatBoostRegressor(iterations=100, learning_rate=0.1)
```

**When to use:**
- State-of-the-art performance needed
- Structured/tabular data
- Kaggle competitions
- Have time for hyperparameter tuning

**Pros:** Usually best performance, handles missing values
**Cons:** Slow training, many hyperparameters, can overfit

---

**SVR (Support Vector Regression)**
```python
from sklearn.svm import SVR

svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
```

**When to use:**
- Small to medium datasets
- Non-linear relationships
- Need robust predictions

**Cons:** Slow on large datasets, requires feature scaling

### Classification Problems

**Logistic Regression**
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1.0, penalty='l2')
```

**When to use:**
- Binary/multi-class classification
- Need probability estimates
- Baseline model
- Linear decision boundaries

**Pros:** Fast, interpretable, probabilistic output
**Cons:** Assumes linear separability

---

**Naive Bayes**
```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# For continuous features
gnb = GaussianNB()

# For count data (text)
mnb = MultinomialNB()

# For binary features
bnb = BernoulliNB()
```

**When to use:**
- Text classification
- High-dimensional data
- Fast training/prediction needed
- Baseline model

**Pros:** Very fast, works with small data, simple
**Cons:** Assumes feature independence (often violated)

---

**K-Nearest Neighbors (KNN)**
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
```

**When to use:**
- Small datasets
- Non-linear decision boundaries
- No training phase needed

**Cons:** Slow prediction, memory intensive, requires scaling

---

**Decision Tree**
```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=5, min_samples_split=10)
```

**When to use:**
- Need interpretability (can visualize)
- Mixed feature types
- Quick prototype

---

**Random Forest**
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
```

**When to use:**
- Default choice for classification
- Imbalanced classes (with class weights)
- Feature importance needed

---

**Gradient Boosting**
```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

xgb = XGBClassifier(n_estimators=100, learning_rate=0.1)
lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.1)
```

**When to use:**
- Best performance needed
- Tabular data
- Have computational resources

---

**SVM (Support Vector Machine)**
```python
from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=1.0, gamma='scale')
```

**When to use:**
- Small to medium datasets
- High-dimensional data
- Clear margin of separation

**Cons:** Slow on large datasets

### Clustering Problems

**K-Means**
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42)
```

**When to use:**
- Spherical clusters
- Known number of clusters
- Large datasets (fast)

---

**DBSCAN**
```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
```

**When to use:**
- Arbitrary cluster shapes
- Outlier detection
- Unknown number of clusters

---

**Hierarchical Clustering**
```python
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
```

**When to use:**
- Small to medium datasets
- Need dendrogram visualization
- Hierarchical structure in data

## Model Complexity Tradeoffs

### Bias-Variance Tradeoff

**High Bias (Underfitting):**
- Model too simple
- Poor performance on both train and test
- Example: Linear model on non-linear data

**High Variance (Overfitting):**
- Model too complex
- Great on train, poor on test
- Example: Deep decision tree

**Sweet spot:**
- Balanced complexity
- Good generalization

```python
import matplotlib.pyplot as plt
import numpy as np

# Illustrate bias-variance tradeoff
train_scores = []
test_scores = []
depths = range(1, 20)

for depth in depths:
    dt = DecisionTreeRegressor(max_depth=depth)
    dt.fit(X_train, y_train)
    train_scores.append(dt.score(X_train, y_train))
    test_scores.append(dt.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, label='Train Score')
plt.plot(depths, test_scores, label='Test Score')
plt.xlabel('Max Depth')
plt.ylabel('R² Score')
plt.legend()
plt.title('Bias-Variance Tradeoff')
plt.show()
```

### Simple vs Complex Models

**Start simple:**
1. Begin with linear models (regression, logistic regression)
2. Try single decision tree
3. Move to ensemble methods if needed
4. Try deep learning only for complex data (images, text, etc.)

**When to increase complexity:**
- Simple models underperform
- Non-linear relationships evident
- Large amounts of data available
- Computational resources available

**When to keep it simple:**
- Need interpretability
- Limited data
- Limited compute
- Simple model performs well enough

## No Free Lunch Theorem

**Theorem:** No single algorithm works best for all problems.

**Implications:**
- Must try multiple algorithms
- Performance depends on problem characteristics
- Domain knowledge helps select candidates
- Empirical evaluation is essential

**Common patterns:**
- **Tabular data:** Gradient boosting usually wins
- **Images:** Convolutional neural networks
- **Text:** Transformers (BERT, GPT)
- **Time series:** ARIMA, Prophet, LSTM
- **Small data:** Simple models (linear, Naive Bayes)

## Decision Flowcharts

### Supervised Learning Decision Tree

```
Start
  ├─ Labeled data?
  │   ├─ Yes (Supervised)
  │   │   ├─ Predicting category? (Classification)
  │   │   │   ├─ < 100k samples?
  │   │   │   │   ├─ Linear separable? → Logistic Regression, SVM
  │   │   │   │   └─ Non-linear? → Random Forest, XGBoost
  │   │   │   └─ > 100k samples?
  │   │   │       ├─ Tabular? → XGBoost, LightGBM
  │   │   │       ├─ Images? → CNN
  │   │   │       └─ Text? → Transformers
  │   │   └─ Predicting number? (Regression)
  │   │       ├─ Linear relationship? → Linear Regression, Ridge
  │   │       └─ Non-linear? → Random Forest, XGBoost, SVR
  │   └─ No (Unsupervised)
  │       ├─ Group similar items? → K-Means, DBSCAN, Hierarchical
  │       ├─ Find outliers? → Isolation Forest, LOF
  │       └─ Reduce dimensions? → PCA, t-SNE, UMAP
```

### Quick Selection Guide

| Situation | Recommended Algorithm |
|-----------|----------------------|
| Small data (< 1k samples) | Linear models, Naive Bayes, KNN |
| Medium data (1k-100k) | Random Forest, SVM, XGBoost |
| Large data (> 100k) | XGBoost, LightGBM, Neural Networks |
| Need interpretability | Linear models, Decision Tree, Rule-based |
| High-dimensional sparse | Naive Bayes, Linear models with L1 |
| Mixed feature types | Tree-based methods |
| Imbalanced classes | Ensemble with class weights, SMOTE |
| Real-time prediction | Linear models, small trees |
| Batch prediction | Any algorithm |

## Ensemble Methods

### Why Ensemble?

**Ensemble** - combining multiple models for better performance.

**Benefits:**
- Reduce overfitting
- Improve accuracy
- More robust predictions
- Often win competitions

### Voting/Averaging

**Classification - Voting:**
```python
from sklearn.ensemble import VotingClassifier

# Hard voting (majority vote)
voting = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('svm', SVC())
    ],
    voting='hard'
)

# Soft voting (average probabilities)
voting = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('svm', SVC(probability=True))
    ],
    voting='soft'
)

voting.fit(X_train, y_train)
```

**Regression - Averaging:**
```python
from sklearn.ensemble import VotingRegressor

voting = VotingRegressor(
    estimators=[
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor()),
        ('gb', GradientBoostingRegressor())
    ]
)
```

### Stacking

**Concept:** Train a meta-model on predictions of base models.

```python
from sklearn.ensemble import StackingClassifier

stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('gb', GradientBoostingClassifier(n_estimators=100)),
        ('svm', SVC(probability=True))
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)
```

**When to use:**
- Diverse base models
- Have computational resources
- Squeeze out last % of performance

### Bagging

**Concept:** Train multiple models on random subsets of data.

```python
from sklearn.ensemble import BaggingClassifier

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    max_features=0.8,
    random_state=42
)
```

**Examples:** Random Forest is bagging of decision trees.

### Boosting

**Concept:** Sequentially train models, each correcting previous errors.

**AdaBoost:**
```python
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=1.0
)
```

**Gradient Boosting:** XGBoost, LightGBM, CatBoost

**When to use:** Almost always for tabular data

## Practical Considerations

### Interpretability vs Performance

**High interpretability:**
- Linear Regression
- Logistic Regression
- Decision Tree (shallow)
- Rule-based models

**Medium interpretability:**
- Random Forest (feature importance)
- Gradient Boosting (SHAP values)

**Low interpretability (Black box):**
- Deep Neural Networks
- SVM with RBF kernel
- Complex ensembles

**Trade-off:**
```python
# If interpretability is critical
if need_interpretability:
    model = LogisticRegression()
else:
    model = XGBClassifier()  # Better performance
```

### Training Time vs Prediction Time

| Algorithm | Training | Prediction |
|-----------|----------|------------|
| Naive Bayes | Very fast | Very fast |
| Linear models | Fast | Very fast |
| KNN | Very fast | Slow |
| Decision Tree | Fast | Fast |
| Random Forest | Medium | Medium |
| XGBoost | Slow | Fast |
| Neural Networks | Very slow | Fast (GPU) |

**Consider:**
- **Batch predictions** - training time matters more
- **Real-time predictions** - inference speed critical
- **Frequent retraining** - fast training important

### Memory Requirements

**Low memory:**
- Linear models
- Naive Bayes
- Single decision tree

**High memory:**
- KNN (stores all training data)
- Random Forest (many trees)
- Neural Networks (many parameters)

### Handling Imbalanced Data

**Techniques:**

```python
# Class weights
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(class_weight='balanced')

# Or manual weights
class_weights = {0: 1, 1: 10}  # Penalize minority class errors more
rf = RandomForestClassifier(class_weight=class_weights)
```

```python
# SMOTE (oversampling)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

```python
# Undersampling
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
```

**Algorithms good for imbalanced data:**
- Random Forest with class weights
- XGBoost with scale_pos_weight
- Logistic Regression with class weights

### Multi-class vs Binary

**Binary classification:**
- Most algorithms handle natively
- Easier to tune
- More metrics available (ROC-AUC)

**Multi-class:**
- Some algorithms need modification (SVM: OvR or OvO)
- Use macro/micro averaging for metrics
- Consider if classes are ordinal

```python
# Multi-class strategies
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

# One-vs-Rest (one classifier per class)
ovr = OneVsRestClassifier(SVC())

# One-vs-One (classifier for each pair)
ovo = OneVsOneClassifier(SVC())
```

### Online Learning

**Algorithms supporting incremental learning:**

```python
from sklearn.linear_model import SGDClassifier

# Can call partial_fit for online learning
sgd = SGDClassifier()

# Initial fit
sgd.fit(X_batch1, y_batch1)

# Update with new data
sgd.partial_fit(X_batch2, y_batch2)
```

**When needed:**
- Data too large for memory
- Continuous data stream
- Model needs frequent updates

## Model Selection Checklist

Before choosing a model, answer:

- [ ] What type of problem? (classification, regression, clustering)
- [ ] How much data? (small, medium, large)
- [ ] How many features? (low, high dimensional)
- [ ] Need interpretability?
- [ ] Computational constraints? (training time, memory)
- [ ] Inference speed requirements?
- [ ] Data characteristics? (linear, non-linear, sparse)
- [ ] Imbalanced classes?
- [ ] Cost of errors? (false positives vs false negatives)

## Example Selection Process

```python
def select_model(problem_type, n_samples, n_features, need_interpretability):
    """Helper function for model selection"""

    if problem_type == 'regression':
        if need_interpretability:
            return LinearRegression()
        elif n_samples < 1000:
            return Ridge()
        else:
            return XGBRegressor()

    elif problem_type == 'classification':
        if need_interpretability:
            return LogisticRegression()
        elif n_samples < 1000:
            return RandomForestClassifier(n_estimators=100)
        else:
            return XGBClassifier()

    elif problem_type == 'clustering':
        if n_samples < 10000:
            return KMeans(n_clusters=5)
        else:
            return MiniBatchKMeans(n_clusters=5)

# Usage
model = select_model(
    problem_type='classification',
    n_samples=50000,
    n_features=100,
    need_interpretability=False
)
```

## References

- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" (Aurélien Géron)
- "The Elements of Statistical Learning" (Hastie, Tibshirani, Friedman)
- scikit-learn Algorithm Cheat Sheet: https://scikit-learn.org/stable/tutorial/machine_learning_map/
- "Applied Predictive Modeling" (Kuhn & Johnson)
