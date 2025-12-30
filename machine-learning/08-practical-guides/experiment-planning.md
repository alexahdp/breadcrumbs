# Experiment Planning

## Table of Contents

1. [The Scientific Method in ML](#the-scientific-method-in-ml)
2. [Defining Objectives](#defining-objectives)
3. [Hypothesis Formulation](#hypothesis-formulation)
4. [Experimental Design](#experimental-design)
5. [Metrics and Evaluation](#metrics-and-evaluation)
6. [Baseline Establishment](#baseline-establishment)
7. [Experiment Execution](#experiment-execution)
8. [Common Pitfalls](#common-pitfalls)
9. [Documentation and Reproducibility](#documentation-and-reproducibility)

## The Scientific Method in ML

Machine learning experiments should follow the scientific method to ensure rigorous, reproducible results.

### The ML Experiment Cycle

1. **Observe** - analyze the problem and existing data
2. **Question** - formulate what you want to improve
3. **Hypothesize** - propose a solution or approach
4. **Predict** - estimate expected outcomes
5. **Test** - run controlled experiments
6. **Analyze** - evaluate results objectively
7. **Iterate** - refine based on findings

**Key principle:** Change one variable at a time to understand what actually drives improvements.

## Defining Objectives

### SMART Goals Framework

Objectives should be:
- **Specific** - clearly defined problem and success criteria
- **Measurable** - quantifiable metrics
- **Achievable** - realistic given constraints
- **Relevant** - aligned with business/research goals
- **Time-bound** - clear deadlines (when applicable)

### Types of ML Objectives

**Business objectives:**
- Reduce customer churn by 15%
- Increase recommendation click-through rate
- Decrease fraud detection false positives

**Technical objectives:**
- Improve model F1-score from 0.75 to 0.85
- Reduce inference latency below 100ms
- Achieve 95% accuracy on test set

**Example:**
```
Bad: "Make the model better"
Good: "Increase precision from 0.72 to 0.80 while maintaining recall above 0.65"
```

### Problem Type Identification

Clearly identify your problem type:
- **Regression** - predicting continuous values
- **Binary classification** - two-class prediction
- **Multi-class classification** - multiple exclusive classes
- **Multi-label classification** - multiple non-exclusive labels
- **Clustering** - grouping similar instances
- **Ranking** - ordering items by relevance

## Hypothesis Formulation

### Creating Testable Hypotheses

A good hypothesis is:
- **Falsifiable** - can be proven wrong
- **Specific** - clearly states the proposed change
- **Measurable** - includes expected quantitative outcome

**Hypothesis template:**
```
If we [change/add/remove X],
then [metric Y] will [increase/decrease] by [Z amount/percent],
because [reasoning based on theory or observation].
```

### Example Hypotheses

**Good:**
- "If we add interaction features between age and income, then model RMSE will decrease by at least 5%, because these features likely have non-linear relationships with house prices."

**Bad:**
- "Adding more features will improve the model" (not specific or measurable)
- "This architecture works well" (not testable)

### Hypothesis Categories

**Data hypotheses:**
- Adding more training examples will reduce overfitting
- Removing outliers beyond 3σ will improve model stability
- Balancing classes will increase minority class recall

**Feature hypotheses:**
- Log-transforming skewed features will improve linear model performance
- Adding polynomial features will capture non-linear relationships
- Feature X is highly correlated with target and should improve predictions

**Model hypotheses:**
- Random Forest will outperform Logistic Regression for this non-linear problem
- Increasing tree depth to 10 will reduce underfitting
- Adding dropout with p=0.3 will reduce overfitting in neural network

## Experimental Design

### Train/Validation/Test Split

**Standard split:**
- **Training set (60-80%)** - model learns from this data
- **Validation set (10-20%)** - used for hyperparameter tuning and model selection
- **Test set (10-20%)** - final evaluation, touched only once

```python
from sklearn.model_selection import train_test_split

# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: separate train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)
# Results in 60% train, 20% validation, 20% test
```

### Cross-Validation Strategies

**K-Fold Cross-Validation:**

$$CV_{score} = \frac{1}{k}\sum_{i=1}^{k} Score_i$$

where each $Score_i$ is computed on a different fold.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
print(f"CV F1: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

**When to use:**
- Standard k-fold: balanced datasets, independent observations
- Stratified k-fold: imbalanced datasets (preserves class distribution)
- Time series split: temporal data (respects time ordering)
- Group k-fold: grouped data (ensures groups don't split across folds)

### Controlling for Randomness

**Set random seeds:**
```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

**Run multiple trials:**
- Run each experiment with different random seeds (e.g., 3-5 times)
- Report mean and standard deviation of results
- Use statistical tests to verify significance

## Metrics and Evaluation

### Selecting Appropriate Metrics

Metrics should align with business objectives and problem characteristics.

**Regression metrics:**
- **MAE** - robust to outliers, interpretable in original units
- **RMSE** - penalizes large errors more heavily
- **R²** - proportion of variance explained

**Classification metrics:**
- **Accuracy** - overall correctness (use only for balanced datasets)
- **Precision** - minimize false positives
- **Recall** - minimize false negatives
- **F1-score** - balance between precision and recall
- **ROC-AUC** - threshold-independent performance

### Primary vs Secondary Metrics

**Primary metric** - the main objective:
- Must directly align with business goal
- Used for final model selection
- Example: F1-score for balanced classification

**Secondary metrics** - monitoring additional aspects:
- Computational cost (training time, inference latency)
- Model interpretability
- Fairness metrics
- Memory usage

### Statistical Significance

Don't trust single-number improvements. Use statistical tests:

**Paired t-test for comparing two models:**
```python
from scipy import stats

# Cross-validation scores for two models
model1_scores = [0.85, 0.87, 0.84, 0.86, 0.85]
model2_scores = [0.88, 0.89, 0.87, 0.88, 0.90]

t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Difference is statistically significant")
```

## Baseline Establishment

### Why Baselines Matter

A baseline provides:
- Reference point for measuring improvement
- Reality check on model complexity
- Quick assessment of problem difficulty

### Types of Baselines

**Simple baselines:**
- **Classification:** majority class predictor, random predictor
- **Regression:** mean predictor, median predictor
- **Time series:** last value, moving average

**Stronger baselines:**
- Simple models: Linear Regression, Logistic Regression
- Previous best model
- Domain-specific heuristics
- Human-level performance

### Example Baseline Implementation

```python
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score

# Classification baseline: most frequent class
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
baseline_acc = accuracy_score(y_test, dummy.predict(X_test))

print(f"Baseline accuracy: {baseline_acc:.3f}")
print(f"Model must beat {baseline_acc:.3f} to be useful")
```

## Experiment Execution

### Systematic Parameter Exploration

**Grid search** - exhaustive search over parameter grid:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV F1: {grid_search.best_score_:.3f}")
```

**Random search** - random sampling (more efficient for large spaces):
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    DecisionTreeClassifier(),
    param_distributions,
    n_iter=50,  # number of random combinations
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)
```

### Experiment Tracking

Track every experiment with:
- **Date and time**
- **Hypothesis being tested**
- **Data version** (size, preprocessing applied)
- **Features used**
- **Model architecture and hyperparameters**
- **Random seed**
- **Results** (all metrics, not just primary)
- **Notes** (observations, issues encountered)

**Using MLflow for tracking:**
```python
import mlflow

mlflow.start_run()

# Log parameters
mlflow.log_param("max_depth", 5)
mlflow.log_param("n_estimators", 100)

# Log metrics
mlflow.log_metric("train_f1", 0.85)
mlflow.log_metric("val_f1", 0.82)
mlflow.log_metric("test_f1", 0.81)

# Log model
mlflow.sklearn.log_model(model, "random_forest_model")

mlflow.end_run()
```

## Common Pitfalls

### Data Leakage

**Definition:** Information from outside the training set influencing model training.

**Common sources:**
1. **Leakage from future** - using future information to predict past
2. **Leakage from test set** - test data influences preprocessing or feature selection
3. **Target leakage** - features that wouldn't be available at prediction time

**Example of leakage:**
```python
# WRONG: Scaling before split (test data influences training)
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled)

# CORRECT: Scaling after split
X_train, X_test = train_test_split(X)
scaler.fit(X_train)  # fit only on training data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Test Set Reuse

**Problem:** Repeatedly evaluating on test set leads to overfitting on test data.

**Solution:**
- Use validation set for model selection
- Touch test set only once for final evaluation
- If you must iterate, create a new held-out test set

### P-Hacking

**Problem:** Running many experiments and cherry-picking significant results.

**Solutions:**
- Pre-register hypotheses before experimenting
- Apply multiple testing corrections (Bonferroni)
- Report all experiments, not just successful ones

**Bonferroni correction:**
$$\alpha_{corrected} = \frac{\alpha}{n}$$

where $n$ is the number of hypotheses tested.

### Ignoring Computational Costs

**Considerations:**
- **Training time** - can you iterate quickly enough?
- **Inference latency** - meets production requirements?
- **Memory footprint** - fits in production environment?
- **Cost** - GPU hours, API calls within budget?

A model with 1% better accuracy but 10x inference time may not be worth it.

## Documentation and Reproducibility

### Experiment Documentation Template

```markdown
## Experiment: [Short descriptive name]

**Date:** 2024-01-15
**Hypothesis:** Adding polynomial features will improve R² by 10%

### Data
- Dataset: housing_v2.csv
- Size: 15,000 samples
- Split: 60/20/20 train/val/test
- Preprocessing: StandardScaler, outlier removal (z > 3)

### Model
- Algorithm: Ridge Regression
- Hyperparameters: alpha=1.0
- Features: 25 (including 10 polynomial features)

### Results
| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| R²     | 0.87  | 0.82       | 0.81 |
| RMSE   | 24.5k | 28.3k      | 29.1k|

### Conclusion
Hypothesis supported. R² improved from 0.73 to 0.81 (+11%).
Polynomial features capturing important non-linearities.

### Next Steps
- Try degree-3 polynomials
- Investigate feature importance
```

### Reproducibility Checklist

Ensure experiments can be reproduced:
- [ ] Random seeds set for all libraries
- [ ] Exact package versions recorded (`requirements.txt`)
- [ ] Data version tracked (checksums, version control)
- [ ] Code version controlled (git commit hash)
- [ ] Preprocessing steps documented
- [ ] Hyperparameters logged
- [ ] Environment details recorded (OS, hardware)

### Version Control Best Practices

```bash
# Tag important experiments
git tag -a exp_v1.2 -m "Baseline Random Forest - F1: 0.85"

# Include data versioning
dvc add data/processed/train.csv
dvc push

# Track experiments in separate branches
git checkout -b experiment/polynomial-features
```

### Requirements File

```txt
# requirements.txt with exact versions
numpy==1.24.3
pandas==2.0.2
scikit-learn==1.3.0
matplotlib==3.7.1
```

## References

- "Guidelines for Responsible and Human-Centered Use of Explainable AI" (Liao et al.)
- "Machine Learning Yearning" (Andrew Ng)
- "Designing Machine Learning Systems" (Chip Huyen)
- "Experimental Design and Analysis" (Howard et al.)
