# Hyperparameter Tuning

## Table of Contents

1. [Introduction](#introduction)
2. [Grid Search](#grid-search)
3. [Random Search](#random-search)
4. [Bayesian Optimization](#bayesian-optimization)
5. [Hyperband and Successive Halving](#hyperband-and-successive-halving)
6. [Genetic Algorithms](#genetic-algorithms)
7. [Advanced Methods](#advanced-methods)
8. [Practical Guidelines](#practical-guidelines)

## Introduction

**Hyperparameters** are model configuration settings that are set before training and control the learning process. They are distinct from **parameters** (model weights), which are learned from data.

### Parameters vs Hyperparameters

**Parameters:**
- Learned from training data
- Examples: weights in neural networks, coefficients in linear regression
- Optimized by training algorithm (gradient descent, etc.)

**Hyperparameters:**
- Set before training
- Examples: learning rate, number of trees, regularization strength
- Must be tuned manually or through search algorithms

### Common Hyperparameters

| Model Type | Common Hyperparameters |
|-----------|------------------------|
| **Linear Models** | Regularization strength (α, λ, C), penalty type (L1/L2) |
| **Decision Trees** | Max depth, min samples split/leaf, max features |
| **Random Forest** | Number of trees, max depth, min samples split, max features |
| **Gradient Boosting** | Learning rate, n_estimators, max depth, subsample |
| **SVM** | C, kernel, gamma, degree (polynomial) |
| **K-NN** | Number of neighbors (k), distance metric, weights |
| **Neural Networks** | Learning rate, batch size, number of layers/neurons, dropout rate |

### Why Tuning Matters

- Default hyperparameters rarely optimal
- Performance can vary dramatically (50% → 90% accuracy)
- Different datasets require different settings
- Proper tuning crucial for model comparison

### Search Space

The **search space** defines the range and type of hyperparameters to explore:

```python
# Example search space
param_space = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None],
    'n_estimators': [50, 100, 200, 500]
}
```

## Grid Search

**Grid Search** exhaustively tries all possible combinations of hyperparameters from predefined values.

### Methodology

Given hyperparameters $\{\lambda_1, \lambda_2, ..., \lambda_k\}$ with possible values:
- $\lambda_1 \in \{v_{1,1}, v_{1,2}, ..., v_{1,n_1}\}$
- $\lambda_2 \in \{v_{2,1}, v_{2,2}, ..., v_{2,n_2}\}$
- ...

Grid search evaluates all $n_1 \times n_2 \times ... \times n_k$ combinations.

### Implementation

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create grid search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,                           # 5-fold cross-validation
    scoring='accuracy',             # Metric to optimize
    n_jobs=-1,                      # Use all CPU cores
    verbose=2,                      # Print progress
    return_train_score=True         # Return training scores
)

# Fit grid search
grid_search.fit(X_train, y_train)

# Best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Use best model
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test score: {test_score:.4f}")
```

### Analyzing Results

```python
import pandas as pd

# Get detailed results
results = pd.DataFrame(grid_search.cv_results_)

# Show top 10 configurations
top_results = results.sort_values('rank_test_score').head(10)
print(top_results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])

# Visualize results
import matplotlib.pyplot as plt

# Example: plot score vs. one parameter
if 'param_max_depth' in results.columns:
    plt.figure(figsize=(10, 6))
    for n_est in param_grid['n_estimators']:
        mask = results['param_n_estimators'] == n_est
        subset = results[mask]
        plt.plot(subset['param_max_depth'], subset['mean_test_score'],
                marker='o', label=f'n_estimators={n_est}')
    plt.xlabel('max_depth')
    plt.ylabel('Mean CV Score')
    plt.legend()
    plt.title('Grid Search Results')
```

### Advantages and Disadvantages

**Advantages:**
- Guaranteed to find best combination in search space
- Reproducible results
- Parallelizable
- Simple to understand and implement

**Disadvantages:**
- Computationally expensive (exponential in number of hyperparameters)
- Wastes computation on unpromising regions
- Curse of dimensionality
- Limited by discrete grid

**Computational cost:**

For $k$ hyperparameters with $n$ values each, and $f$ CV folds:

$$\text{Total models} = n^k \times f$$

Example: 4 hyperparameters, 5 values each, 5-fold CV = $5^4 \times 5 = 3,125$ models

### Grid Search for Neural Networks

```python
from sklearn.neural_network import MLPClassifier

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01],
    'batch_size': [32, 64, 128]
}

grid_search = GridSearchCV(
    MLPClassifier(max_iter=1000, random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
```

## Random Search

**Random Search** samples hyperparameter combinations randomly from specified distributions.

### Methodology

Instead of exhaustive search:
1. Define probability distributions for each hyperparameter
2. Sample $n$ random combinations
3. Evaluate each combination
4. Select best performing configuration

### Mathematical Insight

**Bergstra & Bengio (2012)** showed that random search is more efficient than grid search when:
- Only few hyperparameters significantly affect performance
- Distributions are not known in advance

For $k$ hyperparameters where only $k_{important} \ll k$ matter:
- Grid search with $n$ points per dimension: $n^{k_{important}}$ effective samples
- Random search with $N$ samples: $N$ effective samples for important hyperparameters

### Implementation

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import numpy as np

# Define parameter distributions
param_distributions = {
    'n_estimators': randint(50, 500),                    # Discrete uniform
    'max_depth': randint(3, 20),                         # Discrete uniform
    'min_samples_split': randint(2, 20),                 # Discrete uniform
    'min_samples_leaf': randint(1, 10),                  # Discrete uniform
    'max_features': ['sqrt', 'log2', None],              # Categorical
    'bootstrap': [True, False]                           # Categorical
}

# Random search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=100,                     # Number of parameter settings sampled
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42,
    return_train_score=True
)

random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")
```

### Custom Distributions

```python
from scipy.stats import loguniform

# Log-uniform for learning rate (better for parameters spanning orders of magnitude)
param_distributions = {
    'learning_rate': loguniform(1e-5, 1e-1),  # Sample from log scale
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 20),
    'subsample': uniform(0.5, 0.5),           # Uniform between 0.5 and 1.0
    'colsample_bytree': uniform(0.5, 0.5)
}
```

### Advantages and Disadvantages

**Advantages:**
- More efficient than grid search for high-dimensional spaces
- Better exploration of search space
- Can find better solutions with fewer iterations
- Easy to parallelize
- Can stop anytime (no need to complete full search)

**Disadvantages:**
- No guarantee of finding optimal solution
- Results may vary between runs (unless random_state fixed)
- May miss optimal combination

**When to use:**
- Large hyperparameter space
- Limited computational budget
- Many hyperparameters, few are important
- Continuous hyperparameters

### Comparison: Grid vs Random Search

```python
# Theoretical example showing random search advantage
# Assume true optimal performance at learning_rate=0.01, other params don't matter

# Grid search: 3x3 grid = 9 evaluations
grid_lr = [0.001, 0.01, 0.1]
grid_other = [1, 2, 3]
# Only 1 out of 9 tries the optimal learning_rate

# Random search: 9 random samples
# Each sample has 1/3 chance of being near optimal learning rate
# Probability of finding optimal: 1 - (2/3)^9 ≈ 96%
```

## Bayesian Optimization

**Bayesian Optimization** uses probabilistic models to guide the search toward promising regions of the hyperparameter space.

### Key Concepts

**Surrogate Model:**
- Probabilistic model (typically Gaussian Process) of objective function
- Predicts performance for untried hyperparameters
- Provides uncertainty estimates

**Acquisition Function:**
- Determines which hyperparameters to try next
- Balances exploration (high uncertainty) vs exploitation (high predicted performance)

Common acquisition functions:
- **Expected Improvement (EI)**: Expected improvement over current best
- **Probability of Improvement (PI)**: Probability of improving
- **Upper Confidence Bound (UCB)**: $\mu(x) + \kappa\sigma(x)$

### Algorithm

1. Initialize with random evaluations
2. Fit surrogate model to observed (hyperparameters, performance) pairs
3. Use acquisition function to select next hyperparameters
4. Evaluate objective function at selected point
5. Update surrogate model
6. Repeat steps 3-5 until budget exhausted

### Implementation with Scikit-Optimize

```python
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# Define search space
search_spaces = {
    'n_estimators': Integer(50, 500),
    'max_depth': Integer(3, 20),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Categorical(['sqrt', 'log2']),
    'bootstrap': Categorical([True, False])
}

# Bayesian optimization
bayes_search = BayesSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    search_spaces=search_spaces,
    n_iter=50,                  # Number of parameter settings sampled
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

bayes_search.fit(X_train, y_train)

print(f"Best parameters: {bayes_search.best_params_}")
print(f"Best CV score: {bayes_search.best_score_:.4f}")
```

### Implementation with Optuna

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    """Objective function for Optuna."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }

    model = RandomForestClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()

    return score

# Create study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, n_jobs=-1)

print(f"Best parameters: {study.best_params}")
print(f"Best score: {study.best_value:.4f}")

# Visualization
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
```

### Implementation with Hyperopt

```python
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score

# Define search space
space = {
    'n_estimators': hp.choice('n_estimators', range(50, 500, 50)),
    'max_depth': hp.choice('max_depth', range(3, 20)),
    'min_samples_split': hp.choice('min_samples_split', range(2, 20)),
    'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 10)),
    'max_features': hp.choice('max_features', ['sqrt', 'log2']),
}

def objective(params):
    model = RandomForestClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    # Hyperopt minimizes, so return negative score
    return {'loss': -score, 'status': STATUS_OK}

# Optimize
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,  # Tree-structured Parzen Estimator
    max_evals=100,
    trials=trials
)

print(f"Best parameters: {best}")
```

### Advantages and Disadvantages

**Advantages:**
- More efficient than random/grid search
- Intelligently explores search space
- Handles continuous and discrete parameters
- Works well with limited evaluation budget
- Provides uncertainty estimates

**Disadvantages:**
- More complex to set up
- Surrogate model overhead
- May get stuck in local optima
- Less interpretable than grid search
- Harder to parallelize effectively

**When to use:**
- Expensive objective function (training takes long)
- Limited computational budget
- Continuous hyperparameters
- Need efficient search

## Hyperband and Successive Halving

**Successive Halving** and **Hyperband** adaptively allocate resources to promising configurations by early stopping poor performers.

### Successive Halving

**Algorithm:**
1. Start with $n$ random configurations
2. Train each for small number of iterations
3. Keep top half, discard bottom half
4. Double training time for survivors
5. Repeat until one configuration remains

**Resource allocation:**
- Round 1: $n$ configs, $r$ resources each → $n \times r$ total
- Round 2: $n/2$ configs, $2r$ resources each → $n \times r$ total
- Round 3: $n/4$ configs, $4r$ resources each → $n \times r$ total
- ...

### Hyperband

Extension of successive halving that handles the trade-off between many short runs vs few long runs.

**Key idea:** Run successive halving with different $n$ and $r$ budgets.

### Implementation (Scikit-learn)

```python
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from scipy.stats import randint

param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
}

halving_search = HalvingRandomSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    resource='n_estimators',  # Resource to increase
    max_resources=500,         # Maximum n_estimators
    min_resources=50,          # Minimum n_estimators
    factor=3,                  # Reduction factor
    cv=5,
    random_state=42,
    n_jobs=-1
)

halving_search.fit(X_train, y_train)

print(f"Best parameters: {halving_search.best_params_}")
print(f"Best score: {halving_search.best_score_:.4f}")
```

### Advantages and Disadvantages

**Advantages:**
- Very efficient for large search spaces
- Automatically handles early stopping
- Can evaluate many configurations quickly
- Good for neural networks and iterative algorithms

**Disadvantages:**
- Assumes performance is monotonic with resources
- May eliminate good configurations too early
- Best for algorithms where partial training is meaningful

**When to use:**
- Training is iterative (neural networks, boosting)
- Many configurations to try
- Clear resource that can be varied (epochs, trees, data size)

## Genetic Algorithms

**Genetic Algorithms** use evolutionary principles to search for optimal hyperparameters.

### Key Concepts

- **Population**: Set of hyperparameter configurations
- **Fitness**: Performance score
- **Selection**: Choose best performing configurations
- **Crossover**: Combine configurations to create offspring
- **Mutation**: Random changes to configurations

### Algorithm

1. Initialize random population
2. Evaluate fitness of each individual
3. Select best individuals (parents)
4. Create new individuals via crossover and mutation
5. Replace worst individuals with new ones
6. Repeat steps 2-5 for multiple generations

### Implementation with TPOT

```python
from tpot import TPOTClassifier

# TPOT: Automated ML using genetic programming
tpot = TPOTClassifier(
    generations=5,          # Number of iterations
    population_size=50,     # Number of individuals
    cv=5,
    random_state=42,
    verbosity=2,
    n_jobs=-1,
    max_time_mins=60       # Time budget
)

tpot.fit(X_train, y_train)
print(f"Test score: {tpot.score(X_test, y_test):.4f}")

# Export best pipeline
tpot.export('best_pipeline.py')
```

### Advantages and Disadvantages

**Advantages:**
- Can explore complex search spaces
- Handles interactions between hyperparameters
- Can optimize entire pipeline (preprocessing + model)

**Disadvantages:**
- Computationally expensive
- Many hyperparameters to tune (population size, mutation rate, etc.)
- May not converge to global optimum

## Advanced Methods

### Population-Based Training (PBT)

Combines random search with evolutionary methods:
- Trains population of models in parallel
- Periodically copies weights from high-performing to low-performing models
- Mutates hyperparameters

**Use case:** Deep learning with long training times

### Gradient-Based Optimization

Treat hyperparameters as differentiable and optimize via gradient descent.

**Challenges:**
- Many hyperparameters are discrete
- Requires computing gradients through entire training process
- Computationally expensive

### Multi-Fidelity Optimization

Evaluate configurations at different levels of fidelity:
- Small dataset subset
- Fewer training epochs
- Lower resolution images

**Advantage:** Quickly eliminate poor configurations

### Neural Architecture Search (NAS)

Automated search for optimal neural network architecture:
- Number of layers
- Layer types
- Connections between layers

**Methods:**
- Reinforcement learning
- Evolutionary algorithms
- Gradient-based (DARTS)

## Practical Guidelines

### General Strategy

**1. Start simple:**
```python
# First: Use default parameters
model = RandomForestClassifier()
baseline_score = cross_val_score(model, X_train, y_train, cv=5).mean()
print(f"Baseline: {baseline_score:.4f}")
```

**2. Coarse search:**
```python
# Second: Coarse random search over wide ranges
param_dist = {
    'n_estimators': randint(10, 1000),
    'max_depth': randint(1, 50),
}
random_search = RandomizedSearchCV(model, param_dist, n_iter=50, cv=5)
random_search.fit(X_train, y_train)
```

**3. Fine-tune:**
```python
# Third: Fine-grained grid search around best values
best_params = random_search.best_params_
param_grid = {
    'n_estimators': [best_params['n_estimators'] - 20,
                     best_params['n_estimators'],
                     best_params['n_estimators'] + 20],
    'max_depth': [best_params['max_depth'] - 2,
                  best_params['max_depth'],
                  best_params['max_depth'] + 2],
}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

### Choosing Search Method

| Method | Best For | Computational Budget | Expertise Required |
|--------|----------|---------------------|-------------------|
| **Grid Search** | Small search space, few hyperparameters | High | Low |
| **Random Search** | Large search space, moderate budget | Medium | Low |
| **Bayesian Optimization** | Expensive evaluations, continuous params | Low-Medium | Medium |
| **Hyperband** | Many configs, iterative algorithms | Low | Medium |
| **Genetic Algorithms** | Complex pipelines, non-standard optimization | High | High |

### Common Hyperparameter Ranges

```python
# Decision Trees / Random Forest
{
    'max_depth': [3, 5, 7, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None]
}

# Gradient Boosting (XGBoost, LightGBM)
{
    'n_estimators': [100, 200, 500, 1000],
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.3],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5, 7]
}

# SVM
{
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# Neural Networks (scikit-learn)
{
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50, 25)],
    'activation': ['relu', 'tanh', 'logistic'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128, 256]
}
```

### Search Space Design

**Use log scale for parameters spanning orders of magnitude:**

```python
# Learning rate: 10^-5 to 10^-1
from scipy.stats import loguniform
'learning_rate': loguniform(1e-5, 1e-1)

# Or in grid search
'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
```

**Consider parameter interactions:**

```python
# Max depth and min_samples_leaf interact
# Deep trees should have larger min_samples_leaf
param_grid = [
    {'max_depth': [3, 5], 'min_samples_leaf': [1, 2]},
    {'max_depth': [10, 15], 'min_samples_leaf': [5, 10]},
    {'max_depth': [None], 'min_samples_leaf': [10, 20]}
]
```

### Preventing Overfitting During Tuning

**Use nested cross-validation:**

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# Outer CV for unbiased evaluation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Inner CV for hyperparameter tuning
clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv)

# Outer CV for evaluation
nested_scores = cross_val_score(clf, X, y, cv=outer_cv)
print(f"Nested CV score: {nested_scores.mean():.4f} (+/- {nested_scores.std():.4f})")
```

**Separate validation from test set:**

```python
# Split data three ways
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Tune on train set (uses internal CV)
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Final evaluation on held-out test set (only once!)
final_score = grid_search.score(X_test, y_test)
```

### Computational Efficiency

**1. Parallelize:**
```python
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)  # Use all cores
```

**2. Use early stopping:**
```python
# For XGBoost
param_grid = {
    'n_estimators': [1000],  # Use large value
    'early_stopping_rounds': [50],  # Stop if no improvement
}
```

**3. Progressive search:**
```python
# Start with subset of data
X_subset, _, y_subset, _ = train_test_split(X_train, y_train, train_size=0.2)

# Quick search on subset
quick_search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=3)
quick_search.fit(X_subset, y_subset)

# Refine with full data
refined_search = GridSearchCV(model, refined_param_grid, cv=5)
refined_search.fit(X_train, y_train)
```

**4. Cache results:**
```python
# Store evaluated configurations to avoid recomputation
results_cache = {}

def cached_objective(params):
    params_key = frozenset(params.items())
    if params_key not in results_cache:
        results_cache[params_key] = evaluate_model(params)
    return results_cache[params_key]
```

### Diagnostics and Analysis

**Learning curves vs hyperparameter:**

```python
from sklearn.model_selection import validation_curve

param_range = np.arange(1, 21)
train_scores, val_scores = validation_curve(
    DecisionTreeClassifier(), X, y,
    param_name="max_depth",
    param_range=param_range,
    cv=5
)

plt.plot(param_range, train_scores.mean(axis=1), label='Training')
plt.plot(param_range, val_scores.mean(axis=1), label='Validation')
plt.xlabel('max_depth')
plt.ylabel('Score')
plt.legend()
```

**Parameter importance:**

```python
# With Optuna
importance = optuna.importance.get_param_importances(study)
print(importance)

# Visualize
optuna.visualization.plot_param_importances(study)
```

**Parallel coordinate plot:**

```python
# Visualize relationship between hyperparameters and performance
optuna.visualization.plot_parallel_coordinate(study)
```

### Reporting Results

**Complete reporting:**

```python
print("Hyperparameter Tuning Results:")
print(f"  Method: Grid Search")
print(f"  Search space size: {len(ParameterGrid(param_grid))}")
print(f"  CV folds: 5")
print(f"  Scoring metric: accuracy")
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test, y_test):.4f}")
print(f"\nTotal models trained: {len(grid_search.cv_results_['params'])}")
print(f"Time taken: {grid_search.refit_time_:.2f} seconds")
```

### Common Pitfalls

**1. Tuning on test set:**
```python
# WRONG
best_params = None
best_score = 0
for params in param_grid:
    model.set_params(**params)
    score = model.fit(X_train, y_train).score(X_test, y_test)  # Using test set!
    if score > best_score:
        best_score = score
        best_params = params

# CORRECT
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
test_score = grid_search.score(X_test, y_test)  # Only evaluate once
```

**2. Not scaling data:**
```python
# WRONG
grid_search = GridSearchCV(SVM(), param_grid, cv=5)
grid_search.fit(X_train, y_train)  # SVM needs scaled data!

# CORRECT
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVM())
])
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

**3. Data leakage:**
```python
# WRONG
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fits on entire dataset!
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_scaled, y)

# CORRECT
pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X, y)  # Scaling done within each fold
```

## References

- Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization
- Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms
- Li, L., et al. (2017). Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization
- Feurer, M., & Hutter, F. (2019). Hyperparameter Optimization. In AutoML: Methods, Systems, Challenges
- Akiba, T., et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework
