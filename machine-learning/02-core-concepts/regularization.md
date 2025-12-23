# Regularization

## Table of Contents

1. [Introduction](#introduction)
2. [L2 Regularization (Ridge)](#l2-regularization-ridge)
3. [L1 Regularization (Lasso)](#l1-regularization-lasso)
4. [Elastic Net](#elastic-net)
5. [Dropout](#dropout)
6. [Early Stopping](#early-stopping)
7. [Data Augmentation](#data-augmentation)
8. [Other Regularization Techniques](#other-regularization-techniques)
9. [Choosing Regularization Methods](#choosing-regularization-methods)

## Introduction

**Regularization** is a set of techniques to prevent overfitting by adding constraints to the model, thereby reducing its complexity and improving generalization to new data.

### Why Regularization?

Without regularization, models tend to:
- Memorize training data including noise
- Develop large weight values that are sensitive to small changes
- Perform poorly on unseen data (high variance)

### Key Idea

Regularization adds a penalty term to the loss function that discourages complexity:

$$L_{regularized} = L_{original} + \lambda \cdot R(\theta)$$

where:
- $L_{original}$ is the original loss function
- $R(\theta)$ is the regularization term (penalty on model parameters)
- $\lambda$ is the regularization strength (hyperparameter)

### Types of Regularization

- **Explicit**: Directly modify the loss function (L1, L2)
- **Implicit**: Modify training process (dropout, early stopping, data augmentation)

## L2 Regularization (Ridge)

**L2 Regularization** (also called **Ridge Regularization** or **Weight Decay**) adds a penalty proportional to the square of the magnitude of coefficients.

### Mathematical Formulation

For linear regression, the objective becomes:

$$L_{Ridge} = \frac{1}{2n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda\sum_{j=1}^{p}\theta_j^2$$

Or in vector form:

$$L_{Ridge} = \frac{1}{2n}||y - X\theta||^2_2 + \lambda||\theta||^2_2$$

where:
- $\lambda \geq 0$ is the regularization parameter
- $||\theta||^2_2 = \sum_{j=1}^{p}\theta_j^2$ is the L2 norm (squared)
- Typically we don't penalize the bias term $\theta_0$

### Properties

**Effect on weights:**
- Shrinks all weights toward zero
- Keeps all features (weights rarely become exactly zero)
- Larger $\lambda$ → stronger shrinkage

**Analytical solution:**

For linear regression, Ridge has a closed-form solution:

$$\theta = (X^TX + \lambda I)^{-1}X^Ty$$

The regularization term $\lambda I$ ensures the matrix is invertible even when $X^TX$ is singular.

**When to use:**
- All features are potentially relevant
- Features are correlated (multicollinearity)
- Want smooth, stable model
- Default choice for regularization

### Implementation

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Important: scale features before regularization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge regression
ridge = Ridge(alpha=1.0)  # alpha is λ
ridge.fit(X_train_scaled, y_train)

# Predictions
y_pred = ridge.predict(X_test_scaled)

# Examine coefficients
print("Coefficients:", ridge.coef_)
print("Intercept:", ridge.intercept_)
```

### Finding Optimal Lambda

```python
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

# Try different alpha values
alphas = np.logspace(-4, 4, 50)
scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    score = cross_val_score(ridge, X_train_scaled, y_train, cv=5,
                           scoring='neg_mean_squared_error').mean()
    scores.append(-score)

# Plot validation curve
plt.plot(alphas, scores)
plt.xscale('log')
plt.xlabel('Alpha (λ)')
plt.ylabel('MSE')
plt.title('Ridge Regularization Strength')

# Best alpha
best_alpha = alphas[np.argmin(scores)]
print(f"Optimal alpha: {best_alpha}")
```

### Ridge for Classification

```python
from sklearn.linear_model import RidgeClassifier

# Ridge classifier (uses Ridge regression for classification)
ridge_clf = RidgeClassifier(alpha=1.0)
ridge_clf.fit(X_train_scaled, y_train)

# Or use LogisticRegression with L2 penalty
from sklearn.linear_model import LogisticRegression
logreg_l2 = LogisticRegression(penalty='l2', C=1.0)  # C = 1/λ
logreg_l2.fit(X_train_scaled, y_train)
```

## L1 Regularization (Lasso)

**L1 Regularization** (Least Absolute Shrinkage and Selection Operator - **Lasso**) adds a penalty proportional to the absolute value of coefficients.

### Mathematical Formulation

$$L_{Lasso} = \frac{1}{2n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda\sum_{j=1}^{p}|\theta_j|$$

Or in vector form:

$$L_{Lasso} = \frac{1}{2n}||y - X\theta||^2_2 + \lambda||\theta||_1$$

where $||\theta||_1 = \sum_{j=1}^{p}|\theta_j|$ is the L1 norm.

### Properties

**Effect on weights:**
- Drives some weights exactly to zero
- Performs automatic feature selection
- Creates sparse models
- Larger $\lambda$ → more features eliminated

**No closed-form solution:**
- Must be solved iteratively (e.g., coordinate descent)
- More computationally expensive than Ridge

**When to use:**
- Feature selection is desired
- Believe only subset of features are relevant
- Want interpretable model with fewer features
- High-dimensional data with many irrelevant features

### Implementation

```python
from sklearn.linear_model import Lasso

# Lasso regression
lasso = Lasso(alpha=0.1, max_iter=10000)
lasso.fit(X_train_scaled, y_train)

# Check which features were selected
feature_importance = np.abs(lasso.coef_)
selected_features = feature_importance > 0
print(f"Selected {selected_features.sum()} out of {len(selected_features)} features")

# Features with non-zero coefficients
feature_names = X.columns  # if using pandas
selected_feature_names = feature_names[selected_features]
print("Selected features:", selected_feature_names.tolist())
```

### Lasso Path

Visualize how coefficients change with regularization strength:

```python
from sklearn.linear_model import lasso_path

# Compute Lasso path
alphas, coefs, _ = lasso_path(X_train_scaled, y_train)

# Plot coefficient paths
plt.figure(figsize=(10, 6))
for coef in coefs:
    plt.plot(alphas, coef)
plt.xscale('log')
plt.xlabel('Alpha (λ)')
plt.ylabel('Coefficients')
plt.title('Lasso Coefficient Paths')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
```

### Lasso for Classification

```python
from sklearn.linear_model import LogisticRegression

# Logistic regression with L1 penalty
logreg_l1 = LogisticRegression(penalty='l1', C=1.0, solver='liblinear')
logreg_l1.fit(X_train_scaled, y_train)

# Check sparsity
n_nonzero = np.sum(logreg_l1.coef_ != 0)
print(f"Non-zero coefficients: {n_nonzero}")
```

## Elastic Net

**Elastic Net** combines both L1 and L2 regularization, getting benefits of both.

### Mathematical Formulation

$$L_{ElasticNet} = \frac{1}{2n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda\left(\alpha||\theta||_1 + \frac{1-\alpha}{2}||\theta||^2_2\right)$$

where:
- $\lambda$ controls overall regularization strength
- $\alpha \in [0, 1]$ controls the balance between L1 and L2:
  - $\alpha = 0$: Pure Ridge
  - $\alpha = 1$: Pure Lasso
  - $0 < \alpha < 1$: Mix of both

### Properties

**Advantages over Lasso:**
- Handles correlated features better
- Can select more than $n$ features when $p > n$
- More stable selection

**Advantages over Ridge:**
- Performs feature selection
- Creates sparse models

**When to use:**
- Features are correlated and want to select groups
- Want both regularization and feature selection
- Lasso is too aggressive in feature elimination

### Implementation

```python
from sklearn.linear_model import ElasticNet

# Elastic Net regression
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
# alpha is λ, l1_ratio is α (mix between L1 and L2)
elastic.fit(X_train_scaled, y_train)

# l1_ratio=0.5 means equal weight to L1 and L2
# l1_ratio=0.0 is Ridge
# l1_ratio=1.0 is Lasso
```

### Finding Optimal Parameters

```python
from sklearn.model_selection import GridSearchCV

# Grid search for both alpha and l1_ratio
param_grid = {
    'alpha': np.logspace(-4, 1, 20),
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
}

elastic_cv = GridSearchCV(ElasticNet(max_iter=10000), param_grid, cv=5,
                          scoring='neg_mean_squared_error')
elastic_cv.fit(X_train_scaled, y_train)

print(f"Best alpha: {elastic_cv.best_params_['alpha']}")
print(f"Best l1_ratio: {elastic_cv.best_params_['l1_ratio']}")
```

## L1 vs L2: Geometric Interpretation

### Constraint Regions

**L2 (Ridge):** Circular constraint region
- $||\theta||^2_2 \leq t$
- Solution rarely hits axes (weights rarely zero)

**L1 (Lasso):** Diamond-shaped constraint region
- $||\theta||_1 \leq t$
- Corners on axes encourage sparse solutions

```
L2 Constraint Region          L1 Constraint Region
       θ₂                            θ₂
        │                             │
    ╱───│───╲                      ╱  │  ╲
  ╱     │     ╲                  ╱    │    ╲
 │      │      │                ╱     │     ╲
─┼──────┼──────┼─ θ₁           ─┼─────┼─────┼─ θ₁
 │      │      │                ╲     │     ╱
  ╲     │     ╱                  ╲    │    ╱
    ╲───│───╱                      ╲  │  ╱
        │                             │
```

The contours of the loss function are more likely to hit the corners of the L1 constraint region (where weights are zero), leading to sparse solutions.

## Dropout

**Dropout** is a regularization technique primarily used in neural networks that randomly "drops out" (sets to zero) a fraction of neurons during training.

### How It Works

During training:
1. Each neuron is kept with probability $p$ (or dropped with probability $1-p$)
2. This creates a "thinned" network for each training batch
3. Forces network to learn redundant representations

At test time:
- Use all neurons
- Scale outputs by $p$ to account for more active neurons

### Mathematical Formulation

For a layer with output $h$:

**Training:**

$$\tilde{h} = m \odot h$$

where $m \sim Bernoulli(p)$ is a binary mask.

**Testing:**

$$h_{test} = p \cdot h$$

### Properties

**Why it works:**
- Prevents co-adaptation of neurons
- Approximates training ensemble of many sub-networks
- Reduces reliance on specific neurons
- Similar to L2 regularization in linear models

**When to use:**
- Training neural networks
- Model is overfitting
- Have enough training data

### Implementation in Neural Networks

```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout after activation
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # No dropout on output layer
        return x

# Model automatically handles train vs test mode
model = SimpleNN(100, 256, 10, dropout_rate=0.5)

# Training mode (dropout active)
model.train()
output = model(data)

# Evaluation mode (dropout inactive)
model.eval()
with torch.no_grad():
    predictions = model(test_data)
```

### Keras/TensorFlow Implementation

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(input_dim,)),
    layers.Dropout(0.5),  # Drop 50% of neurons
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),  # Drop 30% of neurons
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Dropout Variants

**Spatial Dropout** (for CNNs):
- Drops entire feature maps instead of individual pixels
- Useful for convolutional layers

```python
from tensorflow.keras.layers import SpatialDropout2D

layers.SpatialDropout2D(0.2)
```

**DropConnect:**
- Drops connections (weights) instead of neurons
- More aggressive regularization

**Variational Dropout:**
- Uses same dropout mask for all timesteps in RNN
- Better for recurrent networks

## Early Stopping

**Early Stopping** monitors model performance on validation set and stops training when performance starts to degrade.

### How It Works

1. Train model while monitoring validation loss
2. Keep track of best validation performance
3. Stop training if validation loss doesn't improve for $n$ epochs (patience)
4. Restore model to best weights

### Mathematical Intuition

Training loss decreases monotonically, but validation loss eventually increases:

```
Loss
  │
  │  Training Loss
  │     ╲
  │      ╲___________
  │
  │  Validation Loss
  │     ╲
  │      ╲
  │       ╲___╱─────
  │          ↑
  │      Stop here
  └────────────────────→ Epochs
```

### Implementation

```python
from sklearn.model_selection import train_test_split

# Split training data into train and validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# PyTorch example
best_val_loss = float('inf')
patience = 10
patience_counter = 0
best_model_weights = None

for epoch in range(max_epochs):
    # Training
    model.train()
    train_loss = train_one_epoch(model, train_loader)

    # Validation
    model.eval()
    val_loss = validate(model, val_loader)

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# Restore best model
model.load_state_dict(best_model_weights)
```

### Keras Implementation

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=1000,
    callbacks=[early_stop]
)

# Training will stop automatically when validation loss stops improving
```

### Practical Considerations

**Patience parameter:**
- Too small: Stop too early, underfit
- Too large: Train too long, overfit
- Typical values: 5-20 epochs

**Monitor metric:**
- Classification: validation accuracy or F1
- Regression: validation MSE/MAE
- Generally: validation loss

## Data Augmentation

**Data Augmentation** artificially increases training data size by creating modified versions of existing data.

### Why It Works

- Increases effective training set size
- Makes model invariant to transformations
- Reduces overfitting
- Acts as regularization without explicit penalty term

### Image Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define augmentation strategy
datagen = ImageDataGenerator(
    rotation_range=20,           # Random rotations up to 20 degrees
    width_shift_range=0.2,       # Horizontal shifts
    height_shift_range=0.2,      # Vertical shifts
    horizontal_flip=True,        # Random horizontal flips
    zoom_range=0.2,             # Random zoom
    shear_range=0.2,            # Shear transformations
    fill_mode='nearest'         # How to fill empty pixels
)

# Fit on training data
datagen.fit(X_train)

# Train with augmented data
model.fit(datagen.flow(X_train, y_train, batch_size=32),
          epochs=50)
```

### Advanced Augmentation (Albumentations)

```python
import albumentations as A

# More sophisticated augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.2),
    A.Rotate(limit=15, p=0.5),
    A.ColorJitter(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Apply to image
augmented = transform(image=image)
augmented_image = augmented['image']
```

### Text Augmentation

```python
# Synonym replacement
def synonym_replacement(text, n=2):
    """Replace n words with synonyms"""
    # Implementation using WordNet or similar
    pass

# Back-translation
def back_translate(text, intermediate_lang='de'):
    """Translate to another language and back"""
    # English -> German -> English
    pass

# Random insertion/deletion
def random_insertion(text, n=1):
    """Insert n random words"""
    pass
```

### Tabular Data Augmentation

```python
# SMOTE for imbalanced classification
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Gaussian noise injection
X_augmented = X_train + np.random.normal(0, 0.01, X_train.shape)
```

## Other Regularization Techniques

### Batch Normalization

Normalizes layer inputs, which acts as regularization:

```python
from tensorflow.keras.layers import BatchNormalization

model = keras.Sequential([
    layers.Dense(256),
    BatchNormalization(),  # Normalize before activation
    layers.Activation('relu'),
    layers.Dense(128),
    BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(num_classes, activation='softmax')
])
```

### Weight Constraints

Constrain weight magnitudes directly:

```python
from tensorflow.keras.constraints import max_norm

# Constrain weights to have max norm of 3
model.add(layers.Dense(64, kernel_constraint=max_norm(3)))
```

### Label Smoothing

Softens hard targets to prevent overconfidence:

```python
# Instead of [0, 1, 0]
# Use [ε/K, 1-ε+ε/K, ε/K] where ε is smoothing parameter

from tensorflow.keras.losses import CategoricalCrossentropy

loss = CategoricalCrossentropy(label_smoothing=0.1)
```

### Noise Injection

Add noise to inputs or weights:

```python
from tensorflow.keras.layers import GaussianNoise

model.add(GaussianNoise(0.1))  # Add Gaussian noise with std=0.1
```

## Choosing Regularization Methods

### Decision Guide

| Scenario | Recommended Regularization |
|----------|---------------------------|
| Linear model, all features relevant | L2 (Ridge) |
| Linear model, feature selection needed | L1 (Lasso) or Elastic Net |
| Correlated features + feature selection | Elastic Net |
| Neural network, image data | Dropout + Data Augmentation + Batch Norm |
| Neural network, tabular data | Dropout + L2 weight decay |
| Small dataset | Strong regularization + Data Augmentation |
| Very deep network | Dropout + Batch Norm + Early Stopping |
| Sequential data (RNN) | Dropout + Early Stopping |

### Combining Techniques

Multiple regularization techniques can be combined:

```python
from tensorflow.keras.regularizers import l2

model = keras.Sequential([
    layers.Dense(256, activation='relu',
                kernel_regularizer=l2(0.01)),  # L2 regularization
    layers.BatchNormalization(),                # Batch normalization
    layers.Dropout(0.5),                       # Dropout
    layers.Dense(128, activation='relu',
                kernel_regularizer=l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# Add early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(X_train, y_train,
                   validation_split=0.2,
                   epochs=100,
                   callbacks=[early_stop])
```

### Hyperparameter Selection

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier

# Grid search for regularization strength
param_grid = {
    'alpha': np.logspace(-6, 2, 20),
    'penalty': ['l1', 'l2', 'elasticnet'],
    'l1_ratio': [0.15, 0.5, 0.85]  # Only used if penalty='elasticnet'
}

grid_search = GridSearchCV(
    SGDClassifier(max_iter=1000),
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
print(f"Best parameters: {grid_search.best_params_}")
```

## Practical Guidelines

### General Principles

1. **Always use some regularization**: Prevents overfitting
2. **Start with default values**: Then tune if needed
3. **Scale features**: Critical for L1/L2 to work properly
4. **Monitor validation performance**: Regularization should improve it
5. **Don't regularize bias term**: Only regularize weights

### Scaling Matters

```python
# BAD: Regularization without scaling
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)  # Features have different scales!

# GOOD: Scale first
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)
```

### Regularization Strength

**Too little regularization ($\lambda$ too small):**
- Model still overfits
- Training and validation error have large gap

**Too much regularization ($\lambda$ too large):**
- Model underfits
- Both training and validation error are high

**Just right:**
- Validation error is minimized
- Small gap between training and validation error

## References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning, Chapter 7
- Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting
- Zou, H., & Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net
