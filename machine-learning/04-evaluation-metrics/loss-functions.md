# Loss Functions

## Table of Contents

1. [Overview](#overview)
2. [Loss vs Metric](#loss-vs-metric)
3. [Regression Loss Functions](#regression-loss-functions)
4. [Classification Loss Functions](#classification-loss-functions)
5. [Advanced Loss Functions](#advanced-loss-functions)
6. [Custom Loss Functions](#custom-loss-functions)
7. [Choosing the Right Loss](#choosing-the-right-loss)

## Overview

**Loss Function** (Cost Function, Objective Function) - a function that measures the difference between predicted and actual values during model training. The learning algorithm minimizes this function to find optimal model parameters.

**Key properties of good loss functions:**
- Differentiable (for gradient-based optimization)
- Convex or at least smooth (easier to optimize)
- Aligned with the evaluation metric
- Computationally efficient
- Numerically stable

## Loss vs Metric

### Loss Function
- Used **during training** to optimize model parameters
- Must be **differentiable** for gradient descent
- Directly affects learning process
- Examples: MSE, Cross-Entropy, Hinge Loss

### Evaluation Metric
- Used **after training** to assess model performance
- Doesn't need to be differentiable
- Measures final model quality
- Examples: Accuracy, F1 Score, AUC

**Important distinction:**

```python
# During training: minimize loss
model.fit(X_train, y_train)  # Uses loss function (e.g., binary cross-entropy)

# After training: evaluate with metrics
accuracy = accuracy_score(y_test, model.predict(X_test))
f1 = f1_score(y_test, model.predict(X_test))
```

**Sometimes they coincide:**
- MSE can be both loss and metric
- Log loss (cross-entropy) relates to probability metrics

**Often they differ:**
- Training with cross-entropy, evaluating with accuracy
- Training with MSE, evaluating with R²

## Regression Loss Functions

### Mean Squared Error (MSE)

**MSE Loss** - the most common regression loss function:

$$L_{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**Gradient:**

$$\frac{\partial L_{MSE}}{\partial \hat{y}_i} = -\frac{2}{n}(y_i - \hat{y}_i)$$

**Properties:**
- Differentiable everywhere
- Convex (single global minimum)
- Penalizes large errors heavily (quadratic)
- Sensitive to outliers

**When to use:**
- Standard regression problems
- When large errors are particularly bad
- Normally distributed errors

**Python example:**

```python
import numpy as np

def mse_loss(y_true, y_pred):
    """Mean Squared Error loss"""
    return np.mean((y_true - y_pred) ** 2)

def mse_gradient(y_true, y_pred):
    """Gradient of MSE with respect to predictions"""
    return -2 * (y_true - y_pred) / len(y_true)

# Example
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

loss = mse_loss(y_true, y_pred)
grad = mse_gradient(y_true, y_pred)

print(f"MSE Loss: {loss:.4f}")
print(f"Gradient: {grad}")
```

**In practice (PyTorch/TensorFlow):**

```python
# PyTorch
import torch
import torch.nn as nn

criterion = nn.MSELoss()
loss = criterion(y_pred, y_true)

# TensorFlow/Keras
from tensorflow.keras.losses import MeanSquaredError

criterion = MeanSquaredError()
loss = criterion(y_true, y_pred)
```

### Mean Absolute Error (MAE)

**MAE Loss** (L1 Loss) - average absolute difference:

$$L_{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**Gradient:**

$$\frac{\partial L_{MAE}}{\partial \hat{y}_i} = -\frac{1}{n} \cdot \text{sign}(y_i - \hat{y}_i)$$

**Properties:**
- Not differentiable at zero (but subgradient exists)
- Convex
- Linear penalty (all errors weighted equally)
- Robust to outliers
- Constant gradient magnitude

**When to use:**
- When outliers are present
- When all errors should be weighted equally
- When you want robustness

**Python example:**

```python
def mae_loss(y_true, y_pred):
    """Mean Absolute Error loss"""
    return np.mean(np.abs(y_true - y_pred))

def mae_gradient(y_true, y_pred):
    """Gradient of MAE"""
    return -np.sign(y_true - y_pred) / len(y_true)

# PyTorch
criterion = nn.L1Loss()
loss = criterion(y_pred, y_true)

# Keras
from tensorflow.keras.losses import MeanAbsoluteError
criterion = MeanAbsoluteError()
```

### Huber Loss

**Huber Loss** - combines MSE and MAE, quadratic for small errors, linear for large errors:

$$L_\delta(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}$$

**Parameter δ** controls the transition point between quadratic and linear.

**Properties:**
- Differentiable everywhere
- Convex
- Less sensitive to outliers than MSE
- More stable gradients than MAE
- Combines benefits of both MSE and MAE

**When to use:**
- When you have some outliers but don't want full MAE robustness
- When you want smooth gradients
- Robust regression

**Python example:**

```python
def huber_loss(y_true, y_pred, delta=1.0):
    """Huber loss"""
    error = y_true - y_pred
    abs_error = np.abs(error)

    quadratic = 0.5 * error ** 2
    linear = delta * abs_error - 0.5 * delta ** 2

    return np.mean(np.where(abs_error <= delta, quadratic, linear))

# PyTorch
criterion = nn.HuberLoss(delta=1.0)

# Keras
from tensorflow.keras.losses import Huber
criterion = Huber(delta=1.0)
```

### Log-Cosh Loss

**Log-Cosh Loss** - logarithm of the hyperbolic cosine of the error:

$$L(y, \hat{y}) = \frac{1}{n}\sum_{i=1}^{n} \log(\cosh(\hat{y}_i - y_i))$$

**Properties:**
- Smooth approximation of MAE
- Differentiable everywhere
- Approximately equal to $(y - \hat{y})^2 / 2$ for small errors
- Approximately equal to $|y - \hat{y}|$ for large errors

**When to use:**
- Alternative to Huber loss
- When you want smoothness and outlier robustness

**Python example:**

```python
def log_cosh_loss(y_true, y_pred):
    """Log-Cosh loss"""
    error = y_pred - y_true
    return np.mean(np.log(np.cosh(error)))

# Keras
from tensorflow.keras.losses import LogCosh
criterion = LogCosh()
```

### Quantile Loss

**Quantile Loss** - for predicting specific quantiles (e.g., median, 90th percentile):

$$L_\tau(y, \hat{y}) = \sum_{i:y_i \geq \hat{y}_i} \tau |y_i - \hat{y}_i| + \sum_{i:y_i < \hat{y}_i} (1-\tau) |y_i - \hat{y}_i|$$

**Parameter τ** (0 < τ < 1) specifies the quantile:
- τ = 0.5: median (MAE)
- τ = 0.9: 90th percentile
- τ = 0.1: 10th percentile

**When to use:**
- Predicting quantiles or intervals
- Asymmetric cost of over/under-prediction
- Uncertainty estimation

**Python example:**

```python
def quantile_loss(y_true, y_pred, tau=0.5):
    """Quantile loss"""
    error = y_true - y_pred
    return np.mean(np.maximum(tau * error, (tau - 1) * error))
```

## Classification Loss Functions

### Binary Cross-Entropy (Log Loss)

**Binary Cross-Entropy** - standard loss for binary classification:

$$L_{BCE} = -\frac{1}{n}\sum_{i=1}^{n}[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

where:
- $y_i \in \{0, 1\}$: true label
- $\hat{y}_i \in [0, 1]$: predicted probability

**Properties:**
- Differentiable
- Convex in logit space
- Penalizes confident wrong predictions heavily
- Outputs are probabilities

**When to use:**
- Binary classification (standard choice)
- When you need probability outputs
- Multi-label classification (apply to each label)

**Python example:**

```python
def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """Binary cross-entropy loss"""
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) +
                    (1 - y_true) * np.log(1 - y_pred))

# PyTorch
criterion = nn.BCELoss()  # Input: probabilities
# Or with logits (more numerically stable)
criterion = nn.BCEWithLogitsLoss()  # Input: raw scores

# Keras
from tensorflow.keras.losses import BinaryCrossentropy
criterion = BinaryCrossentropy()
```

**Example:**

```python
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.4, 0.2])  # Probabilities

loss = binary_cross_entropy(y_true, y_pred)
print(f"BCE Loss: {loss:.4f}")
# Low loss for confident correct predictions (0.9 for 1, 0.1 for 0)
# High loss for wrong or uncertain predictions (0.4 for 1)
```

### Categorical Cross-Entropy

**Categorical Cross-Entropy** - for multi-class classification (one-hot encoded):

$$L_{CCE} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

where:
- $C$: number of classes
- $y_{i,c}$: one-hot encoded true label (1 for correct class, 0 otherwise)
- $\hat{y}_{i,c}$: predicted probability for class $c$

**Properties:**
- Generalizes binary cross-entropy to multiple classes
- Requires one-hot encoded labels
- Outputs sum to 1 (via softmax)

**When to use:**
- Multi-class classification (mutually exclusive classes)
- When you need class probabilities

**Python example:**

```python
def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """Categorical cross-entropy loss"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# PyTorch
criterion = nn.CrossEntropyLoss()  # Combines softmax + cross-entropy

# Keras
from tensorflow.keras.losses import CategoricalCrossentropy
criterion = CategoricalCrossentropy()
```

**Example:**

```python
# 3 classes, 2 samples
y_true = np.array([[1, 0, 0],  # Class 0
                   [0, 0, 1]])  # Class 2

y_pred = np.array([[0.7, 0.2, 0.1],  # Predicts class 0 (good)
                   [0.2, 0.3, 0.5]])  # Predicts class 2 (ok)

loss = categorical_cross_entropy(y_true, y_pred)
print(f"CCE Loss: {loss:.4f}")
```

### Sparse Categorical Cross-Entropy

**Sparse Categorical Cross-Entropy** - same as categorical CE but accepts integer labels:

$$L_{SCCE} = -\frac{1}{n}\sum_{i=1}^{n} \log(\hat{y}_{i,y_i})$$

where $y_i$ is the integer class label.

**Difference from Categorical CE:**
- Input labels are integers (0, 1, 2, ...) instead of one-hot vectors
- Computationally more efficient
- Same mathematical result

**Python example:**

```python
# PyTorch (use CrossEntropyLoss with integer labels)
criterion = nn.CrossEntropyLoss()
y_true = torch.tensor([0, 2])  # Integer labels
y_pred = torch.tensor([[2.0, 1.0, 0.1],
                       [0.5, 1.0, 2.0]])  # Logits
loss = criterion(y_pred, y_true)

# Keras
from tensorflow.keras.losses import SparseCategoricalCrossentropy
criterion = SparseCategoricalCrossentropy(from_logits=True)
```

### Hinge Loss (SVM Loss)

**Hinge Loss** - used in Support Vector Machines:

$$L_{hinge} = \frac{1}{n}\sum_{i=1}^{n} \max(0, 1 - y_i \hat{y}_i)$$

where:
- $y_i \in \{-1, +1\}$: true label
- $\hat{y}_i \in \mathbb{R}$: raw prediction score (not probability)

**Properties:**
- Encourages margin between classes
- Zero loss for correct predictions with confidence
- Linear penalty for mistakes
- Not probabilistic

**When to use:**
- Support Vector Machines
- When you want maximum margin
- When probabilities aren't needed

**Python example:**

```python
def hinge_loss(y_true, y_pred):
    """Hinge loss (y_true should be -1 or 1)"""
    return np.mean(np.maximum(0, 1 - y_true * y_pred))

# PyTorch
criterion = nn.HingeEmbeddingLoss()

# Example
y_true = np.array([1, -1, 1, -1])
y_pred = np.array([0.8, -0.9, 0.3, -0.2])  # Raw scores

loss = hinge_loss(y_true, y_pred)
# Loss is 0 for confident correct predictions (0.8 for 1, -0.9 for -1)
# Loss is positive for weak or wrong predictions (0.3 for 1, -0.2 for -1)
```

### Focal Loss

**Focal Loss** - addresses class imbalance by down-weighting easy examples:

$$L_{focal} = -\frac{1}{n}\sum_{i=1}^{n} \alpha_i (1 - \hat{y}_i)^\gamma y_i \log(\hat{y}_i)$$

where:
- $\gamma \geq 0$: focusing parameter (typically 2)
- $\alpha$: class balancing parameter
- $(1 - \hat{y}_i)^\gamma$: modulating factor

**Properties:**
- Focuses on hard examples
- Down-weights easy examples (confident correct predictions)
- Addresses class imbalance
- Generalizes cross-entropy (γ=0 gives standard CE)

**When to use:**
- Highly imbalanced datasets
- Object detection
- When most examples are easy to classify

**Python example:**

```python
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25, epsilon=1e-15):
    """Focal loss for binary classification"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Compute focal term
    pt = np.where(y_true == 1, y_pred, 1 - y_pred)
    focal_term = (1 - pt) ** gamma

    # Compute cross-entropy
    ce = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

    # Combine
    loss = alpha * focal_term * ce

    return np.mean(loss)

# Example showing focus on hard examples
y_true = np.array([1, 1, 1, 0, 0])
y_pred_easy = np.array([0.95, 0.9, 0.85, 0.1, 0.15])  # Easy examples
y_pred_hard = np.array([0.6, 0.55, 0.65, 0.4, 0.45])  # Hard examples

focal_easy = focal_loss(y_true, y_pred_easy)
focal_hard = focal_loss(y_true, y_pred_hard)

print(f"Focal loss (easy): {focal_easy:.4f}")
print(f"Focal loss (hard): {focal_hard:.4f}")
# Hard examples contribute more to loss
```

## Advanced Loss Functions

### Contrastive Loss

**Contrastive Loss** - for learning embeddings, encourages similar pairs to be close and dissimilar pairs to be far:

$$L_{contrastive} = \frac{1}{2n}\sum_{i=1}^{n}[y_i d_i^2 + (1-y_i)\max(0, m - d_i)^2]$$

where:
- $d_i$: distance between pair $i$
- $y_i$: 1 if similar, 0 if dissimilar
- $m$: margin parameter

**When to use:**
- Siamese networks
- Learning embeddings
- Face recognition, signature verification

### Triplet Loss

**Triplet Loss** - encourages anchor to be closer to positive than to negative:

$$L_{triplet} = \max(0, d(a, p) - d(a, n) + margin)$$

where:
- $a$: anchor sample
- $p$: positive sample (same class as anchor)
- $n$: negative sample (different class)
- $d$: distance metric

**When to use:**
- Learning embeddings
- Face recognition
- Ranking problems

### Dice Loss

**Dice Loss** - based on Dice coefficient, used in segmentation:

$$L_{dice} = 1 - \frac{2|X \cap Y|}{|X| + |Y|} = 1 - \frac{2\sum_{i} p_i g_i}{\sum_{i} p_i + \sum_{i} g_i}$$

where:
- $p_i$: predicted probability
- $g_i$: ground truth (0 or 1)

**When to use:**
- Image segmentation
- Imbalanced segmentation tasks
- Medical imaging

**Python example:**

```python
def dice_loss(y_true, y_pred, smooth=1e-6):
    """Dice loss for segmentation"""
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice
```

## Custom Loss Functions

### Creating Custom Losses

**PyTorch example:**

```python
import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, weight_factor=1.0):
        super(CustomLoss, self).__init__()
        self.weight_factor = weight_factor

    def forward(self, y_pred, y_true):
        # Your custom loss logic
        mse = torch.mean((y_pred - y_true) ** 2)
        mae = torch.mean(torch.abs(y_pred - y_true))

        # Combine losses
        loss = mse + self.weight_factor * mae
        return loss

# Usage
criterion = CustomLoss(weight_factor=0.5)
loss = criterion(predictions, targets)
loss.backward()
```

**TensorFlow/Keras example:**

```python
import tensorflow as tf
from tensorflow.keras.losses import Loss

class CustomLoss(Loss):
    def __init__(self, weight_factor=1.0):
        super().__init__()
        self.weight_factor = weight_factor

    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        return mse + self.weight_factor * mae

# Usage
model.compile(optimizer='adam', loss=CustomLoss(weight_factor=0.5))
```

### Weighted Loss

Weight different samples or classes differently:

```python
def weighted_mse_loss(y_true, y_pred, weights):
    """MSE with sample weights"""
    squared_error = (y_true - y_pred) ** 2
    weighted_error = squared_error * weights
    return np.mean(weighted_error)

# Example: weight important samples more
weights = np.array([1.0, 2.0, 1.0, 3.0])  # Sample 4 is 3x more important
loss = weighted_mse_loss(y_true, y_pred, weights)

# PyTorch
criterion = nn.MSELoss(reduction='none')
loss = criterion(y_pred, y_true)
weighted_loss = (loss * weights).mean()
```

### Multi-Task Loss

Combine multiple losses for multi-task learning:

```python
def multi_task_loss(y_true, y_pred, task_weights=[1.0, 1.0]):
    """Combine losses from multiple tasks"""
    # Task 1: Regression
    loss_regression = torch.mean((y_pred[:, 0] - y_true[:, 0]) ** 2)

    # Task 2: Classification
    loss_classification = nn.functional.cross_entropy(
        y_pred[:, 1:], y_true[:, 1].long()
    )

    # Weighted combination
    total_loss = (task_weights[0] * loss_regression +
                  task_weights[1] * loss_classification)

    return total_loss
```

## Choosing the Right Loss

### For Regression

| Loss | Use Case | Pros | Cons |
|------|----------|------|------|
| MSE | Standard regression | Fast, convex | Sensitive to outliers |
| MAE | Robust regression | Outlier-resistant | Unstable gradients |
| Huber | Some outliers | Balanced | Extra hyperparameter |
| Quantile | Quantile prediction | Flexible | Task-specific |

### For Classification

| Loss | Use Case | Pros | Cons |
|------|----------|------|------|
| Cross-Entropy | Standard classification | Well-studied, probabilistic | Sensitive to imbalance |
| Focal Loss | Imbalanced data | Handles imbalance | More complex |
| Hinge | Maximum margin | Good generalization | No probabilities |

### Decision Tree

```
What is your task?
├─ Regression
│  ├─ Outliers present?
│  │  ├─ Yes → MAE or Huber Loss
│  │  └─ No → MSE
│  └─ Need quantiles? → Quantile Loss
│
└─ Classification
   ├─ Binary
   │  ├─ Balanced classes → Binary Cross-Entropy
   │  └─ Imbalanced → Focal Loss or Weighted BCE
   │
   └─ Multi-class
      ├─ Mutually exclusive → Categorical Cross-Entropy
      ├─ Multi-label → Binary Cross-Entropy per label
      └─ Imbalanced → Focal Loss or Weighted CE
```

### Practical Tips

1. **Start with standard losses**: MSE for regression, cross-entropy for classification
2. **Match loss to metric**: If evaluating with MAE, consider training with MAE
3. **Check for outliers**: Use robust losses (MAE, Huber) if present
4. **Handle imbalance**: Use weighted losses or focal loss
5. **Ensure differentiability**: Custom losses must have gradients
6. **Numerical stability**: Use log-sum-exp trick, clip values
7. **Combine losses carefully**: Weight combination appropriately
8. **Monitor multiple metrics**: Loss alone doesn't tell the whole story

### Common Combinations

```python
# Classification: loss vs metrics
model.compile(
    loss='binary_crossentropy',  # Training objective
    metrics=['accuracy', 'precision', 'recall', 'AUC']  # Evaluation
)

# Regression: loss vs metrics
model.compile(
    loss='mse',  # Training objective
    metrics=['mae', 'mape']  # Evaluation
)

# Multi-task: multiple losses
total_loss = 0.5 * classification_loss + 0.5 * regression_loss
```

## Practical Examples

### Complete Training Loop with Loss

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Model, loss, optimizer
model = MyNeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch_x, batch_y in train_loader:
        # Forward pass
        predictions = model(batch_x)

        # Compute loss
        loss = criterion(predictions, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Print average loss
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    with torch.no_grad():
        val_predictions = model(val_x)
        val_loss = criterion(val_predictions, val_y)
        print(f"Validation Loss: {val_loss.item():.4f}")
```

## References

- "Deep Learning" by Goodfellow, Bengio, and Courville (Chapter 6: Deep Feedforward Networks)
- PyTorch Loss Functions Documentation
- TensorFlow/Keras Loss Functions Documentation
- Lin, T.Y. et al. (2017) "Focal Loss for Dense Object Detection"
- "Dive into Deep Learning" (d2l.ai) - Chapter on Loss Functions
- Janocha, K. and Czarnecki, W.M. (2017) "On Loss Functions for Deep Neural Networks in Classification"
