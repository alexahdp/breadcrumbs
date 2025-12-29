# Dropout

## Table of Contents

1. [Introduction](#introduction)
2. [How Dropout Works](#how-dropout-works)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Why Dropout Prevents Overfitting](#why-dropout-prevents-overfitting)
5. [Implementation Details](#implementation-details)
6. [Variants](#variants)
7. [When to Use](#when-to-use)

## Introduction

**Dropout** - a regularization technique that randomly "drops out" (sets to zero) a subset of neurons during training, preventing co-adaptation of features and reducing overfitting.

Introduced by Hinton et al. in 2012, dropout has become one of the most widely used regularization techniques in deep learning, particularly for fully connected layers.

## How Dropout Works

### Basic Principle

During training, at each iteration:
1. Randomly select neurons with probability $p$ (dropout rate)
2. Set their outputs to zero
3. Scale remaining outputs by $\frac{1}{1-p}$ to maintain expected value

During inference:
- Use all neurons (no dropout)
- No scaling needed (already handled during training)

**Example:** With dropout rate $p = 0.5$:
- 50% of neurons randomly deactivated each iteration
- Different random subset each time
- Outputs scaled by 2x during training

### Visualization

```
Normal Layer:          With Dropout (p=0.5):
[1.0, 2.0, 3.0, 4.0]  →  [2.0, 0.0, 6.0, 0.0]
                          (scaled by 1/(1-0.5) = 2)
```

## Mathematical Formulation

### Training Phase

For a layer with activation $\mathbf{h}$:

**1. Sample binary mask:**

$$\mathbf{r} \sim \text{Bernoulli}(1-p)$$

where each element $r_i \in \{0, 1\}$.

**2. Apply mask and scale:**

$$\tilde{\mathbf{h}} = \frac{\mathbf{r} \odot \mathbf{h}}{1-p}$$

where $\odot$ denotes element-wise multiplication.

### Inference Phase

During inference, use all neurons without scaling:

$$\tilde{\mathbf{h}} = \mathbf{h}$$

The scaling during training ensures the expected output matches inference.

### Expected Value

The scaling factor ensures expected values match:

$$E[\tilde{h}_i] = \frac{1}{1-p} \cdot (1-p) \cdot h_i = h_i$$

## Why Dropout Prevents Overfitting

### 1. Ensemble Effect

Dropout trains an ensemble of $2^n$ sub-networks (where $n$ is number of neurons):
- Each iteration uses a different sub-network
- Final network approximates averaging all sub-networks
- Ensemble methods reduce overfitting

### 2. Prevents Co-Adaptation

**Co-adaptation** - when neurons rely too heavily on specific other neurons.

Dropout forces neurons to:
- Learn more robust features
- Not depend on presence of specific other neurons
- Develop redundant representations

### 3. Adds Noise

Dropout injects noise into the training process:
- Acts as a strong regularizer
- Prevents memorization
- Encourages generalization

### 4. Feature Learning

Forces the network to:
- Learn multiple independent pathways
- Create redundant representations
- Develop robust features that work in different contexts

## Implementation Details

### Dropout Rate Selection

**Common dropout rates:**
- **0.5** - standard choice for fully connected layers
- **0.2-0.3** - for input layers
- **0.1-0.2** - for convolutional layers (or use Spatial Dropout)

**Guidelines:**
- Higher capacity networks can use higher dropout
- Start with 0.5 and adjust based on validation performance
- Too high (> 0.8) - underfitting
- Too low (< 0.1) - minimal regularization effect

### Where to Apply

**Typically applied to:**
- Fully connected (dense) layers
- Before the final output layer
- After activation functions

**Less common:**
- Convolutional layers (use spatial dropout instead)
- Recurrent connections in RNNs (use recurrent dropout)

### Training vs Inference Modes

Critical to distinguish between training and inference:

```python
# PyTorch
model.train()  # Enable dropout
model.eval()   # Disable dropout

# TensorFlow/Keras
model(x, training=True)   # Enable dropout
model(x, training=False)  # Disable dropout
```

## Variants

### Spatial Dropout (Dropout2D/3D)

For convolutional layers, drops entire feature maps instead of individual values:

$$\text{Drop entire channel: } C \times H \times W \to 0$$

**Why:** Neighboring pixels are highly correlated; dropping individual values is less effective.

```python
# PyTorch
nn.Dropout2d(p=0.2)  # For conv layers

# Keras
layers.SpatialDropout2D(0.2)
```

### DropConnect

Drops connections (weights) instead of activations:

$$\mathbf{r} \sim \text{Bernoulli}(1-p)$$
$$\mathbf{y} = (\mathbf{W} \odot \mathbf{r})\mathbf{x}$$

More aggressive regularization than standard dropout.

### Variational Dropout

Uses the same dropout mask across time steps in RNNs:
- Prevents inconsistent dropout across sequence
- Better for recurrent networks

### Alpha Dropout

Designed for SELU activation:
- Maintains self-normalizing property
- Used with Scaled Exponential Linear Units

### Monte Carlo Dropout

Keeps dropout active during inference:
- Multiple forward passes with different masks
- Provides uncertainty estimates
- Used for Bayesian deep learning

$$\mathbb{E}[y] \approx \frac{1}{T}\sum_{t=1}^{T}f(x, \mathbf{r}_t)$$

## When to Use

### Use Dropout When:

- Network is overfitting (train acc >> val acc)
- Working with fully connected layers
- Dataset is relatively small
- Model has high capacity
- Need regularization without reducing model size

### Avoid or Reduce When:

- Model is underfitting
- Using batch normalization (they can interfere)
- Working with convolutional layers (use spatial dropout)
- Training RNNs (use recurrent dropout)
- Very small networks

### Dropout vs Other Regularization

| Technique | Mechanism | When to Use |
|-----------|-----------|-------------|
| Dropout | Random deactivation | Fully connected layers |
| L1/L2 | Weight penalty | Any layer, feature selection (L1) |
| Batch Norm | Normalize activations | CNNs, deep networks |
| Early Stopping | Stop before overfitting | Always monitor |
| Data Augmentation | Increase training data | Limited data |

### Interaction with Batch Normalization

**Important:** BatchNorm and Dropout can interact negatively:
- BatchNorm provides regularization
- Dropout disrupts batch statistics
- Using both may reduce performance

**Recommendation:**
- In modern CNNs: prefer BatchNorm over Dropout
- If using both: lower dropout rate or place dropout after BatchNorm

## Practical Examples

### PyTorch Example

```python
import torch
import torch.nn as nn

class MLPWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)

        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)
        return x

# Training
model.train()
output = model(data)  # Dropout active

# Inference
model.eval()
with torch.no_grad():
    output = model(data)  # Dropout disabled
```

### TensorFlow/Keras Example

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(784,)),
    layers.Dropout(0.5),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(10, activation='softmax')
])

# Dropout automatically handles training vs inference
model.fit(x_train, y_train, epochs=10)  # Dropout on
model.predict(x_test)  # Dropout off
```

### Spatial Dropout for CNNs

```python
# PyTorch
class CNNWithSpatialDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.dropout1 = nn.Dropout2d(p=0.2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.dropout2 = nn.Dropout2d(p=0.2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.dropout1(x)  # Drops entire feature maps

        x = torch.relu(self.conv2(x))
        x = self.dropout2(x)
        return x
```

### Monte Carlo Dropout for Uncertainty

```python
def mc_dropout_predict(model, x, n_samples=100):
    """Get prediction with uncertainty using MC Dropout"""
    model.train()  # Keep dropout active
    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x)
            predictions.append(pred)

    predictions = torch.stack(predictions)
    mean = predictions.mean(dim=0)
    std = predictions.std(dim=0)

    return mean, std
```

## Tuning Dropout Rate

### Grid Search Approach

```python
dropout_rates = [0.1, 0.2, 0.3, 0.5, 0.7]
best_val_acc = 0
best_dropout = 0.5

for p in dropout_rates:
    model = create_model(dropout_rate=p)
    val_acc = train_and_validate(model)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_dropout = p

print(f"Best dropout rate: {best_dropout}")
```

### Adaptive Strategy

Start with standard values and adjust:
1. Train with $p = 0.5$
2. If overfitting persists → increase to 0.6-0.7
3. If underfitting → decrease to 0.3-0.4
4. Monitor validation performance

## Common Mistakes

**1. Forgetting to disable during inference**
```python
# Wrong
model.train()
predictions = model(test_data)  # Dropout still active!

# Correct
model.eval()
predictions = model(test_data)
```

**2. Applying dropout to convolutional layers**
```python
# Less effective
nn.Dropout(0.5)  # For conv layers

# Better
nn.Dropout2d(0.2)  # Spatial dropout
```

**3. Too high dropout with batch normalization**
```python
# May hurt performance
x = BatchNorm()(x)
x = Dropout(0.7)(x)  # Too aggressive with BN

# Better
x = BatchNorm()(x)
x = Dropout(0.2)(x)  # Or remove dropout entirely
```

## References

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) (Srivastava et al., 2014)
- [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580) (Hinton et al., 2012)
- [Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142) (Gal & Ghahramani, 2016)
- [Recurrent Dropout without Memory Loss](https://arxiv.org/abs/1603.05118) (Semeniuta et al., 2016)
