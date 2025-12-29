# Batch Normalization

## Table of Contents

1. [Introduction](#introduction)
2. [The Problem of Internal Covariate Shift](#the-problem-of-internal-covariate-shift)
3. [How Batch Normalization Works](#how-batch-normalization-works)
4. [Benefits](#benefits)
5. [Implementation Details](#implementation-details)
6. [Variants](#variants)
7. [When to Use](#when-to-use)

## Introduction

**Batch Normalization (BatchNorm)** - a technique that normalizes the inputs of each layer to have zero mean and unit variance, reducing internal covariate shift and accelerating training.

Introduced by Sergey Ioffe and Christian Szegedy in 2015, batch normalization has become one of the most important techniques in deep learning, enabling faster training and better model performance.

## The Problem of Internal Covariate Shift

**Internal Covariate Shift** - the change in the distribution of network activations due to parameter updates during training.

Problems caused by covariate shift:
- Requires lower learning rates
- Makes training more sensitive to initialization
- Can cause gradient vanishing or explosion
- Slows down convergence

As the network learns, the distribution of inputs to each layer changes, forcing layers to continuously adapt to new distributions.

## How Batch Normalization Works

### Algorithm

For a mini-batch of activations $\mathcal{B} = \{x_1, x_2, ..., x_m\}$, BatchNorm performs:

**1. Calculate batch statistics:**

$$\mu_\mathcal{B} = \frac{1}{m}\sum_{i=1}^{m}x_i$$

$$\sigma^2_\mathcal{B} = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_\mathcal{B})^2$$

**2. Normalize:**

$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}}$$

where $\epsilon$ is a small constant (e.g., $10^{-5}$) for numerical stability.

**3. Scale and shift:**

$$y_i = \gamma \hat{x}_i + \beta$$

where:
- $\gamma$ - learnable scale parameter
- $\beta$ - learnable shift parameter

The scale and shift parameters allow the network to undo the normalization if needed, giving it the flexibility to learn the optimal representation.

### Training vs Inference

**During training:**
- Statistics ($\mu$, $\sigma^2$) computed from current mini-batch
- Moving averages maintained for inference

**During inference:**
- Use population statistics (moving averages from training)
- Deterministic behavior (no dependency on batch)

Moving average update:

$$\mu_{moving} \leftarrow \alpha \cdot \mu_{moving} + (1-\alpha) \cdot \mu_\mathcal{B}$$

where $\alpha$ is the momentum (typically 0.9 or 0.99).

## Benefits

### 1. Faster Training

- Allows higher learning rates (10-100x)
- Reduces training time significantly
- Less sensitive to initialization

### 2. Regularization Effect

- Acts as a regularizer due to noise from batch statistics
- Can reduce or eliminate need for dropout
- Adds slight stochasticity during training

### 3. Gradient Flow

- Reduces gradient vanishing/explosion
- Enables training of very deep networks
- Smooths the optimization landscape

### 4. Less Sensitivity

- More robust to hyperparameter choices
- Less dependent on careful initialization
- Reduces sensitivity to weight scale

## Implementation Details

### Where to Apply

BatchNorm is typically applied:
- **After linear/conv layers, before activation** (original paper)
- **After activation function** (alternative approach)

Common placement:

```python
# Before activation (original)
x = Dense(units)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# After activation (alternative)
x = Dense(units)(x)
x = Activation('relu')(x)
x = BatchNormalization()(x)
```

### Batch Size Considerations

**Minimum batch size:** BatchNorm requires reasonably sized batches (typically ≥ 16-32).

Problems with small batches:
- Poor statistics estimation
- High variance in $\mu$ and $\sigma^2$
- Unstable training

### Learnable Parameters

Per feature/channel, BatchNorm adds:
- 2 learnable parameters: $\gamma$, $\beta$
- 2 non-trainable parameters: running mean, running variance

For a layer with $n$ features:
- **Total parameters:** $4n$
- **Trainable parameters:** $2n$

## Variants

### Layer Normalization

Normalizes across features instead of batch dimension:

$$\mu_i = \frac{1}{H}\sum_{j=1}^{H}x_{ij}$$

**Use cases:**
- Recurrent networks (RNN, LSTM)
- Small batch sizes
- Variable sequence lengths

### Instance Normalization

Normalizes each sample and channel independently:

$$\mu_{ic} = \frac{1}{HW}\sum_{h,w}x_{ichw}$$

**Use cases:**
- Style transfer
- GANs
- Image generation tasks

### Group Normalization

Divides channels into groups and normalizes within each group:

$$\mu_{ig} = \frac{1}{C_g HW}\sum_{c \in g, h, w}x_{ichw}$$

**Use cases:**
- Small batch sizes (batch size = 1)
- Object detection
- Video models

### Comparison

| Normalization | Normalizes Over | Best For |
|--------------|----------------|----------|
| Batch Norm | Batch, Height, Width | Large batches, CNNs |
| Layer Norm | Features | RNNs, Transformers |
| Instance Norm | Height, Width | Style transfer |
| Group Norm | Groups of channels | Small batches |

## When to Use

### Use Batch Normalization When:

- Training deep neural networks (> 10 layers)
- Using CNNs for computer vision
- Batch size is sufficiently large (≥ 16-32)
- Training from scratch
- Need faster convergence

### Avoid or Replace When:

- Batch size is very small (< 8)
- Using recurrent networks (prefer LayerNorm)
- Fine-tuning with different data distribution
- Online learning (single sample at a time)
- Strong distributional shift between train and test

### Common Pitfalls

**1. Train/Test Discrepancy**
```python
# Wrong: Forgetting to set training mode
model.eval()  # Use population statistics
predictions = model(data)
```

**2. Freezing BatchNorm Incorrectly**
```python
# When fine-tuning, may need to freeze BN statistics
for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.eval()  # Keep in eval mode even during training
```

**3. Using with Small Batches**
- BatchNorm performs poorly with batch size < 8
- Consider GroupNorm or LayerNorm instead

## Practical Example

```python
import torch
import torch.nn as nn

class ConvBlockWithBN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Without BatchNorm: slower training, lower learning rates
class ConvBlockNoBN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
```

### TensorFlow/Keras Example

```python
from tensorflow.keras import layers, models

# Functional API
inputs = layers.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D(2)(x)
```

## References

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) (Ioffe & Szegedy, 2015)
- [Group Normalization](https://arxiv.org/abs/1803.08494) (Wu & He, 2018)
- [Layer Normalization](https://arxiv.org/abs/1607.06450) (Ba et al., 2016)
- [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604) (Santurkar et al., 2018)
