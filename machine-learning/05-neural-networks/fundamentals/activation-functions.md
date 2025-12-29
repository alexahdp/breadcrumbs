# Activation Functions

## Table of Contents

1. [Introduction](#introduction)
2. [Why Activation Functions?](#why-activation-functions)
3. [Common Activation Functions](#common-activation-functions)
4. [Choosing Activation Functions](#choosing-activation-functions)
5. [Advanced Activation Functions](#advanced-activation-functions)
6. [Practical Considerations](#practical-considerations)

## Introduction

**Activation functions** are mathematical functions applied to the output of neurons in neural networks. They introduce non-linearity into the network, enabling it to learn complex patterns and relationships in data.

Without activation functions (or with only linear activations), a neural network would be equivalent to a single-layer linear model, regardless of the number of layers.

### Key Properties

Good activation functions should have:
- **Non-linearity**: enable learning of complex patterns
- **Differentiability**: required for backpropagation
- **Computational efficiency**: fast to compute
- **Zero-centered or normalized output**: helps with optimization
- **No vanishing gradients**: gradients should not become too small

## Why Activation Functions?

### Without Activation Functions

Consider a 2-layer network without activation:

$$\mathbf{h} = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$$
$$\mathbf{y} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2$$

Substituting:

$$\mathbf{y} = \mathbf{W}_2 (\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = \mathbf{W}_2 \mathbf{W}_1 \mathbf{x} + \mathbf{W}_2 \mathbf{b}_1 + \mathbf{b}_2$$

This is equivalent to a single linear layer: $\mathbf{y} = \mathbf{W} \mathbf{x} + \mathbf{b}$

**Conclusion**: Stacking linear transformations without non-linearity produces only a linear transformation.

### With Activation Functions

$$\mathbf{h} = g(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$$
$$\mathbf{y} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2$$

Now the network can learn non-linear decision boundaries and complex patterns.

## Common Activation Functions

### 1. Step Function

**Formula:**
$$\text{step}(x) = \begin{cases} 1 & \text{if } x \geq 0 \\ 0 & \text{if } x < 0 \end{cases}$$

**Derivative:**
$$\text{step}'(x) = 0 \text{ (undefined at } x=0\text{)}$$

**Properties:**
- ✅ Simple, clear threshold
- ❌ Not differentiable (cannot use gradient descent)
- ❌ Zero gradients everywhere (except at discontinuity)

**Use case:** Original perceptron only (not used in modern networks)

### 2. Sigmoid (Logistic)

**Formula:**
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Derivative:**
$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**Range:** $(0, 1)$

**Properties:**
- ✅ Smooth, differentiable
- ✅ Output interpretable as probability
- ❌ Vanishing gradients for large $|x|$
- ❌ Not zero-centered (outputs always positive)
- ❌ Computationally expensive (exp function)

**Use case:** Output layer for binary classification

**Gradient problem:** When $x$ is large (positive or negative), $\sigma'(x) \approx 0$, causing vanishing gradients in deep networks.

### 3. Hyperbolic Tangent (Tanh)

**Formula:**
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{2}{1 + e^{-2x}} - 1$$

**Derivative:**
$$\tanh'(x) = 1 - \tanh^2(x)$$

**Range:** $(-1, 1)$

**Properties:**
- ✅ Zero-centered (better than sigmoid)
- ✅ Smooth, differentiable
- ❌ Still suffers from vanishing gradients
- ❌ Computationally expensive

**Use case:** Hidden layers (especially in RNNs), less common now

**Relationship to sigmoid:** $\tanh(x) = 2\sigma(2x) - 1$

### 4. ReLU (Rectified Linear Unit)

**Formula:**
$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

**Derivative:**
$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

**Range:** $[0, \infty)$

**Properties:**
- ✅ Simple and computationally efficient
- ✅ No vanishing gradient problem for positive values
- ✅ Sparse activation (many neurons output 0)
- ✅ Empirically faster convergence
- ❌ Not zero-centered
- ❌ Dead ReLU problem: neurons can get stuck outputting 0

**Use case:** **Default choice for hidden layers** in most networks

**Dead ReLU problem:** If a neuron's weights cause it to always output negative values, the gradient is always 0, and the neuron never updates (dies). Can be mitigated with proper initialization and learning rate.

### 5. Leaky ReLU

**Formula:**
$$\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$$

where $\alpha$ is a small constant (typically 0.01)

**Derivative:**
$$\text{LeakyReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha & \text{if } x \leq 0 \end{cases}$$

**Properties:**
- ✅ Fixes dying ReLU problem
- ✅ Small gradient for negative values
- ✅ Simple and efficient

**Use case:** Alternative to ReLU when dead neurons are a problem

**Variants:**
- **Parametric ReLU (PReLU)**: $\alpha$ is learned during training
- **Randomized Rleaky ReLU (RReLU)**: $\alpha$ is random during training, fixed during testing

### 6. ELU (Exponential Linear Unit)

**Formula:**
$$\text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}$$

where $\alpha > 0$ (typically 1.0)

**Derivative:**
$$\text{ELU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha e^x = \text{ELU}(x) + \alpha & \text{if } x \leq 0 \end{cases}$$

**Properties:**
- ✅ Smooth everywhere
- ✅ Can produce negative outputs (closer to zero-centered)
- ✅ No dead neurons
- ✅ Robust to noise
- ❌ Computationally more expensive (exp function)

**Use case:** Can outperform ReLU in some cases, especially for small datasets

### 7. GELU (Gaussian Error Linear Unit)

**Formula:**
$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the cumulative distribution function of the standard normal distribution.

**Approximation:**
$$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right]\right)$$

**Properties:**
- ✅ Smooth, non-monotonic
- ✅ Used in state-of-the-art models (BERT, GPT)
- ✅ Better performance than ReLU in many NLP tasks

**Use case:** Transformer models, BERT, GPT

### 8. Swish (SiLU)

**Formula:**
$$\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

**Derivative:**
$$\text{Swish}'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x))$$

**Properties:**
- ✅ Smooth, non-monotonic
- ✅ Self-gated (output depends on input value)
- ✅ Empirically better than ReLU on deep models
- ❌ Computationally more expensive

**Use case:** Deep networks, especially in computer vision

### 9. Softmax

**Formula:** (for a vector $\mathbf{z}$)
$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**Properties:**
- ✅ Outputs sum to 1 (probability distribution)
- ✅ All outputs positive
- ✅ Differentiable
- Used only in output layer

**Use case:** **Output layer for multi-class classification**

**Numerical stability:** Compute as:
$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j=1}^{K} e^{z_j - \max(\mathbf{z})}}$$

### 10. Linear (Identity)

**Formula:**
$$f(x) = x$$

**Derivative:**
$$f'(x) = 1$$

**Use case:** **Output layer for regression tasks**

## Choosing Activation Functions

### General Guidelines

**Hidden Layers:**
1. **Start with ReLU** - best default choice for most problems
2. **Try Leaky ReLU or ELU** if you have dead neurons
3. **Use GELU or Swish** for state-of-the-art performance (at computational cost)
4. **Avoid sigmoid/tanh** in deep networks (vanishing gradients)

**Output Layer:**
- **Regression**: Linear activation
- **Binary classification**: Sigmoid
- **Multi-class classification**: Softmax
- **Multi-label classification**: Sigmoid (one per label)

### Task-Specific Recommendations

| Task | Hidden Layers | Output Layer |
|------|---------------|--------------|
| Image classification | ReLU, Leaky ReLU | Softmax |
| Regression | ReLU, ELU | Linear |
| Binary classification | ReLU | Sigmoid |
| NLP (Transformers) | GELU | Softmax/Linear |
| GANs | Leaky ReLU | Tanh (generator), Sigmoid (discriminator) |
| RNNs/LSTMs | Tanh, ReLU | Task-dependent |

### Network Depth Considerations

- **Shallow networks** (1-3 layers): Any activation works reasonably
- **Deep networks** (10+ layers): Prefer ReLU-based activations
- **Very deep networks** (50+ layers): Use residual connections + batch normalization with ReLU

## Advanced Activation Functions

### Maxout

**Formula:**
$$\text{maxout}(\mathbf{x}) = \max_{i \in [1,k]} \mathbf{w}_i^T \mathbf{x} + b_i$$

Learns the activation function by taking the max over $k$ linear functions.

**Properties:**
- ✅ Very flexible, can approximate any convex function
- ❌ Doubles the number of parameters

### Mish

**Formula:**
$$\text{Mish}(x) = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))$$

**Properties:**
- ✅ Smooth, non-monotonic
- ✅ Self-regularizing
- ✅ Can outperform ReLU and Swish

## Practical Considerations

### Vanishing Gradient Problem

**Problem:** In deep networks with sigmoid/tanh, gradients become very small in early layers.

**Why it happens:**
- Sigmoid/tanh squash large inputs to small ranges
- Their derivatives are small for large $|x|$
- Gradients multiply through layers (chain rule)
- Product of many small numbers → vanishing gradient

**Solutions:**
1. Use ReLU-based activations
2. Batch normalization
3. Residual connections (ResNet)
4. Proper weight initialization

### Exploding Gradient Problem

**Problem:** Gradients become extremely large.

**Solutions:**
1. Gradient clipping
2. Proper weight initialization
3. Batch normalization
4. Lower learning rate

### Implementation Examples

```python
import numpy as np

# ReLU
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Tanh
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Softmax (for vectors)
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # numerical stability
    return exp_x / np.sum(exp_x)

# ELU
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, elu(x, alpha) + alpha)
```

### Using PyTorch

```python
import torch
import torch.nn as nn

# Common activations in PyTorch
activations = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(negative_slope=0.01),
    'elu': nn.ELU(alpha=1.0),
    'gelu': nn.GELU(),
    'silu': nn.SiLU(),  # Swish
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'softmax': nn.Softmax(dim=1)
}

# Example network with different activations
class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 2)

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
```

### Visualization Tips

When visualizing activation functions:
- Plot the function itself: $f(x)$ vs $x$
- Plot the derivative: $f'(x)$ vs $x$
- Check behavior for large positive/negative values
- Observe where gradients are strong vs weak

## Common Mistakes

1. **Using sigmoid/tanh in deep networks** → vanishing gradients
2. **No activation in output layer for regression** ✅ Correct
3. **Using ReLU in output layer for classification** ❌ Wrong
4. **Forgetting numerical stability in softmax** → NaN values
5. **Using high learning rate with sensitive activations** → divergence

## Summary Table

| Activation | Range | Zero-centered | Monotonic | Best for |
|------------|-------|---------------|-----------|----------|
| ReLU | $[0, \infty)$ | No | Yes | Hidden layers (default) |
| Leaky ReLU | $(-\infty, \infty)$ | No | Yes | Hidden layers (avoid dead neurons) |
| ELU | $(-\alpha, \infty)$ | Nearly | Yes | Hidden layers (small datasets) |
| GELU | $(-\infty, \infty)$ | No | No | Transformers, NLP |
| Swish | $(-\infty, \infty)$ | No | No | Deep networks, CV |
| Sigmoid | $(0, 1)$ | No | Yes | Binary classification output |
| Tanh | $(-1, 1)$ | Yes | Yes | RNNs (less common now) |
| Softmax | $(0, 1)$, sum=1 | No | - | Multi-class output |
| Linear | $(-\infty, \infty)$ | Yes | Yes | Regression output |

## References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning"
- Nair, V., & Hinton, G. E. (2010). "Rectified Linear Units Improve Restricted Boltzmann Machines"
- Clevert, D. A., et al. (2015). "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)"
- Hendrycks, D., & Gimpel, K. (2016). "Gaussian Error Linear Units (GELUs)"
- Ramachandran, P., et al. (2017). "Searching for Activation Functions"
