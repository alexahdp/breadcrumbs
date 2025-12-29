# Optimizers

## Table of Contents

1. [Introduction](#introduction)
2. [Gradient Descent](#gradient-descent)
3. [Momentum-Based Methods](#momentum-based-methods)
4. [Adaptive Learning Rate Methods](#adaptive-learning-rate-methods)
5. [Second-Order Methods](#second-order-methods)
6. [Choosing an Optimizer](#choosing-an-optimizer)
7. [Practical Considerations](#practical-considerations)
8. [Implementation](#implementation)

## Introduction

**Optimizers** are algorithms that adjust neural network parameters (weights and biases) to minimize the loss function. They determine how the network learns from gradients computed by backpropagation.

The general update rule is:

$$\theta_{t+1} = \theta_t - \Delta\theta_t$$

where $\theta$ represents all parameters and $\Delta\theta_t$ is computed differently by each optimizer.

### Key Concepts

**Learning Rate ($\alpha$ or $\eta$)**: Controls the step size of parameter updates. Too high → divergence; too low → slow convergence.

**Convergence**: The process of loss decreasing toward a minimum.

**Local vs Global Minimum**: Neural network loss surfaces are non-convex with many local minima.

## Gradient Descent

### Batch Gradient Descent (BGD)

Computes gradients using the **entire training dataset** before updating parameters.

**Update rule**:
$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t)$$

where:
- $\alpha$ is the learning rate
- $\nabla_\theta J(\theta_t)$ is the gradient of the loss function

**Advantages**:
- ✅ Stable convergence
- ✅ Guaranteed to converge to global minimum for convex functions

**Disadvantages**:
- ❌ Very slow for large datasets
- ❌ Requires entire dataset in memory
- ❌ Cannot handle online learning

**When to use**: Small datasets that fit in memory

### Stochastic Gradient Descent (SGD)

Updates parameters using **one training example** at a time.

**Update rule**:
$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t; x^{(i)}, y^{(i)})$$

**Advantages**:
- ✅ Fast updates
- ✅ Can escape shallow local minima (noise helps)
- ✅ Enables online learning

**Disadvantages**:
- ❌ High variance in updates (noisy gradients)
- ❌ May not converge to exact minimum (oscillates)
- ❌ Difficult to choose learning rate

### Mini-Batch Gradient Descent

Computes gradients using **small batches** of training examples (typically 32-512).

**Update rule**:
$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t; x^{(i:i+b)}, y^{(i:i+b)})$$

where $b$ is the batch size.

**Advantages**:
- ✅ Balance between BGD and SGD
- ✅ Reduces variance of updates
- ✅ Efficient computation on GPUs
- ✅ Most commonly used in practice

**Disadvantages**:
- ❌ Requires choosing batch size
- ❌ Still sensitive to learning rate

**Typical batch sizes**: 32, 64, 128, 256, 512

## Momentum-Based Methods

### SGD with Momentum

Accumulates a **velocity vector** in directions with consistent gradients, accelerating convergence.

**Update rules**:
$$v_t = \beta v_{t-1} + \alpha \nabla_\theta J(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_t$$

where:
- $v_t$ is the velocity (exponentially weighted average of gradients)
- $\beta$ is the momentum coefficient (typically 0.9)

**Physical analogy**: A ball rolling down a hill accumulates momentum.

**Advantages**:
- ✅ Faster convergence than vanilla SGD
- ✅ Reduces oscillations
- ✅ Helps escape local minima
- ✅ Dampens noise in gradients

**Disadvantages**:
- ❌ Additional hyperparameter ($\beta$)
- ❌ May overshoot minimum

**Typical values**: $\beta = 0.9$ or $0.95$

### Nesterov Accelerated Gradient (NAG)

"Looks ahead" by computing gradients at the anticipated position.

**Update rules**:
$$v_t = \beta v_{t-1} + \alpha \nabla_\theta J(\theta_t - \beta v_{t-1})$$
$$\theta_{t+1} = \theta_t - v_t$$

**Key difference**: Computes gradient at $\theta_t - \beta v_{t-1}$ instead of $\theta_t$

**Advantages**:
- ✅ More responsive to changes in gradient direction
- ✅ Often faster convergence than standard momentum
- ✅ Reduces overshooting

**Disadvantages**:
- ❌ Slightly more complex
- ❌ Requires two gradient computations per step

## Adaptive Learning Rate Methods

These methods adapt the learning rate for each parameter individually.

### AdaGrad (Adaptive Gradient)

Adapts learning rate based on historical gradients for each parameter.

**Update rules**:
$$G_t = G_{t-1} + (\nabla_\theta J(\theta_t))^2$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{G_t + \epsilon}} \nabla_\theta J(\theta_t)$$

where:
- $G_t$ accumulates squared gradients
- $\epsilon$ is a small constant (e.g., $10^{-8}$) for numerical stability
- Operations are element-wise

**Intuition**: Parameters with large gradients get smaller learning rates; parameters with small gradients get larger learning rates.

**Advantages**:
- ✅ No need to manually tune learning rate
- ✅ Works well with sparse data
- ✅ Different learning rate for each parameter

**Disadvantages**:
- ❌ Learning rate monotonically decreases
- ❌ Can stop learning too early
- ❌ Not suitable for non-convex optimization

**When to use**: Sparse data, NLP tasks

### RMSprop (Root Mean Square Propagation)

Fixes AdaGrad's monotonic learning rate decay by using exponentially weighted moving average.

**Update rules**:
$$E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta)(\nabla_\theta J(\theta_t))^2$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{E[g^2]_t + \epsilon}} \nabla_\theta J(\theta_t)$$

where:
- $E[g^2]_t$ is the exponentially weighted average of squared gradients
- $\beta$ is the decay rate (typically 0.9)

**Advantages**:
- ✅ Fixes AdaGrad's diminishing learning rate
- ✅ Works well in practice
- ✅ Good for RNNs

**Disadvantages**:
- ❌ Additional hyperparameter ($\beta$)
- ❌ Not as popular as Adam

**Typical values**: $\alpha = 0.001$, $\beta = 0.9$

### Adam (Adaptive Moment Estimation)

Combines **momentum** (first moment) and **RMSprop** (second moment).

**Update rules**:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla_\theta J(\theta_t)$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla_\theta J(\theta_t))^2$$

**Bias correction**:
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

**Parameter update**:
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

where:
- $m_t$ is the first moment (mean of gradients)
- $v_t$ is the second moment (uncentered variance of gradients)
- $\beta_1, \beta_2$ are decay rates

**Advantages**:
- ✅ **Most popular optimizer** in deep learning
- ✅ Combines benefits of momentum and adaptive learning rates
- ✅ Bias correction handles initial steps well
- ✅ Works well with minimal tuning

**Disadvantages**:
- ❌ Can converge to suboptimal solutions in some cases
- ❌ May not generalize as well as SGD with momentum

**Typical values**: $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

### AdamW (Adam with Weight Decay)

Adam with **decoupled weight decay** regularization.

**Update rule**:
$$\theta_{t+1} = \theta_t - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

where $\lambda$ is the weight decay coefficient.

**Key difference**: Weight decay is applied directly to parameters, not included in gradient.

**Advantages**:
- ✅ Better regularization than Adam with L2
- ✅ Often better generalization
- ✅ State-of-the-art for transformers

**Typical values**: $\lambda = 0.01$

### Nadam (Nesterov-accelerated Adam)

Combines **Nesterov momentum** with **Adam**.

**Update rule**:
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} (\beta_1 \hat{m}_t + \frac{(1-\beta_1)\nabla_\theta J(\theta_t)}{1-\beta_1^t})$$

**Advantages**:
- ✅ Often converges faster than Adam
- ✅ Better for some tasks

**Disadvantages**:
- ❌ Slightly more complex
- ❌ Not as widely used

### Adadelta

Extension of AdaGrad that seeks to reduce monotonic learning rate decay.

**Update rules**:
$$E[g^2]_t = \rho E[g^2]_{t-1} + (1-\rho)g_t^2$$
$$\Delta\theta_t = -\frac{\sqrt{E[\Delta\theta^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} g_t$$
$$E[\Delta\theta^2]_t = \rho E[\Delta\theta^2]_{t-1} + (1-\rho)\Delta\theta_t^2$$
$$\theta_{t+1} = \theta_t + \Delta\theta_t$$

**Advantages**:
- ✅ No learning rate hyperparameter
- ✅ Robust to hyperparameters

**Disadvantages**:
- ❌ Less popular than Adam/RMSprop
- ❌ Can be slower to converge

## Second-Order Methods

Use second-order derivatives (Hessian) for more informed updates.

### Newton's Method

$$\theta_{t+1} = \theta_t - H^{-1} \nabla_\theta J(\theta_t)$$

where $H$ is the Hessian matrix.

**Advantages**:
- ✅ Fast convergence near minimum
- ✅ Uses curvature information

**Disadvantages**:
- ❌ Extremely expensive to compute $H$ and $H^{-1}$ ($O(n^3)$)
- ❌ Not practical for large neural networks

### L-BFGS (Limited-memory BFGS)

Approximates the Hessian using gradient history.

**Advantages**:
- ✅ Faster convergence than first-order methods
- ✅ Memory efficient (doesn't store full Hessian)

**Disadvantages**:
- ❌ Requires full batch (not suitable for mini-batch)
- ❌ Complex implementation
- ❌ Not commonly used for deep learning

**When to use**: Small datasets, when you can use full-batch

## Choosing an Optimizer

### Quick Decision Guide

**Default choice**: **Adam** or **AdamW**
- Works well for most problems
- Minimal hyperparameter tuning
- Good convergence speed

**For best generalization**: **SGD with momentum**
- Often achieves better test performance
- Requires careful learning rate tuning
- Use with learning rate scheduling

**For transformers/NLP**: **AdamW**
- Current state-of-the-art
- Better regularization

**For RNNs**: **RMSprop** or **Adam**
- Handles vanishing/exploding gradients well

**For sparse data**: **AdaGrad** or **Adam**
- Adaptive learning rates help with sparse features

### Comparison Table

| Optimizer | Speed | Generalization | Tuning Effort | Best For |
|-----------|-------|----------------|---------------|----------|
| SGD | Slow | Excellent | High | Final models, research |
| SGD + Momentum | Medium | Excellent | Medium-High | Final models |
| Adam | Fast | Good | Low | Prototyping, most tasks |
| AdamW | Fast | Very Good | Low | Transformers, modern NNs |
| RMSprop | Fast | Good | Medium | RNNs |
| AdaGrad | Medium | Good | Low | Sparse data |

## Practical Considerations

### Learning Rate Schedules

Adjust learning rate during training for better convergence.

**Step Decay**:
$$\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t/k \rfloor}$$

Reduce learning rate by factor $\gamma$ every $k$ epochs.

**Exponential Decay**:
$$\alpha_t = \alpha_0 e^{-\lambda t}$$

**Cosine Annealing**:
$$\alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_{max} - \alpha_{min})(1 + \cos(\frac{t\pi}{T}))$$

**Warm Restarts**: Periodically reset learning rate to escape local minima.

**Learning Rate Warmup**: Gradually increase learning rate at start of training.

### Gradient Clipping

Prevent exploding gradients by limiting gradient magnitude.

**Clip by value**:
$$g_i = \max(\min(g_i, \text{threshold}), -\text{threshold})$$

**Clip by norm**:
$$g = \begin{cases}
g & \text{if } ||g|| \leq \text{threshold} \\
\frac{g \cdot \text{threshold}}{||g||} & \text{otherwise}
\end{cases}$$

**When to use**: RNNs, very deep networks, when gradients explode

### Batch Size Effects

**Small batches** (32-64):
- More noise in gradients
- Better generalization (regularization effect)
- Slower convergence (more iterations)
- Less memory

**Large batches** (256-512):
- More stable gradients
- Faster per-epoch training
- May generalize worse
- More memory required

**Very large batches** (>1024):
- May require learning rate adjustment
- Can lead to sharp minima (worse generalization)
- Use with caution

### Hyperparameter Tuning Tips

**Learning Rate**:
- Most important hyperparameter
- Start with default: 0.001 for Adam, 0.01 for SGD
- Use learning rate finder
- Reduce if training is unstable

**Batch Size**:
- Start with 32 or 64
- Increase if you have more memory
- Adjust learning rate proportionally

**Momentum/Beta Parameters**:
- Use defaults first ($\beta_1=0.9$, $\beta_2=0.999$ for Adam)
- $\beta_2$ closer to 1 for sparse gradients

## Implementation

### NumPy Implementations

```python
import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.v = {}

    def update(self, params, grads):
        if not self.v:
            for key in params.keys():
                self.v[key] = np.zeros_like(params[key])

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Time step

    def update(self, params, grads):
        if not self.m:
            for key in params.keys():
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

        self.t += 1

        for key in params.keys():
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]

            # Update biased second moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # Update parameters
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class RMSprop:
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = {}

    def update(self, params, grads):
        if not self.cache:
            for key in params.keys():
                self.cache[key] = np.zeros_like(params[key])

        for key in params.keys():
            # Accumulate squared gradients
            self.cache[key] = (self.decay_rate * self.cache[key] +
                             (1 - self.decay_rate) * grads[key] ** 2)

            # Update parameters
            params[key] -= self.lr * grads[key] / (np.sqrt(self.cache[key]) + self.epsilon)
```

### PyTorch Usage

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

# Choose optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Alternative optimizers:
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
# optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Compute gradients

        # Update parameters
        optimizer.step()
```

### Learning Rate Scheduling

```python
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step decay: reduce LR by factor of 0.1 every 30 epochs
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# Reduce on plateau: reduce when validation loss stops improving
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# Training loop with scheduler
for epoch in range(num_epochs):
    train(...)
    val_loss = validate(...)

    scheduler.step()  # or scheduler.step(val_loss) for ReduceLROnPlateau
```

### Gradient Clipping

```python
# Clip gradients by norm
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Clip gradients by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# Example in training loop
for batch_x, batch_y in dataloader:
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)

    optimizer.zero_grad()
    loss.backward()

    # Clip gradients before optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
```

## Debugging Optimizer Issues

### Loss Not Decreasing

**Possible causes**:
1. Learning rate too high → reduce by 10x
2. Learning rate too low → increase by 10x
3. Bad initialization → check weight initialization
4. Wrong loss function → verify loss function matches task
5. Gradient vanishing/exploding → check gradient norms

### Loss Exploding (NaN/Inf)

**Solutions**:
1. Reduce learning rate
2. Apply gradient clipping
3. Check for numerical instability (log of negative, division by zero)
4. Use batch normalization

### Slow Convergence

**Solutions**:
1. Increase learning rate (if stable)
2. Try Adam instead of SGD
3. Use learning rate warmup
4. Increase batch size
5. Better weight initialization

### Poor Generalization

**Solutions**:
1. Try SGD with momentum instead of Adam
2. Add regularization (L2, dropout)
3. Reduce model capacity
4. Increase training data
5. Use data augmentation

## References

- Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization"
- Loshchilov, I., & Hutter, F. (2017). "Decoupled Weight Decay Regularization" (AdamW)
- Duchi, J., et al. (2011). "Adaptive Subgradient Methods for Online Learning" (AdaGrad)
- Tieleman, T., & Hinton, G. (2012). "Lecture 6.5-rmsprop"
- Ruder, S. (2016). "An Overview of Gradient Descent Optimization Algorithms"
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning" - Chapter 8
