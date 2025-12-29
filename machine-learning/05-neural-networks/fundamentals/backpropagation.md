# Backpropagation

## Table of Contents

1. [Introduction](#introduction)
2. [The Chain Rule](#the-chain-rule)
3. [Computational Graph](#computational-graph)
4. [Forward Pass](#forward-pass)
5. [Backward Pass](#backward-pass)
6. [Algorithm Details](#algorithm-details)
7. [Implementation](#implementation)
8. [Common Issues](#common-issues)

## Introduction

**Backpropagation** (backward propagation of errors) is the algorithm used to efficiently compute gradients of the loss function with respect to all parameters in a neural network. It is the cornerstone of training neural networks.

### Why Backpropagation?

Neural networks have many parameters (weights and biases). To train them using gradient descent, we need to compute:

$$\frac{\partial L}{\partial w_{ij}^{[l]}} \quad \text{and} \quad \frac{\partial L}{\partial b_i^{[l]}}$$

for every weight and bias in the network.

**Naive approach**: Compute each partial derivative independently → extremely slow ($O(n^2)$ for $n$ parameters)

**Backpropagation**: Use the chain rule to compute all gradients efficiently in $O(n)$ time

### Key Idea

Backpropagation applies the **chain rule of calculus** to efficiently compute gradients by:
1. Computing outputs layer by layer (forward pass)
2. Computing gradients layer by layer in reverse order (backward pass)
3. Reusing previously computed gradients

## The Chain Rule

### Single Variable Chain Rule

For composite functions $f(g(x))$:

$$\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

**Example:**
$$y = (3x + 2)^2$$

Let $u = 3x + 2$, then $y = u^2$

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = 2u \cdot 3 = 6(3x + 2)$$

### Multivariable Chain Rule

For $z = f(x, y)$ where $x = g(t)$ and $y = h(t)$:

$$\frac{dz}{dt} = \frac{\partial z}{\partial x} \cdot \frac{dx}{dt} + \frac{\partial z}{\partial y} \cdot \frac{dy}{dt}$$

### Vector Chain Rule

For $\mathbf{y} = f(\mathbf{x})$ and $L = g(\mathbf{y})$:

$$\frac{\partial L}{\partial x_i} = \sum_j \frac{\partial L}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i}$$

In matrix form:

$$\frac{\partial L}{\partial \mathbf{x}} = \left(\frac{\partial \mathbf{y}}{\partial \mathbf{x}}\right)^T \frac{\partial L}{\partial \mathbf{y}}$$

## Computational Graph

A **computational graph** represents the sequence of operations in a neural network as a directed acyclic graph (DAG).

### Simple Example

Consider: $L = (w \cdot x + b)^2$

```
x, w, b (inputs)
   ↓
   z = w·x + b
   ↓
   L = z²
```

**Forward pass**: Compute values from inputs to output
**Backward pass**: Compute gradients from output to inputs

### Neural Network as Computational Graph

For a simple 2-layer network:

```
Input x
   ↓
z^[1] = W^[1]·x + b^[1]
   ↓
a^[1] = g^[1](z^[1])
   ↓
z^[2] = W^[2]·a^[1] + b^[2]
   ↓
a^[2] = g^[2](z^[2])
   ↓
Loss L(a^[2], y)
```

## Forward Pass

The forward pass computes the output of the network and all intermediate values needed for backpropagation.

### Notation

For layer $l$:
- Input: $\mathbf{a}^{[l-1]}$ (with $\mathbf{a}^{[0]} = \mathbf{x}$)
- Linear transformation: $\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$
- Activation: $\mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})$
- Output: $\hat{\mathbf{y}} = \mathbf{a}^{[L]}$ (where $L$ is the last layer)

### Forward Pass Algorithm

For each layer $l = 1, 2, \ldots, L$:

1. Compute linear transformation:
   $$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$

2. Apply activation function:
   $$\mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})$$

3. **Cache** $\mathbf{z}^{[l]}$, $\mathbf{a}^{[l]}$, $\mathbf{W}^{[l]}$, $\mathbf{b}^{[l]}$ for backward pass

4. Compute loss (after final layer):
   $$L = \mathcal{L}(\mathbf{a}^{[L]}, \mathbf{y})$$

## Backward Pass

The backward pass computes gradients of the loss with respect to all parameters by applying the chain rule in reverse order.

### Output Layer Gradients

For the output layer $L$:

**Step 1**: Compute gradient of loss w.r.t. output activations:

$$\frac{\partial L}{\partial \mathbf{a}^{[L]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}}$$

This depends on the loss function used.

**Step 2**: Compute gradient w.r.t. pre-activation:

$$\frac{\partial L}{\partial \mathbf{z}^{[L]}} = \frac{\partial L}{\partial \mathbf{a}^{[L]}} \odot g'^{[L]}(\mathbf{z}^{[L]})$$

where $\odot$ denotes element-wise multiplication.

**Step 3**: Compute gradients w.r.t. parameters:

$$\frac{\partial L}{\partial \mathbf{W}^{[L]}} = \frac{\partial L}{\partial \mathbf{z}^{[L]}} (\mathbf{a}^{[L-1]})^T$$

$$\frac{\partial L}{\partial \mathbf{b}^{[L]}} = \frac{\partial L}{\partial \mathbf{z}^{[L]}}$$

### Hidden Layer Gradients

For hidden layer $l$ (working backwards from $L-1$ to $1$):

**Step 1**: Propagate gradient back to activations:

$$\frac{\partial L}{\partial \mathbf{a}^{[l]}} = (\mathbf{W}^{[l+1]})^T \frac{\partial L}{\partial \mathbf{z}^{[l+1]}}$$

**Step 2**: Compute gradient w.r.t. pre-activation:

$$\frac{\partial L}{\partial \mathbf{z}^{[l]}} = \frac{\partial L}{\partial \mathbf{a}^{[l]}} \odot g'^{[l]}(\mathbf{z}^{[l]})$$

**Step 3**: Compute gradients w.r.t. parameters:

$$\frac{\partial L}{\partial \mathbf{W}^{[l]}} = \frac{\partial L}{\partial \mathbf{z}^{[l]}} (\mathbf{a}^{[l-1]})^T$$

$$\frac{\partial L}{\partial \mathbf{b}^{[l]}} = \frac{\partial L}{\partial \mathbf{z}^{[l]}}$$

### Vectorized Notation (Batch)

For $m$ training examples, replace vectors with matrices where each column is one example:

$$\frac{\partial L}{\partial \mathbf{W}^{[l]}} = \frac{1}{m} \frac{\partial L}{\partial \mathbf{Z}^{[l]}} (\mathbf{A}^{[l-1]})^T$$

$$\frac{\partial L}{\partial \mathbf{b}^{[l]}} = \frac{1}{m} \sum_{i=1}^{m} \frac{\partial L}{\partial \mathbf{z}^{[l](i)}}$$

## Algorithm Details

### Complete Backpropagation Algorithm

**Input**: Training data $(\mathbf{x}, \mathbf{y})$, network parameters $\{\mathbf{W}^{[l]}, \mathbf{b}^{[l]}\}_{l=1}^{L}$

**Output**: Gradients $\{\frac{\partial L}{\partial \mathbf{W}^{[l]}}, \frac{\partial L}{\partial \mathbf{b}^{[l]}}\}_{l=1}^{L}$

**Forward Pass**:
```
1. Set a^[0] = x
2. For l = 1 to L:
     z^[l] = W^[l] · a^[l-1] + b^[l]
     a^[l] = g^[l](z^[l])
     Cache z^[l], a^[l]
3. Compute loss L = ℒ(a^[L], y)
```

**Backward Pass**:
```
1. Compute dL/da^[L] from loss function
2. Compute dL/dz^[L] = dL/da^[L] ⊙ g'^[L](z^[L])
3. For l = L down to 1:
     dL/dW^[l] = dL/dz^[l] · (a^[l-1])^T
     dL/db^[l] = dL/dz^[l]
     If l > 1:
       dL/da^[l-1] = (W^[l])^T · dL/dz^[l]
       dL/dz^[l-1] = dL/da^[l-1] ⊙ g'^[l-1](z^[l-1])
4. Return gradients
```

### Loss Function Derivatives

**Mean Squared Error (Regression)**:
$$L = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

$$\frac{\partial L}{\partial \hat{y}} = \frac{1}{m}(\hat{y} - y)$$

**Binary Cross-Entropy (Binary Classification)**:
$$L = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)})]$$

$$\frac{\partial L}{\partial \hat{y}} = -\frac{1}{m} \left(\frac{y}{\hat{y}} - \frac{1-y}{1-\hat{y}}\right)$$

With sigmoid activation in output layer:
$$\frac{\partial L}{\partial z^{[L]}} = \frac{1}{m}(\hat{y} - y)$$

**Categorical Cross-Entropy (Multi-class)**:
$$L = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log(\hat{y}_k^{(i)})$$

With softmax activation:
$$\frac{\partial L}{\partial z^{[L]}} = \frac{1}{m}(\hat{y} - y)$$

### Activation Function Derivatives

**ReLU**:
$$g'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

**Sigmoid**:
$$g'(z) = g(z)(1 - g(z))$$

**Tanh**:
$$g'(z) = 1 - g(z)^2$$

**Leaky ReLU**:
$$g'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \alpha & \text{if } z \leq 0 \end{cases}$$

## Implementation

### NumPy Implementation

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.L = len(layer_sizes) - 1  # number of layers (excluding input)
        self.parameters = {}

        # Initialize parameters
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] = np.random.randn(
                layer_sizes[l], layer_sizes[l-1]
            ) * 0.01
            self.parameters[f'b{l}'] = np.zeros((layer_sizes[l], 1))

    def forward_propagation(self, X):
        """
        Forward propagation

        Returns:
        AL: output of the network
        cache: intermediate values for backpropagation
        """
        cache = {'A0': X}
        A = X

        # Forward through all layers
        for l in range(1, self.L + 1):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']

            Z = np.dot(W, A_prev) + b
            cache[f'Z{l}'] = Z

            # Use ReLU for hidden layers, sigmoid for output
            if l < self.L:
                A = relu(Z)
            else:
                A = sigmoid(Z)

            cache[f'A{l}'] = A

        return A, cache

    def compute_cost(self, AL, Y):
        """Binary cross-entropy loss"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) / m
        return cost

    def backward_propagation(self, AL, Y, cache):
        """
        Backward propagation

        Returns:
        gradients: dictionary containing gradients
        """
        gradients = {}
        m = Y.shape[1]

        # Output layer gradients
        # For sigmoid + binary cross-entropy: dZ = A - Y
        dZ = AL - Y
        gradients[f'dW{self.L}'] = np.dot(dZ, cache[f'A{self.L-1}'].T) / m
        gradients[f'db{self.L}'] = np.sum(dZ, axis=1, keepdims=True) / m

        # Hidden layers gradients
        for l in reversed(range(1, self.L)):
            # Propagate gradient to previous layer
            dA = np.dot(self.parameters[f'W{l+1}'].T, dZ)

            # Gradient through activation function
            dZ = dA * relu_derivative(cache[f'Z{l}'])

            # Compute gradients for this layer
            gradients[f'dW{l}'] = np.dot(dZ, cache[f'A{l-1}'].T) / m
            gradients[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / m

        return gradients

    def update_parameters(self, gradients, learning_rate):
        """Update parameters using gradient descent"""
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] -= learning_rate * gradients[f'dW{l}']
            self.parameters[f'b{l}'] -= learning_rate * gradients[f'db{l}']

    def train(self, X, Y, learning_rate=0.01, num_iterations=1000):
        """Complete training loop"""
        costs = []

        for i in range(num_iterations):
            # Forward propagation
            AL, cache = self.forward_propagation(X)

            # Compute cost
            cost = self.compute_cost(AL, Y)
            costs.append(cost)

            # Backward propagation
            gradients = self.backward_propagation(AL, Y, cache)

            # Update parameters
            self.update_parameters(gradients, learning_rate)

            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")

        return costs
```

### Usage Example

```python
# Create dataset
np.random.seed(42)
X = np.random.randn(2, 100)  # 2 features, 100 examples
Y = (X[0] + X[1] > 0).astype(int).reshape(1, -1)  # Binary labels

# Create and train network
nn = NeuralNetwork([2, 4, 3, 1])  # 2 inputs, 2 hidden layers, 1 output
costs = nn.train(X, Y, learning_rate=0.1, num_iterations=1000)

# Make predictions
AL, _ = nn.forward_propagation(X)
predictions = (AL > 0.5).astype(int)
accuracy = np.mean(predictions == Y)
print(f"Accuracy: {accuracy:.2%}")
```

### Gradient Checking

Verify backpropagation implementation using numerical gradients:

```python
def gradient_check(network, X, Y, epsilon=1e-7):
    """
    Check if backpropagation gradients are correct using numerical gradient

    Returns:
    difference: relative difference between gradients
    """
    # Compute gradients using backpropagation
    AL, cache = network.forward_propagation(X)
    gradients = network.backward_propagation(AL, Y, cache)

    # Flatten parameters and gradients
    params_vector = []
    grads_vector = []

    for l in range(1, network.L + 1):
        params_vector.append(network.parameters[f'W{l}'].flatten())
        params_vector.append(network.parameters[f'b{l}'].flatten())
        grads_vector.append(gradients[f'dW{l}'].flatten())
        grads_vector.append(gradients[f'db{l}'].flatten())

    params_vector = np.concatenate(params_vector)
    grads_vector = np.concatenate(grads_vector)

    # Compute numerical gradients
    num_grads = np.zeros_like(params_vector)

    for i in range(len(params_vector)):
        # Compute f(θ + ε)
        params_vector[i] += epsilon
        # Set parameters and compute cost
        # ... (set parameters in network)
        AL_plus, _ = network.forward_propagation(X)
        cost_plus = network.compute_cost(AL_plus, Y)

        # Compute f(θ - ε)
        params_vector[i] -= 2 * epsilon
        AL_minus, _ = network.forward_propagation(X)
        cost_minus = network.compute_cost(AL_minus, Y)

        # Numerical gradient
        num_grads[i] = (cost_plus - cost_minus) / (2 * epsilon)

        # Restore parameter
        params_vector[i] += epsilon

    # Compute relative difference
    numerator = np.linalg.norm(grads_vector - num_grads)
    denominator = np.linalg.norm(grads_vector) + np.linalg.norm(num_grads)
    difference = numerator / denominator

    if difference < 1e-7:
        print("✓ Gradient check passed!")
    else:
        print("✗ Gradient check failed!")

    return difference
```

## Common Issues

### 1. Vanishing Gradients

**Problem**: Gradients become extremely small in early layers.

**Causes**:
- Sigmoid/tanh activations (derivative < 1)
- Deep networks (many layers)
- Poor weight initialization

**Solutions**:
- Use ReLU-based activations
- Batch normalization
- Residual connections
- Proper initialization (He/Xavier)

### 2. Exploding Gradients

**Problem**: Gradients become extremely large.

**Causes**:
- Poor weight initialization
- High learning rate
- Very deep networks

**Solutions**:
- Gradient clipping
- Lower learning rate
- Batch normalization
- Proper initialization

### 3. Dead Neurons

**Problem**: Neurons always output 0 (with ReLU).

**Causes**:
- Large negative pre-activation values
- High learning rate
- Poor initialization

**Solutions**:
- Use Leaky ReLU or ELU
- Lower learning rate
- Proper initialization

### 4. Numerical Instability

**Problem**: NaN or Inf values during training.

**Causes**:
- Large exponentials in softmax/sigmoid
- Division by zero in log
- Very large gradients

**Solutions**:
- Numerical stability tricks (e.g., subtract max in softmax)
- Gradient clipping
- Check for invalid values

## Computational Complexity

**Forward pass**: $O(N)$ where $N$ is the total number of parameters
**Backward pass**: $O(N)$ (same as forward pass)

**Memory**: Must store all intermediate activations ($O(N)$)

**Efficiency**: Backpropagation is remarkably efficient compared to naive gradient computation, which would be $O(N^2)$.

## Historical Context

- **1970**: Seppo Linnainmaa publishes the reverse mode of automatic differentiation
- **1974**: Paul Werbos develops backpropagation for neural networks in his PhD thesis
- **1986**: Rumelhart, Hinton, and Williams popularize backpropagation
- **1989**: LeCun applies backpropagation to handwritten digit recognition
- **Modern era**: Automatic differentiation frameworks (PyTorch, TensorFlow) handle backpropagation automatically

## References

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning Representations by Back-propagating Errors"
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning" - Chapter 6
- Nielsen, M. "Neural Networks and Deep Learning" - Chapter 2
- LeCun, Y., et al. (1998). "Efficient BackProp"
