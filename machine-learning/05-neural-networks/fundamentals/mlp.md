# Multi-Layer Perceptron (MLP)

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Forward Propagation](#forward-propagation)
4. [Universal Approximation](#universal-approximation)
5. [Training Process](#training-process)
6. [Practical Considerations](#practical-considerations)
7. [Implementation](#implementation)

## Introduction

A **Multi-Layer Perceptron (MLP)** is a feedforward artificial neural network consisting of multiple layers of neurons. Unlike the single-layer perceptron, MLPs can learn non-linear relationships and solve complex problems like the XOR function.

MLPs are also called **feedforward neural networks** or **fully connected networks** because:
- Information flows in one direction (forward) from input to output
- Each neuron in one layer connects to all neurons in the next layer

### Key Characteristics

- **Multiple layers**: input layer, one or more hidden layers, output layer
- **Non-linear activation functions**: enable learning of complex patterns
- **Backpropagation**: used for training
- **Universal approximators**: can approximate any continuous function

## Architecture

### Layer Types

**1. Input Layer**
- Receives the raw input features
- Number of neurons = number of features
- No computation performed, just passes data forward

**2. Hidden Layer(s)**
- Performs computations and feature transformations
- Can have one or more hidden layers
- Each layer learns increasingly abstract representations

**3. Output Layer**
- Produces final predictions
- Number of neurons depends on the task:
  - Binary classification: 1 neuron (with sigmoid)
  - Multi-class classification: $K$ neurons (with softmax), where $K$ is number of classes
  - Regression: 1 or more neurons (linear activation)

### Network Notation

For a network with $L$ layers:

- $L$ = total number of layers (excluding input)
- $n^{[l]}$ = number of neurons in layer $l$
- $\mathbf{W}^{[l]}$ = weight matrix for layer $l$ (shape: $n^{[l]} \times n^{[l-1]}$)
- $\mathbf{b}^{[l]}$ = bias vector for layer $l$ (shape: $n^{[l]} \times 1$)
- $\mathbf{a}^{[l]}$ = activation output from layer $l$
- $\mathbf{z}^{[l]}$ = linear combination (pre-activation) at layer $l$

### Example Architecture

For a network with 4 input features, two hidden layers (5 and 3 neurons), and 2 output classes:

```
Input: 4 features
  ↓
Hidden Layer 1: 5 neurons (ReLU)
  ↓
Hidden Layer 2: 3 neurons (ReLU)
  ↓
Output Layer: 2 neurons (Softmax)
```

Dimensions:
- $\mathbf{W}^{[1]}$: $5 \times 4$, $\mathbf{b}^{[1]}$: $5 \times 1$
- $\mathbf{W}^{[2]}$: $3 \times 5$, $\mathbf{b}^{[2]}$: $3 \times 1$
- $\mathbf{W}^{[3]}$: $2 \times 3$, $\mathbf{b}^{[3]}$: $2 \times 1$

## Forward Propagation

Forward propagation is the process of computing the network's output from the input.

### Single Neuron Computation

For a single neuron $j$ in layer $l$:

$$z_j^{[l]} = \sum_{i=1}^{n^{[l-1]}} w_{ji}^{[l]} a_i^{[l-1]} + b_j^{[l]}$$

$$a_j^{[l]} = g^{[l]}(z_j^{[l]})$$

where $g^{[l]}$ is the activation function for layer $l$.

### Vectorized Layer Computation

For the entire layer $l$:

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$

$$\mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})$$

where:
- $\mathbf{a}^{[0]} = \mathbf{x}$ (input)
- $g^{[l]}$ is applied element-wise

### Complete Forward Pass

For a 3-layer network:

```
Input: x

Layer 1:
  z^[1] = W^[1] · x + b^[1]
  a^[1] = g^[1](z^[1])

Layer 2:
  z^[2] = W^[2] · a^[1] + b^[2]
  a^[2] = g^[2](z^[2])

Output Layer 3:
  z^[3] = W^[3] · a^[2] + b^[3]
  a^[3] = g^[3](z^[3])

Final output: ŷ = a^[3]
```

### Batch Processing

For $m$ training examples, we stack them as columns:

$$\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]}$$

where $\mathbf{A}^{[l]}$ has shape $(n^{[l]}, m)$ and each column is one training example.

## Universal Approximation

### Universal Approximation Theorem

**Theorem**: A feedforward neural network with:
- At least one hidden layer
- A sufficient number of neurons
- A non-linear activation function (e.g., sigmoid, tanh, ReLU)

can approximate any continuous function on compact subsets of $\mathbb{R}^n$ to arbitrary precision.

### Implications

**What it means:**
- MLPs are theoretically capable of learning any function
- Given enough neurons, an MLP can fit any dataset

**What it doesn't guarantee:**
- How to find the weights (training may be difficult)
- How many neurons are needed
- Whether the learned function generalizes to new data

**Practical insight**: Deep networks (many layers) often work better than wide networks (many neurons per layer) because they can learn hierarchical representations more efficiently.

## Training Process

### Loss Functions

**Regression:**
$$L = \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2 \quad \text{(MSE)}$$

**Binary Classification:**
$$L = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)})] \quad \text{(Binary Cross-Entropy)}$$

**Multi-class Classification:**
$$L = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log(\hat{y}_k^{(i)}) \quad \text{(Categorical Cross-Entropy)}$$

### Training Algorithm

1. **Initialize** weights and biases (randomly)
2. **Forward pass**: compute predictions
3. **Compute loss**: measure error
4. **Backward pass**: compute gradients using backpropagation
5. **Update parameters**: adjust weights and biases using optimizer
6. **Repeat** steps 2-5 for multiple epochs

This process is covered in detail in [Backpropagation](./backpropagation.md) and [Optimizers](./optimizers.md).

## Practical Considerations

### Network Depth vs Width

**Shallow networks** (few layers, many neurons):
- Require exponentially more neurons for complex functions
- Less parameter efficient
- May struggle with hierarchical features

**Deep networks** (many layers, moderate neurons):
- Learn hierarchical representations
- More parameter efficient
- Can suffer from vanishing gradients
- Require careful initialization and normalization

**Rule of thumb**: Start with 2-3 hidden layers and adjust based on performance.

### Number of Neurons per Layer

**Guidelines:**
- Hidden layers typically have 10-100 neurons
- Often decrease in size from input to output
- More neurons → more capacity but risk of overfitting

**Common patterns:**
- Pyramid: gradually decrease layer sizes (e.g., 128 → 64 → 32)
- Constant: same size for all hidden layers
- Hourglass: decrease then increase

### Activation Functions

Different layers often use different activation functions:

- **Hidden layers**: ReLU, LeakyReLU, tanh
- **Output layer**:
  - Regression: linear (no activation)
  - Binary classification: sigmoid
  - Multi-class classification: softmax

See [Activation Functions](./activation-functions.md) for detailed coverage.

### Overfitting Prevention

MLPs are prone to overfitting. Use these techniques:

- **Regularization**: L1, L2 (weight decay)
- **Dropout**: randomly deactivate neurons during training
- **Early stopping**: stop training when validation loss increases
- **Data augmentation**: artificially expand training data
- **Batch normalization**: normalize layer inputs

### Weight Initialization

Proper initialization is crucial:

- **Zero initialization**: Bad! All neurons learn the same features
- **Random small values**: Can cause vanishing gradients
- **Xavier/Glorot**: Good for sigmoid/tanh
  $$W \sim \mathcal{N}(0, \frac{1}{n^{[l-1]}})$$
- **He initialization**: Good for ReLU
  $$W \sim \mathcal{N}(0, \frac{2}{n^{[l-1]}})$$

## Implementation

### NumPy Implementation

```python
import numpy as np

class MLP:
    def __init__(self, layer_sizes, activation='relu'):
        """
        Initialize MLP

        Parameters:
        layer_sizes: list of integers, e.g., [4, 5, 3, 2]
                    (4 inputs, 5 neurons in layer 1, 3 in layer 2, 2 outputs)
        activation: 'relu', 'sigmoid', or 'tanh'
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.activation = activation

        # Initialize parameters
        self.parameters = {}
        for l in range(1, self.num_layers + 1):
            # He initialization for ReLU
            self.parameters[f'W{l}'] = np.random.randn(
                layer_sizes[l], layer_sizes[l-1]
            ) * np.sqrt(2 / layer_sizes[l-1])
            self.parameters[f'b{l}'] = np.zeros((layer_sizes[l], 1))

    def _activate(self, Z, activation):
        """Apply activation function"""
        if activation == 'relu':
            return np.maximum(0, Z)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        elif activation == 'tanh':
            return np.tanh(Z)
        else:
            return Z  # linear

    def forward(self, X):
        """
        Forward propagation

        Parameters:
        X: input data, shape (n_features, m_examples)

        Returns:
        AL: output of the network
        cache: intermediate values for backpropagation
        """
        cache = {'A0': X}
        A = X

        # Forward through hidden layers
        for l in range(1, self.num_layers):
            Z = np.dot(self.parameters[f'W{l}'], A) + self.parameters[f'b{l}']
            A = self._activate(Z, self.activation)
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A

        # Output layer (linear for regression, modify for classification)
        L = self.num_layers
        Z = np.dot(self.parameters[f'W{L}'], A) + self.parameters[f'b{L}']
        AL = Z  # linear activation for regression
        cache[f'Z{L}'] = Z
        cache[f'A{L}'] = AL

        return AL, cache

    def predict(self, X):
        """Make predictions"""
        AL, _ = self.forward(X)
        return AL
```

### Scikit-learn Implementation

```python
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (important for neural networks!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create MLP classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),  # Two hidden layers
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=200,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

# Train
mlp.fit(X_train_scaled, y_train)

# Predict
y_pred = mlp.predict(X_test_scaled)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Number of layers: {mlp.n_layers_}")
print(f"Number of iterations: {mlp.n_iter_}")
print(f"Loss: {mlp.loss_:.4f}")
```

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()

        # Build layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Example usage
model = MLP(input_size=20, hidden_sizes=[64, 32], output_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Debugging MLPs

### Common Issues

**1. Loss not decreasing:**
- Learning rate too high or too low
- Bad initialization
- Wrong loss function

**2. Poor validation performance:**
- Overfitting: add regularization, dropout
- Underfitting: increase capacity, train longer

**3. Vanishing/exploding gradients:**
- Use proper initialization
- Use batch normalization
- Use ReLU instead of sigmoid/tanh
- Gradient clipping for exploding gradients

### Monitoring Training

Track these metrics:
- Training loss (should decrease)
- Validation loss (should decrease, then plateau)
- Gradient norms (should not be too small or too large)
- Weight norms (should not explode)

## References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning"
- Nielsen, M. "Neural Networks and Deep Learning"
- Bishop, C. (2006). "Pattern Recognition and Machine Learning"
- Cybenko, G. (1989). "Approximation by Superpositions of a Sigmoidal Function"
