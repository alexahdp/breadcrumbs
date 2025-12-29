# Feedforward Neural Networks

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Dense (Fully Connected) Layers](#dense-fully-connected-layers)
4. [Forward Propagation](#forward-propagation)
5. [Universal Approximation Theorem](#universal-approximation-theorem)
6. [Deep Feedforward Networks](#deep-feedforward-networks)
7. [Common Architectures](#common-architectures)
8. [Practical Considerations](#practical-considerations)

## Introduction

**Feedforward Neural Networks (FNN)**, also known as **Multi-Layer Perceptrons (MLP)**, are the most basic type of artificial neural network architecture. In this network, information moves in only one direction—forward—from input nodes, through hidden nodes (if any), to output nodes. There are no cycles or loops in the network.

**Key characteristics:**
- Information flows in one direction only (no feedback loops)
- Consists of layers of neurons fully connected to adjacent layers
- Foundation for understanding more complex architectures
- Also called **Dense Networks** due to fully connected layers

## Architecture

### Basic Structure

A feedforward network consists of:

1. **Input Layer** - receives the input features
2. **Hidden Layers** - intermediate layers that learn representations
3. **Output Layer** - produces the final prediction

**Single Hidden Layer Network:**

```
Input Layer → Hidden Layer → Output Layer
   (n)      →     (h)      →      (m)
```

where:
- $n$ = number of input features
- $h$ = number of hidden units
- $m$ = number of output units

### Layer Connectivity

Each neuron in layer $l$ is connected to **every** neuron in layer $l+1$, hence the name "fully connected" or "dense" layers.

For a network with $L$ layers:
- Layer 0: Input layer
- Layers 1 to $L-1$: Hidden layers
- Layer $L$: Output layer

## Dense (Fully Connected) Layers

### Mathematical Formulation

For a single dense layer, the transformation is:

$$\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$$

$$\mathbf{a} = f(\mathbf{z})$$

where:
- $\mathbf{x} \in \mathbb{R}^n$ - input vector
- $\mathbf{W} \in \mathbb{R}^{m \times n}$ - weight matrix
- $\mathbf{b} \in \mathbb{R}^m$ - bias vector
- $\mathbf{z}$ - pre-activation (linear combination)
- $f$ - activation function
- $\mathbf{a}$ - activation (output)

### Parameters Count

For a layer with $n$ inputs and $m$ outputs:

$$\text{Parameters} = n \times m + m = m(n + 1)$$

**Example:** Layer with 784 inputs and 128 outputs:
- Weights: $784 \times 128 = 100,352$
- Biases: $128$
- **Total: 100,480 parameters**

## Forward Propagation

### Layer-by-Layer Computation

For a network with $L$ layers:

$$\mathbf{a}^{[0]} = \mathbf{x}$$

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]}\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$

$$\mathbf{a}^{[l]} = f^{[l]}(\mathbf{z}^{[l]})$$

for $l = 1, 2, ..., L$

The final output is:

$$\hat{\mathbf{y}} = \mathbf{a}^{[L]}$$

### Example: Three-Layer Network

**Network structure:** Input(4) → Hidden(3) → Hidden(2) → Output(1)

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

# Forward pass
x = np.array([1.0, 2.0, 3.0, 4.0])  # Input

# Layer 1
W1 = np.random.randn(3, 4)
b1 = np.zeros(3)
z1 = W1 @ x + b1
a1 = relu(z1)

# Layer 2
W2 = np.random.randn(2, 3)
b2 = np.zeros(2)
z2 = W2 @ a1 + b2
a2 = relu(z2)

# Layer 3 (Output)
W3 = np.random.randn(1, 2)
b3 = np.zeros(1)
z3 = W3 @ a2 + b3
y_pred = sigmoid(z3)  # For binary classification
```

## Universal Approximation Theorem

**Theorem:** A feedforward network with:
- A single hidden layer
- Finite number of neurons
- Appropriate activation function (non-polynomial)

can approximate **any continuous function** on compact subsets of $\mathbb{R}^n$ to arbitrary accuracy.

**Implications:**
- Theoretically, one hidden layer is sufficient for any function
- In practice, deeper networks learn better representations
- Depth allows hierarchical feature learning with fewer total parameters

**Practical insight:** While 1 hidden layer can approximate any function, it might require an exponentially large number of neurons. Deep networks achieve the same with fewer parameters.

## Deep Feedforward Networks

### Why Go Deep?

**Advantages of depth:**
1. **Hierarchical feature learning** - early layers learn simple features, deeper layers learn complex combinations
2. **Parameter efficiency** - achieve same expressiveness with fewer parameters
3. **Better generalization** - learned representations are more abstract
4. **Compositional structures** - many real-world functions are compositional

### Typical Depth Guidelines

- **Shallow (1-2 hidden layers):** Simple tabular data, small datasets
- **Medium (3-5 hidden layers):** Moderate complexity problems
- **Deep (6+ hidden layers):** Complex patterns, large datasets, images, sequences

## Common Architectures

### Binary Classification

```
Input(n) → Hidden(h1) → Hidden(h2) → Output(1)
           ReLU          ReLU          Sigmoid
```

**Output activation:** Sigmoid (outputs probability)

**Loss function:** Binary Cross-Entropy

$$L = -\frac{1}{N}\sum_{i=1}^{N}[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

### Multi-Class Classification

```
Input(n) → Hidden(h1) → Hidden(h2) → Output(C)
           ReLU          ReLU          Softmax
```

**Output activation:** Softmax (outputs probability distribution)

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C}e^{z_j}}$$

**Loss function:** Categorical Cross-Entropy

$$L = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{C}y_{ij}\log(\hat{y}_{ij})$$

### Regression

```
Input(n) → Hidden(h1) → Hidden(h2) → Output(m)
           ReLU          ReLU          Linear
```

**Output activation:** Linear (no activation) or ReLU (for positive outputs)

**Loss function:** Mean Squared Error

$$L = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2$$

## Practical Considerations

### Layer Size Selection

**Rule of thumb for hidden layer sizes:**
- Start with layers decreasing in size: e.g., 512 → 256 → 128
- Or maintain constant size across layers
- Output layer size determined by task

**Example architectures for different tasks:**

```python
# Small tabular data (10 features)
model = [10, 64, 32, 1]

# Medium complexity (100 features)
model = [100, 256, 128, 64, 10]

# High-dimensional input (1000+ features)
model = [1000, 512, 256, 128, 64, num_classes]
```

### Activation Functions

**Hidden layers:**
- **ReLU** - default choice, fast, helps with vanishing gradients
- **LeakyReLU** - prevents "dying ReLU" problem
- **ELU** - smoother alternative to ReLU
- **GELU** - used in modern transformers

**Output layer:**
- **Sigmoid** - binary classification (0 to 1)
- **Softmax** - multi-class classification (probability distribution)
- **Linear** - regression
- **Tanh** - outputs in range (-1, 1)

### Weight Initialization

Proper initialization is critical for training deep networks.

**Xavier/Glorot Initialization** (for tanh, sigmoid):

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)$$

**He Initialization** (for ReLU):

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

```python
import numpy as np

# He initialization
def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(
            layer_dims[l],
            layer_dims[l-1]
        ) * np.sqrt(2.0 / layer_dims[l-1])
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))

    return parameters
```

### Regularization Techniques

**L2 Regularization (Weight Decay):**

$$L_{total} = L_{data} + \lambda\sum_{l=1}^{L}||\mathbf{W}^{[l]}||_2^2$$

**Dropout:**
- Randomly set fraction $p$ of activations to zero during training
- Prevents co-adaptation of neurons
- Typical values: $p = 0.2$ to $0.5$

**Early Stopping:**
- Monitor validation loss
- Stop training when validation loss stops improving

### Overfitting Prevention

**Signs of overfitting:**
- Training accuracy much higher than validation accuracy
- Training loss decreasing while validation loss increasing

**Solutions:**
1. **More data** - collect more training examples
2. **Regularization** - L1, L2, dropout
3. **Reduce model complexity** - fewer layers or neurons
4. **Data augmentation** - artificially expand training set
5. **Cross-validation** - k-fold validation for better estimates
6. **Batch normalization** - normalizes layer inputs

### Implementation Example

**Complete feedforward network in PyTorch:**

```python
import torch
import torch.nn as nn

class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_prob=0.5):
        super(FeedforwardNN, self).__init__()

        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_prob))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Example usage
model = FeedforwardNN(
    input_size=784,          # e.g., MNIST flattened images
    hidden_sizes=[512, 256, 128],
    num_classes=10,
    dropout_prob=0.3
)

# For binary classification, use sigmoid at the end
# For multi-class, apply softmax with CrossEntropyLoss (includes softmax)
```

### Training Tips

**Learning rate selection:**
- Start with $\alpha = 0.001$ (Adam optimizer)
- Use learning rate scheduling (reduce on plateau)
- Try learning rate finder to find optimal range

**Batch size:**
- Typical values: 32, 64, 128, 256
- Smaller batches: more noise, better generalization, slower training
- Larger batches: faster training, more stable gradients, may generalize worse

**Monitoring:**
- Track both training and validation metrics
- Use tensorboard or wandb for visualization
- Monitor gradient norms to detect vanishing/exploding gradients

### When to Use Feedforward Networks

**Good for:**
- Tabular data with fixed-size feature vectors
- Classification and regression tasks
- Relatively small to medium datasets
- Problems without spatial or temporal structure

**Not ideal for:**
- Image data (use CNNs instead)
- Sequential data (use RNNs/LSTMs/Transformers)
- Very high-dimensional data without structure
- Problems requiring variable-length inputs

## References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals and Systems.
- Nielsen, M. (2015). Neural Networks and Deep Learning. Determination Press.
- [PyTorch Documentation: Neural Networks](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)
