# Perceptron

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Model](#mathematical-model)
3. [Perceptron Learning Algorithm](#perceptron-learning-algorithm)
4. [Limitations](#limitations)
5. [Practical Implementation](#practical-implementation)

## Introduction

The **perceptron** is the simplest type of artificial neural network, invented by Frank Rosenblatt in 1958. It is a binary linear classifier that maps input features to a binary output (0 or 1).

The perceptron serves as the fundamental building block for more complex neural networks and helped establish the foundation for modern deep learning.

### Biological Inspiration

The perceptron is inspired by biological neurons:
- **Dendrites** → Input features
- **Cell body** → Weighted sum and activation
- **Axon** → Output signal

## Mathematical Model

### Architecture

A perceptron consists of:
- **Input layer**: $n$ input features $x_1, x_2, \ldots, x_n$
- **Weights**: $w_1, w_2, \ldots, w_n$ (learnable parameters)
- **Bias**: $b$ (learnable parameter, also called threshold)
- **Activation function**: step function

### Forward Pass

The perceptron computes its output in two steps:

**Step 1: Weighted Sum**

$$z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^T\mathbf{x} + b$$

where:
- $\mathbf{x} = [x_1, x_2, \ldots, x_n]^T$ is the input vector
- $\mathbf{w} = [w_1, w_2, \ldots, w_n]^T$ is the weight vector
- $b$ is the bias term

**Step 2: Activation**

$$\hat{y} = \begin{cases}
1 & \text{if } z \geq 0 \\
0 & \text{if } z < 0
\end{cases}$$

Alternatively, using the step function:

$$\hat{y} = \text{step}(z) = \text{step}(\mathbf{w}^T\mathbf{x} + b)$$

### Geometric Interpretation

The perceptron creates a **decision boundary** (hyperplane) in the feature space:

$$\mathbf{w}^T\mathbf{x} + b = 0$$

This hyperplane divides the input space into two half-spaces:
- Points on one side are classified as class 1
- Points on the other side are classified as class 0

**Weight vector $\mathbf{w}$** is perpendicular to the decision boundary and points toward the positive class.

**Bias $b$** controls the distance of the hyperplane from the origin.

## Perceptron Learning Algorithm

The perceptron learning algorithm adjusts weights to correctly classify training examples.

### Algorithm Steps

For each training example $(\mathbf{x}^{(i)}, y^{(i)})$:

1. Compute the predicted output: $\hat{y}^{(i)} = \text{step}(\mathbf{w}^T\mathbf{x}^{(i)} + b)$
2. Calculate the error: $e^{(i)} = y^{(i)} - \hat{y}^{(i)}$
3. Update weights and bias:

$$\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot e^{(i)} \cdot \mathbf{x}^{(i)}$$
$$b \leftarrow b + \eta \cdot e^{(i)}$$

where $\eta$ is the **learning rate** (typically between 0.01 and 1).

### Update Rule Interpretation

- If prediction is **correct** ($e^{(i)} = 0$): no update
- If prediction is **incorrect** ($e^{(i)} \neq 0$): weights are adjusted to move the decision boundary toward the correct classification

### Convergence Theorem

**Perceptron Convergence Theorem**: If the training data is **linearly separable**, the perceptron learning algorithm is guaranteed to converge to a solution in a finite number of steps.

However, if the data is **not linearly separable**, the algorithm will never converge and will oscillate indefinitely.

## Limitations

### 1. Linear Separability Requirement

The perceptron can only solve **linearly separable** problems. It cannot learn non-linear decision boundaries.

**Classic Example: XOR Problem**

The XOR (exclusive OR) function cannot be learned by a single perceptron:

| $x_1$ | $x_2$ | XOR |
|-------|-------|-----|
| 0     | 0     | 0   |
| 0     | 1     | 1   |
| 1     | 0     | 1   |
| 1     | 1     | 0   |

No single line can separate the two classes in this 2D space.

### 2. Binary Classification Only

The basic perceptron only performs binary classification. Multi-class problems require extensions or multiple perceptrons.

### 3. No Probabilistic Output

The step function produces only binary outputs (0 or 1), not probability estimates.

### Solutions to Limitations

- **Multi-layer perceptrons (MLPs)**: Stack multiple layers to learn non-linear relationships
- **Kernel methods**: Map data to higher dimensions where it becomes linearly separable
- **Different activation functions**: Replace step function with smooth, differentiable functions

## Practical Implementation

### Python Implementation from Scratch

```python
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the perceptron

        Parameters:
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,) - binary labels (0 or 1)
        """
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training loop
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # Compute linear output
                linear_output = np.dot(x_i, self.weights) + self.bias

                # Apply step function
                y_predicted = self._step_function(linear_output)

                # Update weights and bias
                error = y[idx] - y_predicted
                self.weights += self.lr * error * x_i
                self.bias += self.lr * error

    def predict(self, X):
        """Predict class labels for samples in X"""
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._step_function(linear_output)
        return y_predicted

    def _step_function(self, x):
        """Step activation function"""
        return np.where(x >= 0, 1, 0)
```

### Usage Example

```python
# Create linearly separable dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    flip_y=0,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train perceptron
perceptron = Perceptron(learning_rate=0.1, n_iterations=100)
perceptron.fit(X_train, y_train)

# Make predictions
predictions = perceptron.predict(X_test)

# Calculate accuracy
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.2f}")
```

### Using Scikit-learn

```python
from sklearn.linear_model import Perceptron as SklearnPerceptron
from sklearn.metrics import accuracy_score

# Create and train perceptron
clf = SklearnPerceptron(max_iter=100, eta0=0.1, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Weights: {clf.coef_}")
print(f"Bias: {clf.intercept_}")
```

## Comparison with Logistic Regression

| Aspect | Perceptron | Logistic Regression |
|--------|-----------|---------------------|
| Activation | Step function | Sigmoid function |
| Output | Binary (0 or 1) | Probability [0, 1] |
| Loss function | Misclassification count | Log loss |
| Convergence | Only if linearly separable | Always converges |
| Optimization | Perceptron algorithm | Gradient descent |
| Differentiable | No | Yes |

## Historical Significance

The perceptron sparked both optimism and controversy:

- **1958**: Rosenblatt's perceptron shows promise for pattern recognition
- **1969**: Minsky and Papert's book "Perceptrons" proves limitations (XOR problem)
- **AI Winter**: Funding for neural network research decreased
- **1980s**: Multi-layer perceptrons and backpropagation revived interest
- **Modern era**: Perceptrons remain relevant as building blocks of deep networks

## References

- Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"
- Minsky, M., & Papert, S. (1969). "Perceptrons: An Introduction to Computational Geometry"
- Bishop, C. (2006). "Pattern Recognition and Machine Learning"
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning"
