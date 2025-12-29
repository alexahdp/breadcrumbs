# Neural Networks

Neural networks are computational models inspired by biological neural networks in animal brains. They consist of interconnected nodes (neurons) organized in layers that can learn to perform tasks by processing examples.

## Overview

Neural networks revolutionized machine learning by enabling the automatic discovery of features and representations from raw data. Unlike classical machine learning methods that require manual feature engineering, neural networks can learn hierarchical representations at multiple levels of abstraction.

## Contents

### Fundamentals

Essential building blocks and concepts for understanding neural networks:

1. **[Perceptron](./fundamentals/perceptron.md)** - The simplest neural network unit, forming the foundation for more complex architectures
2. **[Multi-Layer Perceptron (MLP)](./fundamentals/mlp.md)** - Feedforward networks with multiple layers capable of learning non-linear relationships
3. **[Activation Functions](./fundamentals/activation-functions.md)** - Non-linear functions that enable neural networks to learn complex patterns
4. **[Backpropagation](./fundamentals/backpropagation.md)** - The algorithm for training neural networks through gradient descent
5. **[Optimizers](./fundamentals/optimizers.md)** - Algorithms that adjust network weights to minimize loss

### Architectures

(To be added in future sections)

### Training Techniques

(To be added in future sections)

## Key Concepts

**Universal Approximation Theorem**: A neural network with at least one hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of $\mathbb{R}^n$, under mild assumptions on the activation function.

**Deep Learning**: The use of neural networks with multiple hidden layers, enabling the learning of hierarchical feature representations.

**Forward Pass**: The process of computing the output of a neural network by passing input data through all layers.

**Backward Pass**: The process of computing gradients of the loss function with respect to network parameters using the chain rule.

## Historical Context

- **1943**: McCulloch-Pitts neuron - first mathematical model of a neuron
- **1958**: Rosenblatt's perceptron - first trainable neural network
- **1986**: Backpropagation popularized by Rumelhart, Hinton, and Williams
- **2006**: Deep learning renaissance begins with deep belief networks
- **2012**: AlexNet wins ImageNet, marking the beginning of the deep learning era

## When to Use Neural Networks

Neural networks excel at:
- **Image recognition and computer vision**
- **Natural language processing**
- **Speech recognition**
- **Time series prediction**
- **Complex pattern recognition** in high-dimensional data

Consider classical ML methods when:
- Dataset is small (< 1000 samples)
- Interpretability is crucial
- Computational resources are limited
- Problem is well-suited to specific algorithms (e.g., linear relationships)

## Prerequisites

Before diving into neural networks, ensure familiarity with:
- Linear algebra (matrix operations, vectors)
- Calculus (derivatives, gradients, chain rule)
- Probability and statistics
- Basic programming and numerical computing

## References

- Deep Learning (Goodfellow, Bengio, Courville)
- Neural Networks and Deep Learning (Nielsen)
- Pattern Recognition and Machine Learning (Bishop)
- Dive into Deep Learning (Zhang et al.)
