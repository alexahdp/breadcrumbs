# Calculus Basics

Calculus is the mathematical foundation for training machine learning models. Understanding derivatives and gradients is essential for optimization algorithms like gradient descent.

## Table of Contents

1. [Derivatives](#derivatives)
2. [Partial Derivatives](#partial-derivatives)
3. [Gradients](#gradients)
4. [Chain Rule](#chain-rule)
5. [Common Derivatives](#common-derivatives)
6. [Optimization](#optimization)
7. [Taylor Series](#taylor-series)
8. [Applications in Machine Learning](#applications-in-machine-learning)

## Derivatives

### Definition

The **derivative** measures the rate of change of a function with respect to its input:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Geometric interpretation:** The slope of the tangent line to the function at point $x$.

**Physical interpretation:** Instantaneous rate of change.

### Notation

Multiple notations for derivatives:
- $f'(x)$ - Lagrange notation
- $\frac{df}{dx}$ - Leibniz notation
- $\frac{d}{dx}f(x)$ - operator notation
- $D_x f$ - differential operator
- $\dot{f}$ - Newton notation (for time derivatives)

### Derivative Rules

**Constant rule:**
$$\frac{d}{dx}(c) = 0$$

**Power rule:**
$$\frac{d}{dx}(x^n) = nx^{n-1}$$

**Constant multiple rule:**
$$\frac{d}{dx}(cf(x)) = c \cdot f'(x)$$

**Sum rule:**
$$\frac{d}{dx}(f(x) + g(x)) = f'(x) + g'(x)$$

**Product rule:**
$$\frac{d}{dx}(f(x)g(x)) = f'(x)g(x) + f(x)g'(x)$$

**Quotient rule:**
$$\frac{d}{dx}\left(\frac{f(x)}{g(x)}\right) = \frac{f'(x)g(x) - f(x)g'(x)}{g(x)^2}$$

**Chain rule:**
$$\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$$

## Partial Derivatives

When a function has multiple variables, we use **partial derivatives** to measure the rate of change with respect to one variable while keeping others constant.

**Definition:**
$$\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h, y) - f(x, y)}{h}$$

**Example:**
$$f(x, y) = x^2y + 3xy^2$$

$$\frac{\partial f}{\partial x} = 2xy + 3y^2$$

$$\frac{\partial f}{\partial y} = x^2 + 6xy$$

### Notation

For multivariable function $f(x_1, x_2, \ldots, x_n)$:
- $\frac{\partial f}{\partial x_i}$ - partial derivative with respect to $x_i$
- $\partial_i f$ - shorthand notation
- $f_{x_i}$ - subscript notation

## Gradients

The **gradient** is a vector of all partial derivatives of a scalar function:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

**Alternative notation:** $\nabla f = \text{grad } f$

**Properties:**
- Points in direction of steepest ascent
- Magnitude indicates steepness
- Perpendicular to level curves/surfaces

**Example:**
$$f(x, y) = x^2 + xy + y^2$$

$$\nabla f = \begin{bmatrix} 2x + y \\ x + 2y \end{bmatrix}$$

At point $(1, 2)$:
$$\nabla f(1,2) = \begin{bmatrix} 4 \\ 5 \end{bmatrix}$$

### Directional Derivative

The rate of change of $f$ in direction $\vec{v}$:

$$D_{\vec{v}}f = \nabla f \cdot \hat{v}$$

where $\hat{v}$ is the unit vector in direction $\vec{v}$.

**Interpretation:** How fast is $f$ changing if we move in direction $\vec{v}$?

### Gradient Descent

**Gradient descent** moves in the opposite direction of the gradient to find minima:

$$x_{new} = x_{old} - \alpha \nabla f(x_{old})$$

where $\alpha$ is the **learning rate**.

**Intuition:**
- Gradient points uphill (steepest ascent)
- Negative gradient points downhill (steepest descent)
- We step in the downhill direction to minimize the function

## Chain Rule

### Single Variable Chain Rule

For composition $f(g(x))$:

$$\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$$

**Example:**
$$h(x) = \sin(x^2)$$

Let $f(u) = \sin(u)$ and $g(x) = x^2$:

$$h'(x) = \cos(x^2) \cdot 2x = 2x\cos(x^2)$$

### Multivariable Chain Rule

For $z = f(x, y)$ where $x = g(t)$ and $y = h(t)$:

$$\frac{dz}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt}$$

**Example:**
$$z = x^2y, \quad x = t^2, \quad y = t^3$$

$$\frac{dz}{dt} = 2xy \cdot 2t + x^2 \cdot 3t^2 = 2(t^2)(t^3)(2t) + (t^2)^2(3t^2) = 4t^6 + 3t^6 = 7t^6$$

### Vector Chain Rule (Backpropagation)

For $\vec{y} = f(\vec{x})$ and $z = g(\vec{y})$:

$$\frac{\partial z}{\partial x_i} = \sum_j \frac{\partial z}{\partial y_j} \frac{\partial y_j}{\partial x_i}$$

In matrix form:
$$\frac{\partial z}{\partial \vec{x}} = \frac{\partial z}{\partial \vec{y}} \cdot \frac{\partial \vec{y}}{\partial \vec{x}}$$

**This is the foundation of backpropagation in neural networks!**

## Common Derivatives

### Elementary Functions

**Exponential:**
$$\frac{d}{dx}(e^x) = e^x$$

$$\frac{d}{dx}(a^x) = a^x \ln(a)$$

**Logarithmic:**
$$\frac{d}{dx}(\ln x) = \frac{1}{x}$$

$$\frac{d}{dx}(\log_a x) = \frac{1}{x \ln a}$$

**Trigonometric:**
$$\frac{d}{dx}(\sin x) = \cos x$$

$$\frac{d}{dx}(\cos x) = -\sin x$$

$$\frac{d}{dx}(\tan x) = \sec^2 x = \frac{1}{\cos^2 x}$$

**Inverse trigonometric:**
$$\frac{d}{dx}(\arctan x) = \frac{1}{1+x^2}$$

### Activation Functions (ML Specific)

**Sigmoid:**
$$\sigma(x) = \frac{1}{1+e^{-x}}$$

$$\sigma'(x) = \sigma(x)(1-\sigma(x))$$

**Hyperbolic tangent:**
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

$$\tanh'(x) = 1 - \tanh^2(x)$$

**ReLU (Rectified Linear Unit):**
$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \\ \text{undefined} & \text{if } x = 0 \end{cases}$$

In practice, we use 0 at $x=0$.

**Leaky ReLU:**
$$\text{LeakyReLU}(x) = \max(\alpha x, x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$$

$$\text{LeakyReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha & \text{if } x \leq 0 \end{cases}$$

where $\alpha$ is a small constant (e.g., 0.01).

**Softmax:**
$$\text{softmax}(\vec{x})_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

$$\frac{\partial \text{softmax}_i}{\partial x_j} = \begin{cases} \text{softmax}_i(1-\text{softmax}_i) & \text{if } i = j \\ -\text{softmax}_i \cdot \text{softmax}_j & \text{if } i \neq j \end{cases}$$

## Optimization

### Critical Points

**Critical point** - where the derivative is zero or undefined:
$$f'(x) = 0 \quad \text{or} \quad f'(x) \text{ undefined}$$

**For multivariable functions:**
$$\nabla f = \vec{0}$$

### Second Derivative Test

**For single variable:**
- If $f''(x) > 0$: local minimum
- If $f''(x) < 0$: local maximum
- If $f''(x) = 0$: inconclusive (could be inflection point)

**For multivariable functions**, use the **Hessian matrix**:

$$H = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}$$

**Test:**
- If $H$ is positive definite (all eigenvalues > 0): local minimum
- If $H$ is negative definite (all eigenvalues < 0): local maximum
- If $H$ has mixed eigenvalues: saddle point

### Convexity

**Convex function** - the line segment between any two points on the graph lies above the graph:

$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

for all $x, y$ and $\lambda \in [0,1]$.

**Test for convexity:**
- Single variable: $f''(x) \geq 0$ for all $x$
- Multivariable: Hessian is positive semidefinite

**Why it matters:** Convex functions have a unique global minimum, making optimization easier.

**Examples:**
- Convex: $x^2$, $e^x$, $-\log x$ (for $x > 0$)
- Non-convex: $\sin x$, $x^3$, $x^4 - x^2$

## Taylor Series

**Taylor series** approximates a function as an infinite sum:

$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + \cdots$$

**First-order (linear) approximation:**
$$f(x) \approx f(a) + f'(a)(x-a)$$

**Second-order (quadratic) approximation:**
$$f(x) \approx f(a) + f'(a)(x-a) + \frac{1}{2}f''(a)(x-a)^2$$

### Multivariable Taylor Series

Around point $\vec{a}$:

$$f(\vec{x}) \approx f(\vec{a}) + \nabla f(\vec{a})^T(\vec{x}-\vec{a}) + \frac{1}{2}(\vec{x}-\vec{a})^T H(\vec{a})(\vec{x}-\vec{a})$$

where $H$ is the Hessian matrix.

**Applications:**
- Newton's method for optimization
- Understanding behavior near critical points
- Approximating complex functions

## Applications in Machine Learning

### Loss Function Optimization

**Goal:** Minimize loss function $L(\vec{w})$ with respect to weights $\vec{w}$.

**Gradient descent update:**
$$\vec{w}_{t+1} = \vec{w}_t - \alpha \nabla L(\vec{w}_t)$$

where:
- $\alpha$ is the learning rate
- $\nabla L$ is the gradient of the loss

### Backpropagation

**Backpropagation** is repeated application of the chain rule to compute gradients in neural networks.

**Forward pass:** Compute output
$$\vec{y} = f_L(f_{L-1}(\cdots f_2(f_1(\vec{x}))))$$

**Backward pass:** Compute gradients using chain rule
$$\frac{\partial L}{\partial \vec{w}_i} = \frac{\partial L}{\partial \vec{y}} \cdot \frac{\partial \vec{y}}{\partial \vec{h}_L} \cdot \frac{\partial \vec{h}_L}{\partial \vec{h}_{L-1}} \cdots \frac{\partial \vec{h}_{i+1}}{\partial \vec{w}_i}$$

### Example: Single Neuron

**Forward pass:**
$$z = \vec{w}^T\vec{x} + b$$
$$\hat{y} = \sigma(z)$$
$$L = \frac{1}{2}(\hat{y} - y)^2$$

**Backward pass:**
$$\frac{\partial L}{\partial \hat{y}} = \hat{y} - y$$

$$\frac{\partial \hat{y}}{\partial z} = \sigma'(z) = \sigma(z)(1-\sigma(z))$$

$$\frac{\partial z}{\partial \vec{w}} = \vec{x}$$

$$\frac{\partial z}{\partial b} = 1$$

**Chain rule:**
$$\frac{\partial L}{\partial \vec{w}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial \vec{w}} = (\hat{y} - y) \cdot \sigma(z)(1-\sigma(z)) \cdot \vec{x}$$

$$\frac{\partial L}{\partial b} = (\hat{y} - y) \cdot \sigma(z)(1-\sigma(z))$$

### Common Loss Functions and Their Derivatives

**Mean Squared Error (MSE):**
$$L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

$$\frac{\partial L}{\partial \hat{y}_i} = -\frac{2}{n}(y_i - \hat{y}_i)$$

**Binary Cross-Entropy:**
$$L = -\frac{1}{n}\sum_{i=1}^{n}[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

$$\frac{\partial L}{\partial \hat{y}_i} = -\frac{1}{n}\left[\frac{y_i}{\hat{y}_i} - \frac{1-y_i}{1-\hat{y}_i}\right]$$

**Categorical Cross-Entropy (with softmax):**
$$L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

where $\hat{y} = \text{softmax}(z)$.

Combined derivative (softmax + cross-entropy):
$$\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$$

This elegant result is why softmax and cross-entropy are commonly used together!

### Regularization

**L2 Regularization (Ridge):**
$$L_{total} = L_{data} + \lambda \sum_{i} w_i^2$$

$$\frac{\partial L_{total}}{\partial w_i} = \frac{\partial L_{data}}{\partial w_i} + 2\lambda w_i$$

**L1 Regularization (Lasso):**
$$L_{total} = L_{data} + \lambda \sum_{i} |w_i|$$

$$\frac{\partial L_{total}}{\partial w_i} = \frac{\partial L_{data}}{\partial w_i} + \lambda \cdot \text{sign}(w_i)$$

### Newton's Method

**Second-order optimization** using both gradient and Hessian:

$$\vec{w}_{t+1} = \vec{w}_t - H^{-1}\nabla L(\vec{w}_t)$$

where $H$ is the Hessian matrix.

**Advantages:**
- Faster convergence than gradient descent
- Uses curvature information

**Disadvantages:**
- Computing Hessian is expensive: $O(n^2)$ space, $O(n^3)$ time for inversion
- Impractical for large neural networks

## Practical Considerations

### Numerical Gradients

**Finite difference approximation:**
$$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$$

for small $h$ (e.g., $10^{-5}$).

**Use for:**
- Gradient checking (verify backpropagation implementation)
- Debugging
- Functions where analytical gradient is hard to derive

**Don't use for:**
- Training (too slow)
- Production code

### Vanishing and Exploding Gradients

**Vanishing gradients:** Gradients become very small, slowing learning.
- Common with sigmoid/tanh in deep networks
- Solution: ReLU, batch normalization, skip connections

**Exploding gradients:** Gradients become very large, causing instability.
- Common in RNNs
- Solution: gradient clipping, careful initialization

### Computational Graphs

Modern deep learning frameworks (PyTorch, TensorFlow) use **computational graphs**:
1. Build graph during forward pass
2. Automatically compute gradients using chain rule (autograd)
3. No manual derivative calculation needed!

## Summary

Key calculus concepts for machine learning:

1. **Derivatives** - rate of change
2. **Partial derivatives** - multivariable functions
3. **Gradients** - direction of steepest ascent
4. **Chain rule** - backpropagation foundation
5. **Optimization** - finding minima/maxima
6. **Taylor series** - function approximation

## References

- Calculus (James Stewart)
- Mathematics for Machine Learning (Deisenroth, Faisal, Ong)
- Deep Learning (Goodfellow, Bengio, Courville) - Chapter 4
- [3Blue1Brown: Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
