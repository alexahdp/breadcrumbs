# Linear Algebra

## Table of Contents

1. [Vectors](#vectors)
2. [Matrices](#matrices)
3. [Matrix Operations](#matrix-operations)
4. [Special Matrices](#special-matrices)
5. [Matrix Properties](#matrix-properties)
6. [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
7. [Matrix Decomposition](#matrix-decomposition)
8. [Applications in Machine Learning](#applications-in-machine-learning)

## Vectors

### Vector Basics

A **vector** is an ordered array of numbers, representing a point or direction in space.

**Column vector**:
$$\vec{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$

**Row vector**:
$$\vec{v}^T = \begin{bmatrix} v_1 & v_2 & \cdots & v_n \end{bmatrix}$$

### Vector Operations

**Vector Addition**:
$$\vec{a} + \vec{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \\ a_n + b_n \end{bmatrix}$$

**Scalar Multiplication**:
$$c\vec{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{bmatrix}$$

**Vector Magnitude (L2 Norm)**:
$$||\vec{v}|| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \sqrt{\vec{v}^T\vec{v}}$$

**Unit Vector** - a vector with magnitude equal to 1:
$$\hat{v} = \frac{\vec{v}}{||\vec{v}||}$$

### Dot Product (Scalar Product)

**Dot Product** - an operation on two vectors resulting in a scalar, characterizing the lengths of the vectors and the angle between them:

$$\vec{a} \cdot \vec{b} = \sum_{i=1}^{n} a_i b_i = ||\vec{a}|| \cdot ||\vec{b}|| \cdot \cos(\theta)$$

where $\theta$ is the angle between vectors.

**Properties:**
- If $\vec{a} \cdot \vec{b} = 0$, vectors are orthogonal (perpendicular)
- If $\vec{a} \cdot \vec{b} > 0$, angle is acute
- If $\vec{a} \cdot \vec{b} < 0$, angle is obtuse
- Dot product is commutative: $\vec{a} \cdot \vec{b} = \vec{b} \cdot \vec{a}$

**Geometric interpretation:** The dot product measures how much two vectors point in the same direction.

### Cross Product (3D Only)

For vectors in $\mathbb{R}^3$:

$$\vec{a} \times \vec{b} = \begin{bmatrix} a_2b_3 - a_3b_2 \\ a_3b_1 - a_1b_3 \\ a_1b_2 - a_2b_1 \end{bmatrix}$$

Properties:
- Result is perpendicular to both vectors
- Magnitude equals area of parallelogram formed by vectors
- Anti-commutative: $\vec{a} \times \vec{b} = -\vec{b} \times \vec{a}$

## Matrices

A **matrix** is a rectangular array of numbers arranged in rows and columns.

**General form** ($m \times n$ matrix):
$$A = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}$$

- $m$ rows, $n$ columns
- Element at row $i$, column $j$: $a_{ij}$

### Geometric Interpretation

Matrices represent **linear transformations** from one vector space to another. When you multiply a matrix by a vector, you transform that vector.

## Matrix Operations

### Matrix Addition and Subtraction

Only defined for matrices of the same dimensions:

$$A + B = \begin{bmatrix} a_{11}+b_{11} & \cdots \\ \vdots & \ddots \end{bmatrix}$$

### Scalar Multiplication

$$cA = \begin{bmatrix} ca_{11} & ca_{12} & \cdots \\ ca_{21} & ca_{22} & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}$$

### Matrix Multiplication

For $A$ ($m \times n$) and $B$ ($n \times p$), the product $C = AB$ is $m \times p$:

$$c_{ij} = \sum_{k=1}^{n} a_{ik}b_{kj}$$

**Important properties:**
- Not commutative: $AB \neq BA$ (in general)
- Associative: $(AB)C = A(BC)$
- Distributive: $A(B+C) = AB + AC$
- Number of columns in $A$ must equal number of rows in $B$

**Geometric interpretation:** Matrix multiplication represents composition of linear transformations.

### Transpose

**Transpose** - a matrix obtained from the original by swapping rows and columns:

$$A^T_{ij} = A_{ji}$$

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \quad \Rightarrow \quad A^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$$

**Properties:**
- $(A^T)^T = A$
- $(A + B)^T = A^T + B^T$
- $(AB)^T = B^T A^T$ (note the reversal!)
- $(cA)^T = cA^T$

## Special Matrices

### Identity Matrix (I)

**Identity Matrix** - a square matrix with ones on the main diagonal and zeros elsewhere:

$$I_n = \begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}$$

**Properties:**
- $AI = IA = A$ (multiplicative identity)
- Represents the identity transformation (no change)

### Diagonal Matrix (D)

**Diagonal Matrix** - a square matrix where all off-diagonal elements are zero:

$$D = \begin{bmatrix}
d_1 & 0 & \cdots & 0 \\
0 & d_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & d_n
\end{bmatrix}$$

**Geometric interpretation:** Represents scaling along coordinate axes.

### Symmetric Matrix

**Symmetric Matrix** - a square matrix where elements are symmetric about the main diagonal:

$$A = A^T \quad \Leftrightarrow \quad a_{ij} = a_{ji}$$

**Properties:**
- All eigenvalues are real numbers
- Eigenvectors corresponding to different eigenvalues are orthogonal
- Can be diagonalized by orthogonal matrix

### Orthogonal Matrix (Q)

**Orthogonal Matrix** - a square matrix $A$ with real elements such that:

$$A^T A = AA^T = I \quad \Rightarrow \quad A^T = A^{-1}$$

**Properties:**
- Columns (and rows) are orthonormal vectors
- Preserves vector lengths: $||A\vec{x}|| = ||\vec{x}||$
- Preserves dot products: $(A\vec{x}) \cdot (A\vec{y}) = \vec{x} \cdot \vec{y}$
- Represents rotation and/or reflection
- Determinant is ±1

## Matrix Properties

### Determinant

The **determinant** is a scalar value that can be computed from a square matrix. Geometrically, it represents the signed volume of the parallelepiped formed by the matrix's column vectors.

**For 2×2 matrix:**
$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

**For 3×3 matrix:**
$$\det\begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix} = a(ei-fh) - b(di-fg) + c(dh-eg)$$

**Properties:**
- $\det(A^T) = \det(A)$
- $\det(AB) = \det(A) \cdot \det(B)$
- $\det(cA) = c^n \det(A)$ for $n \times n$ matrix
- If $\det(A) = 0$, matrix is singular (not invertible)
- If matrix has linearly dependent rows/columns, $\det(A) = 0$
- $\det(A^{-1}) = \frac{1}{\det(A)}$

**Geometric interpretation:**
- $|\det(A)|$ = volume scaling factor of the transformation
- Sign indicates orientation preservation (positive) or reversal (negative)

### Rank

**Rank** - the maximum number of linearly independent rows or columns in a matrix.

$$\text{rank}(A) = \text{number of linearly independent rows} = \text{number of linearly independent columns}$$

**Properties:**
- $\text{rank}(A) \leq \min(m, n)$ for $m \times n$ matrix
- Full rank: $\text{rank}(A) = \min(m, n)$
- Rank-deficient: $\text{rank}(A) < \min(m, n)$
- $\text{rank}(A) = \text{rank}(A^T)$
- If $A$ is square and full rank, then $A$ is invertible

### Trace

**Trace** - the sum of diagonal elements:

$$\text{tr}(A) = \sum_{i=1}^{n} a_{ii}$$

**Properties:**
- $\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B)$
- $\text{tr}(cA) = c \cdot \text{tr}(A)$
- $\text{tr}(AB) = \text{tr}(BA)$ (cyclic property)
- $\text{tr}(A) = \sum_{i=1}^{n} \lambda_i$ (sum of eigenvalues)

### Inverse Matrix

**Inverse Matrix** ($A^{-1}$) - a matrix such that when multiplied by the original matrix $A$, gives the identity matrix:

$$AA^{-1} = A^{-1}A = I$$

**Conditions for invertibility:**
- Matrix must be square
- Determinant must be non-zero: $\det(A) \neq 0$
- Matrix must have full rank

**Properties:**
- $(A^{-1})^{-1} = A$
- $(AB)^{-1} = B^{-1}A^{-1}$ (note the reversal!)
- $(A^T)^{-1} = (A^{-1})^T$
- $\det(A^{-1}) = \frac{1}{\det(A)}$

**For 2×2 matrix:**
$$A = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \quad \Rightarrow \quad A^{-1} = \frac{1}{ad-bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

## Eigenvalues and Eigenvectors

### Definitions

**Eigenvector** - a non-zero vector $\vec{v}$ that, when a linear transformation $A$ is applied, only changes by a scalar factor:

$$A\vec{v} = \lambda\vec{v}$$

**Eigenvalue** ($\lambda$) - the scalar factor by which the eigenvector is scaled.

**Geometric interpretation:** Eigenvectors are special directions that are only stretched or compressed (not rotated) by the transformation.

### Finding Eigenvalues and Eigenvectors

To find eigenvalues, solve the **characteristic equation**:

$$\det(A - \lambda I) = 0$$

Once eigenvalues are found, find eigenvectors by solving:

$$(A - \lambda I)\vec{v} = \vec{0}$$

**Example** (2×2 matrix):
$$A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$$

Characteristic equation:
$$\det\begin{bmatrix} 4-\lambda & 1 \\ 2 & 3-\lambda \end{bmatrix} = (4-\lambda)(3-\lambda) - 2 = 0$$

$$\lambda^2 - 7\lambda + 10 = 0 \quad \Rightarrow \quad \lambda_1 = 5, \lambda_2 = 2$$

### Spectrum

**Spectrum** - the set of all eigenvalues and their corresponding eigenvectors.

**Properties:**
- Sum of eigenvalues = trace of matrix
- Product of eigenvalues = determinant of matrix
- Eigenvectors corresponding to different eigenvalues are linearly independent

## Matrix Decomposition

### Eigendecomposition

For a square matrix $A$ with $n$ linearly independent eigenvectors:

$$A = Q\Lambda Q^{-1}$$

where:
- $Q$ = matrix whose columns are eigenvectors
- $\Lambda$ = diagonal matrix with eigenvalues on diagonal

**Only works when:**
- Matrix is square
- Matrix has $n$ linearly independent eigenvectors

**Symmetric matrices** always have orthogonal eigenvectors, so:
$$A = Q\Lambda Q^T \quad \text{where } Q^TQ = I$$

### Singular Value Decomposition (SVD)

**SVD** works for any $m \times n$ matrix:

$$A = U\Sigma V^T$$

where:
- $U$ ($m \times m$): left singular vectors (orthogonal)
- $\Sigma$ ($m \times n$): diagonal matrix of singular values
- $V$ ($n \times n$): right singular vectors (orthogonal)

**Properties:**
- Always exists for any matrix
- Singular values $\sigma_i \geq 0$, ordered: $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$
- Rank of $A$ = number of non-zero singular values

**Applications:**
- Dimensionality reduction (PCA)
- Image compression
- Recommender systems
- Pseudoinverse computation

### LU Decomposition

Decomposes matrix into lower and upper triangular matrices:

$$A = LU$$

where:
- $L$: lower triangular matrix
- $U$: upper triangular matrix

**Used for:**
- Solving systems of linear equations
- Computing determinants
- Matrix inversion

### Cholesky Decomposition

For symmetric positive-definite matrix:

$$A = LL^T$$

where $L$ is lower triangular.

**Used for:**
- Efficient solving of systems
- Generating correlated random variables
- Monte Carlo simulations

## Applications in Machine Learning

### Linear Regression

Solving $\vec{y} = X\vec{w}$ for weights $\vec{w}$:

$$\vec{w} = (X^TX)^{-1}X^T\vec{y}$$

This involves:
- Matrix transpose
- Matrix multiplication
- Matrix inversion

### Neural Networks

**Forward pass** (layer computation):
$$\vec{h} = f(W\vec{x} + \vec{b})$$

- $W$: weight matrix
- $\vec{x}$: input vector
- $\vec{b}$: bias vector
- $f$: activation function

Each layer is a matrix multiplication followed by non-linear activation.

### Principal Component Analysis (PCA)

1. Compute covariance matrix: $C = \frac{1}{n}X^TX$
2. Find eigenvalues and eigenvectors of $C$
3. Principal components = eigenvectors with largest eigenvalues
4. Project data: $X_{reduced} = XQ_k$ where $Q_k$ contains top $k$ eigenvectors

### Support Vector Machines

The kernel trick often involves matrix operations:
$$K_{ij} = k(\vec{x}_i, \vec{x}_j)$$

where $K$ is the kernel matrix (Gram matrix).

### Recommendation Systems

Matrix factorization decomposes user-item matrix:
$$R \approx UV^T$$

where:
- $R$: ratings matrix
- $U$: user feature matrix
- $V$: item feature matrix

## Computational Considerations

### Numerical Stability

**Condition Number** - measures sensitivity of matrix operations to numerical errors:

$$\kappa(A) = ||A|| \cdot ||A^{-1}||$$

- $\kappa(A) \approx 1$: well-conditioned
- $\kappa(A) >> 1$: ill-conditioned (avoid inversion)

**Ill-conditioned matrices** arise from:
- Highly correlated features
- Features with very different scales
- Nearly linearly dependent rows/columns

**Solution:** Use regularization (Ridge regression) or SVD instead of direct inversion.

### Computational Complexity

Operation complexities for $n \times n$ matrices:
- Matrix addition: $O(n^2)$
- Matrix multiplication: $O(n^3)$ (naive), $O(n^{2.37})$ (Strassen)
- Matrix inversion: $O(n^3)$
- Eigendecomposition: $O(n^3)$
- SVD: $O(mn^2)$ for $m \times n$ matrix

**Practical implications:**
- Avoid matrix inversion when possible
- Use sparse matrices when applicable
- Consider iterative methods for large systems

## Summary

Linear algebra concepts essential for ML:

1. **Vectors and matrices** - fundamental data structures
2. **Matrix multiplication** - core operation in neural networks
3. **Transpose and inverse** - used in optimization and regression
4. **Eigendecomposition** - PCA, spectral methods
5. **SVD** - dimensionality reduction, matrix completion
6. **Norms and distances** - regularization, similarity measures

## References

- Linear Algebra and Its Applications (Gilbert Strang)
- Introduction to Linear Algebra for Applied Machine Learning (Pablo Caceres)
- Mathematics for Machine Learning (Deisenroth, Faisal, Ong)
- [MIT OpenCourseWare: Linear Algebra](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/)
