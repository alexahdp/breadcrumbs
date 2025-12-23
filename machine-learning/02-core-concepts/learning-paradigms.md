# Learning Paradigms

## Table of Contents

1. [Overview](#overview)
2. [Supervised Learning](#supervised-learning)
3. [Unsupervised Learning](#unsupervised-learning)
4. [Semi-Supervised Learning](#semi-supervised-learning)
5. [Reinforcement Learning](#reinforcement-learning)
6. [Other Paradigms](#other-paradigms)
7. [Choosing the Right Paradigm](#choosing-the-right-paradigm)

## Overview

Learning paradigms define how machine learning models interact with data and learn patterns. The choice of paradigm depends on:
- The type of data available (labeled vs unlabeled)
- The nature of the problem (prediction, clustering, decision-making)
- Available resources (cost of labeling, computational requirements)

## Supervised Learning

**Supervised Learning** is a learning paradigm where the model learns from labeled data - each training example consists of an input and its corresponding correct output (label or target).

### Mathematical Formulation

Given a training dataset:

$$D = \{(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)\}$$

where:
- $x_i \in \mathbb{R}^d$ is the input feature vector
- $y_i$ is the target label
- Goal: learn a function $f: X \rightarrow Y$ that maps inputs to outputs

The learning objective is to minimize the loss function:

$$L(f) = \frac{1}{n}\sum_{i=1}^{n}\ell(f(x_i), y_i)$$

where $\ell$ is a loss function measuring the difference between prediction and true value.

### Types of Supervised Learning

**Classification** - predicting discrete categories:
- Binary classification: $y \in \{0, 1\}$ (spam detection, disease diagnosis)
- Multiclass classification: $y \in \{1, 2, ..., K\}$ (digit recognition, species classification)
- Multilabel classification: $y \in \{0, 1\}^K$ (image tagging, document categorization)

**Regression** - predicting continuous values:
- $y \in \mathbb{R}$ (price prediction, temperature forecasting, age estimation)

### Common Algorithms

- Linear Regression, Logistic Regression
- Support Vector Machines (SVM)
- Decision Trees, Random Forests
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Neural Networks

### Advantages and Limitations

**Advantages:**
- Clear evaluation metrics
- Direct optimization toward target
- Well-established theory and methods
- Strong performance when sufficient labeled data available

**Limitations:**
- Requires labeled data (expensive and time-consuming)
- May not generalize to different distributions
- Limited to predefined categories
- Cannot discover hidden patterns in data structure

### Practical Example

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Supervised learning workflow
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model with labeled data
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

## Unsupervised Learning

**Unsupervised Learning** is a learning paradigm where the model learns patterns from unlabeled data - only inputs are provided, without corresponding outputs.

### Mathematical Formulation

Given a dataset:

$$D = \{x_1, x_2, ..., x_n\}$$

where:
- $x_i \in \mathbb{R}^d$ is the input feature vector
- No labels $y_i$ are provided
- Goal: discover hidden structure or patterns in the data

### Common Tasks

**Clustering** - grouping similar data points:

Objective: partition data into $K$ clusters $C_1, C_2, ..., C_K$ to minimize within-cluster variance:

$$\min \sum_{k=1}^{K}\sum_{x_i \in C_k}||x_i - \mu_k||^2$$

where $\mu_k$ is the centroid of cluster $k$.

**Dimensionality Reduction** - reducing the number of features while preserving information:

Principal Component Analysis (PCA) finds directions of maximum variance:

$$\max_w \frac{w^T\Sigma w}{w^Tw}$$

where $\Sigma$ is the covariance matrix.

**Density Estimation** - learning the probability distribution of data:

$$p(x) = \frac{1}{n}\sum_{i=1}^{n}K(x - x_i)$$

where $K$ is a kernel function.

**Anomaly Detection** - identifying unusual patterns that don't conform to expected behavior.

### Common Algorithms

- **Clustering**: K-Means, DBSCAN, Hierarchical Clustering, Gaussian Mixture Models
- **Dimensionality Reduction**: PCA, t-SNE, UMAP, Autoencoders
- **Association**: Apriori, FP-Growth
- **Density Estimation**: Kernel Density Estimation, Gaussian Mixture Models

### Advantages and Limitations

**Advantages:**
- No need for labeled data
- Can discover hidden patterns
- Useful for exploratory data analysis
- Can work with large amounts of data

**Limitations:**
- Difficult to evaluate objectively
- Results may be hard to interpret
- No clear ground truth
- Requires domain knowledge to validate findings

### Practical Example

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Clustering example
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Dimensionality reduction for visualization
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Visualize clusters in 2D
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='viridis')
plt.title('Clusters in 2D PCA space')
```

## Semi-Supervised Learning

**Semi-Supervised Learning** combines a small amount of labeled data with a large amount of unlabeled data during training.

### Motivation

- Labeled data is expensive to obtain (requires expert annotation)
- Unlabeled data is abundant and cheap
- Unlabeled data can help learn better representations

### Mathematical Formulation

Given:
- Labeled data: $D_L = \{(x_1, y_1), ..., (x_l, y_l)\}$
- Unlabeled data: $D_U = \{x_{l+1}, ..., x_{l+u}\}$
- Typically $u \gg l$ (much more unlabeled data)

Objective function combines supervised and unsupervised losses:

$$L = \sum_{i=1}^{l}\ell(f(x_i), y_i) + \lambda\sum_{i=l+1}^{l+u}\ell_u(f(x_i))$$

where $\ell_u$ is an unsupervised loss and $\lambda$ controls the balance.

### Common Approaches

**Self-Training:**
1. Train model on labeled data
2. Predict labels for unlabeled data
3. Add high-confidence predictions to training set
4. Retrain and repeat

**Co-Training:**
- Use multiple models with different views of data
- Each model labels data for the other

**Graph-Based Methods:**
- Construct graph where similar instances are connected
- Propagate labels through the graph

**Consistency Regularization:**
- Model should produce similar predictions for perturbed versions of the same input

### Practical Example

```python
from sklearn.semi_supervised import LabelPropagation
import numpy as np

# Create semi-supervised dataset
# -1 indicates unlabeled data
y_train = np.copy(y)
y_train[unlabeled_indices] = -1

# Train semi-supervised model
model = LabelPropagation(kernel='knn', n_neighbors=7)
model.fit(X_train, y_train)

# Predict on new data
y_pred = model.predict(X_test)
```

## Reinforcement Learning

**Reinforcement Learning (RL)** is a learning paradigm where an agent learns to make decisions by interacting with an environment to maximize cumulative reward.

### Key Components

- **Agent**: the learner or decision maker
- **Environment**: what the agent interacts with
- **State** $s_t$: the current situation of the agent
- **Action** $a_t$: what the agent can do
- **Reward** $r_t$: feedback signal from the environment
- **Policy** $\pi$: mapping from states to actions

### Mathematical Formulation

**Markov Decision Process (MDP)** is defined by tuple $(S, A, P, R, \gamma)$:
- $S$: set of states
- $A$: set of actions
- $P(s'|s,a)$: transition probability
- $R(s,a)$: reward function
- $\gamma \in [0,1]$: discount factor

**Goal**: Find optimal policy $\pi^*$ that maximizes expected cumulative reward:

$$\pi^* = \arg\max_{\pi} E\left[\sum_{t=0}^{\infty}\gamma^t r_t | \pi\right]$$

**Value Function**: Expected return starting from state $s$ and following policy $\pi$:

$$V^{\pi}(s) = E_{\pi}\left[\sum_{t=0}^{\infty}\gamma^t r_t | s_0 = s\right]$$

**Q-Function**: Expected return for taking action $a$ in state $s$:

$$Q^{\pi}(s,a) = E_{\pi}\left[\sum_{t=0}^{\infty}\gamma^t r_t | s_0 = s, a_0 = a\right]$$

**Bellman Equation**:

$$V^{\pi}(s) = \sum_a \pi(a|s)\sum_{s'}P(s'|s,a)[R(s,a) + \gamma V^{\pi}(s')]$$

### Learning Approaches

**Value-Based Methods:**
- Q-Learning
- Deep Q-Networks (DQN)
- Goal: learn optimal value function

**Policy-Based Methods:**
- REINFORCE
- Actor-Critic
- Proximal Policy Optimization (PPO)
- Goal: directly learn optimal policy

**Model-Based Methods:**
- Learn model of environment
- Use model for planning

### Applications

- Game playing (AlphaGo, Atari games)
- Robotics control
- Autonomous vehicles
- Resource management
- Recommendation systems

### Practical Example

```python
import gymnasium as gym
import numpy as np

# Simple Q-learning example
env = gym.make('FrozenLake-v1')
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1  # learning rate
gamma = 0.99  # discount factor
epsilon = 0.1  # exploration rate

for episode in range(10000):
    state = env.reset()[0]
    done = False

    while not done:
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # Take action and observe result
        next_state, reward, done, truncated, info = env.step(action)

        # Q-learning update
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
        )

        state = next_state
        done = done or truncated
```

## Other Paradigms

### Self-Supervised Learning

Learning representations from unlabeled data by creating pseudo-labels from the data itself.

**Examples:**
- **Masked Language Modeling**: predict masked words in text (BERT)
- **Contrastive Learning**: learn by comparing similar and dissimilar examples (SimCLR)
- **Next Frame Prediction**: predict future frames in video

Objective for contrastive learning:

$$L = -\log\frac{\exp(sim(z_i, z_j)/\tau)}{\sum_{k=1}^{2N}\mathbb{1}_{k \neq i}\exp(sim(z_i, z_k)/\tau)}$$

where $sim$ is similarity function (e.g., cosine similarity) and $\tau$ is temperature parameter.

### Transfer Learning

Leveraging knowledge learned from one task to improve performance on a related task.

**Process:**
1. Pre-train model on large source dataset
2. Fine-tune on smaller target dataset
3. Optionally freeze some layers

**When to use:**
- Limited target domain data
- Source and target tasks are related
- Pre-trained models available

### Meta-Learning (Learning to Learn)

Learning algorithms that can quickly adapt to new tasks with limited data.

**Approaches:**
- Model-Agnostic Meta-Learning (MAML)
- Prototypical Networks
- Matching Networks

### Active Learning

Iteratively selecting the most informative samples for labeling to minimize annotation cost.

**Query Strategies:**
- Uncertainty sampling
- Query-by-committee
- Expected model change

## Choosing the Right Paradigm

| Paradigm | When to Use | Data Requirements |
|----------|-------------|-------------------|
| **Supervised** | Clear target variable, labeled data available | Large labeled dataset |
| **Unsupervised** | Exploratory analysis, no labels available | Unlabeled data |
| **Semi-Supervised** | Limited labels, abundant unlabeled data | Small labeled + large unlabeled |
| **Reinforcement** | Sequential decision making, can interact with environment | Environment simulator or real interaction |
| **Self-Supervised** | Need representations, no labels | Large unlabeled dataset |
| **Transfer** | Limited target data, related source task exists | Pre-trained model + small target dataset |

### Decision Framework

1. **Do you have labeled data?**
   - Yes, plenty → Supervised Learning
   - Yes, but limited → Semi-Supervised or Transfer Learning
   - No → Unsupervised or Self-Supervised Learning

2. **Is it a sequential decision problem?**
   - Yes → Reinforcement Learning
   - No → Continue to next question

3. **What's your goal?**
   - Prediction → Supervised Learning
   - Discover patterns → Unsupervised Learning
   - Learn representations → Self-Supervised Learning

4. **Do you have a related pre-trained model?**
   - Yes → Transfer Learning
   - No → Train from scratch

## Practical Considerations

### Data Requirements

- **Supervised**: Typically needs 1000+ labeled examples per class
- **Unsupervised**: Works with any amount, more data = better patterns
- **Reinforcement**: Needs many interactions (100K - 1M+ steps)
- **Semi-Supervised**: 10-20% labeled can be enough with good unlabeled data

### Computational Cost

- Supervised learning: Moderate (single pass through data)
- Unsupervised learning: Low to moderate
- Reinforcement learning: High (requires many environment interactions)
- Semi-supervised: Moderate to high (iterative refinement)

### Evaluation Challenges

- **Supervised**: Clear metrics (accuracy, F1, RMSE)
- **Unsupervised**: Subjective (silhouette score, domain expertise)
- **Reinforcement**: Average reward over episodes
- **Semi-supervised**: Supervised metrics on labeled validation set

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction
- Zhu, X., & Goldberg, A. B. (2009). Introduction to Semi-Supervised Learning
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning
