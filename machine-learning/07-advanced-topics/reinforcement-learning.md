# Reinforcement Learning

## Table of Contents

1. [Introduction to Reinforcement Learning](#introduction-to-reinforcement-learning)
2. [Core Concepts](#core-concepts)
3. [Markov Decision Processes](#markov-decision-processes)
4. [Value-Based Methods](#value-based-methods)
5. [Policy-Based Methods](#policy-based-methods)
6. [Actor-Critic Methods](#actor-critic-methods)
7. [Exploration vs Exploitation](#exploration-vs-exploitation)
8. [Deep Reinforcement Learning](#deep-reinforcement-learning)
9. [Applications](#applications)

## Introduction to Reinforcement Learning

**Reinforcement Learning (RL)** is a paradigm of machine learning where an agent learns to make decisions by interacting with an environment. Unlike supervised learning (learning from labeled examples) or unsupervised learning (finding patterns), RL learns from trial and error using rewards and penalties.

### Key Characteristics

- **Sequential decision-making**: Actions affect future states
- **Delayed rewards**: Consequences of actions may not be immediate
- **Trial-and-error learning**: Agent explores to find good strategies
- **Trade-offs**: Balance between exploration and exploitation

### Comparison with Other Learning Paradigms

| Aspect | Supervised | Unsupervised | Reinforcement |
|--------|-----------|--------------|---------------|
| Feedback | Labeled data | No labels | Reward signal |
| Goal | Predict output | Find patterns | Maximize reward |
| Examples | Classification | Clustering | Game playing |

## Core Concepts

### The RL Framework

The agent-environment interaction loop:

1. **Agent** observes current **state** $s_t$
2. **Agent** takes **action** $a_t$
3. **Environment** transitions to new **state** $s_{t+1}$
4. **Agent** receives **reward** $r_{t+1}$
5. Repeat

### Key Components

**Agent** - the learner and decision maker

**Environment** - everything the agent interacts with

**State** ($s$) - current situation of the agent:
- Can be fully observable (agent sees complete state)
- Can be partially observable (agent has limited information)

**Action** ($a$) - choices available to the agent:
- **Discrete**: Finite set of actions (move left, right, up, down)
- **Continuous**: Actions in continuous space (steering angle, speed)

**Reward** ($r$) - scalar feedback signal:
- Indicates how good the action was
- Can be sparse (infrequent) or dense (every step)
- Goal: Maximize cumulative reward

**Policy** ($\pi$) - agent's behavior, mapping from states to actions:
- **Deterministic**: $a = \pi(s)$
- **Stochastic**: $a \sim \pi(a|s)$

**Value Function** ($V$ or $Q$) - expected cumulative reward:
- Estimates long-term value of states or state-action pairs

**Model** (optional) - agent's representation of environment dynamics:
- **Model-based RL**: Learns model of environment
- **Model-free RL**: Learns directly from experience

### Return and Discounting

**Return** ($G_t$) - total cumulative reward from time $t$:

$$G_t = r_{t+1} + r_{t+2} + r_{t+3} + ... + r_T$$

**Discounted Return** - future rewards are worth less:

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

where $\gamma \in [0, 1]$ is the **discount factor**:
- $\gamma = 0$: Only immediate rewards matter (myopic)
- $\gamma = 1$: All rewards equally important (far-sighted)
- Typical values: $\gamma \in [0.9, 0.99]$

**Why discount?**
- Mathematical convenience (infinite sums converge)
- Uncertainty about future
- Models preference for immediate rewards

## Markov Decision Processes

**Markov Decision Process (MDP)** - mathematical framework for RL:

$$\text{MDP} = (S, A, P, R, \gamma)$$

where:
- $S$: Set of states
- $A$: Set of actions
- $P$: Transition probability $P(s'|s,a)$
- $R$: Reward function $R(s,a,s')$
- $\gamma$: Discount factor

### Markov Property

**Markov property**: Future is independent of past given present:

$$P(s_{t+1}|s_t, a_t, s_{t-1}, a_{t-1}, ..., s_0, a_0) = P(s_{t+1}|s_t, a_t)$$

The current state contains all relevant information for decision-making.

### Value Functions

**State-value function** - expected return starting from state $s$ following policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi[G_t | s_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} | s_t = s\right]$$

**Action-value function (Q-function)** - expected return starting from state $s$, taking action $a$, then following $\pi$:

$$Q^\pi(s,a) = \mathbb{E}_\pi[G_t | s_t = s, a_t = a]$$

**Optimal value functions**:
$$V^*(s) = \max_\pi V^\pi(s)$$
$$Q^*(s,a) = \max_\pi Q^\pi(s,a)$$

### Bellman Equations

**Bellman equation for $V^\pi$** (recursive relationship):

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

**Bellman equation for $Q^\pi$**:

$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')]$$

**Bellman optimality equation**:

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]$$

$$Q^*(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')]$$

## Value-Based Methods

Learn value functions, derive policy from values.

### Dynamic Programming

Requires complete knowledge of MDP (model-based).

**Policy Iteration**:
1. **Policy Evaluation**: Compute $V^\pi$ for current policy
2. **Policy Improvement**: Update policy greedily: $\pi'(s) = \arg\max_a Q^\pi(s,a)$
3. Repeat until convergence

**Value Iteration**:
Directly compute optimal value function:

$$V(s) \leftarrow \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V(s')]$$

Then extract policy: $\pi(s) = \arg\max_a Q(s,a)$

### Monte Carlo Methods

Learn from complete episodes (model-free):

1. Generate episode following policy $\pi$
2. For each state $s$ visited:
   - Calculate return $G_t$
   - Update value estimate: $V(s) \leftarrow V(s) + \alpha[G_t - V(s)]$

**Advantages**:
- No environment model needed
- Simple to implement

**Disadvantages**:
- Only works for episodic tasks
- High variance
- Slow convergence

### Temporal Difference (TD) Learning

Combines ideas from Monte Carlo and Dynamic Programming.

**TD(0)** - update after each step:

$$V(s_t) \leftarrow V(s_t) + \alpha[r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]$$

**TD error**: $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$

**Advantages**:
- Works for non-episodic tasks
- Lower variance than Monte Carlo
- Can learn online

### Q-Learning

Learn action-value function directly:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]$$

**Key properties**:
- **Off-policy**: Learns optimal policy while following exploratory policy
- **Model-free**: No environment model needed
- Guaranteed to converge to $Q^*$ (under conditions)

**Q-learning algorithm**:
```python
# Initialize Q(s,a) arbitrarily
for episode in episodes:
    s = initial_state
    while not terminal:
        a = epsilon_greedy(Q, s)  # Choose action
        s_next, r = env.step(a)    # Take action
        # Update Q-value
        Q[s,a] = Q[s,a] + alpha * (r + gamma * max(Q[s_next]) - Q[s,a])
        s = s_next
```

### SARSA (State-Action-Reward-State-Action)

**On-policy** TD control:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]$$

**Difference from Q-learning**:
- Uses actual next action $a_{t+1}$ (not max)
- Learns value of policy being followed
- More conservative in risky environments

## Policy-Based Methods

Directly learn policy without value function.

### Policy Gradient

**Policy parameterization**: $\pi(a|s, \theta)$ where $\theta$ are parameters.

**Objective**: Maximize expected return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)]$$

**Policy Gradient Theorem**:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a)]$$

**REINFORCE algorithm**:

$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) G_t$$

**Advantages**:
- Can learn stochastic policies
- Effective in high-dimensional or continuous action spaces
- Better convergence properties

**Disadvantages**:
- High variance
- Sample inefficient
- Can converge to local optima

### Baseline Reduction

Reduce variance by subtracting baseline $b(s)$:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot (Q^{\pi_\theta}(s,a) - b(s))]$$

Common baseline: state value $V(s)$, giving **advantage function**:

$$A(s,a) = Q(s,a) - V(s)$$

## Actor-Critic Methods

Combine value-based and policy-based methods.

### Architecture

**Actor**: Policy $\pi(a|s, \theta)$ - selects actions

**Critic**: Value function $V(s, w)$ or $Q(s,a,w)$ - evaluates actions

**Update rules**:

Critic (TD learning):
$$w \leftarrow w + \alpha_w \delta \nabla_w V(s, w)$$

Actor (policy gradient):
$$\theta \leftarrow \theta + \alpha_\theta \delta \nabla_\theta \log \pi(a|s, \theta)$$

where $\delta = r + \gamma V(s', w) - V(s, w)$ is TD error.

### Advantage Actor-Critic (A2C)

Uses advantage function:

$$A(s,a) = Q(s,a) - V(s) \approx r + \gamma V(s') - V(s)$$

**Benefits**:
- Lower variance than REINFORCE
- More stable than Q-learning
- Works well in practice

### A3C (Asynchronous Advantage Actor-Critic)

Multiple agents learn in parallel:
- Each agent has own copy of environment
- Updates shared global network
- Improves data efficiency and stability

## Exploration vs Exploitation

**Exploitation**: Choose action with highest known value

**Exploration**: Try new actions to discover better options

### Epsilon-Greedy

$$a = \begin{cases}
\arg\max_a Q(s,a) & \text{with probability } 1-\epsilon \\
\text{random action} & \text{with probability } \epsilon
\end{cases}$$

**Variants**:
- Constant $\epsilon$: Simple but never fully exploits
- Decaying $\epsilon$: $\epsilon_t = \epsilon_0 e^{-\lambda t}$
- Adaptive $\epsilon$: Adjust based on learning progress

### Upper Confidence Bound (UCB)

$$a_t = \arg\max_a \left[Q(s,a) + c\sqrt{\frac{\ln t}{N(s,a)}}\right]$$

where:
- $N(s,a)$ is count of action $a$ in state $s$
- $c$ controls exploration level

### Boltzmann Exploration

Probability proportional to estimated value:

$$\pi(a|s) = \frac{\exp(Q(s,a)/\tau)}{\sum_{a'}\exp(Q(s,a')/\tau)}$$

where $\tau$ is **temperature**:
- High $\tau$: More exploration (uniform)
- Low $\tau$: More exploitation (greedy)

## Deep Reinforcement Learning

Combining deep neural networks with RL.

### Deep Q-Network (DQN)

Use neural network to approximate $Q(s,a; \theta)$.

**Key innovations**:

1. **Experience Replay**: Store transitions $(s,a,r,s')$ in replay buffer, sample randomly for training
   - Breaks correlation between consecutive samples
   - Improves data efficiency

2. **Target Network**: Separate network $Q(s,a; \theta^-)$ with frozen weights
   - Updated periodically: $\theta^- \leftarrow \theta$
   - Stabilizes training

**DQN loss**:
$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D}\left[\left(r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta)\right)^2\right]$$

**Improvements**:
- **Double DQN**: Reduces overestimation
- **Dueling DQN**: Separate value and advantage streams
- **Prioritized Experience Replay**: Sample important transitions more often

### Policy Gradient with Neural Networks

**TRPO (Trust Region Policy Optimization)**:
Constrains policy updates to prevent destructive updates:

$$\max_\theta \mathbb{E}[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A(s,a)]$$
$$\text{subject to } \mathbb{E}[KL(\pi_{\theta_{old}} || \pi_\theta)] \leq \delta$$

**PPO (Proximal Policy Optimization)**:
Simpler alternative to TRPO:

$$L(\theta) = \mathbb{E}\left[\min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$

**Benefits**:
- More stable than vanilla policy gradient
- Simpler than TRPO
- State-of-the-art performance

### Continuous Action Spaces

**DDPG (Deep Deterministic Policy Gradient)**:
- Actor-critic for continuous actions
- Deterministic policy: $a = \mu(s|\theta)$
- Critic estimates $Q(s,a)$

**TD3 (Twin Delayed DDPG)**:
Improvements over DDPG:
- Two critics (take minimum to reduce overestimation)
- Delayed policy updates
- Target policy smoothing

**SAC (Soft Actor-Critic)**:
Maximizes both reward and entropy:

$$J(\pi) = \sum_t \mathbb{E}_{(s_t,a_t)\sim\rho_\pi}[r(s_t,a_t) + \alpha H(\pi(\cdot|s_t))]$$

More robust and sample-efficient.

## Applications

### Game Playing

**Atari Games**: DQN achieved human-level performance

**Go**: AlphaGo defeated world champion using:
- Policy networks
- Value networks
- Monte Carlo Tree Search (MCTS)

**StarCraft II**: AlphaStar uses multi-agent RL

### Robotics

**Manipulation**: Grasping, assembly tasks

**Locomotion**: Walking, running robots

**Navigation**: Autonomous navigation

**Challenges**:
- Sample efficiency (real-world data expensive)
- Safety constraints
- Sim-to-real transfer

### Autonomous Vehicles

- Lane keeping
- Path planning
- Adaptive cruise control

### Resource Management

**Data centers**: Cooling optimization (DeepMind reduced Google's cooling costs by 40%)

**Traffic control**: Optimizing traffic light timing

**Energy grids**: Load balancing

### Recommendation Systems

- Personalized recommendations
- Ad placement
- Dynamic pricing

## Challenges and Considerations

### Sample Efficiency

RL often requires millions of samples:
- **Solution**: Model-based RL, transfer learning, meta-learning

### Sparse Rewards

Long delays between actions and rewards:
- **Solution**: Reward shaping, hierarchical RL, curiosity-driven exploration

### Partial Observability

Agent doesn't see full state:
- **Solution**: Recurrent networks (LSTMs), belief states

### Multi-Agent RL

Multiple agents interacting:
- **Challenges**: Non-stationarity, credit assignment
- **Approaches**: Centralized training with decentralized execution

### Safety and Constraints

Ensuring safe exploration:
- **Safe RL**: Constrained optimization
- **Offline RL**: Learn from fixed dataset
- **Human feedback**: Learn from human preferences

## Practical Tips

### Debugging RL

1. Test on simple environments first
2. Monitor learning curves (rewards, losses)
3. Verify policy improvement
4. Check exploration behavior
5. Validate value function estimates

### Hyperparameter Tuning

Important hyperparameters:
- Learning rate
- Discount factor $\gamma$
- Exploration parameters ($\epsilon$, temperature)
- Network architecture
- Batch size, replay buffer size

### Implementation

```python
# Simple Q-learning example
import numpy as np

Q = np.zeros((n_states, n_actions))
alpha = 0.1  # learning rate
gamma = 0.99  # discount factor
epsilon = 0.1  # exploration

for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        # Take action
        next_state, reward, done, _ = env.step(action)

        # Q-learning update
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state
```

## References

- "Reinforcement Learning: An Introduction" (Sutton & Barto, 2018)
- "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013) - DQN paper
- "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- "Mastering the Game of Go with Deep Neural Networks and Tree Search" (Silver et al., 2016)
- OpenAI Spinning Up in Deep RL - https://spinningup.openai.com/
- DeepMind's Advanced Deep Learning & Reinforcement Learning course
