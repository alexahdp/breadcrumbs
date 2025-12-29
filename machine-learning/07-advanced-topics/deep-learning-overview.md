# Deep Learning Overview

## Table of Contents

1. [Advanced Deep Learning Concepts](#advanced-deep-learning-concepts)
2. [Attention Mechanisms](#attention-mechanisms)
3. [Advanced Architectures](#advanced-architectures)
4. [Modern Training Techniques](#modern-training-techniques)
5. [Optimization Advances](#optimization-advances)
6. [Normalization Techniques](#normalization-techniques)
7. [Advanced Regularization](#advanced-regularization)
8. [Meta-Learning](#meta-learning)
9. [Neural Architecture Search](#neural-architecture-search)

## Advanced Deep Learning Concepts

This section covers advanced concepts beyond basic neural networks (covered in section 05). These techniques are used in state-of-the-art models across various domains.

### Depth vs Width

**Deep networks** (many layers):
- Can learn hierarchical representations
- Each layer learns increasingly abstract features
- More parameters but potentially better generalization

**Wide networks** (many neurons per layer):
- Increased capacity at each level
- May require more data to avoid overfitting
- Can be easier to train

**Optimal architecture** depends on task, data availability, and computational constraints.

### Skip Connections and Residual Learning

**Residual connections** allow information to bypass layers:

$$y = F(x, \{W_i\}) + x$$

where $F$ represents the residual function to be learned.

**Benefits**:
- Enables training very deep networks (100+ layers)
- Alleviates vanishing gradient problem
- Allows identity mapping when needed
- Faster convergence

**Used in**: ResNet, DenseNet, Transformer architectures

### Dense Connections

**DenseNet** connects each layer to every subsequent layer:

$$x_l = H_l([x_0, x_1, ..., x_{l-1}])$$

**Benefits**:
- Strengthens feature propagation
- Reduces parameters (compared to ResNet)
- Alleviates vanishing gradient
- Encourages feature reuse

## Attention Mechanisms

**Attention** allows models to focus on relevant parts of input when producing output.

### Basic Attention

Computes weighted combination of values based on query-key similarity:

$$\text{score}(q, k_i) = q^T k_i \quad \text{or} \quad \frac{q^T k_i}{\sqrt{d}}$$

$$\alpha_i = \frac{\exp(\text{score}(q, k_i))}{\sum_j \exp(\text{score}(q, k_j))}$$

$$\text{output} = \sum_i \alpha_i v_i$$

**Components**:
- **Query** $(q)$: What we're looking for
- **Keys** $(k_i)$: What we're comparing against
- **Values** $(v_i)$: What we return

### Self-Attention

Input sequence attends to itself:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $Q$, $K$, $V$ are linear projections of the input.

**Applications**:
- Transformers (NLP, vision)
- Modeling long-range dependencies
- Capturing global context

### Cross-Attention

One sequence attends to another:
- Used in encoder-decoder architectures
- Machine translation: decoder attends to encoder outputs
- Image captioning: text generation attends to image features

### Multi-Head Attention

Performs attention multiple times with different learned projections:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(head_1, ..., head_h)W^O$$

$$head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Benefits**:
- Learns different aspects of relationships
- Increases model capacity
- Allows attention to different representation subspaces

## Advanced Architectures

### Vision Transformers (ViT)

Applies Transformers directly to image patches:

1. Split image into patches (e.g., 16×16)
2. Flatten and project patches to embeddings
3. Add positional embeddings
4. Process through Transformer encoder

**Key insight**: CNNs are not necessary for vision tasks; pure attention can work.

**Performance**:
- Competitive with CNNs when trained on large datasets
- Better scalability
- More interpretable attention patterns

### Graph Neural Networks (GNN)

Process data with graph structure:

$$h_v^{(l+1)} = \sigma\left(W^{(l)} h_v^{(l)} + \sum_{u \in N(v)} \frac{1}{c_{uv}} h_u^{(l)}\right)$$

where:
- $h_v^{(l)}$ is node $v$'s representation at layer $l$
- $N(v)$ is the neighborhood of node $v$
- $c_{uv}$ is a normalization constant

**Applications**:
- Social network analysis
- Molecular property prediction
- Recommendation systems
- Knowledge graphs

**Variants**:
- **GCN** (Graph Convolutional Networks)
- **GAT** (Graph Attention Networks)
- **GraphSAGE** (Graph Sample and Aggregate)

### Neural ODEs (Ordinary Differential Equations)

Model continuous-depth networks:

$$\frac{dh(t)}{dt} = f(h(t), t, \theta)$$

**Benefits**:
- Memory-efficient training
- Adaptive computation
- Continuous-time modeling
- Better for time-series and physics

### Capsule Networks

Use "capsules" (groups of neurons) to represent entities:

$$v_j = \frac{||\mathbf{s}_j||^2}{1 + ||\mathbf{s}_j||^2} \frac{\mathbf{s}_j}{||\mathbf{s}_j||}$$

**Key ideas**:
- Vector outputs (not scalars) encode properties
- Dynamic routing between capsules
- Better equivariance to transformations

**Challenges**:
- Computationally expensive
- Limited adoption compared to CNNs/Transformers

## Modern Training Techniques

### Mixed Precision Training

Uses both 16-bit (FP16) and 32-bit (FP32) floating-point:

**Benefits**:
- 2-3× faster training
- Reduced memory usage
- Enables larger batch sizes
- Minimal accuracy loss

**Techniques**:
- Loss scaling to prevent underflow
- Master weights in FP32
- Gradients computed in FP16

### Gradient Accumulation

Accumulate gradients over multiple mini-batches before updating:

```python
for i, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Benefits**:
- Simulates larger batch sizes
- Useful when GPU memory is limited
- Can improve generalization

### Gradient Checkpointing

Trade computation for memory by recomputing activations:

**Standard backprop**: Store all activations (high memory)

**Gradient checkpointing**: Store subset of activations, recompute others (lower memory, more computation)

**Use case**: Training very deep networks or large batch sizes

### Learning Rate Scheduling

**Cosine Annealing**:
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)$$

**Warm Restarts**: Periodically reset learning rate to high value

**One Cycle Policy**:
1. Increase learning rate to maximum
2. Decrease to minimum
3. Further decrease in final phase

### Label Smoothing

Replace hard targets with soft distributions:

$$y_{smooth} = (1 - \epsilon) \cdot y_{true} + \epsilon / K$$

where $K$ is number of classes and $\epsilon$ is smoothing parameter (e.g., 0.1).

**Benefits**:
- Prevents overconfidence
- Better calibration
- Improved generalization

## Optimization Advances

### Adam Variants

**AdamW** (Adam with decoupled Weight decay):
$$\theta_{t+1} = \theta_t - \eta_t (\frac{m_t}{\sqrt{v_t} + \epsilon} + \lambda\theta_t)$$

Better generalization than standard Adam.

**RAdam** (Rectified Adam): Addresses warm-up issue in Adam

**Lookahead**: Maintains fast and slow weights, combines them periodically

### Second-Order Methods

**Natural Gradient Descent**: Uses Fisher information matrix

**K-FAC** (Kronecker-Factored Approximate Curvature): Efficient second-order optimization

**Benefits**: Better convergence, fewer hyperparameters

**Drawbacks**: Computationally expensive

### Sharpness-Aware Minimization (SAM)

Seeks flat minima for better generalization:

$$\min_\theta \max_{||\epsilon|| \leq \rho} L(\theta + \epsilon)$$

Finds parameters where loss is flat in neighborhood.

## Normalization Techniques

### Batch Normalization (Review)

Normalizes activations across batch dimension:

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

**Issues**:
- Batch size dependent
- Problems with small batches
- Inconsistency between training and inference

### Layer Normalization

Normalizes across feature dimension:

$$\hat{x} = \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}}$$

**Benefits**:
- Batch size independent
- Used in Transformers
- Better for RNNs and small batches

### Group Normalization

Divides channels into groups and normalizes within groups:

$$\hat{x} = \frac{x - \mu_G}{\sqrt{\sigma_G^2 + \epsilon}}$$

**Benefits**:
- Batch size independent
- Works well with small batches
- Good alternative to BatchNorm

### Instance Normalization

Normalizes each sample independently:
- Used in style transfer
- Normalizes per-instance statistics

### Comparison

| Normalization | Dimension | Best For |
|--------------|-----------|----------|
| Batch | Across batch | Large batches, CNNs |
| Layer | Across features | Transformers, RNNs |
| Instance | Per sample | Style transfer |
| Group | Channel groups | Small batches, vision |

## Advanced Regularization

### DropConnect

Randomly drops connections (weights) instead of activations:
- More aggressive than dropout
- Can improve generalization

### Cutout / Random Erasing

Randomly mask rectangular regions in images:
- Forces model to use full context
- Simple data augmentation
- Improves robustness

### MixUp

Creates virtual training examples:

$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j$$
$$\tilde{y} = \lambda y_i + (1 - \lambda) y_j$$

where $\lambda \sim \text{Beta}(\alpha, \alpha)$.

**Benefits**:
- Better calibration
- More robust to adversarial examples
- Improved generalization

### CutMix

Combines MixUp with regional dropout:
- Cut and paste patches from images
- Mix labels proportionally to area

### Stochastic Depth

Randomly drop layers during training:

$$h_{l+1} = h_l + b_l \cdot f_l(h_l)$$

where $b_l \sim \text{Bernoulli}(p_l)$.

**Benefits**:
- Reduces training time
- Regularization effect
- Enables training deeper networks

## Meta-Learning

**Meta-learning** (learning to learn) develops models that can quickly adapt to new tasks.

### Model-Agnostic Meta-Learning (MAML)

Learns initialization that's easily fine-tuned:

$$\theta^* = \arg\min_\theta \sum_{\mathcal{T}_i} L_{\mathcal{T}_i}(f_{\theta_i'})$$

where $\theta_i' = \theta - \alpha \nabla_\theta L_{\mathcal{T}_i}(f_\theta)$

**Process**:
1. Sample tasks
2. Adapt to each task (inner loop)
3. Update meta-parameters (outer loop)

### Few-Shot Learning

Learn from few examples (1-shot, 5-shot, etc.):

**Approaches**:
- **Metric learning**: Learn embedding space (Siamese networks, Prototypical networks)
- **Meta-learning**: MAML, Reptile
- **Data augmentation**: Generate synthetic examples

**Applications**:
- Low-resource scenarios
- Personalization
- Continual learning

## Neural Architecture Search

**NAS** automates architecture design.

### Search Spaces

- **Micro-search**: Find optimal cell structure
- **Macro-search**: Find entire network structure

### Search Strategies

**Random Search**: Baseline approach

**Reinforcement Learning**: Controller generates architectures, trained based on accuracy

**Evolutionary Algorithms**: Mutate and select architectures

**Gradient-Based**: DARTS (Differentiable Architecture Search)
$$\alpha^* = \arg\min_\alpha L_{val}(w^*(\alpha), \alpha)$$

### Evaluation Strategies

**Full training**: Expensive but accurate

**Early stopping**: Train for fewer epochs

**Weight sharing**: Share weights across architectures (One-Shot NAS)

**Performance prediction**: Learn to predict architecture performance

### Challenges

- Computational cost (thousands of GPU hours)
- Search space design
- Transferability across datasets
- Reproducibility

## Practical Considerations

### Model Compression

**Pruning**: Remove unnecessary weights or neurons
- Magnitude-based, gradient-based, or learned

**Quantization**: Use lower precision (INT8 instead of FP32)
- Post-training or quantization-aware training

**Knowledge Distillation**: Train smaller model to mimic larger model
$$L = \alpha L_{CE}(y, \hat{y}) + (1-\alpha) L_{KL}(p_{teacher}, p_{student})$$

**Low-Rank Factorization**: Decompose weight matrices

### Hardware Considerations

**TPUs**: Optimized for matrix operations and Transformers

**GPUs**: General-purpose, good for CNNs and RNNs

**Specialized hardware**: Edge devices, mobile phones

**Distributed training**: Data parallelism, model parallelism

### Framework Choices

**PyTorch**:
- Dynamic computation graphs
- Pythonic and flexible
- Strong research community

**TensorFlow/Keras**:
- Production-ready
- TensorFlow Lite for mobile
- Strong ecosystem

**JAX**:
- Functional programming
- Automatic differentiation
- High-performance computing

## Emerging Directions

### Self-Supervised Learning

Learn representations from unlabeled data:
- **Contrastive learning**: SimCLR, MoCo
- **Masked modeling**: MAE (Masked Autoencoders)
- **Bootstrap methods**: BYOL, SimSiam

### Multimodal Learning

Combining multiple modalities (vision, language, audio):
- CLIP: Vision-language pre-training
- DALL-E: Text-to-image generation
- Flamingo: Visual language models

### Efficient Transformers

Reducing $O(n^2)$ complexity of attention:
- Sparse attention patterns
- Low-rank approximations
- Linear attention mechanisms

## References

- "Deep Residual Learning for Image Recognition" (He et al., 2016)
- "Attention Is All You Need" (Vaswani et al., 2017)
- "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2021)
- "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (Finn et al., 2017)
- "Neural Architecture Search: A Survey" (Elsken et al., 2019)
- Deep Learning (Goodfellow, Bengio, Courville)
- Dive into Deep Learning (Zhang et al.)
