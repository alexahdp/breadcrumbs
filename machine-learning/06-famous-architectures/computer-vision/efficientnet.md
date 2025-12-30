# EfficientNet - Rethinking Model Scaling

## Table of Contents

1. [Overview](#overview)
2. [The Scaling Problem](#the-scaling-problem)
3. [Compound Scaling](#compound-scaling)
4. [Architecture](#architecture)
5. [Mathematical Foundation](#mathematical-foundation)
6. [Implementation](#implementation)
7. [Performance and Efficiency](#performance-and-efficiency)
8. [Impact and Variants](#impact-and-variants)

## Overview

**EfficientNet** is a family of convolutional neural networks that achieves state-of-the-art accuracy while being significantly more efficient than previous models through systematic scaling of network dimensions.

**Authors**: Mingxing Tan and Quoc V. Le (Google Research, Brain Team)

**Year**: 2019

**Achievement**: 84.3% top-1 accuracy on ImageNet with 66M parameters (vs 480M for GPipe)

**Key innovation**: **Compound scaling** - systematically scale depth, width, and resolution together using a principled approach

**Impact**: Established new standard for efficient architectures, inspiring numerous follow-up works

## The Scaling Problem

### Traditional Scaling Methods

To improve CNN performance, researchers typically scale one dimension:

1. **Depth scaling**: Add more layers (e.g., ResNet-18 → ResNet-152)
2. **Width scaling**: Add more channels per layer (e.g., Wide ResNet)
3. **Resolution scaling**: Use larger input images

**Problem**: Scaling only one dimension has diminishing returns and is not optimal.

### Single-Dimension Scaling Limitations

**Depth scaling** (more layers):
- ✅ Captures more complex features
- ❌ Vanishing gradients
- ❌ Diminishing returns (80% accuracy gain in first 20 layers)

**Width scaling** (more channels):
- ✅ Captures more fine-grained features
- ❌ Hard to capture higher-level features
- ❌ Saturates quickly

**Resolution scaling** (larger images):
- ✅ Captures fine-grained patterns
- ❌ Expensive computation
- ❌ Diminishing returns at very high resolution

### The Key Insight

**Observation**: Intuition suggests balancing all dimensions should be more efficient.

**Question**: How to scale depth, width, and resolution systematically?

**Answer**: Compound scaling with fixed relationship between dimensions.

## Compound Scaling

### Compound Scaling Method

Scale all three dimensions together with compound coefficient $\phi$:

$$\text{depth: } d = \alpha^\phi$$
$$\text{width: } w = \beta^\phi$$
$$\text{resolution: } r = \gamma^\phi$$

subject to constraints:

$$\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$$
$$\alpha \geq 1, \beta \geq 1, \gamma \geq 1$$

where:
- $\alpha, \beta, \gamma$ are constants determined by grid search
- $\phi$ is user-specified coefficient to control resources

### Why This Works

**Intuition**: If input image is bigger, network needs:
- **More layers** (depth) to increase receptive field
- **More channels** (width) to capture more fine-grained patterns
- Both work together synergistically

**FLOPs scaling**:
- Depth: $\text{FLOPs} \propto d$
- Width: $\text{FLOPs} \propto w^2$ (both input and output channels)
- Resolution: $\text{FLOPs} \propto r^2$ (height × width)

**Total FLOPs**:

$$\text{FLOPs} \propto d \cdot w^2 \cdot r^2 = (\alpha \cdot \beta^2 \cdot \gamma^2)^\phi$$

Setting $\alpha \cdot \beta^2 \cdot \gamma^2 = 2$ means FLOPs increase by approximately $2^\phi$.

### Finding Optimal $\alpha, \beta, \gamma$

**Two-step process**:

**Step 1**: Fix $\phi = 1$, do small grid search to find $\alpha, \beta, \gamma$

For EfficientNet-B0:
- $\alpha = 1.2$ (depth)
- $\beta = 1.1$ (width)
- $\gamma = 1.15$ (resolution)

**Step 2**: Fix $\alpha, \beta, \gamma$, scale up using different $\phi$ values

| Model | $\phi$ | Depth | Width | Resolution |
|-------|--------|-------|-------|------------|
| B0 | 0 | 1.0 | 1.0 | 224 |
| B1 | 0.5 | 1.1 | 1.1 | 240 |
| B2 | 1 | 1.2 | 1.2 | 260 |
| B3 | 2 | 1.4 | 1.4 | 300 |
| B4 | 3 | 1.8 | 1.8 | 380 |
| B5 | 4 | 2.2 | 2.2 | 456 |
| B6 | 5 | 2.6 | 2.6 | 528 |
| B7 | 6 | 3.1 | 3.1 | 600 |

## Architecture

### EfficientNet-B0 Baseline

EfficientNet-B0 is found using Neural Architecture Search (NAS) optimizing both accuracy and FLOPs.

**Building block**: Mobile Inverted Bottleneck MBConv (from MobileNetV2)

```
Input: 224×224×3
    ↓
Stage 1: Conv 3×3, 32, stride 2
    ↓ (112×112×32)
Stage 2: MBConv1, k3×3, 16  × 1 block
    ↓ (112×112×16)
Stage 3: MBConv6, k3×3, 24  × 2 blocks, stride 2
    ↓ (56×56×24)
Stage 4: MBConv6, k5×5, 40  × 2 blocks, stride 2
    ↓ (28×28×40)
Stage 5: MBConv6, k3×3, 80  × 3 blocks, stride 2
    ↓ (14×14×80)
Stage 6: MBConv6, k5×5, 112 × 3 blocks
    ↓ (14×14×112)
Stage 7: MBConv6, k5×5, 192 × 4 blocks, stride 2
    ↓ (7×7×192)
Stage 8: MBConv6, k3×3, 320 × 1 block
    ↓ (7×7×320)
Stage 9: Conv 1×1, 1280
    ↓
AvgPool + Dropout (0.2)
    ↓
FC 1000
```

**Notation**: MBConv6 k3×3 means:
- MBConv: Mobile Inverted Bottleneck Convolution
- 6: Expansion ratio
- k3×3: Kernel size 3×3

### MBConv Block

**Mobile Inverted Bottleneck Convolution** (from MobileNetV2):

```
Input (C channels)
    ↓
1×1 Conv: Expand to t×C channels
BatchNorm + Swish
    ↓
Depthwise k×k Conv: Process each channel separately
BatchNorm + Swish
    ↓
Squeeze-and-Excitation
    ↓
1×1 Conv: Project back to C channels
BatchNorm
    ↓
+ (Skip connection if stride=1 and same dimensions)
    ↓
Output
```

**Components**:

1. **Expansion**: Increase channels by factor $t$ (typically 6)
2. **Depthwise convolution**: Spatial filtering per channel
3. **Squeeze-and-Excitation**: Channel attention
4. **Projection**: Reduce back to original channels
5. **Skip connection**: Residual connection when applicable

### Squeeze-and-Excitation (SE)

Adds channel-wise attention:

$$\text{SE}(x) = x \odot \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot \text{AvgPool}(x)))$$

where:
- $\odot$ is element-wise multiplication
- $\sigma$ is sigmoid
- AvgPool produces channel-wise statistics
- $W_1, W_2$ are learned weights

**Purpose**: Recalibrate channel-wise features, emphasizing important channels.

## Mathematical Foundation

### Compound Scaling Formulation

**Optimization problem**:

$$\max_{d,w,r} \text{Accuracy}(\mathcal{N}(d, w, r))$$
$$\text{subject to } \mathcal{N}(d, w, r) = \bigodot_{i=1,...,s} \hat{F}_i^{d \cdot \hat{L}_i}(X_{\langle r \cdot \hat{H}_i, r \cdot \hat{W}_i, w \cdot \hat{C}_i \rangle})$$
$$\text{Memory}(\mathcal{N}) \leq \text{target\_memory}$$
$$\text{FLOPs}(\mathcal{N}) \leq \text{target\_flops}$$

where:
- $\mathcal{N}$ is the network
- $\hat{F}_i$ is layer $i$ in baseline network
- $\hat{L}_i$ is number of layers in stage $i$
- $\hat{H}_i, \hat{W}_i, \hat{C}_i$ are baseline height, width, channels

### Depthwise Separable Convolution

MBConv uses **depthwise separable convolutions**:

**Standard convolution** FLOPs:

$$\text{FLOPs}_{\text{standard}} = H \cdot W \cdot C_{in} \cdot C_{out} \cdot k^2$$

**Depthwise separable** FLOPs:

$$\text{FLOPs}_{\text{depthwise}} = H \cdot W \cdot C_{in} \cdot k^2 + H \cdot W \cdot C_{in} \cdot C_{out}$$

**Reduction**:

$$\frac{\text{FLOPs}_{\text{depthwise}}}{\text{FLOPs}_{\text{standard}}} = \frac{1}{C_{out}} + \frac{1}{k^2}$$

For $C_{out} = 256, k = 3$: reduction = $1/256 + 1/9 \approx 11\%$ of original!

### Swish Activation

EfficientNet uses **Swish** activation instead of ReLU:

$$\text{Swish}(x) = x \cdot \sigma(\beta x)$$

where $\sigma$ is sigmoid and $\beta$ is learnable or fixed to 1.

**Properties**:
- Smooth (differentiable everywhere)
- Non-monotonic
- Unbounded above, bounded below
- Better performance than ReLU in many cases

## Implementation

### PyTorch MBConv Block

```python
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),  # Swish
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, expand_ratio, se_ratio=0.25):
        super(MBConvBlock, self).__init__()

        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)

        hidden_dim = in_channels * expand_ratio

        layers = []

        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])

        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size,
                     stride=stride, padding=kernel_size//2,
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        ])

        # Squeeze and excitation
        if se_ratio > 0:
            layers.append(SEBlock(hidden_dim,
                                 reduction=int(1/se_ratio)))

        # Projection phase
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)
```

### Using Pre-trained EfficientNet

```python
from torchvision import models

# Load pre-trained EfficientNet-B0
model = models.efficientnet_b0(pretrained=True)

# For different scales
model_b1 = models.efficientnet_b1(pretrained=True)
model_b7 = models.efficientnet_b7(pretrained=True)

# Transfer learning
for param in model.features.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_new_classes
)

# Fine-tuning
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
```

### Compound Scaling in Code

```python
def scale_dimensions(base_depth, base_width, base_resolution, phi,
                     alpha=1.2, beta=1.1, gamma=1.15):
    """
    Scale network dimensions using compound scaling

    Args:
        phi: Compound coefficient
        alpha, beta, gamma: Scaling coefficients
    """
    depth = int(base_depth * alpha ** phi)
    width = int(base_width * beta ** phi)
    resolution = int(base_resolution * gamma ** phi)

    return depth, width, resolution

# Example: Scale from B0 to B3 (phi=2)
base_depth, base_width, base_resolution = 1.0, 1.0, 224
depth_b3, width_b3, resolution_b3 = scale_dimensions(
    base_depth, base_width, base_resolution, phi=2
)
print(f"B3: depth={depth_b3}, width={width_b3}, resolution={resolution_b3}")
# Output: B3: depth=1.44, width=1.21, resolution=300
```

## Performance and Efficiency

### EfficientNet Family Performance

| Model | Params | FLOPs | Top-1 Acc | Top-5 Acc | Inference (ms) |
|-------|--------|-------|-----------|-----------|----------------|
| B0 | 5.3M | 0.39B | 77.1% | 93.3% | 2.9 |
| B1 | 7.8M | 0.70B | 79.1% | 94.4% | 4.1 |
| B2 | 9.2M | 1.0B | 80.1% | 94.9% | 4.8 |
| B3 | 12M | 1.8B | 81.6% | 95.7% | 6.9 |
| B4 | 19M | 4.2B | 82.9% | 96.4% | 11.6 |
| B5 | 30M | 9.9B | 83.6% | 96.7% | 20.3 |
| B6 | 43M | 19B | 84.0% | 96.8% | 32.1 |
| B7 | 66M | 37B | 84.3% | 97.0% | 51.2 |

### Comparison with Other Architectures

| Model | Params | FLOPs | Top-1 Acc | Efficiency |
|-------|--------|-------|-----------|------------|
| ResNet-50 | 25.6M | 4.1B | 76.0% | Baseline |
| ResNet-152 | 60.2M | 11.6B | 77.6% | ↓ |
| DenseNet-264 | 33M | 5.8B | 77.8% | ~ |
| GPipe | 557M | 128B | 84.3% | ↓↓↓ |
| **EfficientNet-B0** | **5.3M** | **0.39B** | **77.1%** | **↑↑↑** |
| **EfficientNet-B7** | **66M** | **37B** | **84.3%** | **↑↑** |

**Key insights**:
- EfficientNet-B0: Same accuracy as ResNet-50 with 5× fewer params, 10× fewer FLOPs
- EfficientNet-B7: Same accuracy as GPipe with 8× fewer params, 3× fewer FLOPs

### Accuracy vs Efficiency

EfficientNet achieves **better Pareto frontier**: for any given accuracy, it uses fewer resources than alternatives.

**Example**: To reach 80% top-1 accuracy:
- ResNet-152: 60M params, 11.6B FLOPs
- DenseNet-264: 33M params, 5.8B FLOPs
- **EfficientNet-B2**: 9.2M params, 1.0B FLOPs ← **Best!**

## Training Details

### Training Configuration

- **Optimizer**: RMSprop with decay 0.9, momentum 0.9
- **Batch size**: 2048 (distributed training)
- **Learning rate**: 0.256, decayed by 0.97 every 2.4 epochs
- **Weight decay**: 1e-5
- **Dropout**: 0.2 (B0) to 0.5 (B7), scaled with model size
- **Stochastic depth**: Survival probability 0.8
- **Activation**: Swish (SiLU)

### Data Augmentation

**AutoAugment**: Automated data augmentation search

Additional augmentations:
- Random crop and resize
- Random horizontal flip
- Mixup (blend two images)
- Cutout (random masking)

### Exponential Moving Average (EMA)

Use EMA of model weights for inference:

$$\theta_{\text{EMA}} = \beta \cdot \theta_{\text{EMA}} + (1 - \beta) \cdot \theta$$

Typical $\beta = 0.9999$

## Impact and Variants

### Why EfficientNet Matters

1. **Systematic scaling**: Principled approach to scaling, not ad-hoc
2. **SOTA efficiency**: Best accuracy-to-resource ratio
3. **Versatile**: Models from mobile (B0) to high-accuracy (B7)
4. **Transferable**: Excellent for transfer learning

### EfficientNetV2 (2021)

**Improvements**:
- **Fused-MBConv**: Fuse expansion and depthwise for efficiency
- **Progressive training**: Gradually increase image size during training
- **Adaptive regularization**: Adjust dropout/augmentation with image size

**Results**: Faster training, better parameter efficiency, up to 87.3% top-1 accuracy

### Applications

**Transfer learning**:
- Image classification on custom datasets
- Object detection (EfficientDet)
- Semantic segmentation
- Medical imaging

**Edge deployment**:
- B0, B1 for mobile devices
- TensorFlow Lite, ONNX conversion
- Real-time applications

### Influence

**Inspired**:
- NFNets: Normalizer-Free Networks
- CoAtNet: Combines convolutions with attention
- Vision Transformers: Scaled using similar principles

**Design principles adopted widely**:
- Compound scaling
- Neural Architecture Search for baseline
- Squeeze-and-Excitation blocks
- Swish activation

## Comparison: When to Use What

| Use Case | Recommended Model | Reason |
|----------|------------------|--------|
| High accuracy needed | EfficientNet-B7 | SOTA accuracy |
| Balanced performance | EfficientNet-B3/B4 | Best accuracy/speed |
| Mobile/Edge | EfficientNet-B0/B1 | Smallest, fastest |
| Transfer learning | EfficientNet-B0/B3 | Good features, fast fine-tuning |
| Research baseline | ResNet-50 | Standard benchmark |
| Simple architecture | ResNet | Easier to understand/modify |

## References

- Mingxing Tan, Quoc V. Le. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" ICML (2019)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- Mingxing Tan, Quoc V. Le. "EfficientNetV2: Smaller Models and Faster Training" ICML (2021)
- [EfficientNetV2 Paper](https://arxiv.org/abs/2104.00298)
- Mark Sandler et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks" CVPR (2018)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [PyTorch Implementation](https://pytorch.org/vision/stable/models.html#efficientnet)
