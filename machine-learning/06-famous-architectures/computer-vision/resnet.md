# ResNet - Deep Residual Learning

## Table of Contents

1. [Overview](#overview)
2. [The Degradation Problem](#the-degradation-problem)
3. [Residual Learning](#residual-learning)
4. [Architecture](#architecture)
5. [Mathematical Foundation](#mathematical-foundation)
6. [Implementation](#implementation)
7. [Variants and Extensions](#variants-and-extensions)
8. [Impact and Applications](#impact-and-applications)

## Overview

**ResNet** (Residual Network) introduced residual connections (skip connections) that enabled training of extremely deep neural networks (100+ layers) without degradation.

**Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)

**Year**: 2015

**Achievement**: Won ILSVRC 2015 with 3.57% top-5 error (surpassing human-level performance ~5%)

**Key innovation**: **Residual connections** - instead of learning $H(x)$, learn $F(x) = H(x) - x$

**Impact**: One of the most influential papers in deep learning, enabling the very deep networks used today

## The Degradation Problem

### The Paradox of Depth

**Hypothesis**: Deeper networks should perform at least as well as shallower ones
- A deep network can always learn identity mappings for extra layers
- Therefore, it should not perform worse

**Reality**: Plain networks degrade with increasing depth

| Network Depth | Training Error | Test Error |
|---------------|----------------|------------|
| 20 layers | Low | Good |
| 56 layers | **Higher!** | **Worse!** |

This is **not** overfitting (training error also increases)!

### Why Plain Deep Networks Fail

**Problem**: Optimization difficulty
- Gradients vanish or explode in very deep networks
- Hard to learn identity mappings
- Random initialization is far from optimal

**Not the issue**:
- Overfitting (would show low training error)
- Model capacity (deeper should have more capacity)

**Solution**: Make identity mappings easy to learn via residual connections

## Residual Learning

### Core Idea

Instead of learning the desired mapping $H(x)$ directly, learn the **residual** $F(x) = H(x) - x$.

Then: $H(x) = F(x) + x$

**Intuition**:
- It's easier to learn deviations from identity than the full mapping
- If identity is optimal, it's easier to push residual to zero than learn identity

### Residual Block

```
Input x
    ↓
    ├─────────────┐  (skip connection)
    ↓             │
  Conv - BN - ReLU
    ↓             │
  Conv - BN       │
    ↓             │
    +  ← ← ← ← ← ┘  (element-wise addition)
    ↓
   ReLU
    ↓
  Output
```

**Mathematically**:

$$y = F(x, \{W_i\}) + x$$

where:
- $x$ is input
- $F(x, \{W_i\})$ is the residual mapping
- $+$ is element-wise addition

**After ReLU**:

$$\text{output} = \sigma(F(x) + x)$$

where $\sigma$ is ReLU activation.

### Why It Works

1. **Easy identity**: If $F(x) = 0$, we get identity mapping $H(x) = x$
2. **Better gradients**: Skip connections provide direct gradient paths
3. **Ensemble effect**: Network behaves like ensemble of shallower networks
4. **Feature reuse**: Lower-level features propagate directly to higher layers

**Gradient flow**:

During backpropagation through skip connection:

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left(1 + \frac{\partial F}{\partial x}\right)$$

The "+1" ensures gradient always flows, preventing vanishing gradients!

## Architecture

### ResNet-50 Architecture

ResNet uses **bottleneck blocks** for efficiency:

```
Input: 224×224×3
    ↓
Conv1: 7×7, 64, stride 2
    ↓ (112×112×64)
MaxPool: 3×3, stride 2
    ↓ (56×56×64)

Stage 1: [1×1, 64  ]
         [3×3, 64  ] × 3 blocks
         [1×1, 256 ]
    ↓ (56×56×256)

Stage 2: [1×1, 128 ]
         [3×3, 128 ] × 4 blocks
         [1×1, 512 ]
    ↓ (28×28×512)

Stage 3: [1×1, 256 ]
         [3×3, 256 ] × 6 blocks
         [1×1, 1024]
    ↓ (14×14×1024)

Stage 4: [1×1, 512 ]
         [3×3, 512 ] × 3 blocks
         [1×1, 2048]
    ↓ (7×7×2048)

Average Pool: 7×7
    ↓ (2048)
FC: 1000
    ↓
Softmax
```

### ResNet Family

| Model | Layers | Parameters | Top-5 Error | GFLOPs |
|-------|--------|------------|-------------|---------|
| ResNet-18 | 18 | 11.7M | 10.2% | 1.8 |
| ResNet-34 | 34 | 21.8M | 7.7% | 3.7 |
| ResNet-50 | 50 | 25.6M | 5.3% | 4.1 |
| ResNet-101 | 101 | 44.5M | 4.6% | 7.9 |
| ResNet-152 | 152 | 60.2M | 4.5% | 11.6 |

### Building Blocks

**Basic Block** (used in ResNet-18, ResNet-34):

```
x → [3×3 conv, C]
  → [3×3 conv, C]
  → + (with x)
```

**Bottleneck Block** (used in ResNet-50+):

```
x → [1×1 conv, C]    (reduce dimensions)
  → [3×3 conv, C]    (process features)
  → [1×1 conv, 4C]   (expand dimensions)
  → + (with x)
```

**Why bottleneck**:
- Reduces computational cost
- 1×1 convolutions reduce/expand channels
- 3×3 convolution operates on smaller dimension
- Same accuracy with fewer FLOPs

### Projection Shortcuts

When dimensions don't match, use **projection**:

$$y = F(x, \{W_i\}) + W_s x$$

where $W_s$ is a 1×1 convolution (or linear projection) to match dimensions.

**Three options**:
1. **Zero-padding**: Pad identity with zeros (no parameters)
2. **Projection**: Use 1×1 conv when dimensions change
3. **All projection**: Use 1×1 conv for all shortcuts

**Best practice**: Option 2 (projection only when needed)

## Mathematical Foundation

### Forward Propagation

For a residual unit:

$$y_l = h(x_l) + F(x_l, W_l)$$
$$x_{l+1} = f(y_l)$$

where:
- $x_l$ is input to layer $l$
- $F$ is residual function
- $h(x_l) = x_l$ is identity (or projection)
- $f$ is activation (ReLU)

### Backward Propagation

By chain rule, gradient for layer $l$:

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \frac{\partial x_L}{\partial x_l}$$

With residual connections:

$$x_L = x_l + \sum_{i=l}^{L-1} F(x_i, W_i)$$

Therefore:

$$\frac{\partial x_L}{\partial x_l} = 1 + \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1} F(x_i, W_i)$$

**Key insight**: The "+1" ensures gradient doesn't vanish, even if $\frac{\partial F}{\partial x} \to 0$.

### Ensemble Interpretation

ResNet can be viewed as an ensemble of $2^n$ paths (where $n$ is number of blocks):
- Each residual block creates two paths: through block or skip
- Exponential number of paths of varying lengths
- Behaves like ensemble averaging

**Effective depth**: Average path length is much shorter than network depth.

## Implementation

### PyTorch ResNet-50

```python
import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4  # Output channels = input × 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        # Bottleneck layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        # Bottleneck path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Projection shortcut if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add residual
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()

        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None

        # Need projection if dimensions change
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # First block (may downsample)
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def resnet152(num_classes=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)
```

### Using Pre-trained ResNet

```python
from torchvision import models

# Load pre-trained ResNet-50
model = models.resnet50(pretrained=True)

# Feature extraction mode
model.eval()
for param in model.parameters():
    param.requires_grad = False

# Get features before classification
features = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(
    model.maxpool(model.relu(model.bn1(model.conv1(images)))))))))

# Fine-tuning for new task
model.fc = nn.Linear(model.fc.in_features, num_new_classes)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

## Variants and Extensions

### ResNeXt

Introduces **cardinality** (number of paths):

- Multiple parallel branches with same topology
- Aggregates transformations: "split-transform-merge"
- Better accuracy with same complexity

### Wide ResNet

Increases **width** (filters per layer) instead of depth:

- ResNet-50 with 2× filters outperforms ResNet-101
- Faster to train (more parallelizable)
- Trade-off: more memory

### DenseNet

Connects each layer to **every subsequent layer**:

- Extreme feature reuse
- Even better gradient flow
- More parameters but better regularization

### ResNet-D

Improvements to ResNet design:

- Better downsampling (avoid information loss)
- Improved stem (initial layers)
- ~1-2% accuracy improvement

### Pre-activation ResNet

Move batch norm and ReLU **before** convolution:

```
BN → ReLU → Conv → BN → ReLU → Conv → Add
```

**Advantages**:
- Cleaner residual path
- Better gradient flow
- Easier to train very deep networks (1000+ layers)

## Training Details

### Original Training (ImageNet)

- **Optimizer**: SGD with momentum (0.9)
- **Batch size**: 256
- **Weight decay**: 0.0001
- **Learning rate**: 0.1, divided by 10 when error plateaus
- **Training time**: Converges in ~600K iterations
- **Data augmentation**: Scale augmentation, random crops, horizontal flips, color jittering
- **Batch normalization**: After every convolution, before ReLU

### Initialization

**Kaiming Initialization** (by same authors):

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$$

where $n_{\text{in}}$ is number of input connections.

**Designed for ReLU**: Accounts for ReLU killing half the activations.

### Data Augmentation

- **Scale jittering**: Random resize, then 224×224 crop
- **Horizontal flips**: 50% probability
- **Color normalization**: Per-channel mean subtraction
- **PCA color augmentation**: AlexNet-style

**Test time**: 10-crop testing or dense evaluation

## Impact and Applications

### Performance Comparison

| Architecture | Layers | Top-5 Error | vs Human |
|--------------|--------|-------------|----------|
| AlexNet | 8 | 15.3% | Much worse |
| VGG-19 | 19 | 9.0% | Worse |
| GoogLeNet | 22 | 6.7% | Worse |
| ResNet-50 | 50 | 5.3% | Better! |
| ResNet-152 | 152 | **3.57%** | **Much better!** |
| Human | - | ~5% | Baseline |

### Why ResNet Changed Everything

1. **Enabled very deep networks**: 100-1000+ layers possible
2. **Universal technique**: Skip connections used everywhere now
3. **Better optimization**: Easier to train
4. **Transfer learning**: ResNet features work excellently
5. **Foundation**: Basis for modern architectures

### Applications Beyond ImageNet

- **Object detection**: Faster R-CNN, Mask R-CNN use ResNet backbone
- **Segmentation**: U-Net with ResNet encoder
- **Face recognition**: State-of-the-art systems use ResNet
- **Medical imaging**: Transfer learning from ImageNet ResNet
- **Video understanding**: 3D ResNet for temporal modeling

### Modern Usage

**ResNet-50** is the de facto standard:
- Good accuracy/efficiency trade-off
- Fast inference
- Excellent pre-trained weights
- Well-supported in all frameworks

## Key Takeaways

**Core innovation**: Skip connections solve vanishing gradient and enable very deep networks

**Why it works**:
- Makes identity mapping easy to learn
- Provides gradient highway
- Acts as ensemble of paths
- Enables feature reuse

**Legacy**:
- Skip connections are now standard in almost all architectures
- Proved that depth can be pushed to extreme with right design
- Established ResNet as go-to architecture for transfer learning

## References

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition" CVPR (2016)
- [Original Paper](https://arxiv.org/abs/1512.03385)
- Kaiming He et al. "Identity Mappings in Deep Residual Networks" ECCV (2016)
- [Pre-activation ResNet Paper](https://arxiv.org/abs/1603.05027)
- [PyTorch Implementation](https://pytorch.org/vision/stable/models.html#resnet)
