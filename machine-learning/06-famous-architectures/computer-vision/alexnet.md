# AlexNet - The Deep Learning Revolution

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Key Innovations](#key-innovations)
4. [Mathematical Details](#mathematical-details)
5. [Implementation](#implementation)
6. [Training Details](#training-details)
7. [Impact and Legacy](#impact-and-legacy)

## Overview

**AlexNet** is a deep convolutional neural network that won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012, achieving a top-5 error rate of 15.3% compared to 26.2% from the second-place entry.

**Authors**: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton (University of Toronto)

**Year**: 2012

**Significance**: AlexNet demonstrated that deep learning could dramatically outperform traditional computer vision methods, sparking the modern deep learning revolution.

**Key achievements**:
- Cut ImageNet error rate nearly in half
- First large-scale CNN trained on GPUs
- Popularized ReLU, dropout, and data augmentation
- Proved that depth and scale matter

## Architecture

AlexNet consists of 8 learned layers: 5 convolutional and 3 fully connected.

```
Input: 224×224×3 RGB image
    ↓
Conv1: 96 filters (11×11, stride 4) → 55×55×96
ReLU → MaxPool (3×3, stride 2) → LRN
    ↓ (27×27×96)
Conv2: 256 filters (5×5, pad 2) → 27×27×256
ReLU → MaxPool (3×3, stride 2) → LRN
    ↓ (13×13×256)
Conv3: 384 filters (3×3, pad 1) → 13×13×384
ReLU
    ↓
Conv4: 384 filters (3×3, pad 1) → 13×13×384
ReLU
    ↓
Conv5: 256 filters (3×3, pad 1) → 13×13×256
ReLU → MaxPool (3×3, stride 2)
    ↓ (6×6×256 = 9216)
FC6: 4096 units
ReLU → Dropout (0.5)
    ↓
FC7: 4096 units
ReLU → Dropout (0.5)
    ↓
FC8: 1000 units (ImageNet classes)
Softmax
    ↓
Output: Class probabilities
```

### Architecture Summary

| Layer | Type | Output Size | Kernel | Stride | Padding | Parameters |
|-------|------|-------------|--------|--------|---------|------------|
| Input | Image | 224×224×3 | - | - | - | 0 |
| Conv1 | Conv+ReLU+Pool | 27×27×96 | 11×11 | 4 | 0 | 34,944 |
| Conv2 | Conv+ReLU+Pool | 13×13×256 | 5×5 | 1 | 2 | 307,456 |
| Conv3 | Conv+ReLU | 13×13×384 | 3×3 | 1 | 1 | 885,120 |
| Conv4 | Conv+ReLU | 13×13×384 | 3×3 | 1 | 1 | 1,327,488 |
| Conv5 | Conv+ReLU+Pool | 6×6×256 | 3×3 | 1 | 1 | 884,992 |
| FC6 | Dense+ReLU+Dropout | 4096 | - | - | - | 37,752,832 |
| FC7 | Dense+ReLU+Dropout | 4096 | - | - | - | 16,781,312 |
| FC8 | Dense | 1000 | - | - | - | 4,097,000 |

**Total parameters**: ~62 million (mostly in fully connected layers)

## Key Innovations

### 1. ReLU Activation

AlexNet was the first successful large-scale CNN to use **ReLU** (Rectified Linear Unit) instead of tanh or sigmoid.

$$\text{ReLU}(x) = \max(0, x)$$

**Advantages**:
- **Faster training**: 6x faster than tanh for same accuracy
- **No saturation**: Doesn't suffer from vanishing gradients for positive values
- **Simple computation**: Just thresholding at zero
- **Better gradient flow**: Enables training deeper networks

**Comparison**:

| Activation | Formula | Range | Gradient Issue |
|------------|---------|-------|----------------|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | (0, 1) | Vanishing gradients |
| Tanh | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | (-1, 1) | Vanishing gradients |
| ReLU | $\max(0, x)$ | [0, ∞) | Dying ReLU (for x<0) |

### 2. Dropout Regularization

**Dropout** randomly sets activations to zero during training with probability $p$ (typically 0.5).

$$\tilde{h}_i = \begin{cases}
0 & \text{with probability } p \\
h_i & \text{with probability } 1-p
\end{cases}$$

At test time, multiply activations by $(1-p)$ to account for all neurons being active.

**Purpose**:
- Prevents overfitting
- Forces network to learn robust features
- Acts like ensemble of multiple networks
- Applied to FC6 and FC7 with $p=0.5$

**Effect**: Reduced overfitting despite 60M parameters trained on only 1.2M images.

### 3. Local Response Normalization (LRN)

Normalizes activations across nearby feature maps:

$$b_{x,y}^i = a_{x,y}^i \left/ \left(k + \alpha \sum_{j=\max(0, i-n/2)}^{\min(N-1, i+n/2)} (a_{x,y}^j)^2\right)^\beta \right.$$

where:
- $a_{x,y}^i$ is the activity of neuron at position $(x,y)$ in feature map $i$
- $n$ is the normalization window size
- $k, \alpha, \beta$ are hyperparameters (k=2, α=10⁻⁴, β=0.75, n=5)

**Purpose**: Implements lateral inhibition (neurons competing with neighbors)

**Note**: LRN is mostly obsolete now, replaced by batch normalization.

### 4. Overlapping Pooling

Unlike traditional non-overlapping pooling, AlexNet used **overlapping max pooling**:
- Window size: 3×3
- Stride: 2 (not 3)

This creates overlap between pooling windows, slightly reducing overfitting.

**Output size calculation**:

$$\text{Output} = \left\lfloor \frac{\text{Input} - \text{Pool Size}}{\text{Stride}} \right\rfloor + 1$$

Example: $\frac{55 - 3}{2} + 1 = 27$

### 5. Data Augmentation

Crucial for preventing overfitting with limited data:

**Training time augmentation**:
1. **Random crops**: Extract random 224×224 patches from 256×256 images
2. **Horizontal flips**: Mirror images randomly
3. **PCA color augmentation**: Alter RGB channel intensities

**PCA color augmentation**:
Add multiples of principal components of RGB pixel values:

$$I'_{xy} = I_{xy} + [p_1, p_2, p_3][\alpha_1\lambda_1, \alpha_2\lambda_2, \alpha_3\lambda_3]^T$$

where:
- $p_i$ are eigenvectors of RGB covariance matrix
- $\lambda_i$ are eigenvalues
- $\alpha_i \sim \mathcal{N}(0, 0.1)$ are random variables

**Test time**: Average predictions over 10 crops (4 corners + center, plus horizontal flips)

### 6. GPU Training

**First large-scale CNN trained on GPUs** (2 NVIDIA GTX 580 GPUs with 3GB memory each).

**Model parallelism**:
- Network split across 2 GPUs
- GPUs communicate only at certain layers
- Conv2, Conv4, Conv5: each GPU processes half the filters from previous layer
- Conv3, FC layers: each GPU sees all feature maps from previous layer

This parallelization enabled training on 1.2M images in reasonable time (~6 days).

## Mathematical Details

### Convolution Operation

For input $X$ and filter $W$:

$$Y_{i,j,k} = \sum_{c=1}^{C} \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} W_{k,c,m,n} \cdot X_{c, s \cdot i + m, s \cdot j + n} + b_k$$

where:
- $(i,j)$ are output spatial coordinates
- $k$ is output channel
- $c$ is input channel
- $s$ is stride
- $K$ is kernel size

### Output Dimension Calculation

$$\text{Output Size} = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$

where:
- $W$ is input width/height
- $K$ is kernel size
- $P$ is padding
- $S$ is stride

### Receptive Field

The receptive field grows with each layer:

**AlexNet receptive field growth**:
- Conv1: 11×11
- After Pool1: 19×19
- Conv2: 51×51
- After Pool2: 67×67
- Conv5: 195×195
- After Pool5: 227×227 (entire input!)

## Implementation

### Modern PyTorch Implementation

```python
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

### Using Pre-trained AlexNet

```python
from torchvision import models

# Load pre-trained model
model = models.alexnet(pretrained=True)

# For feature extraction (freeze weights)
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for new task
model.classifier[6] = nn.Linear(4096, num_new_classes)

# Now train only the final layer
optimizer = torch.optim.Adam(model.classifier[6].parameters(), lr=0.001)
```

## Training Details

### Dataset

**ImageNet ILSVRC-2010**:
- 1.2 million training images
- 50,000 validation images
- 150,000 test images
- 1000 object categories
- Variable resolution (downsampled to 256×256)

### Training Hyperparameters

- **Optimizer**: Stochastic Gradient Descent (SGD) with momentum
- **Batch size**: 128
- **Momentum**: 0.9
- **Weight decay**: 0.0005
- **Learning rate**: 0.01, divided by 10 when validation error plateaus
- **Training time**: ~6 days on 2 GPUs

**Weight decay as regularization**:

$$L_{total} = L_{data} + \lambda \sum_{i} w_i^2$$

where $\lambda = 0.0005$.

### Initialization

- **Weights**: Gaussian distribution with mean 0, std 0.01
- **Biases**:
  - Layers 2, 4, 5, and FC layers: initialized to 1
  - Remaining layers: initialized to 0

### Loss Function

**Cross-entropy loss** for multi-class classification:

$$L = -\frac{1}{N}\sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

where:
- $N$ is batch size
- $C$ is number of classes
- $y_{i,c}$ is true label (one-hot encoded)
- $\hat{y}_{i,c}$ is predicted probability

## Impact and Legacy

### Competition Results (ILSVRC 2012)

| Rank | Method | Top-5 Error |
|------|--------|-------------|
| 1 | AlexNet | 15.3% |
| 2 | Traditional CV | 26.2% |

**Improvement**: 10.9 percentage points (42% relative improvement)

### What AlexNet Proved

1. **Deep learning works**: Given enough data and compute, deep networks outperform hand-crafted features
2. **Depth matters**: Deeper networks learn better representations
3. **GPUs enable scale**: Hardware acceleration is crucial for deep learning
4. **Data augmentation is essential**: Prevents overfitting with limited data

### Influence on Later Architectures

**Direct descendants**:
- **ZFNet** (2013): Visualized and improved AlexNet
- **VGG** (2014): Took depth to 16-19 layers
- **GoogLeNet** (2014): More efficient architecture
- **ResNet** (2015): Enabled 100+ layer networks

**Techniques still used**:
- ReLU activation (universal)
- Dropout (common regularization)
- Data augmentation (standard practice)
- GPU training (essential)

### What's Changed Since AlexNet

**Improvements**:
- **Batch normalization** replaced LRN
- **Smaller kernels**: 3×3 instead of 11×11, 5×5
- **Deeper networks**: 50-200 layers common
- **Better optimizers**: Adam, AdamW instead of SGD
- **Better architectures**: ResNet, EfficientNet, ViT
- **More data**: ImageNet-21k, web-scale datasets

**What stayed**:
- ReLU and variants (LeakyReLU, ELU, etc.)
- Dropout for regularization
- Data augmentation strategies
- Transfer learning paradigm

## Comparison with Other Architectures

| Architecture | Year | Layers | Parameters | Top-5 Error | Key Innovation |
|--------------|------|--------|------------|-------------|----------------|
| AlexNet | 2012 | 8 | 60M | 15.3% | ReLU, Dropout, GPU |
| VGG-16 | 2014 | 16 | 138M | 7.3% | Very deep, small filters |
| GoogLeNet | 2014 | 22 | 6.8M | 6.7% | Inception modules |
| ResNet-50 | 2015 | 50 | 25M | 3.6% | Residual connections |
| EfficientNet-B7 | 2019 | - | 66M | 2.0% | Compound scaling |

## Limitations

1. **Large fully connected layers**: Most parameters in FC layers (not efficient)
2. **Large input stride**: 11×11 kernels with stride 4 lose information
3. **LRN**: Doesn't help much, adds computation
4. **No batch normalization**: Makes training less stable
5. **Simple architecture**: No skip connections or multi-scale features

Despite limitations, AlexNet remains historically significant as the architecture that sparked the deep learning revolution.

## References

- Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton. "ImageNet Classification with Deep Convolutional Neural Networks" NIPS (2012)
- [Original Paper](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
- [PyTorch Implementation](https://pytorch.org/vision/stable/models.html#alexnet)
