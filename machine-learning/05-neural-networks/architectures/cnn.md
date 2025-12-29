# Convolutional Neural Networks (CNN)

## Table of Contents

1. [Introduction](#introduction)
2. [Motivation and Intuition](#motivation-and-intuition)
3. [Convolutional Layers](#convolutional-layers)
4. [Pooling Layers](#pooling-layers)
5. [CNN Architecture](#cnn-architecture)
6. [Parameter Calculation](#parameter-calculation)
7. [Receptive Field](#receptive-field)
8. [Common Architectures](#common-architectures)
9. [Practical Considerations](#practical-considerations)

## Introduction

**Convolutional Neural Networks (CNNs)** are a specialized type of neural network designed to process data with grid-like topology, such as images. CNNs use a mathematical operation called **convolution** instead of general matrix multiplication in at least one of their layers.

**Key innovations:**
- **Local connectivity** - neurons connect only to a local region of the input
- **Parameter sharing** - same weights used across different positions
- **Translation invariance** - features detected anywhere in the input
- **Hierarchical feature learning** - low-level to high-level features

**Primary applications:**
- Image classification
- Object detection and localization
- Image segmentation
- Video analysis
- Medical image analysis
- Face recognition

## Motivation and Intuition

### Why Not Regular Feedforward Networks?

**Problems with fully connected networks for images:**

1. **Too many parameters** - A 224×224 RGB image has 150,528 inputs. A single hidden layer with 1,000 neurons would require 150 million parameters!

2. **No spatial structure** - Flattening an image loses spatial relationships between pixels

3. **No translation invariance** - A cat in top-left corner vs bottom-right requires learning the same features twice

### Biological Inspiration

CNNs are inspired by the **visual cortex**:
- **Receptive fields** - neurons respond to stimuli in a limited region of visual field
- **Simple cells** - detect edges and orientations in specific locations
- **Complex cells** - similar to simple cells but with spatial invariance
- **Hierarchical processing** - from edges to shapes to objects

## Convolutional Layers

### Convolution Operation

A **2D convolution** applies a filter (kernel) across the input:

$$S(i,j) = (I * K)(i,j) = \sum_m \sum_n I(i+m, j+n) \cdot K(m,n)$$

where:
- $I$ - input image
- $K$ - kernel (filter)
- $S$ - feature map (output)
- $(i,j)$ - position in output
- $(m,n)$ - position in kernel

### Key Components

**Filter/Kernel:**
- Small matrix of learnable weights (e.g., 3×3, 5×5, 7×7)
- Slides across the input to detect features
- Multiple filters learn different features

**Stride:**
- Step size when sliding the filter
- Stride = 1: move one pixel at a time
- Stride = 2: move two pixels, reducing output size

**Padding:**
- Add borders to the input
- **Valid padding** (no padding): output size < input size
- **Same padding**: output size = input size (for stride=1)

### Output Dimension Calculation

For a convolutional layer:

$$n_{out} = \left\lfloor\frac{n_{in} + 2p - k}{s}\right\rfloor + 1$$

where:
- $n_{in}$ - input size (width or height)
- $p$ - padding
- $k$ - kernel size
- $s$ - stride
- $n_{out}$ - output size

**Example:**
- Input: 32×32
- Kernel: 5×5
- Padding: 2
- Stride: 1

$$n_{out} = \left\lfloor\frac{32 + 2(2) - 5}{1}\right\rfloor + 1 = 32$$

### Feature Maps

**Multiple channels:**
- Input: $H \times W \times C_{in}$ (height × width × channels)
- Kernel: $k \times k \times C_{in} \times C_{out}$
- Output: $H' \times W' \times C_{out}$

Each filter produces one feature map. With $C_{out}$ filters, we get $C_{out}$ feature maps.

**Example visualization:**

```
Input: 32×32×3 (RGB image)
         ↓
Conv1: 32 filters, 3×3, stride=1, padding=1
         ↓
Output: 32×32×32 (32 feature maps)
```

### Mathematical Formulation

For a convolutional layer with activation:

$$\mathbf{z}^{[l]}_{i,j,k} = \sum_{c=1}^{C_{in}}\sum_{m=0}^{k-1}\sum_{n=0}^{k-1} \mathbf{W}^{[l]}_{m,n,c,k} \cdot \mathbf{a}^{[l-1]}_{i \cdot s + m, j \cdot s + n, c} + b^{[l]}_k$$

$$\mathbf{a}^{[l]}_{i,j,k} = f(\mathbf{z}^{[l]}_{i,j,k})$$

where:
- $(i,j)$ - spatial position in output
- $k$ - output channel index
- $c$ - input channel index
- $s$ - stride

### Example Implementation

```python
import torch
import torch.nn as nn

# Single convolutional layer
conv = nn.Conv2d(
    in_channels=3,      # RGB input
    out_channels=64,    # 64 filters
    kernel_size=3,      # 3×3 kernels
    stride=1,
    padding=1
)

# Input: batch_size × channels × height × width
x = torch.randn(32, 3, 224, 224)  # 32 images, 224×224 RGB
output = conv(x)
print(output.shape)  # torch.Size([32, 64, 224, 224])
```

## Pooling Layers

**Pooling** reduces the spatial dimensions of feature maps, providing:
- Translation invariance
- Computational efficiency
- Overfitting reduction

### Max Pooling

Takes the **maximum value** in each pooling window:

$$y_{i,j,k} = \max_{(m,n) \in R_{i,j}} a_{m,n,k}$$

where $R_{i,j}$ is the pooling region.

**Common configuration:** 2×2 window, stride=2 (reduces size by half)

### Average Pooling

Takes the **average value** in each pooling window:

$$y_{i,j,k} = \frac{1}{|R_{i,j}|}\sum_{(m,n) \in R_{i,j}} a_{m,n,k}$$

### Global Pooling

**Global Average Pooling (GAP):**
- Pools each feature map to a single value
- Average across entire spatial dimensions
- Often used before final classification layer
- Reduces overfitting compared to fully connected layers

$$y_k = \frac{1}{H \times W}\sum_{i=1}^{H}\sum_{j=1}^{W} a_{i,j,k}$$

### Pooling Output Size

$$n_{out} = \left\lfloor\frac{n_{in} - k}{s}\right\rfloor + 1$$

**Example:**
- Input: 32×32
- Pooling: 2×2, stride=2
- Output: 16×16

### Example Implementation

```python
# Max pooling
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Average pooling
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# Global average pooling
gap = nn.AdaptiveAvgPool2d((1, 1))

x = torch.randn(32, 64, 56, 56)
pooled = max_pool(x)
print(pooled.shape)  # torch.Size([32, 64, 28, 28])
```

## CNN Architecture

### Typical Architecture Pattern

```
Input → [Conv → Activation → Pool] × N → Flatten → [FC → Activation] × M → Output
```

**Standard flow:**
1. **Convolutional blocks** - extract features
   - Spatial dimensions decrease
   - Number of channels increases
2. **Flatten** - convert to 1D vector
3. **Fully connected layers** - classification
4. **Output layer** - final predictions

### Example Architecture

**Simple CNN for MNIST:**

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)    # 28×28×1 → 28×28×32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)   # 14×14×32 → 14×14×64

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)  # Reduces size by half

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Activation
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Block 1
        x = self.relu(self.conv1(x))    # 28×28×32
        x = self.pool(x)                 # 14×14×32

        # Block 2
        x = self.relu(self.conv2(x))    # 14×14×64
        x = self.pool(x)                 # 7×7×64

        # Flatten
        x = x.view(-1, 64 * 7 * 7)      # 3136

        # Fully connected
        x = self.relu(self.fc1(x))      # 128
        x = self.dropout(x)
        x = self.fc2(x)                  # 10

        return x
```

### Modern Design Principles

**1×1 Convolutions:**
- Reduce/increase channel dimensions
- Add non-linearity without changing spatial size
- Used in Inception and ResNet

**Batch Normalization:**
- Normalize inputs to each layer
- Accelerates training
- Reduces sensitivity to initialization
- Place after Conv, before activation

**Skip Connections (ResNets):**
- Allow gradients to flow directly through network
- Enable training very deep networks
- $\mathbf{y} = f(\mathbf{x}) + \mathbf{x}$

## Parameter Calculation

### Convolutional Layer Parameters

$$\text{Parameters} = (k \times k \times C_{in} + 1) \times C_{out}$$

where:
- $k$ - kernel size
- $C_{in}$ - input channels
- $C_{out}$ - output channels
- $+1$ - for bias term

**Example:**
- Kernel: 3×3
- Input channels: 64
- Output channels: 128

Parameters = $(3 \times 3 \times 64 + 1) \times 128 = 73,856$

### Fully Connected Layer Parameters

$$\text{Parameters} = (n_{in} + 1) \times n_{out}$$

**Example:**
- Input: 7×7×512 = 25,088
- Output: 4,096

Parameters = $25,088 \times 4,096 + 4,096 = 102,764,544$

**This is why modern CNNs avoid large FC layers!**

### Memory Calculation

**Activations memory** (for one sample in forward pass):

$$\text{Memory} = H \times W \times C \times 4 \text{ bytes}$$

(assuming 32-bit floats)

**Example:** 224×224×64 feature map = 12.8 MB per sample

## Receptive Field

**Receptive field** - the region of the input that affects a particular neuron in a deeper layer.

### Calculation

For a sequence of convolutional layers:

$$RF_l = RF_{l-1} + (k_l - 1) \times \prod_{i=1}^{l-1} s_i$$

where:
- $RF_l$ - receptive field at layer $l$
- $k_l$ - kernel size at layer $l$
- $s_i$ - stride at layer $i$

**Example:**
- Layer 1: kernel=3, stride=1 → RF = 3
- Layer 2: kernel=3, stride=1 → RF = 5
- Layer 3: kernel=3, stride=2 → RF = 7
- Layer 4: kernel=3, stride=1 → RF = 11

### Importance

- Larger receptive field captures more context
- Deep networks build large receptive fields gradually
- Essential for capturing object-level features

## Common Architectures

### LeNet-5 (1998)

**First successful CNN** for digit recognition:

```
Input(32×32×1) → Conv(6@5×5) → Pool(2×2) →
Conv(16@5×5) → Pool(2×2) →
FC(120) → FC(84) → Output(10)
```

### VGG (2014)

**Key idea:** Use only 3×3 convolutions, go deeper

```
Input(224×224×3)
↓
[Conv3×3(64)] × 2 → Pool
[Conv3×3(128)] × 2 → Pool
[Conv3×3(256)] × 3 → Pool
[Conv3×3(512)] × 3 → Pool
[Conv3×3(512)] × 3 → Pool
↓
FC(4096) → FC(4096) → FC(1000)
```

**Advantages:** Simple, uniform architecture
**Disadvantages:** 138M parameters, memory intensive

### ResNet Basics

**Key innovation:** Skip connections solve vanishing gradient

**Residual block:**

$$\mathbf{y} = f(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

Instead of learning $H(\mathbf{x})$, learn residual $f(\mathbf{x}) = H(\mathbf{x}) - \mathbf{x}$

## Practical Considerations

### Data Preprocessing

**Normalization:**
```python
# ImageNet normalization
transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

**Augmentation:**
```python
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(224, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
```

### Hyperparameter Selection

**Kernel size:**
- **3×3** - most common, can capture complex patterns when stacked
- **5×5, 7×7** - larger receptive field, used in early layers
- **1×1** - channel-wise operations, dimensionality reduction

**Number of filters:**
- Common pattern: 64 → 128 → 256 → 512
- Double filters when spatial size halves (via pooling)

**Depth:**
- Start shallow, gradually increase
- Modern networks: 18-152 layers (ResNets)
- Diminishing returns beyond certain depth without skip connections

### Training Tips

**Learning rate:**
- Start with 0.001 (Adam) or 0.1 (SGD with momentum)
- Use learning rate scheduling (step decay, cosine annealing)

**Batch size:**
- Larger batches for CNNs: 64-256
- Limited by GPU memory
- Batch normalization sensitive to batch size

**Regularization:**
- Dropout: 0.5 in FC layers, 0.2-0.3 in Conv layers
- L2 weight decay: 1e-4 to 1e-5
- Data augmentation is crucial

### Common Issues

**Overfitting:**
- More data augmentation
- Stronger regularization
- Reduce model complexity
- Early stopping

**Vanishing gradients:**
- Use ReLU instead of sigmoid/tanh
- Batch normalization
- Skip connections (ResNet)
- Proper initialization (He initialization)

**Memory issues:**
- Reduce batch size
- Reduce input size
- Use gradient checkpointing
- Mixed precision training

### Transfer Learning

**Using pre-trained models:**

```python
import torchvision.models as models

# Load pre-trained ResNet
model = models.resnet50(pretrained=True)

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Fine-tune on new dataset
```

**When to use:**
- Small datasets (<10k images)
- Similar domain to pre-training data
- Limited computational resources

## References

- LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition.
- Krizhevsky, A., Sutskever, I., & Hinton, G. (2012). ImageNet classification with deep convolutional neural networks.
- Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition.
- He, K., et al. (2016). Deep residual learning for image recognition.
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
