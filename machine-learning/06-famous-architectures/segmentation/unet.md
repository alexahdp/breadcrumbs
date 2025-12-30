# U-Net - Convolutional Networks for Biomedical Image Segmentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Key Components](#key-components)
4. [Mathematical Details](#mathematical-details)
5. [Implementation](#implementation)
6. [Training Strategy](#training-strategy)
7. [Applications and Impact](#applications-and-impact)
8. [Variants](#variants)

## Overview

**U-Net** is a convolutional neural network architecture designed for biomedical image segmentation. It achieves precise segmentation with very few training images through extensive data augmentation.

**Authors**: Olaf Ronneberger, Philipp Fischer, Thomas Brox (University of Freiburg)

**Year**: 2015

**Original task**: Cell segmentation in microscopy images

**Achievement**: Won ISBI 2015 Cell Tracking Challenge with large margin

**Key innovation**: **Encoder-decoder architecture with skip connections** - combines high-resolution features from contracting path with upsampled features from expanding path

**Why "U-Net"**: The architecture's shape resembles the letter "U"

## Architecture

### Overall Structure

U-Net consists of two main paths:

1. **Contracting path** (left side): Captures context through downsampling
2. **Expanding path** (right side): Enables precise localization through upsampling
3. **Skip connections**: Copy features from contracting to expanding path

```
Input: 572×572×1
          ↓
    ┌─────────────┐
    │   Encoder   │  Conv→ReLU→Conv→ReLU→MaxPool (repeat)
    │  (Contract) │  64 → 128 → 256 → 512 → 1024
    └─────────────┘
          ↓
    ┌─────────────┐
    │  Bottleneck │  1024 channels
    └─────────────┘
          ↓
    ┌─────────────┐
    │   Decoder   │  UpConv→Concat→Conv→ReLU (repeat)
    │   (Expand)  │  1024 → 512 → 256 → 128 → 64
    └─────────────┘
          ↓
Output: 388×388×2
```

### Detailed Architecture

```
                    Input 572×572×1
                          ↓
    ┌──────────────── Conv 3×3, ReLU → Conv 3×3, ReLU (64) ────────┐
    │                     ↓ MaxPool 2×2                            │
    │        ┌────── Conv 3×3, ReLU → Conv 3×3, ReLU (128) ────┐   │
    │        │            ↓ MaxPool 2×2                         │   │
    │        │   ┌── Conv 3×3, ReLU → Conv 3×3, ReLU (256) ──┐ │   │
    │        │   │        ↓ MaxPool 2×2                       │ │   │
    │        │   │  ┌ Conv 3×3, ReLU → Conv 3×3, ReLU (512) ┐ │ │   │
    │        │   │  │     ↓ MaxPool 2×2                      │ │ │   │
    │        │   │  │ Conv 3×3, ReLU → Conv 3×3, ReLU (1024)│ │ │   │
    │        │   │  │     ↓ UpConv 2×2                       │ │ │   │
    │        │   │  └→ Concat → Conv 3×3 → Conv 3×3 (512) ←─┘ │ │   │
    │        │   │        ↓ UpConv 2×2                         │ │   │
    │        │   └────→ Concat → Conv 3×3 → Conv 3×3 (256) ←──┘ │   │
    │        │            ↓ UpConv 2×2                           │   │
    │        └────────→ Concat → Conv 3×3 → Conv 3×3 (128) ←────┘   │
    │                    ↓ UpConv 2×2                               │
    └────────────────→ Concat → Conv 3×3 → Conv 3×3 (64) ←─────────┘
                          ↓
                    Conv 1×1 (2 classes)
                          ↓
                   Output 388×388×2
```

### Architecture Details by Level

| Level | Operation | Input Size | Output Size | Channels |
|-------|-----------|------------|-------------|----------|
| 0 | Input | 572×572 | 572×572 | 1 |
| 1 | Conv→Conv | 572×572 | 568×568 | 64 |
| 1 | MaxPool | 568×568 | 284×284 | 64 |
| 2 | Conv→Conv | 284×284 | 280×280 | 128 |
| 2 | MaxPool | 280×280 | 140×140 | 128 |
| 3 | Conv→Conv | 140×140 | 136×136 | 256 |
| 3 | MaxPool | 136×136 | 68×68 | 256 |
| 4 | Conv→Conv | 68×68 | 64×64 | 512 |
| 4 | MaxPool | 64×64 | 32×32 | 512 |
| 5 | Conv→Conv (bottleneck) | 32×32 | 28×28 | 1024 |
| 4↑ | UpConv | 28×28 | 56×56 | 512 |
| 4↑ | Concat + Conv→Conv | 56×56 + crop | 52×52 | 512 |
| 3↑ | UpConv | 52×52 | 104×104 | 256 |
| 3↑ | Concat + Conv→Conv | 104×104 + crop | 100×100 | 256 |
| 2↑ | UpConv | 100×100 | 200×200 | 128 |
| 2↑ | Concat + Conv→Conv | 200×200 + crop | 196×196 | 128 |
| 1↑ | UpConv | 196×196 | 392×392 | 64 |
| 1↑ | Concat + Conv→Conv | 392×392 + crop | 388×388 | 64 |
| Out | Conv 1×1 | 388×388 | 388×388 | 2 |

**Note**: Original U-Net uses **valid convolutions** (no padding), so spatial dimensions decrease with each conv layer.

## Key Components

### 1. Contracting Path (Encoder)

**Purpose**: Capture context and semantic information

**Structure** (repeated):
- 3×3 convolution (unpadded)
- ReLU activation
- 3×3 convolution (unpadded)
- ReLU activation
- 2×2 max pooling (stride 2) for downsampling

**Channel progression**: 64 → 128 → 256 → 512 → 1024

**Effect**:
- Spatial resolution decreases
- Receptive field increases
- Feature depth increases

### 2. Bottleneck

**Deepest layer** with highest feature depth (1024 channels)

Captures the most abstract, high-level semantic information.

### 3. Expanding Path (Decoder)

**Purpose**: Enable precise localization

**Structure** (repeated):
- 2×2 up-convolution (transposed convolution) - halves feature channels
- Concatenation with corresponding cropped feature map from contracting path
- 3×3 convolution (unpadded)
- ReLU activation
- 3×3 convolution (unpadded)
- ReLU activation

**Channel progression**: 1024 → 512 → 256 → 128 → 64

**Effect**:
- Spatial resolution increases
- Combines high-level semantic info with low-level spatial info

### 4. Skip Connections

**Critical innovation**: Copy and concatenate features from encoder to decoder

**Purpose**:
- Preserve spatial information lost during downsampling
- Help gradients flow during backpropagation
- Combine coarse semantic info with fine spatial detail

**Implementation**: Features from encoder are **cropped** to match decoder size (due to valid convolutions)

### 5. Final Layer

**1×1 convolution**: Maps 64 feature channels to desired number of classes

For binary segmentation: 2 channels (background and foreground)

## Mathematical Details

### Skip Connection Operation

At each level of decoder:

$$\text{Output}_i = \text{Conv}(\text{Concat}(\text{UpConv}(x_i), \text{Crop}(x_{enc,i})))$$

where:
- $x_i$ is input from previous decoder level
- $x_{enc,i}$ is feature map from corresponding encoder level
- $\text{Crop}$ ensures spatial dimensions match
- $\text{Concat}$ along channel dimension

### Up-Convolution (Transposed Convolution)

**Transposed convolution** upsamples feature maps:

For 2×2 kernel, stride 2:

$$y_{i,j} = \sum_{m,n} x_{\lfloor i/2 \rfloor + m, \lfloor j/2 \rfloor + n} \cdot w_{m,n}$$

**Effect**: Doubles spatial dimensions, halves channels

**Alternative**: Can use bilinear upsampling + regular convolution

### Loss Function

**Pixel-wise softmax** over final feature map:

$$p_k(x) = \frac{\exp(a_k(x))}{\sum_{k'=1}^{K} \exp(a_{k'}(x))}$$

where:
- $a_k(x)$ is activation in channel $k$ at pixel $x$
- $K$ is number of classes

**Cross-entropy loss** with class weighting:

$$L = -\sum_{x \in \Omega} w(x) \log(p_{l(x)}(x))$$

where:
- $\Omega$ is set of all pixels
- $l(x)$ is true label at pixel $x$
- $w(x)$ is weight map

### Weight Map for Separation

To separate touching objects, U-Net introduces **weight map**:

$$w(x) = w_c(x) + w_0 \cdot \exp\left(-\frac{(d_1(x) + d_2(x))^2}{2\sigma^2}\right)$$

where:
- $w_c(x)$ is class balancing weight
- $d_1(x)$ is distance to nearest cell boundary
- $d_2(x)$ is distance to second nearest cell boundary
- $w_0, \sigma$ are hyperparameters (typically $w_0=10, \sigma=5$)

**Purpose**: Force network to learn small separation borders between touching cells

## Implementation

### PyTorch U-Net

```python
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Two consecutive 3×3 convolutions with ReLU"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(UNet, self).__init__()

        # Encoder (contracting path)
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder (expanding path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)  # 1024 from concat

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Final output
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)  # Skip connection
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        return self.out(dec1)
```

### U-Net with Batch Normalization

Modern variant with batch normalization:

```python
class DoubleConvBN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
```

### Using U-Net for Segmentation

```python
# Create model
model = UNet(in_channels=3, out_channels=num_classes)

# Loss function
criterion = nn.CrossEntropyLoss()

# For imbalanced classes, use weighted loss
class_weights = torch.FloatTensor([1.0, 5.0])  # Weight rare class higher
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Or use Dice loss for better boundary detection
class DiceLoss(nn.Module):
    def forward(self, pred, target):
        smooth = 1.
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        return 1 - ((2. * intersection + smooth) /
                   (pred.sum() + target.sum() + smooth))

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for images, masks in dataloader:
    outputs = model(images)
    loss = criterion(outputs, masks)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Inference
model.eval()
with torch.no_grad():
    prediction = model(image)
    segmentation = torch.argmax(prediction, dim=1)
```

## Training Strategy

### Data Augmentation

**Critical** for U-Net since medical datasets are small (30 training images in original paper!)

**Augmentations**:
1. **Elastic deformations**: Random displacement fields
2. **Rotation**: Random angles
3. **Scaling**: Random zoom
4. **Flipping**: Horizontal and vertical
5. **Gaussian noise**: Add random noise
6. **Brightness/contrast**: Random adjustments

**Elastic deformation** formula:

$$\phi(x, y) = (x + \alpha \cdot G_\sigma(x,y), y + \alpha \cdot G_\sigma(x,y))$$

where $G_\sigma$ is Gaussian filter with std $\sigma$, and $\alpha$ controls deformation strength.

### Overlap-Tile Strategy

For large images, use **overlap-tile strategy**:

1. Extract overlapping patches (tiles)
2. Segment each patch
3. Stitch predictions together
4. Overlapping regions provide context for seamless segmentation

**Mirroring**: Mirror image at borders to provide context for edge pixels

### Training Configuration

**Original paper**:
- **Optimizer**: SGD with momentum (0.99)
- **Learning rate**: High (unspecified), decreased over time
- **Weight initialization**: Gaussian with std $\sqrt{2/N}$ where $N$ is number of incoming nodes
- **Batch size**: 1 (due to data augmentation, each image is different)
- **Loss**: Weighted cross-entropy

**Modern training**:
- **Optimizer**: Adam with lr=1e-4
- **Batch normalization**: After each conv layer
- **Dropout**: 0.5 in bottleneck (for regularization)
- **Mixed loss**: Combination of Cross-Entropy and Dice loss

### Dice Coefficient

**Dice coefficient** measures overlap between prediction and ground truth:

$$\text{Dice} = \frac{2|A \cap B|}{|A| + |B|}$$

Range: [0, 1], where 1 is perfect overlap.

**Dice loss**: $L_{Dice} = 1 - \text{Dice}$

## Applications and Impact

### Original Application

**Biomedical image segmentation**:
- Cell segmentation in microscopy
- Neuron detection in EM stacks
- Cell tracking

**Results on ISBI challenges**:
- PhC-U373: IOU 92% (previous best: 83%)
- DIC-HeLa: IOU 77.5% (previous best: 46%)

### Wide Adoption

**Medical imaging**:
- Organ segmentation (liver, kidney, brain)
- Tumor detection and segmentation
- Blood vessel segmentation
- Lesion segmentation
- X-ray, CT, MRI analysis

**Beyond medical**:
- Satellite image segmentation
- Autonomous driving (road/lane detection)
- Industrial defect detection
- Document analysis

### Why U-Net Became Standard

1. **Works with small datasets**: Data augmentation + architecture design
2. **Precise boundaries**: Skip connections preserve spatial detail
3. **End-to-end**: Direct pixel-to-pixel mapping
4. **Fast**: Efficient inference
5. **Flexible**: Easy to modify for different tasks
6. **Simple**: Clear, understandable architecture

## Variants

### U-Net++

**Nested U-Net** with dense skip connections:
- Skip connections at multiple levels
- Reduces semantic gap between encoder and decoder
- Improved performance but slower

### Attention U-Net

Adds **attention gates** to skip connections:
- Learn to focus on relevant features
- Suppress irrelevant regions
- Better for challenging datasets

### 3D U-Net

Extends U-Net to **3D volumetric data**:
- 3D convolutions instead of 2D
- For CT/MRI volumes
- Segment 3D structures

### ResUNet

Combines U-Net with **ResNet**:
- Residual blocks instead of plain convolutions
- Better gradient flow
- Can go deeper

### U-Net with Different Encoders

Replace encoder with pre-trained backbone:
- **ResNet encoder**: Better feature extraction
- **EfficientNet encoder**: More efficient
- **Transfer learning**: Leverage ImageNet pre-training

```python
import segmentation_models_pytorch as smp

# U-Net with ResNet34 encoder
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=num_classes
)
```

## Comparison with Other Segmentation Methods

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| FCN | Fully convolutional | First end-to-end segmentation | Coarse outputs |
| SegNet | Encoder-decoder | Memory efficient | Less precise |
| **U-Net** | Encoder-decoder + skip | Precise, works on small data | Needs augmentation |
| DeepLab | Atrous convolutions | Multi-scale | More complex |
| Mask R-CNN | Instance segmentation | Handles instances | Slower, more complex |

## Key Takeaways

**Architecture**: Symmetric encoder-decoder with skip connections

**Innovation**: Skip connections bridge semantic gap, enabling precise localization

**Strength**: Achieves excellent results with very few training images

**Applications**: Standard for medical image segmentation, widely used elsewhere

**Legacy**: Inspired countless variants and remains highly relevant today

## References

- Olaf Ronneberger, Philipp Fischer, Thomas Brox. "U-Net: Convolutional Networks for Biomedical Image Segmentation" MICCAI (2015)
- [Original Paper](https://arxiv.org/abs/1505.04597)
- [U-Net implementation collection](https://github.com/milesial/Pytorch-UNet)
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
