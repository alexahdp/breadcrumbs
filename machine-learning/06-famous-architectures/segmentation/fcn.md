# FCN - Fully Convolutional Networks for Semantic Segmentation

## Table of Contents

1. [Overview](#overview)
2. [Core Concept](#core-concept)
3. [Architecture](#architecture)
4. [Key Innovations](#key-innovations)
5. [Mathematical Details](#mathematical-details)
6. [Implementation](#implementation)
7. [Training and Evaluation](#training-and-evaluation)
8. [Impact and Limitations](#impact-and-limitations)

## Overview

**FCN** (Fully Convolutional Networks) pioneered end-to-end, pixel-to-pixel learning for semantic segmentation. It was the first to show that classification networks (like VGG, AlexNet) could be adapted for dense prediction.

**Authors**: Jonathan Long, Evan Shelhamer, Trevor Darrell (UC Berkeley)

**Year**: 2015

**Achievement**: State-of-the-art results on PASCAL VOC, NYU-D, SIFT Flow

**Key innovation**: **Replace fully connected layers with convolutional layers** and use transposed convolutions for upsampling

**Impact**: Established the foundation for modern semantic segmentation methods

## Core Concept

### From Classification to Segmentation

**Traditional CNN classification**:
```
Input → Conv layers → Flatten → FC layers → Class scores
224×224×3 → ... → 7×7×512 → Flatten → 4096 → 4096 → 1000
```

**FCN for segmentation**:
```
Input → Conv layers → 1×1 Conv (replace FC) → Upsample → Dense prediction
224×224×3 → ... → 7×7×512 → 7×7×21 → 224×224×21
```

### Key Insight

**Any fully connected layer can be viewed as a convolution** with a kernel covering the entire input region.

**FC layer**: $4096 \times (7 \times 7 \times 512)$ parameters
↓ equivalent to ↓
**1×1 conv**: $4096$ filters of size $7 \times 7 \times 512$

**Advantages of conversion**:
1. Accepts any input size (not fixed)
2. Outputs spatial map (not single vector)
3. Efficient for dense prediction

### Semantic Segmentation

**Task**: Assign each pixel to a semantic category

**Input**: RGB image (H × W × 3)
**Output**: Label map (H × W) where each pixel has a class label

**Challenges**:
- Classification: "What?" (semantic information)
- Localization: "Where?" (spatial information)
- Multi-scale: Objects at different scales
- Efficiency: Must process every pixel

## Architecture

### FCN Variants

FCN has three main variants based on skip connections:

1. **FCN-32s**: No skip connections (coarse)
2. **FCN-16s**: One skip connection from pool4 (better)
3. **FCN-8s**: Two skip connections from pool3 and pool4 (finest)

### FCN-32s (Baseline)

Start with VGG-16, convert to fully convolutional:

```
Input: H×W×3
    ↓
VGG-16 Conv Layers (5 blocks)
    pool1: /2
    pool2: /4
    pool3: /8
    pool4: /16
    pool5: /32
    ↓
Conv7: 1×1 conv, 4096 (replacing FC6)
    ↓
Conv7: 1×1 conv, 4096 (replacing FC7)
    ↓
Score: 1×1 conv, 21 (num classes)
    ↓ (H/32 × W/32 × 21)
Upsample 32× (transposed conv)
    ↓
Output: H×W×21
```

**Problem**: 32× upsampling from very coarse features loses spatial detail.

### FCN-16s (Skip Connection from Pool4)

Add skip connection from pool4:

```
Input: H×W×3
    ↓
VGG Conv Layers
    ↓ pool4 (H/16 × W/16)
    ├──────────→ 1×1 conv, 21 (pool4 scores)
    ↓ pool5              ↓
Conv7 (4096)            ↓
Conv7 (4096)            ↓
Score (21)              ↓
    ↓ (H/32 × W/32)     ↓
Upsample 2×             ↓
    ↓                   ↓
    └────── + ←─────────┘  (element-wise addition)
          ↓
    Upsample 16×
          ↓
Output: H×W×21
```

**Improvement**: Combines coarse semantic info (conv7) with finer spatial info (pool4).

### FCN-8s (Skip Connections from Pool3 and Pool4)

Add second skip connection from pool3:

```
Input
  ↓
VGG Conv
  ↓ pool3 (H/8 × W/8)
  ├───────────────────→ 1×1 conv, 21
  ↓ pool4 (H/16)        ↓
  ├────────→ 1×1 conv   ↓
  ↓ pool5        ↓      ↓
Conv7 (4096)     ↓      ↓
Conv7 (4096)     ↓      ↓
Score (21)       ↓      ↓
  ↓ (H/32)       ↓      ↓
Upsample 2×      ↓      ↓
  ↓              ↓      ↓
  └──── + ←──────┘      ↓
       ↓ (H/16)         ↓
   Upsample 2×          ↓
       ↓                ↓
       └──── + ←────────┘
            ↓ (H/8)
       Upsample 8×
            ↓
        Output: H×W×21
```

**Improvement**: Finest predictions by combining features from multiple scales.

### Architecture Comparison

| Model | Upsampling | Skip Connections | mIOU (PASCAL VOC) |
|-------|------------|------------------|-------------------|
| FCN-32s | 32× | None | 59.4% |
| FCN-16s | 2× then 16× | pool4 | 62.4% |
| FCN-8s | 2× then 2× then 8× | pool3, pool4 | 62.7% |

**Trend**: More skip connections → finer segmentation → better accuracy

## Key Innovations

### 1. Fully Convolutional Architecture

**Replace FC layers with 1×1 convolutions**:

**Original VGG FC layer**:
- Input: 7×7×512 = 25,088 dimensional vector
- FC: 4096 neurons
- Parameters: 25,088 × 4096 ≈ 103M

**Equivalent 1×1 conv**:
- Input: 7×7×512 feature map
- Conv: 4096 filters of 1×1×512
- Output: 7×7×4096 feature map
- Parameters: same, but works on any size!

### 2. Transposed Convolution (Deconvolution)

**Purpose**: Upsample coarse feature maps to original input size

**Transposed convolution** performs learned upsampling:

For stride $s$:
$$\text{Output size} = s \times (\text{Input size} - 1) + k$$

where $k$ is kernel size.

**Example**: 2× upsampling with 4×4 kernel, stride 2:
- Input: 7×7
- Output: 2×(7-1) + 4 = 16×16

**Alternative to**:
- Fixed interpolation (bilinear, nearest)
- Unpooling
- Sub-pixel convolution

### 3. Skip Connections

**Problem**: Deep networks lose spatial information

**Solution**: Combine features from different depths

**Skip architecture**:
- Shallow layers: Fine spatial details, simple features
- Deep layers: Coarse spatial, complex semantic features
- Combination: Best of both worlds

**Implementation**: Element-wise addition after upsampling and 1×1 conv to match dimensions

### 4. Transfer Learning

**Key idea**: Initialize with ImageNet pre-trained weights

**Procedure**:
1. Start with VGG-16 pre-trained on ImageNet
2. Convert FC layers to 1×1 conv
3. Initialize upsampling layers (bilinear or random)
4. Fine-tune end-to-end on segmentation dataset

**Benefits**:
- Faster convergence
- Better generalization
- Works with smaller datasets

## Mathematical Details

### Fully Convolutional Operation

For classification network outputting $C$ classes:

$$f(x; \theta) \in \mathbb{R}^C$$

For FCN outputting $H \times W$ spatial map:

$$f(x; \theta) \in \mathbb{R}^{H \times W \times C}$$

**Each spatial location** gets independent classification:

$$p_c(i,j) = \frac{\exp(f_c(i,j))}{\sum_{c'} \exp(f_{c'}(i,j))}$$

### Transposed Convolution

**Forward pass**: Upsample input

For input $x$ and kernel $w$:

$$y = \text{conv\_transpose}(x, w, \text{stride}=s)$$

**Relationship to regular convolution**: Transposed conv with stride $s$ is equivalent to:
1. Insert $s-1$ zeros between input pixels
2. Apply regular convolution
3. Crop to desired size

**Learnable upsampling**: Weights $w$ learned during training.

### Skip Connection Fusion

At each skip level:

$$h_{\text{fused}} = h_{\text{coarse}} + W_{\text{skip}} * h_{\text{fine}}$$

where:
- $h_{\text{coarse}}$ is upsampled deeper layer
- $h_{\text{fine}}$ is feature from shallower layer
- $W_{\text{skip}}$ is 1×1 conv to match channels
- $*$ is convolution operation

### Loss Function

**Pixel-wise softmax with cross-entropy**:

$$L = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

where:
- $N = H \times W$ is number of pixels
- $C$ is number of classes
- $y_{i,c}$ is ground truth (one-hot)
- $\hat{y}_{i,c}$ is predicted probability

**Class balancing**: Weight rare classes higher to handle imbalance.

## Implementation

### PyTorch FCN-8s

```python
import torch
import torch.nn as nn
import torchvision.models as models

class FCN8s(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN8s, self).__init__()

        # Load pre-trained VGG16
        vgg = models.vgg16(pretrained=True)

        # Encoder (VGG16 features)
        self.features = vgg.features

        # Separate pools for skip connections
        # Pool3: after 16 layers
        # Pool4: after 23 layers
        # Pool5: after 30 layers

        # Replace FC layers with 1×1 convolutions
        self.conv6 = nn.Conv2d(512, 4096, 7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.conv7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        # Score layers (1×1 conv for classification)
        self.score_fr = nn.Conv2d(4096, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)

        # Upsampling layers (transposed convolutions)
        self.upscore2 = nn.ConvTranspose2d(
            num_classes, num_classes, 4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            num_classes, num_classes, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            num_classes, num_classes, 16, stride=8, bias=False)

    def forward(self, x):
        # Extract features at different scales
        pool3 = None
        pool4 = None

        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 16:  # After pool3
                pool3 = x
            if i == 23:  # After pool4
                pool4 = x

        pool5 = x

        # FC layers as convolutions
        x = self.drop6(self.relu6(self.conv6(pool5)))
        x = self.drop7(self.relu7(self.conv7(x)))

        # Score
        x = self.score_fr(x)

        # Upsample and add pool4
        x = self.upscore2(x)
        upscore2 = x  # Store for size reference

        pool4_score = self.score_pool4(pool4)
        x = upscore2[:, :, 5:5+pool4_score.size(2),
                      5:5+pool4_score.size(3)]  # Crop
        x = x + pool4_score

        # Upsample and add pool3
        x = self.upscore_pool4(x)
        upscore_pool4 = x

        pool3_score = self.score_pool3(pool3)
        x = upscore_pool4[:, :, 9:9+pool3_score.size(2),
                          9:9+pool3_score.size(3)]  # Crop
        x = x + pool3_score

        # Final upsampling
        x = self.upscore8(x)
        x = x[:, :, 31:31+x.size(2), 31:31+x.size(3)]  # Crop to input size

        return x
```

### Simpler Modern Implementation

```python
class SimpleFCN(nn.Module):
    def __init__(self, num_classes=21):
        super(SimpleFCN, self).__init__()

        # Encoder: Use ResNet as backbone
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        # Decoder: Upsampling path
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(64, num_classes, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

### Using Pre-trained FCN

```python
from torchvision.models.segmentation import fcn_resnet50

# Load pre-trained FCN with ResNet50 backbone
model = fcn_resnet50(pretrained=True)

# For custom number of classes
model.classifier[4] = nn.Conv2d(512, num_classes, 1)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for images, masks in dataloader:
    outputs = model(images)['out']  # FCN returns dict
    loss = criterion(outputs, masks)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Training and Evaluation

### Training Details (Original Paper)

- **Base network**: VGG-16 pre-trained on ImageNet
- **Optimizer**: SGD with momentum (0.9)
- **Learning rate**: 1e-4 (10× smaller for pre-trained layers)
- **Weight decay**: 5e-4
- **Batch size**: 20 images
- **Initialization**:
  - Pre-trained layers: ImageNet weights
  - New layers (upsampling): Bilinear interpolation
  - Score layers: Zero mean, small std
- **Training time**: ~3 days on NVIDIA Tesla K40

### Data Augmentation

- **Random crops**: Extract patches from images
- **Horizontal flips**: Mirror images
- **Color jittering**: Adjust brightness, contrast, saturation

### Evaluation Metrics

**Mean Intersection over Union (mIOU)**:

$$\text{IOU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}$$

$$\text{mIOU} = \frac{1}{C}\sum_{c=1}^{C} \text{IOU}_c$$

where:
- $TP_c$ = True positives for class $c$
- $FP_c$ = False positives
- $FN_c$ = False negatives
- $C$ = Number of classes

**Pixel Accuracy**:

$$\text{Pixel Acc} = \frac{\sum_c TP_c}{\text{Total pixels}}$$

### Results (PASCAL VOC 2011)

| Method | mIOU | Pixel Acc |
|--------|------|-----------|
| FCN-32s | 59.4% | 89.1% |
| FCN-16s | 62.4% | 90.0% |
| FCN-8s | 62.7% | 90.3% |
| Previous SOTA | 52.6% | - |

**Improvement**: ~10% mIOU over previous best!

## Impact and Limitations

### Impact

**Revolutionary contribution**:
1. **First end-to-end learning** for semantic segmentation
2. **Showed classification networks transfer** to dense prediction
3. **Skip connections** became standard architecture pattern
4. **Established benchmarks** for segmentation

**Influenced**:
- **U-Net**: Similar skip connection idea for medical imaging
- **SegNet**: Encoder-decoder with pooling indices
- **DeepLab**: Atrous convolutions for multi-scale
- **PSPNet**: Pyramid pooling module
- **All modern segmentation**: Built on FCN foundation

### Limitations

1. **Coarse predictions**: Even FCN-8s has 8× downsampling
   - Loses fine detail
   - Boundaries not sharp

2. **No multi-scale context**: Single-scale processing
   - Misses objects at different scales
   - Fixed receptive field

3. **Skip connections are simple**: Just addition
   - Later works use more sophisticated fusion
   - No learned attention

4. **Inefficient upsampling**: Many transposed conv parameters
   - Can cause checkerboard artifacts
   - Modern methods use bilinear + conv

5. **No instance segmentation**: Only semantic labels
   - Can't distinguish individual objects
   - Requires methods like Mask R-CNN

### Modern Improvements

**Subsequent architectures addressed limitations**:

- **DeepLab**: Atrous convolutions for multi-scale without downsampling
- **PSPNet**: Pyramid pooling for multi-scale context
- **U-Net**: Better skip connections with concatenation
- **RefineNet**: Multi-path refinement for sharp boundaries
- **EfficientPS**: Efficient panoptic segmentation

**But FCN principles remain**:
- Fully convolutional processing
- Transfer learning from classification
- Skip connections
- Upsampling for dense prediction

## Comparison with Other Methods

| Architecture | Year | Key Feature | Pros | Cons |
|--------------|------|-------------|------|------|
| **FCN** | 2015 | First end-to-end | Simple, foundational | Coarse outputs |
| SegNet | 2015 | Pooling indices | Memory efficient | No skip features |
| U-Net | 2015 | Symmetric skip connections | Precise boundaries | Medical imaging focus |
| DeepLab v3+ | 2018 | Atrous convolutions | Multi-scale, sharp | More complex |
| HRNet | 2019 | High-resolution throughout | Finest details | Expensive |

## Key Takeaways

**Core innovation**: Convert classification networks to fully convolutional for dense prediction

**Architecture**: Encoder (VGG) + Upsampling + Skip connections

**Breakthrough**: First to achieve strong semantic segmentation with end-to-end learning

**Legacy**:
- Established foundation for all modern segmentation
- Skip connections became standard
- Showed power of transfer learning

**Modern usage**: While surpassed, FCN concepts used in all segmentation architectures

## References

- Jonathan Long, Evan Shelhamer, Trevor Darrell. "Fully Convolutional Networks for Semantic Segmentation" CVPR (2015)
- [Original Paper](https://arxiv.org/abs/1411.4038)
- [Caffe Implementation](https://github.com/shelhamer/fcn.berkeleyvision.org)
- [PyTorch Implementation](https://pytorch.org/vision/stable/models.html#semantic-segmentation)
- PASCAL VOC Segmentation Challenge: [http://host.robots.ox.ac.uk/pascal/VOC/](http://host.robots.ox.ac.uk/pascal/VOC/)
