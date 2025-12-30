# Famous Neural Network Architectures

This section covers landmark neural network architectures that have shaped the field of deep learning. These architectures represent breakthrough innovations in computer vision and image processing.

## Overview

Famous architectures are important to study because they:
- Introduce novel concepts and techniques that have become standard practice
- Demonstrate solutions to specific challenges (vanishing gradients, computational efficiency, etc.)
- Provide proven starting points for transfer learning
- Show the evolution of deep learning over time

## Architecture Categories

### Computer Vision

Classification architectures that learn to recognize and categorize images:

- **[LeNet](./computer-vision/lenet.md)** - The pioneering convolutional neural network (1998)
- **[AlexNet](./computer-vision/alexnet.md)** - The deep learning revolution breakthrough (2012)
- **[VGG](./computer-vision/vgg.md)** - Deep networks with simple, uniform architecture (2014)
- **[ResNet](./computer-vision/resnet.md)** - Residual connections enabling very deep networks (2015)
- **[Inception](./computer-vision/inception.md)** - Multi-scale feature extraction (2014-2016)
- **[EfficientNet](./computer-vision/efficientnet.md)** - Optimally scaled networks (2019)

### Segmentation

Architectures designed for pixel-level prediction tasks:

- **[U-Net](./segmentation/unet.md)** - Encoder-decoder for medical image segmentation (2015)
- **[FCN](./segmentation/fcn.md)** - Fully Convolutional Networks for semantic segmentation (2015)
- **[Mask R-CNN](./segmentation/mask-rcnn.md)** - Instance segmentation and object detection (2017)

## Historical Timeline

```
1998 ─── LeNet-5
         │
2012 ─── AlexNet ────┐
         │           │
2014 ─── VGG        │
         Inception   │  ImageNet Era
         │           │
2015 ─── ResNet     │
         U-Net       │
         FCN        ┘
         │
2017 ─── Mask R-CNN
         │
2019 ─── EfficientNet
```

## Key Innovations by Architecture

| Architecture | Key Innovation | Impact |
|--------------|----------------|---------|
| LeNet | Convolutional layers + pooling | Foundation of CNNs |
| AlexNet | ReLU, Dropout, GPU training | Proved deep learning works |
| VGG | Very deep uniform architecture | Showed depth matters |
| ResNet | Residual connections | Enabled 100+ layer networks |
| Inception | Multi-scale convolutions | Improved efficiency |
| EfficientNet | Compound scaling | SOTA with fewer parameters |
| U-Net | Skip connections in encoder-decoder | Medical imaging standard |
| FCN | Fully convolutional for segmentation | Pioneered semantic segmentation |
| Mask R-CNN | Instance segmentation | Combined detection + segmentation |

## Common Architectural Patterns

### Building Blocks

1. **Convolutional Layers** - Extract spatial features
2. **Pooling Layers** - Reduce spatial dimensions
3. **Activation Functions** - Introduce non-linearity (ReLU, etc.)
4. **Normalization** - Batch normalization, layer normalization
5. **Regularization** - Dropout, weight decay

### Design Patterns

- **Sequential stacking** (VGG) - Simple, deep stacks of layers
- **Residual connections** (ResNet) - Skip connections to avoid degradation
- **Multi-branch** (Inception) - Parallel paths with different operations
- **Encoder-decoder** (U-Net) - Contracting then expanding path
- **Compound scaling** (EfficientNet) - Balance depth, width, resolution

## Choosing an Architecture

### For Image Classification

- **Starting point**: ResNet-50 or EfficientNet-B0
- **High accuracy needed**: ResNet-152, EfficientNet-B7
- **Limited resources**: MobileNet, EfficientNet-B0
- **Transfer learning**: Any pre-trained ImageNet model

### For Segmentation

- **Medical imaging**: U-Net variants
- **Semantic segmentation**: FCN, DeepLab
- **Instance segmentation**: Mask R-CNN
- **Real-time applications**: U-Net with smaller backbone

## Transfer Learning

Most famous architectures are available as pre-trained models:

```python
# PyTorch example
from torchvision import models

# Load pre-trained ResNet
model = models.resnet50(pretrained=True)

# Load pre-trained VGG
model = models.vgg16(pretrained=True)

# Load pre-trained EfficientNet
model = models.efficientnet_b0(pretrained=True)
```

## Performance Considerations

### Model Size vs Accuracy Trade-off

- **Smallest**: MobileNet (~4M parameters)
- **Medium**: ResNet-50 (~25M parameters)
- **Large**: ResNet-152 (~60M parameters)
- **Very Large**: EfficientNet-B7 (~66M parameters)

### Training Time

Factors affecting training time:
- Network depth (more layers = slower)
- Input resolution (higher = slower)
- Batch size (larger = faster per sample)
- Hardware (GPUs, TPUs)

## Modern Trends (Beyond Scope)

While not covered in detail, be aware of:
- **Vision Transformers (ViT)** - Attention-based architectures
- **Neural Architecture Search (NAS)** - Automated architecture design
- **Efficient architectures** - MobileNet, ShuffleNet for mobile devices
- **Self-supervised learning** - Pre-training without labels

## References

- Yann LeCun et al. "Gradient-Based Learning Applied to Document Recognition" (1998)
- Alex Krizhevsky et al. "ImageNet Classification with Deep CNNs" (2012)
- Karen Simonyan, Andrew Zisserman. "Very Deep CNNs for Large-Scale Image Recognition" (2014)
- Kaiming He et al. "Deep Residual Learning for Image Recognition" (2015)
- Christian Szegedy et al. "Going Deeper with Convolutions" (2015)
- Olaf Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- Jonathan Long et al. "Fully Convolutional Networks for Semantic Segmentation" (2015)
- Kaiming He et al. "Mask R-CNN" (2017)
- Mingxing Tan, Quoc V. Le. "EfficientNet: Rethinking Model Scaling for CNNs" (2019)
