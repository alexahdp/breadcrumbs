# Data Augmentation

## Table of Contents

1. [Introduction](#introduction)
2. [Why Data Augmentation Works](#why-data-augmentation-works)
3. [Image Augmentation Techniques](#image-augmentation-techniques)
4. [Advanced Augmentation Methods](#advanced-augmentation-methods)
5. [Text and Audio Augmentation](#text-and-audio-augmentation)
6. [Implementation Strategies](#implementation-strategies)
7. [Best Practices](#best-practices)

## Introduction

**Data Augmentation** - a technique that artificially increases the size and diversity of training data by creating modified versions of existing samples through various transformations.

Data augmentation helps:
- Reduce overfitting
- Improve model generalization
- Create more robust models
- Compensate for limited training data

**Key Principle:** Apply transformations that preserve the label while changing the input in realistic ways.

## Why Data Augmentation Works

### 1. Regularization Effect

Augmentation acts as a regularizer:
- Adds noise to training process
- Prevents memorization
- Forces model to learn invariant features

### 2. Increases Effective Dataset Size

Without augmentation:
- Limited to original samples
- Model sees same data repeatedly
- Easy to overfit

With augmentation:
- Virtually infinite variations
- Each epoch sees different data
- Harder to memorize

### 3. Improves Invariance

Teaches model to be invariant to:
- Small rotations and translations
- Lighting conditions
- Scale changes
- Noise and distortions

### 4. Better Decision Boundaries

Creates smoother decision boundaries:
- Fills gaps between training samples
- Reduces sensitivity to small perturbations
- More robust to distribution shift

## Image Augmentation Techniques

### Basic Geometric Transformations

**1. Horizontal Flip**
- Mirror image along vertical axis
- Probability: 0.5
- **Use for:** Most natural images, objects
- **Avoid for:** Text, directional data (e.g., arrows)

**2. Vertical Flip**
- Mirror image along horizontal axis
- Probability: 0.5
- **Use for:** Satellite imagery, medical scans
- **Avoid for:** Natural photos (objects rarely upside down)

**3. Rotation**
- Rotate by random angle
- Range: typically ±15° to ±45°
- **Formula:** Apply rotation matrix

$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

**4. Translation (Shifting)**
- Shift image horizontally/vertically
- Range: ±10% to ±20% of image size
- Helps with object localization

**5. Scaling (Zoom)**
- Zoom in or out
- Range: 0.8x to 1.2x
- Creates multi-scale training

**6. Shearing**
- Slant image along axis
- Range: ±20°
- Simulates perspective changes

### Color and Intensity Transformations

**1. Brightness Adjustment**
- Add/subtract constant to pixel values
- Range: ±30% of mean brightness

$$I_{new} = I_{old} + \delta$$

where $\delta \in [-k, k]$

**2. Contrast Adjustment**
- Multiply pixel values by factor
- Range: 0.7 to 1.3

$$I_{new} = \alpha \cdot I_{old}$$

where $\alpha$ is contrast factor

**3. Saturation Adjustment**
- Modify color intensity
- Range: 0.7 to 1.3
- Converts to HSV, modifies S channel

**4. Hue Shift**
- Rotate color wheel
- Range: ±10° to ±30°
- In HSV color space

**5. Gamma Correction**
- Non-linear intensity transformation

$$I_{new} = I_{old}^{\gamma}$$

where $\gamma \in [0.7, 1.5]$

**6. Channel Shift**
- Add random values to RGB channels independently
- Simulates lighting variations

### Noise and Blurring

**1. Gaussian Noise**
- Add random Gaussian noise

$$I_{new} = I_{old} + \mathcal{N}(0, \sigma^2)$$

where $\sigma \in [0, 0.05]$

**2. Salt and Pepper Noise**
- Randomly set pixels to min/max values
- Probability: 0.01 to 0.05

**3. Gaussian Blur**
- Apply Gaussian smoothing
- Kernel size: 3x3 to 7x7
- Simulates out-of-focus images

**4. Motion Blur**
- Directional blur
- Simulates camera motion

### Cropping Techniques

**1. Random Crop**
- Extract random patch
- Size: 70% to 100% of original
- Most common augmentation

**2. Center Crop**
- Extract center region
- Used during inference
- Typically 224x224 for ImageNet models

**3. Five Crop / Ten Crop**
- 4 corners + center (+ flips)
- Test-time augmentation
- Average predictions

**4. Random Resized Crop**
- Crop then resize to target size
- Scale: 0.08 to 1.0
- Aspect ratio: 3/4 to 4/3

## Advanced Augmentation Methods

### Cutout and Masking

**1. Cutout**
- Randomly mask square regions
- Size: 10% to 30% of image
- Forces model to use diverse features

**Implementation:**
```python
def cutout(image, size=16):
    h, w = image.shape[:2]
    x = np.random.randint(w)
    y = np.random.randint(h)

    x1 = np.clip(x - size // 2, 0, w)
    x2 = np.clip(x + size // 2, 0, w)
    y1 = np.clip(y - size // 2, 0, h)
    y2 = np.clip(y + size // 2, 0, h)

    image[y1:y2, x1:x2] = 0
    return image
```

**2. Random Erasing**
- Similar to cutout, random value fill
- Rectangle aspect ratio varies

**3. Grid Mask**
- Mask in grid pattern
- Preserves spatial structure

### Mixup and CutMix

**1. Mixup**
- Blend two images and labels

$$\tilde{x} = \lambda x_i + (1-\lambda)x_j$$
$$\tilde{y} = \lambda y_i + (1-\lambda)y_j$$

where $\lambda \sim \text{Beta}(\alpha, \alpha)$, typically $\alpha = 0.2$

**Benefits:**
- Smoother decision boundaries
- Better calibration
- Reduced overfitting

**2. CutMix**
- Cut and paste patches between images
- Label proportional to patch area

$$\lambda = \frac{|B|}{|I|}$$

where $|B|$ is patch area, $|I|$ is image area

**3. MixUp variants**
- **Manifold Mixup:** Mix in feature space
- **PuzzleMix:** Optimal transport-based mixing

### AutoAugment and Learned Policies

**AutoAugment**
- Use reinforcement learning to find best augmentation policy
- Dataset-specific optimal transformations
- Computationally expensive to search

**RandAugment**
- Simplified version of AutoAugment
- Two hyperparameters: N (operations) and M (magnitude)
- Randomly select N operations with magnitude M

**AugMax**
- Adversarial approach
- Select augmentations that maximize loss
- More challenging training samples

### Domain-Specific Augmentations

**Medical Imaging:**
- Elastic deformations
- Intensity windowing
- Histogram equalization
- Specific anatomical transformations

**Satellite Imagery:**
- Multi-spectral band shuffling
- Cloud simulation
- Seasonal variations
- Shadow augmentation

**Document Analysis:**
- Perspective distortion
- Ruled lines
- Paper texture
- Watermarks

## Text and Audio Augmentation

### Text Augmentation

**1. Synonym Replacement**
- Replace words with synonyms
- Use WordNet or word embeddings

```python
# Example
Original: "The cat sat on the mat"
Augmented: "The feline sat on the carpet"
```

**2. Random Insertion**
- Insert random synonyms
- Maintains sentence meaning

**3. Random Swap**
- Swap positions of words
- Limited swaps to preserve grammar

**4. Random Deletion**
- Randomly remove words
- Probability: 0.1 to 0.3

**5. Back Translation**
- Translate to another language and back
- Creates paraphrases

**6. Contextual Word Embeddings**
- Use BERT to generate replacements
- Context-aware substitutions

### Audio Augmentation

**1. Time Stretching**
- Change speed without affecting pitch
- Factor: 0.8 to 1.2

**2. Pitch Shifting**
- Change pitch without affecting speed
- Semitones: ±2 to ±4

**3. Adding Noise**
- White noise, environmental sounds
- SNR: 10-30 dB

**4. Time Masking**
- Mask random time segments
- SpecAugment technique

**5. Frequency Masking**
- Mask random frequency bands
- In spectrogram domain

**6. Room Simulation**
- Add reverb and echo
- Simulates different environments

## Implementation Strategies

### PyTorch with torchvision

```python
from torchvision import transforms

# Training augmentation pipeline
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))
])

# Validation transform (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Albumentations (Advanced Library)

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# More advanced augmentation pipeline
transform = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.2,
        rotate_limit=15,
        p=0.5
    ),
    A.OneOf([
        A.GaussianBlur(blur_limit=3),
        A.MotionBlur(blur_limit=3),
        A.MedianBlur(blur_limit=3),
    ], p=0.3),
    A.OneOf([
        A.OpticalDistortion(distort_limit=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
    ], p=0.3),
    A.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
        p=0.5
    ),
    A.CoarseDropout(
        max_holes=8,
        max_height=32,
        max_width=32,
        fill_value=0,
        p=0.5
    ),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])
```

### TensorFlow/Keras

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create data generator with augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    preprocessing_function=lambda x: x / 255.0
)

# Validation generator (no augmentation)
val_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: x / 255.0
)

# Flow from directory
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```

### Custom Augmentation

```python
import numpy as np
import cv2

class CustomAugmentation:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if np.random.random() < self.p:
            # Apply custom transformation
            image = self.apply_transformation(image)
        return image

    def apply_transformation(self, image):
        # Example: Add grid pattern
        h, w = image.shape[:2]
        grid_size = 20

        for i in range(0, h, grid_size):
            cv2.line(image, (0, i), (w, i), (128, 128, 128), 1)
        for j in range(0, w, grid_size):
            cv2.line(image, (j, 0), (j, h), (128, 128, 128), 1)

        return image
```

### Mixup Implementation

```python
import torch
import numpy as np

def mixup_data(x, y, alpha=1.0):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Training loop with mixup
for inputs, targets in train_loader:
    inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.2)
    outputs = model(inputs)
    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
```

## Best Practices

### 1. Start Conservative

Begin with mild augmentation:
```python
# Conservative
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomRotation(degrees=10)

# Later, increase if needed
transforms.RandomRotation(degrees=30)
```

### 2. Match Domain Characteristics

**Natural images:**
- Flips, rotations, color jitter
- Crops and scales

**Medical images:**
- Elastic deformations
- Intensity variations
- Minimal geometric changes (preserve anatomy)

**Text/Documents:**
- Perspective transforms
- No color augmentation
- Careful with rotations

### 3. Validate Augmentation Quality

Always visualize augmented samples:
```python
# Display augmented batch
import matplotlib.pyplot as plt

def show_augmented_batch(dataset, n=16):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        image, label = dataset[i]
        ax.imshow(image.permute(1, 2, 0))
        ax.set_title(f"Label: {label}")
        ax.axis('off')
    plt.show()

show_augmented_batch(train_dataset)
```

### 4. No Augmentation for Validation/Test

Critical: Only augment training data
```python
# Training: with augmentation
train_transform = get_train_transforms()

# Validation/Test: no augmentation
val_transform = get_val_transforms()  # Only resize + normalize
```

### 5. Balance Augmentation Strength

**Too weak:** Little benefit, still overfits
**Too strong:** Distorts data, hurts performance
**Just right:** Improves generalization

**Rule of thumb:** Augmented images should still be recognizable by humans

### 6. Progressive Augmentation

Start simple, increase complexity:
```python
# Early training: mild augmentation
# Later training: stronger augmentation
# Can use curriculum learning approach
```

### 7. Test-Time Augmentation (TTA)

Improve inference by averaging multiple augmented predictions:
```python
def predict_with_tta(model, image, n_augmentations=5):
    model.eval()
    predictions = []

    for _ in range(n_augmentations):
        augmented = test_time_transform(image)
        with torch.no_grad():
            pred = model(augmented)
        predictions.append(pred)

    # Average predictions
    final_pred = torch.stack(predictions).mean(dim=0)
    return final_pred
```

### 8. Monitor Impact

Track metrics with and without augmentation:
- Training loss/accuracy
- Validation loss/accuracy
- Overfitting gap (train acc - val acc)

## Common Mistakes

**1. Augmenting validation data**
```python
# Wrong
val_dataset = Dataset(transform=train_transform)

# Correct
val_dataset = Dataset(transform=val_transform)
```

**2. Label-breaking augmentations**
```python
# Wrong for digit classification
transforms.RandomRotation(degrees=90)  # 6 becomes 9!

# Better
transforms.RandomRotation(degrees=15)  # Preserve label
```

**3. Unrealistic transformations**
```python
# Too extreme
transforms.ColorJitter(hue=0.5)  # Changes colors drastically
transforms.RandomRotation(degrees=180)  # Upside down

# More reasonable
transforms.ColorJitter(hue=0.1)
transforms.RandomRotation(degrees=15)
```

**4. Ignoring computational cost**
- Complex augmentations can slow training
- Use efficient libraries (Albumentations)
- Consider preprocessing offline

**5. Not adapting to task**
- Medical: preserve anatomical structure
- Satellite: different augmentations than natural images
- Text: rotation usually inappropriate

## Choosing Augmentation Strategies

### Decision Framework

| Dataset Size | Augmentation Strength |
|-------------|---------------------|
| < 1,000 | Strong augmentation + mixup |
| 1,000-10,000 | Moderate to strong |
| 10,000-100,000 | Moderate |
| > 100,000 | Light to moderate |

### Task-Specific Guidelines

**Classification:**
- Standard geometric + color transformations
- Mixup for improved calibration

**Object Detection:**
- Careful with crops (don't cut objects)
- Bounding box-aware transforms
- Mosaic augmentation

**Segmentation:**
- Geometric transforms (apply to mask too)
- Elastic deformations
- Consistent transforms for image and mask

**Few-Shot Learning:**
- Very strong augmentation
- Domain-specific techniques
- Synthetic data generation

## References

- [Cutout: Regularization Method](https://arxiv.org/abs/1708.04552) (DeVries & Taylor, 2017)
- [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412) (Zhang et al., 2017)
- [CutMix: Regularization Strategy to Train Strong Classifiers](https://arxiv.org/abs/1905.04899) (Yun et al., 2019)
- [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501) (Cubuk et al., 2019)
- [RandAugment: Practical automated data augmentation](https://arxiv.org/abs/1909.13719) (Cubuk et al., 2020)
- [Albumentations: Fast and Flexible Image Augmentations](https://www.mdpi.com/2078-2489/11/2/125) (Buslaev et al., 2020)
