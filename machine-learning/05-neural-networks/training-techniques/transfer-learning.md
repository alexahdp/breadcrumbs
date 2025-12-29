# Transfer Learning

## Table of Contents

1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [When to Use Transfer Learning](#when-to-use-transfer-learning)
4. [Strategies](#strategies)
5. [Common Pre-trained Models](#common-pre-trained-models)
6. [Implementation Approaches](#implementation-approaches)
7. [Fine-tuning Techniques](#fine-tuning-techniques)
8. [Domain Adaptation](#domain-adaptation)

## Introduction

**Transfer Learning** - a machine learning technique where a model trained on one task is repurposed and adapted for a second related task, leveraging knowledge learned from the first task.

Instead of training from scratch, transfer learning allows us to:
- Start with pre-trained weights
- Adapt to new tasks with less data
- Achieve better performance faster
- Reduce computational requirements

**Key Insight:** Low-level features (edges, textures, shapes) learned on one task are often transferable to other tasks.

## Core Concepts

### Source and Target Domains

**Source Domain** - the original task/dataset used for pre-training:
- Usually large-scale dataset (ImageNet, COCO)
- General-purpose features
- Extensive training time and resources

**Target Domain** - the new task/dataset you want to solve:
- Often smaller dataset
- Task-specific requirements
- Limited resources or time

### Feature Hierarchy

Neural networks learn hierarchical features:

**Lower layers** (closer to input):
- General features: edges, corners, colors
- Highly transferable across tasks
- Usually frozen or lightly fine-tuned

**Middle layers**:
- More complex patterns: textures, parts
- Moderately transferable
- Often fine-tuned

**Higher layers** (closer to output):
- Task-specific features
- Less transferable
- Usually replaced or heavily fine-tuned

### Positive vs Negative Transfer

**Positive Transfer** - pre-training improves performance on target task:
- Source and target are related
- Shared low-level features
- Sufficient target data for adaptation

**Negative Transfer** - pre-training hurts performance:
- Domains are too different
- Pre-trained features are misleading
- Better to train from scratch

## When to Use Transfer Learning

### Decision Matrix

| Target Dataset Size | Similarity to Source | Strategy |
|-------------------|---------------------|----------|
| Small | High | Freeze most layers, train classifier |
| Small | Low | Extract features, train new model |
| Large | High | Fine-tune entire network |
| Large | Low | Train from scratch or light fine-tuning |

### Use Transfer Learning When:

- **Limited training data** (< 10,000 samples)
- **Limited computational resources**
- **Similar domain** to pre-trained model
- **Need quick results** (faster convergence)
- **Standard computer vision** tasks

### Train from Scratch When:

- **Very large dataset** (> 1M samples)
- **Very different domain** from existing models
- **Unique architecture** required
- **Plenty of computational resources**
- **Novel task** with no similar pre-training

## Strategies

### 1. Feature Extraction (Frozen Base)

Use pre-trained model as fixed feature extractor:

**Approach:**
- Freeze all convolutional layers
- Remove original classifier
- Add new classifier for your task
- Train only the new layers

**When to use:**
- Very small dataset (< 1,000 samples)
- High similarity to source domain
- Limited computational resources

**Advantages:**
- Fast training
- Less prone to overfitting
- Requires minimal data

**Disadvantages:**
- May not fully adapt to target domain
- Limited performance improvement

### 2. Fine-tuning

Unfreeze some layers and train them:

**Approach:**
- Start with pre-trained weights
- Optionally freeze early layers
- Train the rest with small learning rate
- Gradually unfreeze more layers

**When to use:**
- Medium dataset (1,000-100,000 samples)
- Moderate similarity to source domain
- Need better performance than feature extraction

**Advantages:**
- Better adaptation to target domain
- Improved performance
- Retains general features

**Disadvantages:**
- Risk of overfitting on small data
- Requires careful learning rate tuning
- More training time

### 3. Full Network Training

Train entire network with pre-trained initialization:

**Approach:**
- Initialize with pre-trained weights
- Unfreeze all layers
- Train with lower learning rate than from scratch
- Allow all weights to update

**When to use:**
- Large dataset (> 100,000 samples)
- Different domain from source
- Need maximum performance

**Advantages:**
- Maximum flexibility
- Best potential performance
- Full adaptation to target domain

**Disadvantages:**
- Requires large dataset
- Risk of forgetting useful features
- Longer training time

## Common Pre-trained Models

### Computer Vision

**Image Classification:**

**VGG (VGG16, VGG19)**
- Simple architecture, deep layers
- Good for feature extraction
- Large model size (138M parameters)
- Trained on ImageNet

**ResNet (ResNet50, ResNet101, ResNet152)**
- Residual connections, very deep
- Excellent general-purpose model
- Moderate size (25-60M parameters)
- State-of-the-art baseline

**Inception (InceptionV3, InceptionResNetV2)**
- Multiple filter sizes per layer
- Efficient and accurate
- Complex architecture

**EfficientNet (B0-B7)**
- Compound scaling method
- Best accuracy/efficiency tradeoff
- Modern choice for new projects
- Smaller models, better performance

**MobileNet (V1, V2, V3)**
- Designed for mobile/edge devices
- Very small and fast
- Depthwise separable convolutions
- Great for resource-constrained environments

**Object Detection:**
- **YOLO (v3, v4, v5, v8)** - real-time detection
- **Faster R-CNN** - high accuracy detection
- **SSD** - balanced speed/accuracy
- **EfficientDet** - efficient detection

**Segmentation:**
- **U-Net** - medical imaging
- **DeepLab** - semantic segmentation
- **Mask R-CNN** - instance segmentation

### Natural Language Processing

**Transformers:**
- **BERT** - bidirectional encoding
- **GPT** - generative pre-training
- **T5** - text-to-text transfer
- **RoBERTa** - optimized BERT

**Word Embeddings:**
- **Word2Vec** - word vectors
- **GloVe** - global vectors
- **FastText** - subword embeddings

### Pre-training Datasets

**ImageNet**
- 1.4M images, 1000 classes
- Natural images
- Standard for vision models

**COCO (Common Objects in Context)**
- 330K images
- Object detection, segmentation
- 80 object categories

**Places365**
- 1.8M images
- Scene recognition
- 365 scene categories

## Implementation Approaches

### PyTorch Example

```python
import torch
import torch.nn as nn
import torchvision.models as models

# 1. Feature Extraction (Frozen Base)
def create_feature_extractor(num_classes):
    # Load pre-trained ResNet
    model = models.resnet50(pretrained=True)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model

# 2. Fine-tuning
def create_finetuned_model(num_classes, freeze_layers=7):
    model = models.resnet50(pretrained=True)

    # Freeze early layers
    layers = list(model.children())[:freeze_layers]
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = False

    # Replace classifier
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model

# 3. Different learning rates for different layers
def get_optimizer_with_different_lrs(model, base_lr=1e-4):
    # Lower learning rate for pre-trained layers
    # Higher learning rate for new classifier

    params = [
        {'params': model.layer1.parameters(), 'lr': base_lr * 0.1},
        {'params': model.layer2.parameters(), 'lr': base_lr * 0.1},
        {'params': model.layer3.parameters(), 'lr': base_lr * 0.5},
        {'params': model.layer4.parameters(), 'lr': base_lr},
        {'params': model.fc.parameters(), 'lr': base_lr * 10}
    ]

    return torch.optim.Adam(params)
```

### TensorFlow/Keras Example

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

# 1. Feature Extraction
def create_feature_extractor(num_classes, input_shape=(224, 224, 3)):
    # Load pre-trained model without top layer
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze base model
    base_model.trainable = False

    # Add new classifier
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# 2. Fine-tuning
def create_finetuned_model(num_classes, input_shape=(224, 224, 3)):
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze early layers, unfreeze later layers
    base_model.trainable = True

    # Freeze first 100 layers
    for layer in base_model.layers[:100]:
        layer.trainable = False

    # Build model
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)  # Use base model in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    return model

# Two-stage training
model = create_feature_extractor(num_classes=10)

# Stage 1: Train only new layers
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_data, epochs=10)

# Stage 2: Unfreeze and fine-tune
base_model = model.layers[0]
base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_data, epochs=10)
```

## Fine-tuning Techniques

### 1. Learning Rate Scheduling

Use different learning rates for different parts:

**Strategy:**
- **New layers:** higher learning rate (e.g., 1e-3)
- **Middle layers:** medium learning rate (e.g., 1e-4)
- **Early layers:** low learning rate (e.g., 1e-5)

```python
# PyTorch
optimizer = torch.optim.SGD([
    {'params': model.fc.parameters(), 'lr': 1e-3},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.layer3.parameters(), 'lr': 1e-5}
])
```

### 2. Gradual Unfreezing

Progressively unfreeze layers during training:

```python
def gradual_unfreeze(model, optimizer, train_loader, epochs_per_stage=5):
    # Stage 1: Train only classifier
    train_stage(model, optimizer, train_loader, epochs_per_stage)

    # Stage 2: Unfreeze last conv block
    for param in model.layer4.parameters():
        param.requires_grad = True
    train_stage(model, optimizer, train_loader, epochs_per_stage)

    # Stage 3: Unfreeze more layers
    for param in model.layer3.parameters():
        param.requires_grad = True
    train_stage(model, optimizer, train_loader, epochs_per_stage)
```

### 3. Discriminative Fine-tuning

Different learning rates for each layer group:

$$lr_{layer_i} = lr_{base} \times decay^{n-i}$$

where $n$ is total layers and $decay < 1$ (e.g., 0.95).

### 4. Batch Normalization Handling

**Important:** When fine-tuning, handle BatchNorm carefully:

```python
# Option 1: Keep BN layers frozen
for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.eval()

# Option 2: Update BN statistics
for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.train()
```

### 5. Data Preprocessing

Use same preprocessing as pre-training:

```python
# For ImageNet pre-trained models
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]     # ImageNet std
    )
])
```

## Domain Adaptation

### Handling Domain Shift

**Domain Shift** - when training and test data come from different distributions.

**Strategies:**

**1. Data Augmentation**
- Match target domain characteristics
- Color jittering, rotation, scaling
- Domain-specific augmentations

**2. Multi-task Learning**
- Train on both source and target
- Share representations
- Task-specific heads

**3. Adversarial Training**
- Domain-adversarial neural networks (DANN)
- Make features domain-invariant
- Confuse domain classifier

**4. Self-supervised Pre-training**
- Pre-train on target domain without labels
- Rotation prediction, jigsaw puzzles
- Then fine-tune with limited labels

## Best Practices

### 1. Start Simple
```python
# Begin with feature extraction
# Then gradually increase complexity
# Finally, full fine-tuning if needed
```

### 2. Monitor for Overfitting
- Use validation set from target domain
- Early stopping
- Regularization (dropout, weight decay)

### 3. Learning Rate Guidelines
- **Feature extraction:** 1e-3 to 1e-2
- **Fine-tuning:** 1e-5 to 1e-4
- **New layers:** 10-100x higher than base

### 4. Save Checkpoints
```python
# Save best model based on validation
if val_acc > best_val_acc:
    torch.save(model.state_dict(), 'best_model.pth')
    best_val_acc = val_acc
```

### 5. Experiment Tracking

Test multiple strategies:
- Different pre-trained models
- Different freezing strategies
- Different learning rates
- Different augmentation

## Common Pitfalls

**1. Wrong preprocessing**
- Must match pre-training preprocessing
- Normalization values matter

**2. Too high learning rate**
- Destroys pre-trained features
- Use 10-100x lower than training from scratch

**3. Not freezing BatchNorm**
- Small batches can break statistics
- Consider freezing BN in early layers

**4. Ignoring domain mismatch**
- Medical images ≠ natural images
- May need domain adaptation

**5. Insufficient fine-tuning data**
- Need minimum data for fine-tuning
- Consider few-shot learning methods

## References

- [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792) (Yosinski et al., 2014)
- [A Survey on Transfer Learning](https://ieeexplore.ieee.org/document/5288526) (Pan & Yang, 2010)
- [Deep Learning](https://www.deeplearningbook.org/) (Goodfellow et al., 2016) - Chapter 15
- [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) (Howard & Ruder, 2018)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (Tan & Le, 2019)
