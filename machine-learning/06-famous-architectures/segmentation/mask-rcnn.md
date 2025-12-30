# Mask R-CNN - Instance Segmentation

## Table of Contents

1. [Overview](#overview)
2. [Background: From R-CNN to Mask R-CNN](#background-from-r-cnn-to-mask-rcnn)
3. [Architecture](#architecture)
4. [Key Components](#key-components)
5. [Mathematical Details](#mathematical-details)
6. [Implementation](#implementation)
7. [Training and Evaluation](#training-and-evaluation)
8. [Applications and Impact](#applications-and-impact)

## Overview

**Mask R-CNN** extends Faster R-CNN to perform instance segmentation by adding a branch for predicting segmentation masks on each Region of Interest (RoI), in parallel with the existing branch for classification and bounding box regression.

**Authors**: Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick (Facebook AI Research)

**Year**: 2017

**Achievement**: Won COCO 2016 instance segmentation, bounding box detection, and person keypoint challenges

**Key innovation**: **RoIAlign** + **parallel mask prediction branch**

**Task**: **Instance segmentation** - detect and segment each object instance separately

## Background: From R-CNN to Mask R-CNN

### Evolution of R-CNN Family

**R-CNN (2014)**: Region-based CNN
- Selective search for region proposals
- CNN features for each region
- SVM classifier
- **Problem**: Slow (47s per image)

**Fast R-CNN (2015)**: Faster training and inference
- Single CNN for entire image
- RoI pooling to extract features per region
- **Problem**: Region proposals still slow

**Faster R-CNN (2015)**: End-to-end with RPN
- Region Proposal Network (RPN) for proposals
- Shares conv features with detector
- **~10× faster** than Fast R-CNN
- **Limitation**: Only bounding boxes, not masks

**Mask R-CNN (2017)**: Add instance segmentation
- Extends Faster R-CNN
- Adds mask prediction branch
- RoIAlign for pixel-level accuracy
- **Capability**: Bounding boxes + class labels + instance masks

### Semantic vs Instance Segmentation

**Semantic Segmentation** (FCN, U-Net):
- Classify each pixel into categories
- No distinction between instances
- Output: Single segmentation map

**Instance Segmentation** (Mask R-CNN):
- Detect each object instance
- Segment each instance separately
- Output: Multiple masks (one per instance)

**Example**:
- Image with 3 people
- Semantic: All pixels labeled "person"
- Instance: 3 separate person masks

## Architecture

### Overall Pipeline

```
Input Image
    ↓
┌─────────────────────────────┐
│  Backbone (ResNet + FPN)    │  Extract features
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  Region Proposal Network    │  Propose object locations
│         (RPN)               │
└─────────────────────────────┘
    ↓ (RoIs)
┌─────────────────────────────┐
│       RoIAlign              │  Extract features for each RoI
└─────────────────────────────┘
    ↓
    ├──────────────┬──────────────┐
    ↓              ↓              ↓
┌────────┐   ┌──────────┐   ┌─────────┐
│  Class │   │   Box    │   │  Mask   │  Three parallel heads
└────────┘   └──────────┘   └─────────┘
    ↓              ↓              ↓
 Class label   Bounding box  Segmentation mask
```

### Detailed Architecture

**1. Backbone**: ResNet-50 or ResNet-101 + FPN
- Extracts multi-scale feature maps
- FPN provides features at multiple resolutions

**2. Region Proposal Network (RPN)**:
- Slides over feature map
- Proposes ~2000 RoIs per image
- Predicts objectness score and box refinement

**3. RoIAlign**:
- Extracts fixed-size features for each RoI
- Preserves pixel-level spatial correspondence
- Key improvement over RoIPool

**4. Head Networks**:
- **Box head**: Bounding box regression + classification
- **Mask head**: FCN for mask prediction (per class)

### Mask R-CNN with ResNet-FPN

```
Input: 800×800×3
    ↓
ResNet-50-FPN Backbone
    ↓ Multi-scale features
RPN
    ↓ ~2000 RoIs
RoIAlign (7×7 for box, 14×14 for mask)
    ↓
    ├────────────────────┬────────────────────┐
    ↓                    ↓                    ↓
Box Head (FC)      Class Head (FC)      Mask Head (FCN)
    ↓                    ↓                    ↓
7×7×256 → FC → FC   Softmax (C+1)      Conv→Conv→Conv→Deconv
    ↓                    ↓                    ↓
4 coords           Class scores        28×28×C masks
```

## Key Components

### 1. Feature Pyramid Network (FPN)

**Purpose**: Multi-scale feature extraction

**Structure**: Top-down pathway with lateral connections

```
ResNet Backbone         FPN Top-down
    ↓                       ↑
  C5 (2048 ch) ──→ P5 ────┤ (256 ch)
    ↓           ↗          │
  C4 (1024 ch) ──→ P4 ────┤
    ↓           ↗          │
  C3 (512 ch)  ──→ P3 ────┤
    ↓           ↗          │
  C2 (256 ch)  ──→ P2 ────┘
```

**Each pyramid level P**:
1. 1×1 conv on C (reduce to 256 channels)
2. Upsample higher-level P by 2×
3. Add (element-wise)
4. 3×3 conv to reduce aliasing

**Advantage**: Detects objects at multiple scales efficiently

### 2. Region Proposal Network (RPN)

**Purpose**: Generate object proposals

**Process**:
1. Slide 3×3 window over feature map
2. At each location, predict $k$ anchor boxes (different scales/ratios)
3. For each anchor:
   - **Objectness score**: Is it an object? (binary)
   - **Box refinement**: Adjust anchor to fit object

**Anchors**: Pre-defined boxes at multiple scales (32², 64², 128², 256², 512²) and ratios (1:1, 1:2, 2:1)

**Output**: ~2000 region proposals (after NMS)

### 3. RoIAlign (Critical Innovation)

**Problem with RoIPool**: Quantization causes misalignment
- RoI boundaries quantized to integers
- Feature map bins quantized
- Result: Spatial misalignment hurts mask accuracy

**RoIAlign solution**: Avoid quantization
1. Keep RoI boundaries at floating point
2. Use bilinear interpolation to sample feature values
3. Compute max or average over sampled values

**Mathematical formulation**:

For each bin in the output grid:
$$y = \frac{1}{n}\sum_{i=1}^{n} \text{bilinear\_interpolate}(x, p_i)$$

where $p_i$ are sampling points (4 per bin typically).

**Impact**: +1-2% mask AP improvement (critical for pixel-level accuracy)

### 4. Mask Head

**Architecture**: Small FCN applied to each RoI

```
Input: 14×14×256 (RoIAlign output)
    ↓
Conv 3×3, 256, ReLU
    ↓
Conv 3×3, 256, ReLU
    ↓
Conv 3×3, 256, ReLU
    ↓
Conv 3×3, 256, ReLU
    ↓
Deconv 2×2, stride 2 (upsample)
    ↓
Conv 1×1, C (per-class masks)
    ↓
Output: 28×28×C
```

**Key design choices**:
- **Per-class masks**: Predict C binary masks (one per class)
- **No class competition**: Mask prediction decoupled from classification
- **Small FCN**: Efficient, sufficient for small RoIs

### 5. Multi-task Loss

**Total loss**: Combination of RPN and detection losses

$$L = L_{cls} + L_{box} + L_{mask}$$

**Classification loss** (cross-entropy):
$$L_{cls} = -\log p_u$$

where $u$ is true class, $p_u$ is predicted probability.

**Bounding box loss** (smooth L1):
$$L_{box} = \text{smooth}_{L1}(t^u - v)$$

where $t^u$ is predicted box, $v$ is ground truth.

**Mask loss** (binary cross-entropy per pixel):
$$L_{mask} = -\frac{1}{m^2}\sum_{i,j} [y_{ij}\log\hat{y}_{ij}^k + (1-y_{ij})\log(1-\hat{y}_{ij}^k)]$$

where:
- $m = 28$ (mask resolution)
- $k$ is ground truth class
- $y_{ij}$ is ground truth mask
- $\hat{y}_{ij}^k$ is predicted mask for class $k$

**Key insight**: Only compute mask loss for ground truth class (decouples mask and class prediction)

## Mathematical Details

### RoIAlign Bilinear Interpolation

For sampling point $(x, y)$ (floating point coordinates):

$$f(x, y) = \sum_{i,j} w_{ij} \cdot I[i, j]$$

where:
- $i = \lfloor x \rfloor, \lceil x \rceil$
- $j = \lfloor y \rfloor, \lceil y \rceil$
- $w_{ij}$ are bilinear interpolation weights

**Bilinear weights**:
$$w_{ij} = \max(0, 1-|x-i|) \cdot \max(0, 1-|y-j|)$$

### Non-Maximum Suppression (NMS)

**Purpose**: Remove duplicate detections

**Algorithm**:
1. Sort proposals by score (descending)
2. Select highest-scoring box
3. Remove boxes with IoU > threshold (0.7) with selected box
4. Repeat until no boxes remain

**IoU (Intersection over Union)**:
$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

### Mask Prediction

For each RoI, predict $C$ binary masks (28×28 each):

$$M = \{M_1, M_2, ..., M_C\}$$

At inference:
1. Classify RoI → get class $k$
2. Use only mask $M_k$
3. Threshold at 0.5 to get binary mask
4. Resize to RoI size

This **decoupling** (separate masks per class) is crucial for good performance.

## Implementation

### PyTorch Mask R-CNN (Simplified)

```python
import torch
import torch.nn as nn
from torchvision.ops import RoIAlign
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class MaskRCNN(nn.Module):
    def __init__(self, num_classes=81):  # COCO: 80 classes + background
        super(MaskRCNN, self).__init__()

        # Backbone: ResNet-50 + FPN
        self.backbone = resnet_fpn_backbone('resnet50', pretrained=True)

        # RPN head
        self.rpn = RegionProposalNetwork(...)

        # RoIAlign
        self.roi_align_box = RoIAlign(output_size=7, sampling_ratio=2)
        self.roi_align_mask = RoIAlign(output_size=14, sampling_ratio=2)

        # Box head
        self.box_head = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True)
        )
        self.box_predictor = nn.Linear(1024, num_classes * 4)
        self.cls_score = nn.Linear(1024, num_classes)

        # Mask head
        self.mask_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, images, targets=None):
        # Extract features
        features = self.backbone(images)

        # Generate proposals
        proposals = self.rpn(features, images)

        # RoIAlign for boxes
        box_features = self.roi_align_box(
            features, proposals, images.image_sizes)
        box_features = box_features.flatten(start_dim=1)

        # Box prediction
        box_features = self.box_head(box_features)
        class_logits = self.cls_score(box_features)
        box_regression = self.box_predictor(box_features)

        # RoIAlign for masks
        mask_features = self.roi_align_mask(
            features, proposals, images.image_sizes)

        # Mask prediction
        mask_logits = self.mask_head(mask_features)

        if self.training:
            return self.compute_loss(
                class_logits, box_regression, mask_logits, targets)
        else:
            return self.postprocess(
                class_logits, box_regression, mask_logits, proposals)
```

### Using Pre-trained Mask R-CNN

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Load pre-trained model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Inference
with torch.no_grad():
    predictions = model(images)

# predictions is a list of dicts (one per image):
# {
#   'boxes': FloatTensor[N, 4],
#   'labels': Int64Tensor[N],
#   'scores': FloatTensor[N],
#   'masks': FloatTensor[N, 1, H, W]
# }

# Visualize
for i, pred in enumerate(predictions):
    boxes = pred['boxes']
    labels = pred['labels']
    scores = pred['scores']
    masks = pred['masks']

    # Keep only high-confidence detections
    keep = scores > 0.7
    boxes = boxes[keep]
    labels = labels[keep]
    masks = masks[keep]

    # Apply masks
    for mask in masks:
        # Threshold at 0.5
        mask = mask[0] > 0.5
        # Overlay on image
        ...
```

### Training Mask R-CNN

```python
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Create model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.train()

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop
for epoch in range(num_epochs):
    for images, targets in dataloader:
        # targets is a list of dicts:
        # {
        #   'boxes': FloatTensor[N, 4],
        #   'labels': Int64Tensor[N],
        #   'masks': UInt8Tensor[N, H, W]
        # }

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass (returns loss dict during training)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    lr_scheduler.step()
```

## Training and Evaluation

### Training Configuration

**Original paper**:
- **Backbone**: ResNet-50-FPN or ResNet-101-FPN
- **Optimizer**: SGD with momentum (0.9)
- **Learning rate**: 0.02, decreased by 10 at 60k and 80k iterations
- **Weight decay**: 0.0001
- **Batch size**: 16 images (8 GPUs × 2 images/GPU)
- **Image scale**: Shorter side 800 pixels
- **Training iterations**: 90k (COCO)
- **Training time**: ~32 hours on 8 GPUs

### Data Augmentation

- **Horizontal flips**: 50% probability
- **Scale jittering**: Random scales
- **No other augmentation** (to isolate architecture improvements)

### Evaluation Metrics

**Average Precision (AP)**: Primary metric

$$\text{AP} = \frac{1}{|R|}\sum_{r \in R} p(r)$$

where $R$ is set of recall values, $p(r)$ is precision at recall $r$.

**COCO metrics**:
- **AP**: Average over IoU thresholds [0.5:0.05:0.95]
- **AP₅₀**: AP at IoU = 0.5
- **AP₇₅**: AP at IoU = 0.75
- **APₛ, APₘ, APₗ**: AP for small, medium, large objects

**Mask-specific**: Same metrics, but IoU computed on masks instead of boxes

### Results (COCO 2016)

**Instance segmentation (mask AP)**:

| Method | Backbone | mask AP | mask AP₅₀ | mask AP₇₅ |
|--------|----------|---------|-----------|-----------|
| FCIS | ResNet-101 | 29.2 | 49.5 | - |
| **Mask R-CNN** | ResNet-50-FPN | **33.7** | **54.5** | **35.4** |
| **Mask R-CNN** | ResNet-101-FPN | **35.7** | **56.5** | **37.8** |

**Bounding box detection (box AP)**:

| Method | Backbone | box AP | box AP₅₀ | box AP₇₅ |
|--------|----------|--------|----------|----------|
| Faster R-CNN | ResNet-101-FPN | 36.2 | 59.1 | 39.0 |
| **Mask R-CNN** | ResNet-101-FPN | **39.8** | **62.3** | **43.4** |

**Key insight**: Adding mask branch also improves box detection!

## Applications and Impact

### Applications

**Computer Vision**:
- **Instance segmentation**: Primary task
- **Object detection**: State-of-the-art results
- **Keypoint detection**: Human pose estimation (add keypoint head)
- **Panoptic segmentation**: Combine with semantic segmentation

**Real-world uses**:
- **Autonomous driving**: Detect and segment vehicles, pedestrians
- **Medical imaging**: Segment organs, tumors, cells
- **Robotics**: Object manipulation, scene understanding
- **Video analysis**: Track and segment objects in video
- **Agriculture**: Crop monitoring, fruit counting
- **Retail**: Product recognition and counting

### Human Pose Estimation

Mask R-CNN easily extends to **keypoint detection**:

Add **keypoint head** (similar to mask head):
- Input: RoIAlign features
- Output: K heatmaps (one per keypoint type)
- Loss: Cross-entropy on heatmaps

**COCO Keypoint Challenge 2016**: Mask R-CNN won with large margin

### Why Mask R-CNN Succeeded

1. **Simple and general**: Clean extension of Faster R-CNN
2. **Strong performance**: State-of-the-art across multiple tasks
3. **Flexible**: Easy to add new heads (masks, keypoints, etc.)
4. **Well-engineered**: RoIAlign, FPN, proper loss design
5. **Fast**: 5 fps on single GPU (200ms per image)

### Impact on Research

**Set new standard** for:
- Instance segmentation (main contribution)
- Object detection (improved Faster R-CNN)
- Multi-task learning (masks + boxes + classes)

**Influenced**:
- **Panoptic segmentation**: Combine instance + semantic
- **Video instance segmentation**: Track instances over time
- **3D object detection**: Adapt to 3D data
- **Detectron2**: Official implementation framework

## Comparison with Other Methods

### Instance Segmentation Methods

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **Mask R-CNN** | Two-stage (proposals then segment) | High accuracy, flexible | Slower |
| YOLACT | One-stage | Fast (30+ fps) | Lower accuracy |
| PANet | Extends Mask R-CNN | Better accuracy | More complex |
| PointRend | Fine boundaries | Sharp edges | Slower |
| SOLOv2 | One-stage | Fast, simple | Lower accuracy on small objects |

### Mask R-CNN vs Semantic Segmentation

| Aspect | Semantic (FCN, U-Net) | Instance (Mask R-CNN) |
|--------|----------------------|----------------------|
| Output | Single segmentation map | Multiple instance masks |
| Instances | Not distinguished | Each object separate |
| Speed | Faster | Slower (more complex) |
| Use case | Scene parsing | Object-centric tasks |

## Key Takeaways

**Core contribution**: Extends Faster R-CNN with mask prediction branch

**Key innovations**:
- **RoIAlign**: Fixes spatial misalignment
- **Decoupled mask prediction**: Per-class binary masks
- **Multi-task learning**: Joint training improves all tasks

**Architecture**: ResNet-FPN backbone + RPN + RoIAlign + parallel heads

**Performance**: State-of-the-art instance segmentation and object detection

**Impact**: Standard for instance segmentation, influenced many follow-up works

**When to use**:
- Need instance-level segmentation
- High accuracy more important than speed
- Multi-task learning (detection + segmentation + keypoints)

## References

- Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick. "Mask R-CNN" ICCV (2017)
- [Original Paper](https://arxiv.org/abs/1703.06870)
- [Detectron2 Implementation](https://github.com/facebookresearch/detectron2)
- [PyTorch Implementation](https://pytorch.org/vision/stable/models.html#mask-r-cnn)
- Ross Girshick et al. "Fast R-CNN" ICCV (2015)
- Shaoqing Ren et al. "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" NIPS (2015)
- Tsung-Yi Lin et al. "Feature Pyramid Networks for Object Detection" CVPR (2017)
- COCO Dataset: [https://cocodataset.org/](https://cocodataset.org/)
