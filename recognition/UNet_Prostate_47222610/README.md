# 2D Prostate Segmentation using Improved UNet on HipMRI Dataset

**Author:** Chia Jou Lu  
**Course:** COMP3710 – Pattern Analysis, The University of Queensland (2025)

This project implements an **Improved Unet** architecture for automated prostate segmentation from MRI images using the HipMRI Study dataset. The goal is to achieve a Dice similarity coefficient of ≥ 0.75 on the prostate label (Class 3) in the test set.

## Problem Description

Medical image segmentation is crucial for radiotherapy planning in prostate cancer. This project segments four anatomical regions from 2D magnetic resonance imaging (MRI) slices:
- Class 0: Background
- Class 1: Peripheral Zone (Body Outline)
- Class 2: Transition Zone (Bone)
- Class 3: **Prostate (Primary Target)**

The Improved UNet architecture enhances the original UNet through architectural improvements.

## Model Architecture

### Improved UNet vs Standard UNet
The improved UNet incorporates several improvements over the original UNet:

**Key Improvements:**
1. **Deeper Network**: There are 5 levels of encoding/ decoding in Improved UNet, but only 4 in standard UNet.
2. **Residual Connections**: Skip connections using residual blocks for better gradient flow.
3. **Instance Normalization**: More stable than batch normalization for small batch sizes.
4. **Leaky ReLU**: Prevents the ReLU function from failing on negative slopes (alpha = 0.01).
5. **Deep Supervision**: Additional loss at intermediate decoder layers.
6. **Context Module**: Additional context aggregation at bottleneck.

### Network Architecture:
```
Input: (N, 1, 256, 128) - Grayscale MRI images

[Encoder Path - Downsampling]
    Level 0: ResidualDoubleConv: 1 -> 64 channels (256×128)
        MaxPool2d(2×2)
    Level 1: ResidualDoubleConv: 64 -> 128 channels (128×64)
        MaxPool2d(2×2)
    Level 2: ResidualDoubleConv: 128 -> 256 channels (64×32)
        MaxPool2d(2×2)
    Level 3: ResidualDoubleConv: 256 -> 512 channels (32×16)
        MaxPool2d(2×2)
    Level 4 (Bottleneck): ResidualDoubleConv: 512 -> 1024 channels (16×8)

[Context Aggregation Module]
    Parallel dilated convolutions with rates [1, 2, 4, 8]
    Receptive fields: 3×3, 7×7, 15×15, 31×31
    Aggregated multi-scale features (1024 channels)

[Decoder Path with Deep Supervision]
    Level 3: TransposeConv + Skip + ResidualDoubleConv: 1024 -> 512 (32×16)
        ├─ Auxiliary Output: DSV4 (512 -> 4 classes)

    Level 2: TransposeConv + Skip + ResidualDoubleConv: 512 -> 256 (64×32)
        ├─ Auxiliary Output: DSV3 (256 -> 4 classes)

    Level 1: TransposeConv + Skip + ResidualDoubleConv: 256 -> 128 (128×64)
        ├─ Auxiliary Output: DSV2 (128 -> 4 classes)

    Level 0: TransposeConv + Skip + ResidualDoubleConv: 128 -> 64 (256×128)
        ├─ Auxiliary Output: DSV1 (64 -> 4 classes)

[Output Layer]
    1×1 Convolution: 64 -> 4 channels
    Output: (N, 4, 256, 128) - Class logits
```

### Key Architectural Components

**1. Residual Blocks**
- Two 3x3 convolutions with skip connections
- Enables gradient flow in deep networks

**2. Instance Normalization**
- Normalizes per sample
- More stable than Batch Normalization for medical imaging

**3. Context Aggregation**
- Parallel dilated convolutions at bottleneck
- Captures features at multiple scales (3x3 to 31x31)

**4. Deep Supervision**
- Auxiliary outputs at 5 decoder levels
- Loss weights: 1.0, 0.8, 0.6, 0.4, 0.2

## Dataset

**Source**: HipMRI Study on Prostate Cancer 

**Format**: NIfTI (.nii.gz)

**Data Splits**:
- Training: 11,460 slices
- Validation: 660 slices
- Testing: 540 slices

**Preprocessing**:
1. Load NIFTI files with nibabel
2. Resize to 256x128
3. Z-score normalization: '(img - mean) / std'
4. Clean invalid labels (≥4 -> class 0)
5. One-hot encode to 4 classes

## Dependencies
```bash
torch>=2.0.0
numpy>=1.24.0
nibabel>=5.0.0
matplotlib>=3.7.0
opencv-python>=4.7.0
tqdm>=4.65.0
```

## Project Structure
```
UNet_Prostate_47222610/
├── README.md               # This file
├── dataset.py              # Data loading and preprocessing for MRI slices
├── modules.py              # Improved UNet architecture
├── predict.py              # Testing and visualization
└── train.py                # Training with deep supervision
```

## Reproducibility

### Training Configuration

**Fixed Hyperparameters**:
- Architecture: Improved UNet (5-level encoder/decoder)
- Epochs: 30
- Batch size: 16
- Learning rate: 1e-4 (Adam optimizer)
- Weight decay: 1e-5 (L2 regularization)
- Loss function: CrossEntropyLoss + Deep Supervision
- Image size: 256×128
- Number of classes: 4

### File Outputs Summary

After training and evaluation:
```
UNet_Prostate_47222610/
├── improved_unet_best.pth           # Best model 
├── improved_unet_final.pth          # Final model 
├── improved_unet_epoch_*.pth        # Checkpoints 
├── logs/
│   └── improved_unet_*.out          # Training logs (text)
└── results/
    ├── training_curves.png          # Loss/Dice plots
    └── prediction_batch_*.png       # Sample predictions
```

## Results

### Test Set Performance
| Class | Region | Dice |
|-------|--------|------|




### Visualizations




## References



## Academic Integrity



## Author
**Student Name**: Chia Jou Lu
**Student ID**: 47222610
**Course**: COMP3710 Pattern Recognition 
**Institution**: The University of Queensland  
**Date**: November 2025








