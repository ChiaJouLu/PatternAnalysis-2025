# 2D Prostate Segmentation using Improved UNet on HipMRI Dataset

**Author:** Chia Jou Lu  
**Course:** COMP3710 – Pattern Analysis, The University of Queensland (2025)

This project implements an Improved Unet architecture for automated prostate segmentation from MRI images using the HipMRI Study dataset. The goal is to achieve a Dice similarity coefficient of ≥ 0.75 on the prostate label (Class 3) in the test set.

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

### Architecture Overview:
```
Input (1, 256, 128)
    
[Encoder Path]
  Level 0: 1 -> 64 channels (256x128)
  Level 1: 64 -> 128 channels (128×64)
  Level 2: 128 -> 256 channels (64×32)
  Level 3: 256 -> 512 channels (32×16)
  Level 4: 512 -> 1024 channels (16×8) 
```


## Requirements

See `requirements.txt`

## Usage

TODO: Add usage instructions

### Training
```bash
python train.py
```

### Prediction
```bash
python predict.py
```

## Results

TODO: Add training results
- Dice coefficient on test set
- Training plots
- Sample predictions

## References

TODO: Add references
