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

## Dataset

TODO: Describe the HipMRI dataset
- Number of samples
- Image size
- Number of classes

## Model Architecture

TODO: Describe UNet2D architecture
- Encoder layers
- Bottleneck
- Decoder layers

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
