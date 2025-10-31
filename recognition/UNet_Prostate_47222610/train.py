"""
Train Unet for prostate segmentation.

This is the training script for the HipMRI dataset.
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

def dice_coefficient(predicted, target):
    """
    Calculate Dice coefficient for segmentation.

    Args:
        predicted: Predicted segmentated image.
        target: Ground truth segmentated image.

    Returns:
        Dice coefficient value as a float on range [0, 1].
    """
    if predicted.shape == target.shape:
        batch, num_class = predicted.shape[:2]
        pred_flat = predicted.reshape(batch, num_class, -1)
        tar_flat = target.reshape(batch, num_class, -1)

        # 0 intersect 1 = 0, only 1 intersect 1 = 1, same as multiplier
        intersection = (pred_flat * tar_flat).sum(axis=2)
        denominator = pred_flat.sum(axis=2) + tar_flat.sum(axis=2)

        # Handle denominator = 0
        dice = np.where(
            denominator == 0, 
            0, 
            (2. * intersection) / denominator)

        # Overall average, without further classes distinction
        return dice.mean()




if __name__ == "__main__":
    """
    Main function that runs some small tests.
    """
    # Test dice_coefficient function
    # Simulate 2 images, 2 classes, each image is 4x4
    pred = np.array([
        [[[0,1,0,0],
          [1,1,0,0],
          [0,0,1,1],
          [0,0,0,1]],
         
         [[1,0,0,0],
          [0,0,0,0],
          [0,1,1,0],
          [1,1,0,0]]],
        
        [[[0,1,0,1],
          [1,1,1,0],
          [0,0,0,1],
          [0,0,1,1]],

         [[1,0,0,0],
          [0,0,0,1],
          [1,1,0,0],
          [0,1,1,0]]]
    ])

    target = np.array([
        [[[0,1,0,0],
          [1,1,0,0],
          [0,0,1,0],
          [0,0,0,1]],

         [[1,0,0,0],
          [0,0,0,0],
          [0,1,1,1],
          [1,1,0,0]]],

        [[[0,1,0,1],
          [1,1,0,0],
          [0,0,0,1],
          [0,0,1,1]],

         [[1,0,0,0],
          [0,0,0,1],
          [1,1,0,0],
          [0,1,1,0]]]
    ])

    dice = dice_coefficient(pred, target)
    print(f"Dice Coefficient = {dice:.4f}")
