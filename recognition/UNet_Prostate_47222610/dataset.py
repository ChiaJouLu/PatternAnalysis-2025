"""
Dataset loader for HipMRI Prostate Segmentation
Author: s4722261
"""

import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np

# TODO: Implement HipMRIDataset class
# Hint: Load .nii.gz files, normalize images, return (image, label) pairs

# Path to data
DATA_PATH = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data'

