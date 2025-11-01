"""
Dataset Loader for HipMRI Prostate Segmentation

This module provides data loading utilities for the HipMRI prostate MRI dataset
using ONLY the provided utility functions from the assignment Appendix B.

Author: 47222610
Date: October 2025
Assignment: Pattern Recognition Project - 2D Prostate Segmentation

References:
    - Assignment Appendix B: Provided utility functions
    - NIfTI file format: https://nifti.nimh.nih.gov/
    - Nibabel library: https://nipy.org/nibabel/
"""
import numpy as np
import nibabel as nib
from tqdm import tqdm

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    
    return res

# load medical image functions
def load_data_2D(imageNames, normImage=False, categorical=False, 
                 dtype=np.float32, getAffines=False, early_stop=False):
    """
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory ↘
    untitled folder usage.

    normImage: bool (normalise the image 0.0 -1.0)
    early_stop: Stop loading pre-maturely, leaves arrays mostly empty, for quick ↘
        loading and testing scripts.
    """
    affines = []
    
    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)
    
    for i, inName in enumerate(tqdm(imageNames, desc='Loading images')):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage
        
        affines.append(affine)
        if i > 20 and early_stop:
            break
    
    if getAffines:
        return images, affines
    else:
        return images
