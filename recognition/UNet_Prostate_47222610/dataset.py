"""
Dataset Loader for HipMRI Prostate Segmentation

This module provides data loading utilities for the HipMRI prostate MRI dataset
using ONLY the provided utility functions from the assignment Appendix B.

Author: 47222610
Date: October 2025
Assignment: Pattern Recognition Project - 2D Prostate Segmentation

Note:
    Part of the documentation style and test scaffolding were written
    with assistance from ChatGPT 4.1. All implementation logic
    and verification were completed by the author.

References:
    - Assignment Appendix B: Provided utility functions
    - NIfTI file format: https://nifti.nimh.nih.gov/
    - Nibabel library: https://nipy.org/nibabel/
"""

import numpy as np
import nibabel as nib
from tqdm import tqdm


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """
    Convert label array to one-hot encoded channels.
    
    Provided utility function from assignment materials (Appendix B).
    Creates separate binary channels for each unique class label.
    
    Args:
        arr (np.ndarray): Input label array with integer class labels
        dtype: Data type for output array (default: np.uint8)
        
    Returns:
        np.ndarray: One-hot encoded array with shape (*arr.shape, num_classes)
        
    Example:
        >>> labels = np.array([[0, 1], [1, 2]])
        >>> one_hot = to_channels(labels)
        >>> print(one_hot.shape)
        (2, 2, 3)  # 3 classes: 0, 1, 2
        >>> print(one_hot[:, :, 1])  # Channel for class 1
        [[0 1]
         [1 0]]
         
    Note:
        For HipMRI prostate data, classes are:
        0 = background
        1 = body outline
        2 = bone
        3 = prostate
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    
    return res


def load_data_2D(imageNames, normImage=False, categorical=False, 
                 dtype=np.float32, getAffines=False, early_stop=False):
    """
    Load medical image data from NIfTI files into pre-allocated arrays.
    
    Provided utility function from assignment materials (Appendix B).
    This function loads all images at once into memory using a pre-allocated
    array to avoid excessive memory usage. Displays a progress bar during loading.
    
    The function is designed for batch loading and is suitable for datasets that
    can fit into memory. For the HipMRI 2D dataset, this is appropriate as the
    processed 2D slices are memory-efficient.
    
    Args:
        imageNames (list): List of file paths to NIfTI images (.nii.gz files)
        normImage (bool): Whether to normalize images using Z-score normalization.
            If True, applies (image - mean) / std to each image independently.
            Recommended: True for training neural networks.
            Default: False
        categorical (bool): Whether to convert labels to one-hot encoding.
            If True, uses to_channels() to create separate channels per class.
            Recommended: True for segmentation labels.
            Default: False
        dtype: NumPy data type for output array. 
            Recommended: np.float32 for images, np.uint8 for labels.
            Default: np.float32
        getAffines (bool): Whether to return affine transformation matrices.
            Affine matrices contain spatial information about image orientation.
            Useful for saving predictions back to NIfTI format.
            Default: False
        early_stop (bool): Stop after loading 20 images (for quick testing).
            Useful for debugging code without loading the full dataset.
            Default: False
        
    Returns:
        np.ndarray: Array of images with shape:
            - (num_images, height, width) if categorical=False
            - (num_images, height, width, num_classes) if categorical=True
        list: Affine matrices (only returned if getAffines=True)
        
    Raises:
        FileNotFoundError: If any image file in imageNames doesn't exist
        ValueError: If images have inconsistent dimensions
        
    Note:
        - For HipMRI data, automatically handles 3D files by taking first slice
        - Z-score normalization formula: (x - μ) / σ for each image independently
        - Progress bar shows loading progress using tqdm
        - Memory pre-allocation prevents memory fragmentation
        
    Example:
        >>> import glob
        >>> 
        >>> # Get file paths
        >>> train_files = glob.glob('/path/to/train/*.nii.gz')
        >>> 
        >>> # Load all images with normalization
        >>> images = load_data_2D(train_files, normImage=True)
        >>> print(f"Loaded {images.shape[0]} images")
        >>> print(f"Image shape: {images.shape[1:3]}")
        >>> 
        >>> # Load labels with one-hot encoding
        >>> label_files = glob.glob('/path/to/labels/*.nii.gz')
        >>> labels = load_data_2D(label_files, categorical=True)
        >>> print(f"Number of classes: {labels.shape[-1]}")
        >>> 
        >>> # Load with affine matrices (for saving predictions later)
        >>> images, affines = load_data_2D(train_files, 
        >>>                                 normImage=True, 
        >>>                                 getAffines=True)
        >>> print(f"Got {len(affines)} affine matrices")
        
    Technical Details:
        Memory Usage:
            For HipMRI 2D data (256x128 images):
            - 1000 images × 256 × 128 × 4 bytes (float32) ≈ 131 MB
            - Very memory efficient for modern systems
            
        Normalization:
            Z-score normalization is applied per-image:
            normalized = (image - image.mean()) / (image.std() + 1e-8)
            The small epsilon (1e-8) prevents division by zero.
            
        One-Hot Encoding:
            For a 256×128 label image with 4 classes:
            Original: (256, 128) with values [0, 1, 2, 3]
            One-hot: (256, 128, 4) with binary channels
    """
    affines = []
    
    # Get fixed size from first image
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    
    # Handle 3D data (remove extra dimension if present)
    # Some NIfTI files may have shape (H, W, 1) instead of (H, W)
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]
    
    # Pre-allocate array based on categorical flag
    # This prevents memory fragmentation and is more efficient
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)
    
    # Load all images with progress bar
    for i, inName in enumerate(tqdm(imageNames, desc='Loading images')):
        # Load NIfTI file
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # Read from disk only
        affine = niftiImage.affine
        
        # Handle 3D data in HipMRI_study dataset
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]
        
        # Convert to specified data type
        inImage = inImage.astype(dtype)
        
        # Normalize if requested (Z-score normalization)
        # Formula: (x - μ) / σ
        # This standardizes the image to have zero mean and unit variance
        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        
        # Convert to one-hot encoding if categorical
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage
        
        # Store affine matrix for later use
        affines.append(affine)
        
        # Early stop for testing (loads only ~20 images)
        if i > 20 and early_stop:
            break
    
    # Return images with or without affine matrices
    if getAffines:
        return images, affines
    else:
        return images


if __name__ == "__main__":
    """
    Test script to verify dataset loading functionality.
    
    This script tests the dataset loader on a small subset of the HipMRI data
    to ensure everything works correctly before training.
    
    Run with: python dataset.py
    """
    import glob
    
    print("="*70)
    print("Testing HipMRI Dataset Loader")
    print("="*70)
    
    # Data paths on Rangpur (as per assignment specifications)
    DATA_PATH = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data'
    
    # Get file lists (using only first 10 for testing)
    print("\n1. Loading file paths...")
    train_images = sorted(glob.glob(f'{DATA_PATH}/keras_slices_train/*.nii.gz'))[:10]
    train_labels = sorted(glob.glob(f'{DATA_PATH}/keras_slices_seg_train/*.nii.gz'))[:10]
    
    print(f"   Found {len(train_images)} training images")
    print(f"   Found {len(train_labels)} training labels")
    
    # Test loading images
    print("\n2. Testing image loading with normalization...")
    images = load_data_2D(train_images, normImage=True, early_stop=True)
    print(f"   ✓ Loaded images shape: {images.shape}")
    print(f"   ✓ Images dtype: {images.dtype}")
    print(f"   ✓ Images range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Test loading labels
    print("\n3. Testing label loading with one-hot encoding...")
    labels = load_data_2D(train_labels, categorical=True, early_stop=True)
    print(f"   ✓ Loaded labels shape: {labels.shape}")
    print(f"   ✓ Labels dtype: {labels.dtype}")
    print(f"   ✓ Number of classes: {labels.shape[-1]}")
    
    # Test with affines
    print("\n4. Testing affine matrix loading...")
    images, affines = load_data_2D(train_images[:3], getAffines=True, early_stop=True)
    print(f"   ✓ Images shape: {images.shape}")
    print(f"   ✓ Number of affines: {len(affines)}")
    print(f"   ✓ Affine matrix shape: {affines[0].shape}")
    
    print("\n" + "="*70)
    print("All tests passed! ✓")
    print("="*70)
    print("\nDataset loader is ready for use in training.")
