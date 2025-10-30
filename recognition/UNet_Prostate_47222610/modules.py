"""
UNet Model Implementation for 2D Prostate Segmentation

This module implements the UNet architecture for medical image segmentation.
The UNet model is specifically designed for biomedical image segmentation tasks
and has become the standard architecture for such applications.

Architecture:
    The UNet consists of:
    - Encoder (contracting path): Captures context through downsampling
    - Decoder (expanding path): Enables precise localization through upsampling
    - Skip connections: Combine high-resolution features from encoder with 
      upsampled features in decoder

Paper Reference:
    Ronneberger, O., Fischer, P., & Brox, T. (2015).
    U-Net: Convolutional Networks for Biomedical Image Segmentation.
    MICCAI 2015.
    https://arxiv.org/abs/1505.04597

Author: 47222610
Date: October 2025
Assignment: Pattern Recognition Project - 2D Prostate Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    This is the Double Convolution Block, which is a basic building block of UNet.
    Each convolution uses 3×3 kernels with padding=1 to preserve spatial dimensions.
    Batch normalisation helps training stability and convergence.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        
    Architecture:
        Input: (N, in_channels, H, W)
        -> Conv2d(3×3, padding=1)
        -> BatchNorm2d
        -> ReLU
        -> Conv2d(3×3, padding=1)
        -> BatchNorm2d
        -> ReLU
        Output: (N, out_channels, H, W)
        
    Example:
        >>> block = DoubleConv(1, 64)
        >>> x = torch.randn(4, 1, 256, 128)  # Batch of 4 images, 256x128
        >>> out = block(x)
        >>> print(out.shape)  # torch.Size([4, 64, 256, 128])
    """
    
    def __init__(self, in_channels, out_channels):
        """
        Initialize the double convolution block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(DoubleConv, self).__init__()
        
        self.double_conv = nn.Sequential(
            # First Convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Second Convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, feature_map):
        """
        Forward pass through the double convolution block.
        
        Args:
            feature_map (torch.Tensor): Input tensor of shape (N, C_in, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H, W)
        """
        return self.double_conv(feature_map)

