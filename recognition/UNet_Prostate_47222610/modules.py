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


class Downsampling(nn.Module):
    """
    Downsampling block in UNet encoder, which captures context while reducing 
    spatial resolution.
    
    This block performs:
    1. MaxPooling (2×2) to reduce spatial dimensions in a half
    2. Extracts features at the new resolution using double convolution
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        
    Architecture:
        Input: (N, in_channels, H, W)
        -> MaxPool2d(2×2)  # Reduces to (H/2, W/2)
        -> DoubleConv
        Output: (N, out_channels, H/2, W/2)
        
    Example:
        >>> down = Downsampling(64, 128)
        >>> feature_map = torch.randn(4, 64, 256, 128)
        >>> out = down(feature_map)
        >>> print(out.shape)  # torch.Size([4, 128, 128, 64])
    """
    
    def __init__(self, in_channels, out_channels):
        """
        Initialize the downsampling block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(Downsampling, self).__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # Downsample use 2x2
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, feature_map):
        """
        Forward pass through the downsampling block.
        
        Args:
            feature_map (torch.Tensor): Input tensor of shape (N, C_in, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H/2, W/2)
        """
        return self.maxpool_conv(feature_map)
