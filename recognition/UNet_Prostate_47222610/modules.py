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

class Upsampling(nn.Module):
    """
    Upsampling block in UNet decoder.
    
    This block performs:
    1. Upsampling (2×2) to increase spatial dimensions by 2 times
    2. Skip Connection: Concatenation with corresponding encoder features
    3. Refines the combined features using DoubleConv
    
    Args:
        in_channels (int): Number of input channels from previous decoder layer
        out_channels (int): Number of output channels
        
    Architecture:
        Decoder input: (N, in_channels, H, W)
        -> ConvTranspose2d(2×2, stride=2)  # Upsample to (H*2, W*2)
        
        Encoder skip: (N, in_channels, H*2, W*2)
        
        Concatenate: (N, in_channels*2, H*2, W*2)
        -> DoubleConv
        Output: (N, out_channels, H*2, W*2)
        
    Example:
        >>> upsampling = Upsampling(128, 64)
        >>> feature_map1 = torch.randn(4, 128, 64, 32)   # From previous decoder layer
        >>> feature_map2 = torch.randn(4, 128, 128, 64)  # From encoder (skip connection)
        >>> out = upsampling(feature_map1, feature_map2)
        >>> print(out.shape)  # torch.Size([4, 64, 128, 64])
    """
    
    def __init__(self, in_channels, out_channels):
        """
        Initialize the upsampling block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(Upsampling, self).__init__()
        
        # Transposed convolution for upsampling
        self.upsampling = nn.ConvTranspose2d(in_channels, in_channels // 2, 
                                     kernel_size=2, stride=2)
        
        # DoubleConv after concatenation
        # Input is in_channels (in_channels//2 from upsampling + in_channels//2 from skip)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, feature_map1, feature_map2):
        """
        Forward pass through the upsampling block.
        
        Args:
            feature_map1 (torch.Tensor): Input from previous decoder layer (N, C, H, W)
            feature_map2 (torch.Tensor): Skip connection from encoder (N, C, H*2, W*2)
            
        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H*2, W*2)
            
        Note:
            If feature_map1 and feature_map2 have slightly different spatial dimensions due to
            odd-sized inputs, feature_map1 will be padded to match feature_map2's dimensions.
        """
        # Upsample feature_map1
        feature_map1 = self.upsampling(feature_map1)
        
        # Handle potential size mismatch due to odd dimensions
        # Calculate padding needed to match feature_map2's spatial dimensions
        diffY = feature_map2.size()[2] - feature_map1.size()[2]  # Height difference
        diffX = feature_map2.size()[3] - feature_map1.size()[3]  # Width difference
        
        # Pad feature_map1 if needed 
        feature_map1 = F.pad(feature_map1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate 
        feature_map = torch.cat([feature_map2, feature_map1], dim=1)
        
        # Apply double convolution
        return self.conv(feature_map)
