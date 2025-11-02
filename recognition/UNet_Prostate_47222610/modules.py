"""
Improved UNet Model Implementation for 2D Prostate Segmentation

This module implements the Improved UNet architecture based on Isensee et al. 2018.

Key improvements over standard UNet:
    - Instance Normalization instead of Batch Normalization
    - Leaky ReLU instead of ReLU
    - Residual connections in encoder/decoder blocks
    - Context aggregation module with dilated convolutions
    - Deep supervision at multiple scales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDoubleConv(nn.Module):
    """
    Improved Double Convolution Block with residual connections.

    Key improvements:
        - Instance Normalization for better stability with small batches.
        - Leaky ReLU to prevent dying neurons.
        - Residual connection for better gradient flow.
        
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        
    """
    
    def __init__(self, in_channels, out_channels):
        super(ResidualDoubleConv, self).__init__()
        
        self.double_conv = nn.Sequential(
            # First Convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels), # Instance Norm instead of Batch Norm
            nn.LeakyReLU(negative_slope=0.01, inplace=True), # Leaky ReLU 
            
            # Second Convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        # 1x1 conv for residual if channel dimensions change
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()
    
    def forward(self, feature_map):
        """
        Forward pass through the double convolution block.
        """
        residual = self.residual_conv(feature_map)
        out = self.double_conv(feature_map)
        return out + residual  # Residual connection


class ContextModule(nn.Module):
    """
    Context Aggregation Module using dilated convolutions.
    
    Args:
        channels (int): Number of channels
    """

    def __init__(self, channels):
        super(ContextModule, self).__init__()
        
        # Dilated convolutions with different rates (1, 2, 4, 8)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(channels, channels, 3, padding=4, dilation=4)
        self.conv8 = nn.Conv2d(channels, channels, 3, padding=8, dilation=8)
        
        self.norm = nn.InstanceNorm2d(channels)
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, feature_map):
        feature_map1 = self.activation(self.norm(self.conv1(feature_map)))
        feature_map2 = self.activation(self.norm(self.conv2(feature_map)))
        feature_map4 = self.activation(self.norm(self.conv4(feature_map)))
        feature_map8 = self.activation(self.norm(self.conv8(feature_map)))
        
        return feature_map + feature_map1 + feature_map2 + feature_map4 + feature_map8


class Downsampling(nn.Module):
    """
    Downsampling block with residual connections.
    """
    
    def __init__(self, in_channels, out_channels):
        super(Downsampling, self).__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # Downsample use 2x2
            ResidualDoubleConv(in_channels, out_channels)
        )
    
    def forward(self, feature_map):
        """
        Forward pass through the downsampling block.
        """
        return self.maxpool_conv(feature_map)


class Upsampling(nn.Module):
    """
    Upsampling block with residual connections.
    """
    
    def __init__(self, in_channels, out_channels):
        super(Upsampling, self).__init__()
        
        self.upsampling = nn.ConvTranspose2d(in_channels, in_channels // 2, 
                                     kernel_size=2, stride=2)
        
        # DoubleConv after concatenation
        self.conv = ResidualDoubleConv(in_channels, out_channels)
    
    def forward(self, feature_map1, feature_map2):
        """
        Forward pass through the upsampling block.
        """
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
        
        return self.conv(feature_map)


class OutConv(nn.Module):
    """
    Output convolution layer for final segmentation mask.
    """
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        
        # 1×1 convolution to map to output classes
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, feature_map):
        """
        Forward pass through the output convolution.
        """
        return self.conv(feature_map)


class ImprovedUNet(nn.Module):
    """
    Improved UNet architecture.
    
    Key improvements over standard UNet:
    1. Instance Normalization instead of Batch Normalization.
    2. Leaky ReLU activation.
    3. Residual connections in all conv blocks.
    4. Context aggregation module at bottleneck.
    5. Deep supervision at decoder levels.
    
    Args:
        n_channels (int): Number of input channels
        n_classes (int): Number of output classes
        deep_supervision (bool): Whether to use deep supervision
    """
    
    def __init__(self, n_channels, n_classes, deep_supervision=True):
        super(ImprovedUNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.deep_supervision = deep_supervision
        
        # Encoder 
        self.initConv = ResidualDoubleConv(n_channels, 64) # Initial convolution
        self.down1 = Downsampling(64, 128)                 # 256×128 -> 128×64
        self.down2 = Downsampling(128, 256)                # 128×64 -> 64×32
        self.down3 = Downsampling(256, 512)                # 64×32 -> 32×16
        self.down4 = Downsampling(512, 1024)               # 32×16 -> 16×8 

        # Context module at bottleneck
        self.context = ContextModule(1024)
        
        # Decoder 
        self.up1 = Upsampling(1024, 512)                # 16×8 -> 32×16
        self.up2 = Upsampling(512, 256)                 # 32×16 -> 64×32
        self.up3 = Upsampling(256, 128)                 # 64×32 -> 128×64
        self.up4 = Upsampling(128, 64)                  # 128×64 -> 256×128
        
        # Output 
        self.outConv = OutConv(64, n_classes)      # Final 1×1 conv to n_classes

        # Deep supervision outputs 
        if self.deep_supervision:
            self.dsv4 = nn.Conv2d(512, n_classes, 1)
            self.dsv3 = nn.Conv2d(256, n_classes, 1)
            self.dsv2 = nn.Conv2d(128, n_classes, 1)
            self.dsv1 = nn.Conv2d(64, n_classes, 1)
    
    def forward(self, feature_map):
        """
        Forward pass through the ImprovedUNet.
        """
        # Encoder 
        feature_map1 = self.initConv(feature_map) # 64 channels, same size
        feature_map2 = self.down1(feature_map1)   # 128 channels, 1/2 size
        feature_map3 = self.down2(feature_map2)   # 256 channels, 1/4 size
        feature_map4 = self.down3(feature_map3)   # 512 channels, 1/8 size
        feature_map5 = self.down4(feature_map4)   # 1024 channels, 1/16 size 

        # Context aggregation
        feature_map5 = self.context(feature_map5)
        
        # Decoder with skip connections 
        d4 = self.up1(feature_map5, feature_map4)  # 512 channels, 1/8 size
        d3 = self.up2(d4, feature_map3)   # 256 channels, 1/4 size
        d2 = self.up3(d3, feature_map2)   # 128 channels, 1/2 size
        d1 = self.up4(d2, feature_map1)   # 64 channels, original size
        
        # Output 
        logits = self.outConv(d1) 

        # Deep supervision outputs
        if self.deep_supervision and self.training:
            # Upsample intermediate outputs to match target size
            dsv4 = F.interpolate(self.dsv4(d4), size=feature_map.shape[2:], 
                                mode='bilinear', align_corners=False)
            dsv3 = F.interpolate(self.dsv3(d3), size=feature_map.shape[2:], 
                                mode='bilinear', align_corners=False)
            dsv2 = F.interpolate(self.dsv2(d2), size=feature_map.shape[2:], 
                                mode='bilinear', align_corners=False)
            dsv1 = F.interpolate(self.dsv1(d1), size=feature_map.shape[2:], 
                                mode='bilinear', align_corners=False)
            
            return logits, dsv1, dsv2, dsv3, dsv4
        else:
            return logits


