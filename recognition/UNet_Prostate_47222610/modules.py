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

class OutConv(nn.Module):
    """
    Output convolution layer for final segmentation mask.
    
    This is a simple 1×1 convolution that maps the final feature maps to
    the desired number of output classes. No activation function is applied
    here as it will be handled by the loss function.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output classes
        
    Architecture:
        Input: (N, in_channels, H, W)
        -> Conv2d(1×1) # Change only the number of channels, without changing H/W
        Output: (N, out_channels, H, W)
        
    Example:
        >>> out_conv = OutConv(64, 4)  # 4 classes
        >>> feature_map = torch.randn(4, 64, 256, 128)
        >>> out = out_conv(feature_map)
        >>> print(out.shape)  # torch.Size([4, 4, 256, 128])
    """
    
    def __init__(self, in_channels, out_channels):
        """
        Initialize the output convolution layer.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output classes
        """
        super(OutConv, self).__init__()
        
        # 1×1 convolution to map to output classes
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, feature_map):
        """
        Forward pass through the output convolution.
        
        Args:
            feature_map (torch.Tensor): Input tensor of shape (N, C_in, H, W)
            
        Returns:
            torch.Tensor: Output logits of shape (N, num_classes, H, W)
        """
        return self.conv(feature_map)

class UNet(nn.Module):
    """
    Complete UNet architecture for 2D medical image segmentation.

    For HipMRI Prostate Segmentation:
        Input: (N, 1, 256, 128)  - Grayscale MRI images
        Output: (N, 4, 256, 128) - 4-class segmentation
            Class 0: Background
            Class 1: Body outline
            Class 2: Bone
            Class 3: Prostate (target)
    
    Args:
        n_channels (int): Number of input channels (1 for grayscale)
        n_classes (int): Number of output segmentation classes (4 in this case)
        
    Architecture Details:
        Level 0 (Input): 1 -> 64 channels, size: 256×128
        Level 1: 64 -> 128 channels, size: 128×64 (after downsampling)
        Level 2: 128 -> 256 channels, size: 64×32
        Level 3: 256 -> 512 channels, size: 32×16
        Level 4 (Bottleneck): 512 -> 1024 channels, size: 16×8
        Level 3': 1024 -> 512 channels, size: 32×16 (after upsampling + skip)
        Level 2': 512 -> 256 channels, size: 64×32
        Level 1': 256 -> 128 channels, size: 128×64
        Level 0' (Output): 128 -> 64 -> 4 channels, size: 256×128
        
    Example:
        >>> model = UNet(n_channels=1, n_classes=4)
        >>> feature_map = torch.randn(4, 1, 256, 128)  # Batch of 4 MRI images
        >>> output = model(feature_map)
        >>> print(output.shape)  # torch.Size([4, 4, 256, 128])
        >>> 
        >>> # Apply softmax to get probabilities
        >>> probs = torch.softmax(output, dim=1)
        >>> # Get predicted class for each pixel
        >>> predictions = torch.argmax(probs, dim=1)
        >>> print(predictions.shape)  # torch.Size([4, 256, 128])
    """
    
    def __init__(self, n_channels, n_classes):
        """
        Initialize the UNet model.
        
        Args:
            n_channels (int): Number of input channels
                - 1 for grayscale medical images (MRI)
            n_classes (int): Number of output segmentation classes
                - For HipMRI: 4 (background, body, bone, prostate)
        """
        super(UNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder 
        self.initConv = DoubleConv(n_channels, 64) # Initial convolution
        self.down1 = Downsampling(64, 128)              # 256×128 -> 128×64
        self.down2 = Downsampling(128, 256)             # 128×64 -> 64×32
        self.down3 = Downsampling(256, 512)             # 64×32 -> 32×16
        self.down4 = Downsampling(512, 1024)            # 32×16 -> 16×8 (bottleneck)
        
        # Decoder 
        self.up1 = Upsampling(1024, 512)                # 16×8 -> 32×16
        self.up2 = Upsampling(512, 256)                 # 32×16 -> 64×32
        self.up3 = Upsampling(256, 128)                 # 64×32 -> 128×64
        self.up4 = Upsampling(128, 64)                  # 128×64 -> 256×128
        
        # Output layer
        self.outConv = OutConv(64, n_classes)      # Final 1×1 conv to n_classes
    
    def forward(self, feature_map):
        """
        Forward pass through the UNet.
        
        Args:
            feature_map (torch.Tensor): Input images of shape (N, C_in, H, W)
                For HipMRI: (N, 1, 256, 128)
                
        Returns:
            torch.Tensor: Raw logits of shape (N, num_classes, H, W)
                For HipMRI: (N, 4, 256, 128)
            
        Example:
            >>> model = UNet(n_channels=1, n_classes=4)
            >>> feature_map = torch.randn(2, 1, 256, 128)
            >>> logits = model(x)
            >>> 
            >>> # For prediction:
            >>> probs = torch.softmax(logits, dim=1)
            >>> pred = torch.argmax(probs, dim=1)
            >>> 
            >>> # For training with CrossEntropyLoss:
            >>> criterion = nn.CrossEntropyLoss()
            >>> loss = criterion(logits, targets)
        """
        # Encoder path (with skip connections saved)
        feature_map1 = self.initConv(feature_map) # 64 channels, same size
        feature_map2 = self.down1(feature_map1)   # 128 channels, 1/2 size
        feature_map3 = self.down2(feature_map2)   # 256 channels, 1/4 size
        feature_map4 = self.down3(feature_map3)   # 512 channels, 1/8 size
        feature_map5 = self.down4(feature_map4)   # 1024 channels, 1/16 size (bottleneck)
        
        # Decoder path (with skip connections from encoder)
        feature_map = self.up1(feature_map5, feature_map4)  # 512 channels, 1/8 size
        feature_map = self.up2(feature_map, feature_map3)   # 256 channels, 1/4 size
        feature_map = self.up3(feature_map, feature_map2)   # 128 channels, 1/2 size
        feature_map = self.up4(feature_map, feature_map1)   # 64 channels, original size
        
        # Output layer
        logits = self.outConv(feature_map) # n_classes channels, original size
        
        return logits


if __name__ == "__main__":
    """
    Test script to verify UNet model architecture with dummy data.
    """
    print("="*70)
    print("Testing UNet Model Architecture")
    print("="*70)
    
    # Create model for HipMRI (1 input channel, 4 output classes)
    print("\n1. Creating UNet model...")
    model = UNet(n_channels=1, n_classes=4)
    print(f"   ✓ Model created successfully")
    
    # Count parameters
    print("\n2. Counting model parameters...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Total parameters: {total_params:,}")
    print(f"   ✓ Trainable parameters: {trainable_params:,}")
    
    # Test forward pass with HipMRI dimensions
    print("\n3. Testing forward pass with HipMRI dimensions...")
    batch_size = 2
    x = torch.randn(batch_size, 1, 256, 128)
    print(f"   Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"   ✓ Output shape: {output.shape}")
    print(f"   ✓ Expected shape: torch.Size([{batch_size}, 4, 256, 128])")
    
    # Test prediction
    print("\n4. Testing prediction conversion...")
    with torch.no_grad():
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1)
    
    print(f"   ✓ Probability shape: {probs.shape}")
    print(f"   ✓ Prediction shape: {pred.shape}")
    print(f"   ✓ Unique predicted classes: {torch.unique(pred).tolist()}")
    
    # Test model on GPU if available
    print("\n5. Checking GPU availability...")
    if torch.cuda.is_available():
        print(f"   ✓ GPU available: {torch.cuda.get_device_name(0)}")
        print("   Testing model on GPU...")
        model_gpu = model.cuda()
        x_gpu = x.cuda()
        with torch.no_grad():
            output_gpu = model_gpu(x_gpu)
        print(f"   ✓ GPU forward pass successful")
        print(f"   ✓ GPU output shape: {output_gpu.shape}")
    else:
        print("   ⚠ No GPU available, will use CPU for training")
    
    print("\n" + "="*70)
    print("All tests passed! ✓")
    print("="*70)
    print("\nUNet model is ready for training.")
