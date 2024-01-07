"""
Implementation of Denoising Diffusion Probabilistic Models (DDPM) using Pytorch.

DDPM's network architecture is a UNet 
"""
import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    Each Layer has two conv layers and graphNorm. Also downsample
    """

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        # Multiply number of channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Downsample feature map resolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        return self.relu(out)
    

class ResidualBlockWithAttention(nn.Module):
    """
    ResBlock with self attention added on
    """


class UNet(nn.Module):
    def __init__(self, in_features, out_features, in_channels=3):
        """
        Initialize all component models of the neural network.
        Sequence of 6 blocks of 5 ResNet layers.
        As seen in PixelCNN++ paper.
        """
        super().__init__()

        # Encoder Layers (assume 256 image size)
        ## 1. ResBlock to convert to 64 channels.
        ## 2. Two back to back ResBlocks
        ## 3. Downsample from 256 x 256 x 64 to 128 x 128 x 128
        ## 4. 64 x 64 x 128
        ## 5. 32 x 32 x 256
        ## Self Attention here! after first Resblock
        ## 6. 16 x 16 x 256
        ## 7. 8  x 8  x 512
        ## 8. 4  x 4  x 512

        # Decoder Layers
        # Reverse everything in the encoder layers.
        # Account for skip connections for num channels

        # Convert 256 x 256 x 64 back to 256 x 256 x 3 again using last ResBlock

        # Also add the positional embedding!
