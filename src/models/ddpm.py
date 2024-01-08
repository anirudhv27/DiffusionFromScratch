"""
Implementation of Denoising Diffusion Probabilistic Models (DDPM) using Pytorch.

DDPM's network architecture is a UNet 
"""
import math

import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    """
    Positional Encoding to be added at each residual block.
    Take t (number of diffusion timesteps) as an input
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.linear1 = nn.Linear(emb_dim, emb_dim * 4)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(emb_dim * 4, emb_dim * 4)

    def forward(self, t: torch.Tensor):
        """
        Arguments:
            t: torch.Tensor with shape [n]. Represents timesteps given to diffusion model.

        Learn linear transformation of sinusoidal embedding of t.
        """
        assert len(t.shape) == 1
        half_dim = self.emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        emb = self.linear2(self.act(self.linear1(emb)))

        assert emb.shape == (t.shape[0], self.emb_dim * 4)
        return emb

class ResidualBlock(nn.Module):
    """
    Each Layer has two conv layers and graphNorm. Also downsample
    """

    def __init__(
        self, in_channels: int, out_channels: int, num_groups=32, dropout_p=0.1
    ) -> None:
        super().__init__()
        # Multiply number of channels
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.act = nn.SiLU()  #
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(dropout_p)

        # Downsample feature map resolution
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t) -> torch.Tensor:
        residual = self.skip(x)

        out = self.conv1(self.act(self.gn1(x)))
        # Add timestep embedding

        out = self.act(self.gn2(x))
        out = self.dropout(out)
        out = self.conv2(out)

        assert out.shape == residual.shape

        return out + residual


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
