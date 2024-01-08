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

class Downsample(nn.Module):
    '''
    Conv2d to downsample in half.
    '''
    pass

class Upsample(nn.Module):
    '''
    Conv2d to upsample image 2x
    '''
    pass

class ResidualBlock(nn.Module):
    """
    Implement each Residual Block and Sampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups=32,
        dropout_p=0.1,
        has_attn=False,
    ) -> None:
        super().__init__()
        # Multiply number of channels
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.act = nn.SiLU()  #
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(dropout_p)

        # Linear Layer for sinusoidal embedding
        self.time_dense = nn.Linear(out_channels, out_channels)

        # Downsample feature map resolution
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(
                in_channels, out_channels, 1
            )  # Learn expansion of data to out_channels dimension!

        if has_attn:
            self.attn = nn.MultiheadAttention(out_channels, num_heads=1)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, emb_t: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        out = self.conv1(self.act(self.gn1(x)))  # Shape = (B x out_channels x H x W)

        # Add timestep embedding
        emb_t = self.time_dense(self.act(emb_t))  # Shape = (B x out_channels)

        out = out + emb_t[:, :, None, None]

        out = self.act(self.gn2(x))
        out = self.dropout(out)
        out = self.conv2(out)

        assert out.shape == residual.shape

        return out + residual

class DownsampleBlock(nn.Module):
    '''
    Residual Block + Downsampling. 
    '''
    pass

class UpsampleBlock(nn.Module):
    '''
    Residual Block + Upsampling. Shape Layers to allow skip connections.
    '''
    pass

class BottleneckBlock(nn.Module):
    '''
    Bottleneck Block. Res + Attn + Res. All shapes stay the same.
    '''
    pass

class DiffusionUNet(nn.Module):
    """
    Compose ResidualBlocks in UNet for noise prediction.
    """

    def __init__(self, in_channels=3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 128)

        self.time_embed = SinusoidalEmbedding()
        
        # Encoder Layers:

        # Bottleneck Layer

        # Decoder Layers


    def forward(self, img: torch.Tensor, t: torch.Tensor):
        # TODO: Fix Positional Embedding t. Only compute linear transform once, use the same embedding in all layers.
        # Implement forward logic through all layers. Make sure to implement skip connections!        
        emb_t = self.time_embed(t)
        
