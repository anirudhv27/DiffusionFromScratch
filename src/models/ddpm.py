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

        self.linear1 = nn.Linear(emb_dim // 4, emb_dim)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, t: torch.Tensor):
        """
        Arguments:
            t: torch.Tensor with shape [n]. Represents timesteps given to diffusion model.

        Learn linear transformation of sinusoidal embedding of t.
        """
        assert len(t.shape) == 1
        half_dim = self.emb_dim // 8
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        emb = self.linear2(self.act(self.linear1(emb)))

        assert emb.shape == (t.shape[0], self.emb_dim)
        return emb


class Downsample(nn.Module):
    """
    Conv2d to downsample in half.
    """

    def __init__(self, num_channels):
        self.conv = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Upsample(nn.Module):
    """
    Conv2d to Upsample by 2x
    """

    def __init__(self, num_channels):
        self.conv_transpose = nn.ConvTranspose2d(
            num_channels, num_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor):
        return self.conv_transpose(x)


class ResidualBlock(nn.Module):
    """
    Implement each Residual Block and Sampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dims: int,
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
        self.time_dense = nn.Linear(time_dims, out_channels)

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


class DiffusionUNet(nn.Module):
    """
    Compose ResidualBlocks in UNet for noise prediction.
    """

    def __init__(self, in_channels=3, num_groups=32):
        """
        Initialization for a 32 x 32 dataset.
        """
        super().__init__()
        channels_list = [128, 128, 256, 256, 256]
        resolutions_list = [32, 32, 16, 8, 4]
        TIME_EMBED_NCHAN = channels_list[0] * 4

        self.conv1 = nn.Conv2d(in_channels, channels_list[0])
        self.time_embed = SinusoidalEmbedding(
            TIME_EMBED_NCHAN
        )  # Start at 4x initial dimension. Max number of channels in this network's bottleneck.

        # Encoder Layers:
        self.enc_res1 = ResidualBlock(
            channels_list[0], channels_list[1], TIME_EMBED_NCHAN
        )
        self.ds1 = Downsample(channels_list[1])  # 16
        self.enc_res2 = ResidualBlock(
            channels_list[1], channels_list[2], TIME_EMBED_NCHAN, has_attn=True
        )
        self.ds2 = Downsample(channels_list[2])  # 8
        self.enc_res3 = ResidualBlock(
            channels_list[2], channels_list[3], TIME_EMBED_NCHAN
        )
        self.ds3 = Downsample(channels_list[3])  # 4
        self.enc_res4 = ResidualBlock(
            channels_list[3], channels_list[4], TIME_EMBED_NCHAN
        )

        # Bottleneck Layer
        self.mid_res1 = ResidualBlock(
            channels_list[-1], channels_list[-1], TIME_EMBED_NCHAN, has_attn=True
        )
        self.mid_res2 = ResidualBlock(
            channels_list[-1], channels_list[-1], TIME_EMBED_NCHAN
        )

        # Decoder Layers
        self.dec_res4 = ResidualBlock(
            channels_list[4], channels_list[3], TIME_EMBED_NCHAN
        )
        self.us3 = Upsample(channels_list[3])  # 8
        self.dec_res3 = ResidualBlock(
            channels_list[3], channels_list[2], TIME_EMBED_NCHAN
        )
        self.us2 = Upsample(channels_list[2])  # 16
        self.dec_res2 = ResidualBlock(
            channels_list[2], channels_list[1], TIME_EMBED_NCHAN, has_attn=True
        )
        self.us1 = Upsample(channels_list[1])  # 32
        self.dec_res1 = ResidualBlock(
            channels_list[1],
            channels_list[0],
            TIME_EMBED_NCHAN,
        )

        self.gn = nn.GroupNorm(num_groups, channels_list[0])
        self.act = nn.SiLU()
        self.final_conv = nn.Conv2d(
            in_channels=channels_list[0], out_channels=in_channels
        )

    def forward(self, img: torch.Tensor, t: torch.Tensor):
        # Implement forward logic through all layers. Make sure to implement skip connections!
        emb_t = self.time_embed(t)
