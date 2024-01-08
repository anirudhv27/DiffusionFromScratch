"""
Implementation of Denoising Diffusion Probabilistic Models (DDPM) using Pytorch.

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

    def __init__(self, in_channels=3, img_channels_list=[128, 256, 256, 256], attn_layers=[False, True, False, False], start_ch=128, num_groups=32):
        """
        Initialization for a 32 x 32 dataset.
        """
        super().__init__()
        TIME_EMBED_NCHAN = start_ch * 4

        self.conv1 = nn.Conv2d(in_channels, start_ch)
        self.time_embed = SinusoidalEmbedding(
            TIME_EMBED_NCHAN
        )  # Start at 4x initial dimension. Max number of channels in this network's bottleneck.

        # Encoder Layers:
        self.enc_layers = []
        self.downsample_layers = []
        
        in_chan = start_ch
        for i, out_chan in enumerate(img_channels_list):
            self.enc_layers.append(ResidualBlock(in_chan, out_chan, TIME_EMBED_NCHAN, has_attn=attn_layers[i]))
            if i < len(img_channels_list) - 1:
                self.downsample_layers.append(Downsample(out_chan))
            in_chan = out_chan

        # Bottleneck Layer
        self.mid_res1 = ResidualBlock(
            img_channels_list[-1], img_channels_list[-1], TIME_EMBED_NCHAN, has_attn=True
        )
        self.mid_res2 = ResidualBlock(
            img_channels_list[-1], img_channels_list[-1], TIME_EMBED_NCHAN
        )

        # Decoder Layers
        self.dec_layers = []
        self.upsample_layers = []
        in_chan = img_channels_list[-1]
        for i, out_chan in reversed(list(enumerate(img_channels_list))):
            self.dec_layers.append(ResidualBlock(in_chan, out_chan, TIME_EMBED_NCHAN, has_attn=attn_layers[i]))
            if i > 0:
                self.upsample_layers.append(Upsample(out_chan))
            in_chan = out_chan

        self.gn = nn.GroupNorm(num_groups, img_channels_list[0])
        self.act = nn.SiLU()
        self.final_conv = nn.Conv2d(
            in_channels=img_channels_list[0], out_channels=in_channels
        )

    def forward(self, img: torch.Tensor, t: torch.Tensor):
        emb_t = self.time_embed(t)
        img_conv = self.conv1(img)

        # Downsample
        enc_imgs = [] # 
        enc_imgs.append(img_conv)
        
        for i, enc_layer in enumerate(self.enc_layers):
            enc_img = enc_layer(enc_imgs[-1], emb_t)
            if i < len(self.enc_layers):
                enc_img = self.downsample_layers[i](enc_img)
            enc_imgs.append(enc_imgs)
        
        # Middle Block
        out = self.mid_res2(self.mid_res1(enc_imgs[-1]))
        
        # Upsample Block
        for i, dec_layer in enumerate(self.dec_layers):
            res = enc_imgs.pop()
            in_tensor = torch.cat([out, res], dim=1)
            out = dec_layer(in_tensor, emb_t)
            if i < len(self.dec_layers):
                out = self.upsample_layers[i](out)
        
        out = self.gn(out)
        out = self.act(out)
        in_tensor = torch.cat([out, enc_imgs[-1]], dim=1)
        return self.final_conv(in_tensor, emb_t)