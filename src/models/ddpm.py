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
        super().__init__()
        self.conv = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        return self.conv(x)


class Upsample(nn.Module):
    """
    Conv2d to Upsample by 2x
    """

    def __init__(self, num_channels):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            num_channels, num_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
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
        self.dropout = nn.Dropout(dropout_p)

        # Linear Layer for sinusoidal embedding
        self.time_dense = nn.Linear(time_dims, out_channels)

        # Downsample feature map resolution
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(
                in_channels, out_channels, kernel_size=1
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

        out = self.gn2(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)

        assert out.shape == residual.shape

        return self.attn(out + residual)

class AttentionBlock(nn.Module):
    pass

class DiffusionUNet(nn.Module):
    """
    Compose ResidualBlocks in UNet for noise prediction.
    """

    def __init__(
        self,
        img_channels=3,
        start_ch=64,
        chan_mults=(1, 2, 2, 4),
        attn_layers=(False, False, False, False),
        blocks_per_res=2,
    ):
        """
        Initialization for a 32 x 32 dataset.
        """
        super().__init__()
        n_scales = len(chan_mults)
        self.conv1 = nn.Conv2d(img_channels, start_ch, kernel_size=3, padding=1)
        self.time_embed = SinusoidalEmbedding(
            start_ch * 4
        )  # Start at 4x initial dimension. Max number of channels in this network's bottleneck.

        # Encoder Layers:
        enc_layers = []
        out_chan = start_ch
        in_chan = start_ch
        for i, (mult, attn) in enumerate(zip(chan_mults, attn_layers)):
            out_chan = in_chan * mult
            for _ in range(blocks_per_res):
                enc_layers.append(
                    ResidualBlock(in_chan, out_chan, start_ch * 4, has_attn=attn)
                )
                in_chan = out_chan

            if i < n_scales - 1:
                enc_layers.append(Downsample(in_chan))

        self.encoder = nn.ModuleList(enc_layers)

        # Bottleneck Layer
        self.mid_res1 = ResidualBlock(
            in_chan,
            in_chan,
            start_ch * 4,
            has_attn=False,
        )
        self.mid_res2 = ResidualBlock(in_chan, in_chan, start_ch * 4)

        # Decoder Layers
        dec_layers = []
        in_chan = out_chan
        for i, (mult, attn) in reversed(list(enumerate(zip(chan_mults, attn_layers)))):
            out_chan = in_chan
            for _ in range(blocks_per_res):
                dec_layers.append(
                    ResidualBlock(
                        in_chan + out_chan,
                        out_chan,
                        start_ch * 4,
                        has_attn=attn,
                    )
                )
                
            out_chan = in_chan // mult
            dec_layers.append(
                ResidualBlock(
                    in_chan + out_chan,
                    out_chan,
                    start_ch * 4,
                    has_attn=attn,
                )
            )
            in_chan = out_chan
            if i > 0:
                dec_layers.append(Upsample(out_chan))

        self.decoder = nn.ModuleList(dec_layers)

        self.gn = nn.GroupNorm(8, start_ch)  # As per DDPM Paper
        self.act = nn.SiLU()
        self.final_conv = nn.Conv2d(
            in_channels=start_ch,
            out_channels=img_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, img: torch.Tensor, t: torch.Tensor):
        emb_t = self.time_embed(t)
        out = self.conv1(img)

        # Downsample
        enc_imgs = []
        enc_imgs.append(out)

        for enc_layer in self.encoder:
            out = enc_layer(out, emb_t)
            enc_imgs.append(out)

        # Middle Block
        out = self.mid_res2(self.mid_res1(out, emb_t), emb_t)

        # Upsample Block
        for dec_layer in self.decoder:
            if isinstance(dec_layer, Upsample):
                out = dec_layer(out, emb_t)
            else:
                res = enc_imgs.pop()
                out = torch.cat([out, res], dim=1)
                out = dec_layer(out, emb_t)
                
        out = self.gn(out)
        out = self.act(out)
        return self.final_conv(out)


if __name__ == "__main__":
    img = torch.randint(0, 256, (1, 3, 32, 32)).float()
    t = torch.randint(0, 100, (1,)).float()
    model = DiffusionUNet()
    out = model(img, t)
    print(out.shape)
