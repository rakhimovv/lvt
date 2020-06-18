import torch
import torch.nn as nn

from vidgen.layers import norm_layer
from . import Generator
from .build import GENERATOR_REGISTRY
from ...config import CfgNode


class ResBlock(nn.Module):
    def __init__(self, dim, dim_res, norm="BN", use_spectral_norm=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            norm_layer(nn.Conv2d(dim, dim_res, 3, 1, 1), norm, use_spectral_norm=use_spectral_norm),
            nn.ReLU(True),
            norm_layer(nn.Conv2d(dim_res, dim, 1), norm, use_spectral_norm=use_spectral_norm),
        )

    def forward(self, x):
        return x + self.block(x)


@GENERATOR_REGISTRY.register()
class ResDecoder(Generator):
    """
    ref impl: https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
    """

    @classmethod
    def from_config(cls, cfg: CfgNode, **kwargs):
        return cls(
            in_channels=cfg.MODEL.GENERATOR.IN_CHANNELS,
            nf=cfg.MODEL.GENERATOR.NF,
            res_channels=cfg.MODEL.GENERATOR.RES_CHANNELS,
            out_channels=cfg.MODEL.GENERATOR.OUT_CHANNELS,
            norm=cfg.MODEL.GENERATOR.NORM,
            use_spectral_norm=cfg.MODEL.GENERATOR.SPECTRAL,
            n_layers=cfg.MODEL.GENERATOR.N_LAYERS,
            out_activation=kwargs.get('out_activation', cfg.MODEL.GENERATOR.OUT_ACTIVATION),
            stride=kwargs.get('stride', 4)
        )

    def __init__(self, in_channels, nf, res_channels, out_channels, norm, use_spectral_norm, n_layers, out_activation,
                 stride):
        super().__init__()

        self.layers = [norm_layer(nn.Conv2d(in_channels, nf, 3, 1, 1), norm, use_spectral_norm=use_spectral_norm), ]
        for i in range(n_layers):
            self.layers.append(ResBlock(nf, res_channels, norm))
        self.layers.append(nn.ReLU(True))
        if stride == 4:
            self.layers.extend([
                norm_layer(nn.ConvTranspose2d(nf, nf // 2, 4, 2, 1), norm, use_spectral_norm=use_spectral_norm),
                nn.ReLU(True),
                nn.ConvTranspose2d(nf // 2, out_channels, 4, 2, 1)
            ])
        elif stride == 2:
            self.layers.extend([
                norm_layer(nn.ConvTranspose2d(nf, out_channels, 4, 2, 1), norm, use_spectral_norm=use_spectral_norm),
            ])
        else:
            raise ValueError
        if out_activation == "":
            pass
        elif out_activation == "sigmoid":
            self.layers.append(nn.Sigmoid())
        elif out_activation == "tanh":
            self.layers.append(nn.Tanh())
        else:
            raise ValueError
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


@GENERATOR_REGISTRY.register()
class ResShuffleDecoder(Generator):

    @classmethod
    def from_config(cls, cfg: CfgNode, **kwargs):
        return cls(
            in_channels=cfg.MODEL.GENERATOR.IN_CHANNELS,
            nf=cfg.MODEL.GENERATOR.NF,
            res_channels=cfg.MODEL.GENERATOR.RES_CHANNELS,
            out_channels=cfg.MODEL.GENERATOR.OUT_CHANNELS,
            norm=cfg.MODEL.GENERATOR.NORM,
            use_spectral_norm=cfg.MODEL.GENERATOR.SPECTRAL,
            n_layers=cfg.MODEL.GENERATOR.N_LAYERS,
            out_activation=kwargs.get('out_activation', cfg.MODEL.GENERATOR.OUT_ACTIVATION),
            stride=kwargs.get('stride', 4)
        )

    def __init__(self, in_channels, nf, res_channels, out_channels, norm, use_spectral_norm, n_layers, out_activation,
                 stride):
        super().__init__()

        self.layers = [norm_layer(nn.Conv2d(in_channels, nf, 3, 1, 1), norm, use_spectral_norm=use_spectral_norm), ]
        for i in range(n_layers):
            self.layers.append(ResBlock(nf, res_channels, norm))
        self.layers.append(nn.ReLU(True))
        if stride == 4:
            self.layers.extend([
                norm_layer(nn.Conv2d(nf, nf // 2 * 4, 3, 1, 1), norm, use_spectral_norm=use_spectral_norm),
                nn.PixelShuffle(2),
                nn.ReLU(True),
                nn.Conv2d(nf // 2, out_channels * 4, 3, 1, 1),
                nn.PixelShuffle(2),
            ])
        elif stride == 2:
            self.layers.extend([
                norm_layer(nn.Conv2d(nf, out_channels * 4, 3, 1, 1), norm, use_spectral_norm=use_spectral_norm),
                nn.PixelShuffle(2),
            ])
        else:
            raise ValueError
        if out_activation == "":
            pass
        elif out_activation == "sigmoid":
            self.layers.append(nn.Sigmoid())
        elif out_activation == "tanh":
            self.layers.append(nn.Tanh())
        else:
            raise ValueError
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


@GENERATOR_REGISTRY.register()
class VQVAE2Decoder(Generator):

    @classmethod
    def from_config(cls, cfg: CfgNode, **kwargs):
        return cls(
            embed_dim=cfg.MODEL.CODEBOOK.DIM,
            nf=cfg.MODEL.GENERATOR.NF,
            res_channels=cfg.MODEL.GENERATOR.RES_CHANNELS,
            out_channels=cfg.MODEL.GENERATOR.OUT_CHANNELS,
            norm=cfg.MODEL.GENERATOR.NORM,
            use_spectral_norm=cfg.MODEL.GENERATOR.SPECTRAL,
            n_layers=cfg.MODEL.GENERATOR.N_LAYERS,
            out_activation=cfg.MODEL.GENERATOR.OUT_ACTIVATION,
        )

    def __init__(self, embed_dim, nf, res_channels, out_channels, norm, use_spectral_norm, n_layers, out_activation):
        super().__init__()
        self.upsample_t = norm_layer(nn.ConvTranspose2d(embed_dim, embed_dim, 4, 2, 1), norm, use_spectral_norm)
        self.dec = ResDecoder(embed_dim + embed_dim, nf, res_channels, out_channels, norm, use_spectral_norm, n_layers,
                              out_activation, stride=4)

    def forward(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat((upsample_t, quant_b), dim=1)
        x_tilde = self.dec(quant)
        return x_tilde
