import torch.nn as nn

from vidgen.config import CfgNode
from vidgen.layers import norm_layer
from . import Encoder
from .build import ENCODER_REGISTRY
from ..generator import ResDecoder


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


@ENCODER_REGISTRY.register()
class ResEncoder(Encoder):
    """
    ref impl: https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
    """

    @classmethod
    def from_config(cls, cfg: CfgNode, **kwargs):
        return cls(
            in_channels=kwargs.get('in_channels', cfg.MODEL.ENCODER.IN_CHANNELS),
            nf=cfg.MODEL.ENCODER.NF,
            res_channels=cfg.MODEL.ENCODER.RES_CHANNELS,
            norm=cfg.MODEL.ENCODER.NORM,
            use_spectral_norm=cfg.MODEL.ENCODER.SPECTRAL,
            n_layers=cfg.MODEL.ENCODER.N_LAYERS,
            out_activation=cfg.MODEL.ENCODER.OUT_ACTIVATION,
            stride=kwargs.get('stride', 4)
        )

    def __init__(self, in_channels, nf, res_channels, norm, use_spectral_norm, n_layers, out_activation, stride):
        super().__init__()
        if stride == 4:
            self.layers = [
                norm_layer(nn.Conv2d(in_channels, nf // 2, 4, 2, 1), norm, use_spectral_norm=use_spectral_norm),
                nn.ReLU(True),
                norm_layer(nn.Conv2d(nf // 2, nf, 4, 2, 1), norm, use_spectral_norm=use_spectral_norm),
                nn.ReLU(True),
                norm_layer(nn.Conv2d(nf, nf, 3, 1, 1), norm, use_spectral_norm=use_spectral_norm),
            ]
        elif stride == 2:
            self.layers = [
                norm_layer(nn.Conv2d(in_channels, nf // 2, 4, 2, 1), norm, use_spectral_norm=use_spectral_norm),
                nn.ReLU(True),
                norm_layer(nn.Conv2d(nf // 2, nf, 3, 1, 1), norm, use_spectral_norm=use_spectral_norm),
            ]
        else:
            raise ValueError
        for i in range(n_layers):
            self.layers.append(ResBlock(nf, res_channels, norm))
        if out_activation == "":
            pass
        elif out_activation == "sigmoid":
            self.layers.append(nn.Sigmoid())
        elif out_activation == "relu":
            self.layers.append(nn.ReLU(inplace=True))
        elif out_activation == "tanh":
            self.layers.append(nn.Tanh())
        else:
            raise ValueError
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


@ENCODER_REGISTRY.register()
class VQVAE2Encoder(Encoder):

    @classmethod
    def from_config(cls, cfg: CfgNode, **kwargs):
        return cls(
            in_channel=cfg.MODEL.ENCODER.IN_CHANNELS,
            channel=cfg.MODEL.ENCODER.NF,
            n_res_block=cfg.MODEL.ENCODER.N_LAYERS,
            n_res_channel=cfg.MODEL.ENCODER.RES_CHANNELS,
            embed_dim=cfg.MODEL.CODEBOOK.DIM,
            norm=cfg.MODEL.ENCODER.NORM,
            use_spectral_norm=cfg.MODEL.ENCODER.SPECTRAL,
            out_activation=cfg.MODEL.ENCODER.OUT_ACTIVATION,
        )

    def __init__(self, in_channel, channel, n_res_block, n_res_channel, embed_dim, norm, use_spectral_norm,
                 out_activation):
        super().__init__()
        self.enc_b = ResEncoder(in_channel, channel, n_res_channel, norm, use_spectral_norm, n_res_block,
                                out_activation, stride=4)
        self.enc_t = ResEncoder(channel, channel, n_res_channel, norm, use_spectral_norm, n_res_block, out_activation,
                                stride=2)
        self.quantize_conv_t = norm_layer(nn.Conv2d(channel, embed_dim, 1), norm, use_spectral_norm)
        self.dec_t = ResDecoder(embed_dim, channel, n_res_channel, embed_dim, norm, use_spectral_norm, n_res_block,
                                out_activation="", stride=2)
        self.quantize_conv_b = norm_layer(nn.Conv2d(embed_dim + channel, embed_dim, 1), norm, use_spectral_norm)

    def forward(self, x, mode):
        if mode == "enc_b":
            return self.enc_b(x)
        elif mode == "enc_t":
            return self.enc_t(x)
        elif mode == "quantize_conv_t":
            return self.quantize_conv_t(x)
        elif mode == "dec_t":
            return self.dec_t(x)
        elif mode == "quantize_conv_b":
            return self.quantize_conv_b(x)
        else:
            raise ValueError
