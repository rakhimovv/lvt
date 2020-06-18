import numpy as np
import torch.nn as nn

from vidgen.layers import norm_layer
from .build import Generator, GENERATOR_REGISTRY
from ...config import CfgNode


@GENERATOR_REGISTRY.register()
class ConvDecoder(Generator):

    @classmethod
    def from_config(cls, cfg: CfgNode, **kwargs):
        return cls(
            in_channels=cfg.MODEL.GENERATOR.IN_CHANNELS,
            nf=cfg.MODEL.GENERATOR.NF,
            out_channels=cfg.MODEL.GENERATOR.OUT_CHANNELS,
            norm=cfg.MODEL.GENERATOR.NORM,
            use_spectral_norm=cfg.MODEL.GENERATOR.SPECTRAL,
            n_layers=cfg.MODEL.GENERATOR.N_LAYERS,
            out_activation=cfg.MODEL.GENERATOR.OUT_ACTIVATION
        )

    def __init__(self, in_channels, nf, out_channels, norm, use_spectral_norm, n_layers, out_activation):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))

        self.layers = []
        kp = in_channels
        for scale in range(n_layers - 1, -1, -1):
            k = nf << scale
            self.layers.append(norm_layer(nn.Conv2d(kp, k, kw, stride=1, padding=pw), norm,
                                          use_spectral_norm=use_spectral_norm))
            self.layers.append(nn.LeakyReLU(0.2, True))
            self.layers.append(norm_layer(nn.Conv2d(k, k, kw, stride=1, padding=pw), norm,
                                          use_spectral_norm=use_spectral_norm))
            self.layers.append(nn.LeakyReLU(0.2, True))
            self.layers.append(nn.Upsample(scale_factor=2))
            kp = k
        self.layers.append(norm_layer(nn.Conv2d(kp, nf, kw, stride=1, padding=pw), norm="",
                                      use_spectral_norm=use_spectral_norm))
        self.layers.append(norm_layer(nn.Conv2d(kp, out_channels, kw, stride=1, padding=pw), norm="",
                                      use_spectral_norm=use_spectral_norm))
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
