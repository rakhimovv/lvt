import numpy as np
import torch.nn as nn

from vidgen.layers import norm_layer
from . import Encoder
from .build import ENCODER_REGISTRY
from ...config import CfgNode


@ENCODER_REGISTRY.register()
class ConvEncoder(Encoder):
    """
    reference impl: https://gist.github.com/kylemcdonald/e8ca989584b3b0e6526c0a737ed412f0
    """

    @classmethod
    def from_config(cls, cfg: CfgNode, **kwargs):
        return cls(
            in_channels=cfg.MODEL.ENCODER.IN_CHANNELS,
            nf=cfg.MODEL.ENCODER.NF,
            out_channels=cfg.MODEL.ENCODER.OUT_CHANNELS,
            norm=cfg.MODEL.ENCODER.NORM,
            use_spectral_norm=cfg.MODEL.ENCODER.SPECTRAL,
            n_layers=cfg.MODEL.ENCODER.N_LAYERS,
            out_activation=cfg.MODEL.ENCODER.OUT_ACTIVATION
        )

    def __init__(self, in_channels, nf, out_channels, norm, use_spectral_norm, n_layers, out_activation):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))

        self.layers = [
            norm_layer(nn.Conv2d(in_channels, nf, kw, stride=1, padding=pw), norm, use_spectral_norm=use_spectral_norm),
            nn.LeakyReLU(0.2, True)
        ]
        kp = nf
        for i in range(n_layers):
            k = nf << i
            self.layers.append(norm_layer(nn.Conv2d(kp, k, kw, stride=1, padding=pw), norm,
                                          use_spectral_norm=use_spectral_norm))
            self.layers.append(nn.LeakyReLU(0.2, True))
            self.layers.append(norm_layer(nn.Conv2d(k, k, kw, stride=1, padding=pw), norm,
                                          use_spectral_norm=use_spectral_norm))
            self.layers.append(nn.LeakyReLU(0.2, True))
            self.layers.append(nn.AvgPool2d(2))
            kp = k
        k = nf << n_layers
        self.layers.append(norm_layer(nn.Conv2d(kp, k, kw, stride=1, padding=pw), norm,
                                      use_spectral_norm=use_spectral_norm))
        self.layers.append(nn.LeakyReLU(0.2, True))
        self.layers.append(norm_layer(nn.Conv2d(k, out_channels, kw, stride=1, padding=pw), norm,
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
