
"""
Wrappers around on some nn functions.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from .batch_norm import get_norm


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


def get_out_channel(layer):
    if hasattr(layer, 'out_channels'):
        return getattr(layer, 'out_channels')
    return layer.weight.size(0)


def norm_layer(layer, norm, use_spectral_norm):
    if use_spectral_norm:
        layer = spectral_norm(layer)
    if norm:
        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        # FIXME in case of group norm??
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)
        _norm_layer = get_norm(norm, get_out_channel(layer))
        return nn.Sequential(layer, _norm_layer)
    else:
        return layer
