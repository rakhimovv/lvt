from .autoregressive import (
    AUTOREGRESSIVE_REGISTRY,
    build_autoregressive
)
from .encoder import (
    ENCODER_REGISTRY,
    build_encoder
)
from .generator import (
    GENERATOR_REGISTRY,
    build_generator
)
from .loss import *

_EXCLUDE = {"torch"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]

# assert (
#                torch.Tensor([1]) == torch.Tensor([2])
#        ).dtype == torch.bool, "Your Pytorch is too old. Please update to contain https://github.com/pytorch/pytorch/pull/21113"
