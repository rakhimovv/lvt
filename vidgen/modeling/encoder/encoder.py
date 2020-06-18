from abc import ABCMeta

import torch.nn as nn

__all__ = ["Encoder"]


class Encoder(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for network backbones.
    """

    def __init__(self):
        """
        The `__init__` method of any subclass can specify its own set of arguments.
        """
        super().__init__()

    def forward(self, x):
        """
        Subclasses must override this method, but adhere to the same return type.

        Returns: Tensor
        """
        pass
