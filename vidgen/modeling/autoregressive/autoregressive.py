from abc import ABCMeta

import torch.nn as nn

__all__ = ["Autoregressive"]


class Autoregressive(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for autoregressive models
    """

    def __init__(self):
        """
        The `__init__` method of any subclass can specify its own set of arguments.
        """
        super().__init__()

    def forward(self, *args, **kwargs):
        """
        Subclasses must override this method, but adhere to the same return type.

        Returns: Tensor
        """
        pass
