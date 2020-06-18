import logging

from vidgen.utils.registry import Registry
from .autoregressive import Autoregressive

AUTOREGRESSIVE_REGISTRY = Registry("AUTOREGRESSIVE")
AUTOREGRESSIVE_REGISTRY.__doc__ = """
Registry for autoregressive models.

1. A :class:`vidgen.config.CfgNode`

It must returns an instance of :class:`Autoregressive`.
"""


def build_autoregressive(cfg, **kwargs):
    """
    Build an autoregressive model from `cfg.MODEL.AUTOREGRESSIVE.NAME`.

    Returns:
        an instance of :class:`Autoregressive`
    """

    autoregressive_name = cfg.MODEL.AUTOREGRESSIVE.NAME
    autoregressive = AUTOREGRESSIVE_REGISTRY.get(autoregressive_name).from_config(cfg, **kwargs)
    assert isinstance(autoregressive, Autoregressive)
    logger = logging.getLogger(__name__)
    logger.info("#params in autoregressive: {}M".format(sum(p.numel() for p in autoregressive.parameters()) / 1e6))
    return autoregressive
