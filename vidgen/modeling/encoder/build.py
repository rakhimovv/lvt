import logging

from vidgen.utils.registry import Registry
from .encoder import Encoder

ENCODER_REGISTRY = Registry("ENCODER")
ENCODER_REGISTRY.__doc__ = """
Registry for encoders.

1. A :class:`vidgen.config.CfgNode`

It must returns an instance of :class:`Encoder`.
"""


def build_encoder(cfg, **kwargs):
    """
    Build a encoder from `cfg.MODEL.ENCODER.NAME`.

    Returns:
        an instance of :class:`Encoder`
    """
    encoder_name = cfg.MODEL.ENCODER.NAME
    encoder = ENCODER_REGISTRY.get(encoder_name).from_config(cfg, **kwargs)
    assert isinstance(encoder, Encoder)
    logger = logging.getLogger(__name__)
    logger.info("#params in encoder: {}M".format(sum(p.numel() for p in encoder.parameters()) / 1e6))
    return encoder
