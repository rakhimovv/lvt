
import logging

from vidgen.utils.registry import Registry
from .generator import Generator

GENERATOR_REGISTRY = Registry("GENERATOR")
GENERATOR_REGISTRY.__doc__ = """
Registry for generators, which generate imgaes

The registered object will be a callable that accepts two arguments:

1. A :class:`vidgen.config.CfgNode`

It must returns an instance of :class:`Generator`.
"""


def build_generator(cfg, **kwargs):
    """
    Build a generator from `cfg.MODEL.GENERATOR.NAME`.

    Returns:
        an instance of :class:`Generator`
    """
    generator_name = cfg.MODEL.GENERATOR.NAME
    generator = GENERATOR_REGISTRY.get(generator_name).from_config(cfg, **kwargs)
    assert isinstance(generator, Generator)
    logger = logging.getLogger(__name__)
    logger.info("#params in generator: {}M".format(sum(p.numel() for p in generator.parameters()) / 1e6))
    return generator
