import os

from vidgen.data import DatasetCatalog, MetadataCatalog
from vidgen.utils.image import get_video_paths, get_image_paths


def load_bair(root, phase, load_images):
    """
    Returns list of dicts
    each dict contains:
        video_path: path to the video
        image_paths: sorted list of image paths
    """
    if load_images:
        return get_image_paths(os.path.join(root, phase))
    return get_video_paths(os.path.join(root, phase))


def register_bair(name, root, phase, load_images):
    DatasetCatalog.register(name, lambda: load_bair(root, phase, load_images))
    MetadataCatalog.get(name).set(root=root)
