import os

from vidgen.data import DatasetCatalog, MetadataCatalog
from vidgen.utils.image import get_video_paths, get_image_paths


def load_kinetics(root, phase, load_images):
    """
    Returns list of dicts
    each dict contains:
        video_path: path to the video
        image_paths: sorted list of image paths
    """
    if load_images:
        return get_image_paths(os.path.join(root, phase), is_kinetics=True)
    return get_video_paths(os.path.join(root, phase), is_kinetics=True)


def register_kinetics(name, root, phase, load_images):
    DatasetCatalog.register(name, lambda: load_kinetics(root, phase, load_images))
    MetadataCatalog.get(name).set(root=root)