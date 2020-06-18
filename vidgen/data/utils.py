# -*- coding: utf-8 -*-


"""
Common data processing utilities that are used in a
typical object detection data pipeline.
"""
import logging

from PIL import Image

from . import transforms as T
from .catalog import MetadataCatalog


class SizeMismatchError(ValueError):
    """
    When loaded image has difference width/height compared with annotation.
    """


def check_image_size(dataset_dict, image):
    """
    Raise an error if the image does not match the size specified in the dict.
    """
    if "width" in dataset_dict or "height" in dataset_dict:
        image_wh = (image.shape[1], image.shape[0])
        expected_wh = (dataset_dict["width"], dataset_dict["height"])
        if not image_wh == expected_wh:
            raise SizeMismatchError(
                "Mismatched (W,H){}, got {}, expect {}".format(
                    " for image " + dataset_dict["file_name"]
                    if "file_name" in dataset_dict
                    else "",
                    image_wh,
                    expected_wh,
                )
            )

    # To ensure bbox always remap to original image size
    if "width" not in dataset_dict:
        dataset_dict["width"] = image.shape[1]
    if "height" not in dataset_dict:
        dataset_dict["height"] = image.shape[0]


def check_metadata_consistency(key, dataset_names):
    """
    Check that the datasets have consistent metadata.

    Args:
        key (str): a metadata key
        dataset_names (list[str]): a list of dataset names

    Raises:
        AttributeError: if the key does not exist in the metadata
        ValueError: if the given datasets do not have the same metadata values defined by key
    """
    if len(dataset_names) == 0:
        return
    logger = logging.getLogger(__name__)
    entries_per_dataset = [getattr(MetadataCatalog.get(d), key) for d in dataset_names]
    for idx, entry in enumerate(entries_per_dataset):
        if entry != entries_per_dataset[0]:
            logger.error(
                "Metadata '{}' for dataset '{}' is '{}'".format(key, dataset_names[idx], str(entry))
            )
            logger.error(
                "Metadata '{}' for dataset '{}' is '{}'".format(
                    key, dataset_names[0], str(entries_per_dataset[0])
                )
            )
            raise ValueError("Datasets have different metadata '{}'!".format(key))


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Now it includes resizing and flipping.

    Returns:
        list[TransformGen]
    """

    preprocess_mode = cfg.INPUT.PREPROCESS_MODE
    if is_train:
        load_size = cfg.INPUT.LOAD_SIZE_TRAIN
    else:
        load_size = cfg.INPUT.LOAD_SIZE_TEST
    crop_size = cfg.INPUT.CROP_SIZE
    aspect_ratio = cfg.INPUT.ASPECT_RATIO
    no_flip = cfg.INPUT.NO_FLIP
    interp = Image.NEAREST

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if 'resize' in preprocess_mode:
        tfm_gens.append(T.Resize(shape=load_size, interp=interp))
    elif 'scale_width' in preprocess_mode:
        tfm_gens.append(T.ResizeWidth(width_length=load_size, interp=interp))
    elif 'scale_shortside' in preprocess_mode:
        tfm_gens.append(T.ResizeShortestEdge(short_edge_length=load_size, interp=interp))

    if 'crop' in preprocess_mode:
        tfm_gens.append(T.RandomCrop(crop_type="absolute", crop_size=crop_size))

    if preprocess_mode == 'none':
        tfm_gens.append(T.MakePower2(base=32))
    if preprocess_mode == 'fixed':
        tfm_gens.append(T.ResizeWidth(width_length=crop_size, aspect_ratio=aspect_ratio, interp=interp))

    if is_train and not no_flip:
        tfm_gens.append(T.RandomFlip(horizontal=True))

    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))

    return tfm_gens
