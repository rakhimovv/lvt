import copy
import os
import random

import numpy as np
import torch

import vidgen.utils.image
from vidgen.modeling.autoregressive.vt_utils import slice_mask, subscale_order, visible_abc_mask, ss_shift

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]


class ShortVideoException(Exception):
    pass


class DatasetMapper:
    """
    A callable which takes a dataset dict and map it into a format used by the model.
    """

    def __init__(self, cfg, is_train=True):
        self.cfg = cfg
        self.is_train = is_train
        self.img_format = cfg.INPUT.FORMAT
        self.n_frames = cfg.INPUT.N_FRAMES_PER_VIDEO_TRAIN if is_train else cfg.INPUT.N_FRAMES_PER_VIDEO_TEST
        self.scale_zeroone = cfg.INPUT.SCALE_TO_ZEROONE
        self.prepare_slices = cfg.INPUT.PREPARE_SLICES_TRAIN and is_train
        if self.prepare_slices:
            self.abc2idx = None
            self.pad_value = cfg.MODEL.AUTOREGRESSIVE.VT.PAD_VALUE
            self.kernel = cfg.MODEL.AUTOREGRESSIVE.VT.KERNEL
            self.n_prime = cfg.MODEL.AUTOREGRESSIVE.VT.N_PRIME
        assert self.n_frames > 0 or self.n_frames == -1

    def start_end(self, n):
        if n < self.n_frames:
            raise ShortVideoException

        start = 0 if (self.n_frames == -1 or not self.is_train) else random.randint(0, n - self.n_frames)
        end = n if self.n_frames == -1 else start + self.n_frames
        return slice(start, end)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in vidgen Dataset format.

        Returns:
            dict: a format that builtin models in vidgen accept
        """
        try:
            dataset_dict = copy.deepcopy(dataset_dict)

            if "class" in dataset_dict:
                dataset_dict["class"] = torch.tensor(dataset_dict["class"]).long()

            if "image" in dataset_dict:
                dataset_dict["image"] = dataset_dict["image"].astype('float32')
                if self.scale_zeroone:
                    dataset_dict["image"] /= 255.

            elif "latent_paths" in dataset_dict:
                n = len(dataset_dict["latent_paths"])
                video = [np.load(path) for path in dataset_dict["latent_paths"][self.start_end(n)]]
                dataset_dict["image_sequence"] = np.stack(video, axis=0)

            elif "latent_names" in dataset_dict:
                n = len(dataset_dict["latent_names"])
                video_root = dataset_dict["video_root"]
                video = [np.load(video_root + '/' + lname) for lname in dataset_dict["latent_names"][self.start_end(n)]]
                dataset_dict["image_sequence"] = np.stack(video, axis=0)

            elif "image_sequence" in dataset_dict:
                n = len(dataset_dict["image_sequence"])
                dataset_dict["image_sequence"] = dataset_dict["image_sequence"][self.start_end(n)].astype('float32')
                if self.scale_zeroone:
                    dataset_dict["image_sequence"] /= 255.

            elif "image_path" in dataset_dict:
                file_name = dataset_dict["image_path"]
                dataset_dict["image"] = np.ascontiguousarray(
                    vidgen.utils.image.read_image(file_name, self.img_format).transpose(2, 0, 1)).astype('float32')
                if self.scale_zeroone:
                    dataset_dict["image"] /= 255.

            # elif "image_paths" in dataset_dict:
            #     n = len(dataset_dict["image_paths"])
            #     video = [np.ascontiguousarray(utils.read_image(file_name, self.img_format).transpose(2, 0, 1)) for file_name
            #              in dataset_dict["image_paths"][self.start_end(n)]]
            #     dataset_dict["image_sequence"] = np.stack(video, axis=0).astype('float32')
            #     if self.scale_zeroone:
            #         dataset_dict["image_sequence"] /= 255.

            elif "image_names" in dataset_dict:
                n = len(dataset_dict["image_names"])
                video_root = dataset_dict["video_root"]
                video = [np.ascontiguousarray(
                    vidgen.utils.image.read_image(os.path.join(video_root, file_name), self.img_format).transpose(2, 0,
                                                                                                                  1))
                    for
                    file_name
                    in dataset_dict["image_names"][self.start_end(n)]]
                dataset_dict["image_sequence"] = np.stack(video, axis=0).astype('float32')
                if self.scale_zeroone:
                    dataset_dict["image_sequence"] /= 255.

            if self.prepare_slices:
                assert "image_sequence" in dataset_dict
                assert not self.scale_zeroone
                st, sh, sw = self.cfg.MODEL.AUTOREGRESSIVE.VT.STRIDE
                video = dataset_dict["image_sequence"]
                video = torch.as_tensor(video)[None, ...]
                _, T, nc, H, W = video.size()
                assert T % st == 0 and H % sh == 0 and W % sw == 0
                t, h, w = T // st, H // sh, W // sw
                video = video.transpose(1, 2)  # 1, nc, T, H, W

                is_single_frame = (t == 1 and sh == 1 and sw == 1)
                a = random.randint(self.n_prime, st - 1) if is_single_frame else random.randint(0, st - 1)
                b = random.randint(0, sh - 1)
                c = random.randint(0, sw - 1)
                if self.abc2idx is None:
                    _, self.abc2idx = subscale_order(st, sh, sw)
                slice_idx = self.abc2idx[(a, b, c)]

                smask = slice_mask(a, b, c, st, sh, sw, T, H, W, dtype=torch.bool)  # 1, 1, T, H, W
                slice = video.masked_select(smask).clone().view(1, nc, t, h, w)

                vmask = visible_abc_mask(a, b, c, st, sh, sw, T, H, W, dtype=torch.bool)  # 1, 1, T, H, W
                video = video.masked_fill(~vmask, self.pad_value)
                video = ss_shift(video, a, b, c, st, sh, sw, T, H, W, *self.kernel, pad_value=self.pad_value)

                ignore_mask = torch.zeros(size=(1, 1, T, H, W), dtype=torch.bool)
                if self.n_prime > 0:
                    ignore_mask[:, :, :self.n_prime, :, :] = 1
                ignore_mask = ignore_mask.masked_select(smask).clone().view(1, 1, t, h, w)

                # [0] to delete batch dim
                dataset_dict["context"] = video[0].long()
                dataset_dict["slice_idx"] = torch.tensor(slice_idx).long()
                dataset_dict["slice"] = slice[0].long()
                dataset_dict["ignore_mask"] = ignore_mask[0]
                del dataset_dict["image_sequence"]

            return dataset_dict
        except ShortVideoException:
            return None
