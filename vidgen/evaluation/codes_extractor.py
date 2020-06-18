import logging
import os
from collections import OrderedDict

import numpy as np
import torch
from fvcore.common.file_io import PathManager

from vidgen.utils.comm import is_main_process, synchronize
from .evaluator import DatasetEvaluator
from ..utils.lables import KINETICS_IDX_LABEL


class CodesExtractor(DatasetEvaluator):
    """
    Extract codes from latent model
    """

    def __init__(self, dataset_name, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
        """
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

    def reset(self):
        pass

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            latent = output['latent']  # 20, ?, 16, 16
            if latent.dim() == 3:
                latent = latent.unsqueeze(1)  # 20, 1, 16, 16

            v_idx = input['video_idx']
            if 'class' in input:
                class_name = KINETICS_IDX_LABEL[input['class']]
                video_dir = os.path.join(self._output_dir, self._dataset_name, class_name, f'video_{v_idx}')
            else:
                video_dir = os.path.join(self._output_dir, self._dataset_name, f'video_{v_idx}')
            PathManager.mkdirs(video_dir)

            for frame_idx in range(latent.size(0)):
                latent_frame = latent[frame_idx].detach().cpu().numpy()
                latent_frame_path = os.path.join(video_dir, f'{frame_idx}.npy')
                np.save(latent_frame_path, latent_frame)

    def evaluate(self):

        if self._distributed:
            synchronize()
            if not is_main_process():
                return

        return OrderedDict({"latents": {}})
