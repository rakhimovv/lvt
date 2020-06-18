import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from vidgen.utils.comm import all_gather, is_main_process, synchronize
from .evaluator import DatasetEvaluator


class MSEEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, distributed, output_dir=None):
        self._logger = logging.getLogger(__name__)

        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._mse = 0
        self._n_pixels = 0

    def reset(self):
        self._mse = 0
        self._n_pixels = 0

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            try:
                target = torch.as_tensor(input['image'])
            except KeyError:
                target = torch.as_tensor(input['image_sequence'])
            reconstruction = output['reconstruction']

            self._mse += F.mse_loss(reconstruction.detach().cpu(), target, reduction='sum')
            self._n_pixels += np.prod(target.shape)

    def evaluate(self):

        if self._distributed:
            synchronize()
            self._mse = all_gather(self._mse)
            self._n_pixels = all_gather(self._n_pixels)
            if not is_main_process():
                return

            self._mse = np.sum(self._mse)
            self._n_pixels = np.sum(self._n_pixels)

        result = {}
        result["MSE"] = self._mse / self._n_pixels
        results = OrderedDict({"reconstruction": result})
        self._logger.info(results)
        return results
