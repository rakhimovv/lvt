import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from vidgen.utils.comm import all_gather, is_main_process, synchronize
from .evaluator import DatasetEvaluator


class BitsEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, distributed, output_dir=None):
        self._logger = logging.getLogger(__name__)

        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._ce = 0
        self._n_pixels = 0

    def reset(self):
        self._ce = 0
        self._n_pixels = 0

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            if 'logits' in output:
                logits = output['logits']  # nc, nv, T, H, W
                target = torch.as_tensor(input['image_sequence'], device=logits.device)  # T, nc, H, W
                target = target.transpose(0, 1)  # nc, T, H, W
                ignore_mask = output['ignore_mask']  # 1, T, H, W
                nc = target.size(0)
                for k in range(nc):
                    ce = F.cross_entropy(logits[k].unsqueeze(0), target[k].unsqueeze(0),
                                         reduction='none')  # 1, T, H, W
                    ce = ce.masked_select(~ignore_mask)
                    self._ce += ce.sum().item()
                    self._n_pixels += np.prod(ce.shape)
            else:
                raise ValueError

    def evaluate(self):
        if self._distributed:
            synchronize()
            self._ce = all_gather(self._ce)
            self._n_pixels = all_gather(self._n_pixels)
            if not is_main_process():
                return

            self._ce = np.sum(self._ce)
            self._n_pixels = np.sum(self._n_pixels)

        result = {}
        result["bits_per_dim"] = (self._ce / np.log(2)) / self._n_pixels
        results = OrderedDict({"likelihood": result})
        self._logger.info(results)
        return results
