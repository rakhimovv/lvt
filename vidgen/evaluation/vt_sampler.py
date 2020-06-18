import logging
import os
import time
from collections import OrderedDict

import numpy as np
import torch
from fvcore.common.checkpoint import Checkpointer
from fvcore.common.file_io import PathManager

from vidgen.utils.comm import is_main_process, synchronize
from .evaluator import DatasetEvaluator
from ..config import get_cfg
from ..modeling.meta_arch import build_model
from ..utils.image import save_image


class VTSampler(DatasetEvaluator):
    """
    Save sampled codes
    """

    def __init__(self, cfg, dataset_name, distributed, output_dir=None):
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

        vq_cfg = get_cfg()
        vq_cfg.merge_from_file(cfg.TEST.VT_SAMPLER.VQ_VAE.CFG)
        self.vqvae = build_model(vq_cfg)
        Checkpointer(self.vqvae.encoder).resume_or_load(cfg.TEST.VT_SAMPLER.VQ_VAE.ENCODER_WEIGHTS, resume=False)
        Checkpointer(self.vqvae.generator).resume_or_load(cfg.TEST.VT_SAMPLER.VQ_VAE.GENERATOR_WEIGHTS, resume=False)
        Checkpointer(self.vqvae.codebook).resume_or_load(cfg.TEST.VT_SAMPLER.VQ_VAE.CODEBOOK_WEIGHTS, resume=False)
        self.vqvae.set_generator_requires_grad(False)
        self.vqvae.eval()
        self.scale_to_zeroone = vq_cfg.INPUT.SCALE_TO_ZEROONE

    def reset(self):
        pass

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            samples = output['samples']  # list of size num_samples with el: # nc, t, h, w
            v_idx = input['video_idx']
            for sample_idx in range(len(samples)):
                sample = samples[sample_idx].squeeze(0)  # t, h, w if nc == 1 or nc, t, h, w
                if sample.dim() == 4:
                    sample = sample.transpose(0, 1)  # t, nc, h, w
                self.vqvae.to(sample.device)
                code = sample.detach().cpu().numpy()
                sample = self.vqvae.decode(sample)  # T, 3, H, W
                sample = self.vqvae.back_normalizer(sample)  # T, 3, H, W
                if self.scale_to_zeroone:
                    sample = sample * 255
                sample.clamp_(0.0, 255.0)
                sample = sample.permute(0, 2, 3, 1).contiguous()  # T, H, W, 3
                sample = sample.detach().cpu().numpy().astype(np.uint8)

                video_dir = os.path.join(self._output_dir, "samples", self._dataset_name,
                                         f'video_{sample_idx}_{v_idx}')
                PathManager.mkdirs(video_dir)
                np.save(os.path.join(video_dir, f'codes.npy'), code)
                for frame_idx in range(len(sample)):
                    frame_path = os.path.join(video_dir, f'{frame_idx}.png')
                    for i in range(10):
                        try:
                            save_image(sample[frame_idx], frame_path)
                            break
                        except OSError:
                            print(f'sleep 3 sec and try again #{i}')
                            time.sleep(3)
                            continue

    def evaluate(self):
        if self._distributed:
            synchronize()
            if not is_main_process():
                return

        return OrderedDict({"samples": {}})
