import os

import torch.nn.functional as F
from fvcore.common.checkpoint import Checkpointer
from fvcore.common.file_io import PathManager
from torch.nn.parallel import DistributedDataParallel

from . import AutoEncoderModel, META_ARCH_REGISTRY
from .. import PixelLoss
from ..vq import VQEmbedding
from ..vq.vq_embedding import DVQEmbedding
from ...solver import build_lr_scheduler
from ...solver.build import build_optimizer


@META_ARCH_REGISTRY.register()
class VQVAEModel(AutoEncoderModel):
    """
    ref impl: https://github.com/ritheshkumar95/pytorch-vqvae
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.use_codebook_ema = cfg.MODEL.CODEBOOK.EMA

        if cfg.MODEL.CODEBOOK.NUM == 1:
            self.codebook = VQEmbedding(cfg.MODEL.CODEBOOK.SIZE, cfg.MODEL.CODEBOOK.DIM, self.use_codebook_ema)
        else:
            self.codebook = DVQEmbedding(cfg.MODEL.CODEBOOK.NUM, cfg.MODEL.CODEBOOK.SIZE, cfg.MODEL.CODEBOOK.DIM,
                                         self.use_codebook_ema)

        if self.use_codebook_ema:
            self._set_requires_grad(self.codebook.parameters(), False)

        self.pixel_loss = PixelLoss(cfg)  # TODO move it to ae
        self.beta = cfg.MODEL.CODEBOOK.BETA


        self.to(self.device)

    def train(self, mode=True):
        super().train(mode)
        self.codebook.train(mode)
        return self

    def wrap_parallel(self, device_ids, broadcast_buffers):
        super().wrap_parallel(device_ids, broadcast_buffers)
        if not self.use_codebook_ema:
            self.codebook = DistributedDataParallel(self.codebook, device_ids=device_ids,
                                                    broadcast_buffers=broadcast_buffers)

    def _generator_parameters(self):
        params = super()._generator_parameters()
        if not self.use_codebook_ema:
            params += list(self.codebook.parameters())
        return params

    def forward(self, data, mode='inference'):
        return super().forward(data, mode)

    def compute_generator_loss(self, x):
        loss_dict, x, x_tilde = self.compute_supervised_loss(x, return_x=True)

        return loss_dict

    def compute_supervised_loss(self, x, return_x=False):
        loss_dict = {}
        is_seq = len(x.size()) == 5
        if is_seq:
            b, t, c, h, w = x.size()
            x = x.view(b * t, c, h, w)
            z_e_x = self.encoder(x)
        else:
            z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook(z_e_x, "st")
        x_tilde = self.generator(z_q_x_st)

        # Reconstruction loss
        loss_dict['loss_reconstruction'] = self.pixel_loss(x_tilde, x)

        # Vector quantization objective
        if not self.use_codebook_ema:
            loss_dict['loss_dict'] = F.mse_loss(z_q_x, z_e_x.detach())

        # Commitment objective
        loss_dict['loss_commitment'] = self.beta * F.mse_loss(z_e_x, z_q_x.detach())

        if return_x:
            return loss_dict, x, x_tilde
        else:
            return loss_dict

    def encode(self, x):
        if len(x.size()) == 5:
            b, t, c, h, w = x.size()
            z_e_x = self.encoder(x.view(b * t, c, h, w))
            latents = self.codebook(z_e_x)  # b * t, h,  w
            return latents.view(b, t, *latents.size()[1:])
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.forward(latents, mode="emb").permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)
        x_tilde = self.generator(z_q_x)
        return x_tilde

    def configure_optimizers_and_checkpointers(self):
        o, c = super().configure_optimizers_and_checkpointers()

        if not self.use_codebook_ema:
            optimizer_c = build_optimizer(self.codebook, self.cfg, suffix="_G")
            scheduler_c = build_lr_scheduler(self.cfg, optimizer_c)
            o += [
                {"optimizer": optimizer_c, "scheduler": scheduler_c, "type": "generator"},
            ]

        PathManager.mkdirs(os.path.join(self.cfg.OUTPUT_DIR, 'netC'))
        c += [
            {"checkpointer": Checkpointer(self.codebook, os.path.join(self.cfg.OUTPUT_DIR, 'netC')),
             "pretrained": self.cfg.MODEL.CODEBOOK.WEIGHTS, },
        ]

        return o, c
