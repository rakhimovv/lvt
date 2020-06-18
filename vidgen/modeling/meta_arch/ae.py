import os

import numpy as np
import torch
import torch.nn.functional as F
from fvcore.common.checkpoint import Checkpointer
from fvcore.common.file_io import PathManager
from torch import nn
from torch.nn import init
from torch.nn.parallel import DistributedDataParallel

from vidgen.modeling import build_encoder, build_generator
from vidgen.solver import build_lr_scheduler
from . import META_ARCH_REGISTRY
from ...solver.build import build_optimizer
from ...utils.events import get_event_storage
from ...utils.image import tensor2im


@META_ARCH_REGISTRY.register()
class AutoEncoderModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.encoder = build_encoder(cfg)
        self.init_weights(self.encoder, cfg.MODEL.INIT_TYPE)
        self.generator = build_generator(cfg)
        self.init_weights(self.generator, cfg.MODEL.INIT_TYPE)

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(1, num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(1, num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.back_normalizer = lambda y: y * pixel_std + pixel_mean
        self.vis_period = cfg.VIS_PERIOD
        self.to(self.device)

    @staticmethod
    def init_weights(module, init_type="normal", slope=0.2):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == "normal":
                    std = 1 / np.sqrt((1 + slope ** 2) * np.prod(m.weight.data.shape[:-1]))
                    m.weight.data.normal_(std=std)
                elif init_type == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight.data)
                else:
                    raise ValueError
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        module.apply(init_func)

        # propagate to children
        for m in module.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, slope)

    def train(self, mode=True):
        self.training = mode
        self.encoder.train(mode)
        self.generator.train(mode)
        return self

    def wrap_parallel(self, device_ids, broadcast_buffers):
        self.encoder = DistributedDataParallel(self.encoder, device_ids=device_ids,
                                               broadcast_buffers=broadcast_buffers)
        self.generator = DistributedDataParallel(self.generator, device_ids=device_ids,
                                                 broadcast_buffers=broadcast_buffers)
    def _generator_parameters(self):
        params = list(self.encoder.parameters())
        params += list(self.generator.parameters())
        return params

    def _set_requires_grad(self, params, requires_grad):
        for p in params:
            p.requires_grad = requires_grad

    def set_generator_requires_grad(self, requires_grad):
        self._set_requires_grad(self._generator_parameters(), requires_grad)

    def visualize_training(self, x):
        # visualize only 3 images
        if x.dim() == 4:
            x = x[:3]
        elif x.dim() == 5:
            x = x[0][:3]
        x_reconstruct = self.encode_decode(x)

        storage = get_event_storage()
        for h in x_reconstruct:
            image = tensor2im(h, normalize=self.cfg.MODEL.GENERATOR.OUT_ACTIVATION == "tanh", tile=True)
            if len(image.shape) == 2:
                image = image[:, :, None]
            storage.put_image("reconstruction", image.transpose(2, 0, 1))

    def forward(self, data, mode='inference'):
        x = self.preprocess_data(data)

        if mode == 'generator':
            iter = get_event_storage().iter
            if self.vis_period > 0 and iter > 0 and iter % self.vis_period == 0:
                with torch.no_grad():
                    self.visualize_training(x)

            loss = self.compute_generator_loss(x)
            return loss
        elif mode == 'supervised':
            return self.compute_supervised_loss(x)
        elif mode == 'encoder':
            return self.encode(x)
        elif mode == 'encoder_decoder':
            return self.encode_decode(x)
        elif mode == 'interpolate_first_last':
            return self.interpolate_first_last(x)
        elif mode == 'inference':
            # latent = self.encode(x)
            # output = []
            # for i in range(latent.size(0)):
            #     output.append({
            #         'latent': latent[i]
            #     })
            # return output

            out, latent = self.encode_decode(x, return_latent=True)
            if out.dim() == 4:
                out = self.back_normalizer(out)
            elif out.dim() == 5:
                b, t, c, h, w = out.size()
                out = self.back_normalizer(out.view(b * t, c, h, w)).view(b, t, c, h, w)
            else:
                raise ValueError
            if self.cfg.INPUT.SCALE_TO_ZEROONE:
                out.clamp_(0.0, 1.0)
            else:
                out.clamp_(0.0, 255.0)
            output = []
            for i in range(out.size(0)):
                output.append({
                    'reconstruction': out[i],
                    'latent': latent[i]
                })
            return output
        else:
            raise ValueError("|mode| is invalid")

    def preprocess_data(self, data):
        if "image" in data[0].keys():
            real_image = [torch.as_tensor(x["image"], device=self.device) for x in data]
            real_image = torch.stack(real_image, dim=0)
            real_image = self.normalizer(real_image)
            real_image = real_image.contiguous()
            return real_image
        elif "image_sequence" in data[0].keys():
            real_seq = [torch.as_tensor(x["image_sequence"], device=self.device) for x in data]
            real_seq = torch.stack(real_seq, dim=0)
            b, t, c, h, w = real_seq.size()
            real_seq = real_seq.view(b * t, c, h, w)
            real_seq = self.normalizer(real_seq)
            real_seq = real_seq.view(b, t, c, h, w)
            real_seq = real_seq.contiguous()
            return real_seq
        else:
            raise ValueError

    def compute_generator_loss(self, x):
        loss_dict = {}
        is_seq = len(x.size()) == 5
        if is_seq:
            b, t, c, h, w = x.size()
            x = x.view(b * t, c, h, w)
            h = self.encoder(x)
        else:
            h = self.encoder(x)
        out = self.generator(h)
        loss_dict['loss_ae_mse'] = F.mse_loss(out, x)
        return loss_dict

    def compute_supervised_loss(self, real_h):
        return self.compute_generator_loss(real_h)

    def encode(self, x):
        if len(x.size()) == 5:
            b, t, c, h, w = x.size()
            result = self.encoder(x.view(b * t, c, h, w))
            return result.view(b, t, *result.size()[1:])
        return self.encoder(x)

    def encode_decode(self, x, return_latent=False):
        if len(x.size()) == 5:
            b, t, c, h, w = x.size()
            latent = self.encode(x.view(b * t, c, h, w))
            out = self.decode(latent).view(b, t, c, h, w)
            latent = latent.view(b, t, *latent.size()[1:])
        else:
            latent = self.encode(x)
            out = self.decode(latent)

        if return_latent:
            return out, latent
        return out

    def interpolate_first_last(self, x):
        b = x.size(0)
        if len(x.size()) == 5:
            result = []
            for i in range(b):
                result.append(self.interpolate_first_last(x[i]))
            return torch.stack(result, dim=0)
        alphas = torch.tensor(np.linspace(0, 1, b)).to(self.device).view(b, 1, 1, 1).float()
        start = self.encoder(x[0].unsqueeze(0))
        end = self.encoder(x[-1].unsqueeze(0))
        zmix = self.lerp(start, end, alphas)
        return self.generator(zmix)

    @staticmethod
    def lerp(start, end, weights):
        return start + weights * (end - start)

    def configure_optimizers_and_checkpointers(self):
        optimizer_e = build_optimizer(self.encoder, self.cfg, suffix="_G")
        scheduler_e = build_lr_scheduler(self.cfg, optimizer_e)

        optimizer_g = build_optimizer(self.generator, self.cfg, suffix="_G")
        scheduler_g = build_lr_scheduler(self.cfg, optimizer_g)

        PathManager.mkdirs(os.path.join(self.cfg.OUTPUT_DIR, 'netE'))
        PathManager.mkdirs(os.path.join(self.cfg.OUTPUT_DIR, 'netG'))
        c = [
            {"checkpointer": Checkpointer(self.encoder, os.path.join(self.cfg.OUTPUT_DIR, 'netE')),
             "pretrained": self.cfg.MODEL.ENCODER.WEIGHTS, },
            {"checkpointer": Checkpointer(self.generator, os.path.join(self.cfg.OUTPUT_DIR, 'netG')),
             "pretrained": self.cfg.MODEL.GENERATOR.WEIGHTS, },
        ]

        o = [
            {"optimizer": optimizer_e, "scheduler": scheduler_e, "type": "generator"},
            {"optimizer": optimizer_g, "scheduler": scheduler_g, "type": "generator"},
        ]
        return o, c
