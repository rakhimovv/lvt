import os

import numpy as np
import torch
import torch.nn.functional as F
from fvcore.common.checkpoint import Checkpointer
from fvcore.common.file_io import PathManager
from torch import nn
from torch.nn import init
from torch.nn.parallel import DistributedDataParallel

from vidgen.modeling import build_autoregressive
from vidgen.modeling.autoregressive.vt_utils import visible_abc_mask, slice_mask, ss_shift, subscale_order
from vidgen.solver import build_lr_scheduler
from . import META_ARCH_REGISTRY
from ...solver.build import build_optimizer
from ...utils.events import get_event_storage
from ...utils.image import tensor2im


@META_ARCH_REGISTRY.register()
class VideoTransformerModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model = build_autoregressive(cfg)
        self.init_weights(self.model, cfg.MODEL.INIT_TYPE)
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
        self.model.train(mode)
        return self

    def wrap_parallel(self, device_ids, broadcast_buffers):
        self.model = DistributedDataParallel(self.model, device_ids=device_ids,
                                             broadcast_buffers=broadcast_buffers)

    def _video_old2new(self, video):
        for i in range(self.nc):
            vsize = video[:, i, ...].size()
            old2new = self.__getattr__(f'old2new_{i}')
            video[:, i, ...] = torch.index_select(old2new, dim=0, index=video[:, i, ...].contiguous().view(-1)).view(
                vsize)
        return video

    def _video_new2old(self, video):
        for i in range(self.nc):
            vsize = video[:, i, ...].size()
            new2old = self.__getattr__(f'new2old_{i}')
            video[:, i, ...] = torch.index_select(new2old, dim=0, index=video[:, i, ...].contiguous().view(-1)).view(
                vsize)
        return video

    @torch.no_grad()
    def sample_video(self, video, temp=1.0, n_prime=1, class_idx=None):
        """
        video: (B, nc, T, H, W)
        temp: sampling temperature
        n_prime: number of frames given outside
        """
        video = video.to(self.device)

        pad_value = self.cfg.MODEL.AUTOREGRESSIVE.VT.PAD_VALUE
        stride = self.cfg.MODEL.AUTOREGRESSIVE.VT.STRIDE
        kernel = self.cfg.MODEL.AUTOREGRESSIVE.VT.KERNEL

        st, sh, sw = stride
        idx2abc, _ = subscale_order(st, sh, sw)

        device = video.device
        B, nc, T, H, W = video.size()

        t, h, w = T // st, H // sh, W // sw

        # 1 if pixel value is provided outside, 0 otherwise
        prime_mask = torch.zeros(1, 1, T, H, W, dtype=torch.bool, device=video.device)  # 1, 1, T, H, W
        if n_prime > 0:
            prime_mask[:, :, :n_prime, :, :] = True

        for slice_idx in range(st * sh * sw):
            a, b, c = idx2abc[slice_idx]

            # get slice
            smask = slice_mask(a, b, c, st, sh, sw, T, H, W, dtype=torch.bool, device=device)  # 1, 1, T, H, W
            slice = video.masked_select(smask).view(B, nc, t, h, w)
            prime_mask_slice = prime_mask.masked_select(smask).view(1, 1, t, h, w)

            # get context
            vmask = visible_abc_mask(a, b, c, st, sh, sw, T, H, W, dtype=torch.bool, device=device)  # 1, 1, T, H, W
            context = video.masked_fill(~vmask, pad_value)
            context = ss_shift(context, a, b, c, st, sh, sw, T, H, W, *kernel, pad_value=pad_value)

            # context is the same for each slice, use caching
            zl = None
            slice_idx_t = torch.tensor(slice_idx, device=device).view(1).expand(context.size(0))
            for ti in range(t):
                for hi in range(h):
                    for wi in range(w):
                        if prime_mask_slice[0, 0, ti, hi, wi] == True:
                            continue
                        pred, zl = self.model(context, slice, slice_idx_t, mode="sample_pixel", pixel=(ti, hi, wi),
                                              zl=zl,
                                              temp=temp, class_idx=class_idx)
                        slice[:, :, ti, hi, wi] = pred

            # insert generated slice to video
            video = video.masked_scatter(smask, slice.view(-1))

        return video

    @torch.no_grad()
    def sample_slice(self, context, slice_idx, slice_size, temp=0.9, class_idx=None):
        """
        context: (1, nc, T, H, W)
        slice_idx: int
        temp: sampling temperature
        """

        slice = torch.zeros(size=slice_size, device=context.device, dtype=torch.long)
        t, h, w = slice.size()[2:]
        zl = None
        for ti in range(t):
            for hi in range(h):
                for wi in range(w):
                    pred, zl = self.model(context, slice, slice_idx, mode="sample_pixel", pixel=(ti, hi, wi),
                                          zl=zl,
                                          temp=temp, class_idx=class_idx)
                    slice[:, :, ti, hi, wi] = pred

        return slice

    @torch.no_grad()
    def visualize_training(self, context, slice, slice_idx, class_idx=None):
        self.train(False)

        for i in range(len(slice_idx)):
            if slice_idx[i] != 0:
                break

        nv = self.cfg.MODEL.AUTOREGRESSIVE.VT.NV

        context = context[i].unsqueeze(0)  # 1, nc, T, H, W
        slice_idx = slice_idx[i].detach().view(1)
        sampled_slice = self.sample_slice(context, slice_idx, slice.size(), class_idx=class_idx)[0]  # nc, t, h, w
        sampled_slice = tensor2im(sampled_slice.transpose(0, 1) / nv, normalize=False)

        slice = slice[i]  # nc, t, h, w
        gt_slice = tensor2im(slice.transpose(0, 1) / nv, normalize=False)

        storage = get_event_storage()
        storage.put_image("gt_slice", gt_slice)
        storage.put_image("sampled_slice", sampled_slice)
        self.train(True)

    def forward(self, data, mode='inference'):
        if mode == 'supervised':
            context, slice, slice_idx, ignore_mask, class_idx = self.preprocess_data(data)

            iter = get_event_storage().iter
            if self.vis_period > 0 and iter > 0 and iter % self.vis_period == 0:
                with torch.no_grad():
                    self.visualize_training(context, slice, slice_idx, class_idx)

            return self.compute_supervised_loss(context, slice, slice_idx, ignore_mask, iter, class_idx)
        elif mode == 'inference':
            B = len(data)
            output = []
            for i in range(B):
                output.append({})
            if "BitsEvaluator" in self.cfg.TEST.EVALUATORS:
                output = self.calculate_logits_for_entire_video(data, output)
            if "VTSampler" in self.cfg.TEST.EVALUATORS:
                output = self.sample_videos(
                    data, output,
                    n_prime=self.cfg.TEST.VT_SAMPLER.N_PRIME,
                    num_samples=self.cfg.TEST.VT_SAMPLER.NUM_SAMPLES
                )
            assert len(output[0]) > 0
            return output
        else:
            raise ValueError("|mode| is invalid")

    def sample_videos(self, data, output, n_prime=5, num_samples=1):
        video = [torch.as_tensor(x["image_sequence"], device=self.device) for x in data]
        video = torch.stack(video, dim=0)
        video = video.transpose(1, 2).contiguous()  # B, nc, T, H, W
        video[:, :, n_prime:, :, :] = 0

        class_idx = None
        if "class" in data[0]:
            class_idx = [x["class"].to(self.device) for x in data]
            class_idx = torch.stack(class_idx, dim=0)  # b

        samples = []
        for i in range(num_samples):
            samples.append(self.sample_video(video.clone(), n_prime=n_prime, class_idx=class_idx))  # B, nc, T, H, W
        B = video.size(0)
        assert B == len(output)
        for i in range(B):
            output[i]['samples'] = [sample[i] for sample in samples]  # list of size num_samples with el: # nc, T, H, W
        return output

    def calculate_logits_for_entire_video(self, data, output):
        mode = "logits"
        video = [torch.as_tensor(x["image_sequence"], device=self.device) for x in data]
        video = torch.stack(video, dim=0)

        class_idx = None
        if "class" in data[0]:
            class_idx = [x["class"].to(self.device) for x in data]
            class_idx = torch.stack(class_idx, dim=0)  # b

        B, T, nc, H, W = video.size()
        video = video.transpose(1, 2).contiguous()  # B, nc, T, H, W

        pad_value = self.cfg.MODEL.AUTOREGRESSIVE.VT.PAD_VALUE
        stride = self.cfg.MODEL.AUTOREGRESSIVE.VT.STRIDE
        kernel = self.cfg.MODEL.AUTOREGRESSIVE.VT.KERNEL
        n_prime = self.cfg.MODEL.AUTOREGRESSIVE.VT.N_PRIME
        nv = self.cfg.MODEL.AUTOREGRESSIVE.VT.NV

        st, sh, sw = stride
        idx2abc, _ = subscale_order(st, sh, sw)
        t, h, w = T // st, H // sh, W // sw

        logits = torch.zeros(B, nc, nv, T, H, W, device=video.device)  # zeros or not zeros, does not matter

        for slice_idx in range(st * sh * sw):
            a, b, c = idx2abc[slice_idx]

            smask = slice_mask(a, b, c, st, sh, sw, T, H, W, dtype=torch.bool, device=video.device)  # 1, 1, T, H, W
            slice = video.masked_select(smask).clone().view(B, nc, t, h, w)

            vmask = visible_abc_mask(a, b, c, st, sh, sw, T, H, W, dtype=torch.bool,
                                     device=video.device)  # 1, 1, T, H, W
            context = video.masked_fill(~vmask, pad_value)
            context = ss_shift(context, a, b, c, st, sh, sw, T, H, W, *kernel, pad_value=pad_value)

            slice_idx_t = torch.tensor(slice_idx, device=video.device).view(1).expand(context.size(0))
            pred = self.model(context, slice, slice_idx_t, mode=mode,
                              class_idx=class_idx)  # list of size nc: el: (B, nv, t, h, w)

            for k in range(nc):
                logits[:, k, ...] = logits[:, k, ...].masked_scatter(smask, pred[k].view(-1))

        ignore_mask = torch.zeros(1, T, H, W, dtype=torch.bool, device=video.device)  # 1, T, H, W
        if n_prime > 0:
            ignore_mask[:, :n_prime, :, :] = True

        assert B == len(output)
        for i in range(B):
            output[i]['ignore_mask'] = ignore_mask
            output[i]['logits'] = logits[i]

        return output

    def preprocess_data(self, data):
        context = [x["context"].to(self.device) for x in data]
        context = torch.stack(context, dim=0)  # b, nc, T, H, W
        slice = [x["slice"].to(self.device) for x in data]
        slice = torch.stack(slice, dim=0)  # b, nc, t, h, w
        slice_idx = [x["slice_idx"].to(self.device) for x in data]
        slice_idx = torch.stack(slice_idx, dim=0)  # b
        ignore_mask = [x["ignore_mask"].to(self.device) for x in data]
        ignore_mask = torch.stack(ignore_mask, dim=0)  # b, 1, t, h, w

        class_idx = None
        if "class" in data[0]:
            class_idx = [x["class"].to(self.device) for x in data]
            class_idx = torch.stack(class_idx, dim=0)  # b

        return context, slice, slice_idx, ignore_mask, class_idx

    def compute_supervised_loss(self, context, slice, slice_idx, ignore_mask, iter, class_idx):
        loss_dict = {}
        drop_mask = None

        target = torch.masked_fill(slice, ignore_mask, self.cfg.MODEL.IGNORE_INDEX)  # (b, nc, t, h, w)
        pred = self.model(context, slice, slice_idx, drop_mask=drop_mask,
                          class_idx=class_idx)  # list of size nc: el: (b, nv, t, h, w)
        nc = len(pred)
        loss = 0
        for k in range(nc):
            loss += F.cross_entropy(pred[k], target[:, k, ...], ignore_index=self.cfg.MODEL.IGNORE_INDEX)
        loss = loss / nc
        loss_dict['loss_cross_entropy'] = loss
        return loss_dict

    def configure_optimizers_and_checkpointers(self):
        optimizer_g = build_optimizer(self.model, self.cfg, suffix="_G")
        scheduler_g = build_lr_scheduler(self.cfg, optimizer_g)

        PathManager.mkdirs(os.path.join(self.cfg.OUTPUT_DIR, 'netG'))
        c = [
            {"checkpointer": Checkpointer(self.model, os.path.join(self.cfg.OUTPUT_DIR, 'netG')),
             "pretrained": self.cfg.MODEL.GENERATOR.WEIGHTS, },
        ]
        o = [
            {"optimizer": optimizer_g, "scheduler": scheduler_g, "type": "generator"},
        ]
        return o, c
