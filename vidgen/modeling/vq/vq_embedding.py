import torch
import torch.nn as nn

from vidgen.utils import comm
from .vq_utils import vq, vq_st
from ...layers.batch_norm import AllReduce


class VQEmbedding(nn.Module):
    def __init__(self, K, D, ema):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1. / K, 1. / K)
        self.K = K

        self.ema = ema
        if self.ema:
            self.eps = 1e-5
            self.decay = 0.99
            self.register_buffer('running_size', torch.zeros(K))
            self.register_buffer('running_sum', self.embedding.weight.detach())

    def forward(self, z_e_x, mode=""):
        if mode == "":
            z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
            latents = vq(z_e_x_, self.embedding.weight)
            return latents
        elif mode == "st":
            return self._straight_through(z_e_x)
        elif mode == "emb":
            return self.embedding(z_e_x)
        else:
            raise ValueError

    def _straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        if self.ema:
            # Use EMA to update the embedding vectors
            with torch.no_grad():
                device = indices.device
                size = torch.zeros_like(self.running_size, dtype=indices.dtype, device=device)
                size.index_add_(dim=0, index=indices, source=torch.ones_like(indices, device=device))
                if comm.get_world_size() > 1:
                    size = AllReduce.apply(size)
                self.running_size.data.mul_(self.decay).add_(1 - self.decay, size)

                sum = torch.zeros_like(self.running_sum, dtype=z_e_x_.dtype, device=device)
                b, h, w, c = z_e_x_.size()
                sum.index_add_(dim=0, index=indices, source=z_e_x_.view(b * h * w, c))
                if comm.get_world_size() > 1:
                    sum = AllReduce.apply(sum)
                self.running_sum.data.mul_(self.decay).add_(1 - self.decay, sum)

                n = self.running_size.sum()
                size_ = (self.running_size + self.eps) / (n + self.K * self.eps) * n
                self.embedding.weight.data.copy_(self.running_sum / size_.unsqueeze(1))

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
                                               dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar


class DVQEmbedding(nn.Module):
    def __init__(self, num, K, D, ema):
        super().__init__()
        assert D % num == 0
        self.num = num
        self.D = D
        self.ve = nn.ModuleList([VQEmbedding(K, D // num, ema) for i in range(num)])

    def forward(self, z_e_x, mode=""):
        assert z_e_x.dim() == 4
        if mode == "":
            latents = []
            for i, part in enumerate(z_e_x.split(self.D // self.num, dim=1)):
                latents.append(self.ve[i](part))
            return torch.stack(latents, dim=1)
        elif mode == "st":
            r1 = []
            r2 = []
            for i, part in enumerate(z_e_x.split(self.D // self.num, dim=1)):
                z_q_x_st, z_q_x = self.ve[i]._straight_through(part)
                r1.append(z_q_x_st)
                r2.append(z_q_x)
            return torch.cat(r1, dim=1), torch.cat(r2, dim=1)
        elif mode == "emb":
            result = []
            for i in range(self.num):
                # latents = z_e_x
                result.append(self.ve[i].embedding(z_e_x[:, i, ...]))
            return torch.cat(result, dim=-1)
        else:
            raise ValueError
