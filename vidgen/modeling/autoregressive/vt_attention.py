import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, num_dims=3, min_timescale=1.0, max_timescale=1.0e4):
        super().__init__()
        assert d_model >= (num_dims * 2), 'd_model should be >= then 2*num_dims'
        self.d_model = d_model
        self.num_dims = num_dims  # t, h, w

        self.num_timescales = self.d_model // (self.num_dims * 2)

        log_timescale_increment = np.log(max_timescale / min_timescale) / self.num_timescales
        inv_timescales = min_timescale * torch.exp(
            (torch.arange(self.num_timescales).float() * -log_timescale_increment))

        self.register_buffer('inv_timescales', inv_timescales)

    def forward(self, x):
        for dim in range(self.num_dims):
            length = x.shape[dim + 2]  # add 1 to exclude batch dim and c dim

            # cos or sin of (positions * inv_timescales)
            position = torch.arange(length, dtype=torch.float, device=x.device)

            scaled_time = position.view(-1, 1) * self.inv_timescales.view(1, -1)

            signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 1)

            # add padding to signal up to d_model
            prepad = dim * 2 * self.num_timescales
            postpad = self.d_model - (dim + 1) * 2 * self.num_timescales
            signal = F.pad(signal, (prepad, postpad)).T

            # match dim of signal and x
            signal = signal.unsqueeze(0)
            for _ in range(dim):
                signal = signal.unsqueeze(-2)

            for _ in range(self.num_dims - 1 - dim):
                signal = signal.unsqueeze(-1)

            x += signal
        return x

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, da):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = math.sqrt(da)

    def forward(self, q, k, v, B, M=None):
        # q, k, v: na, b, thw, da
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temper  # na, b, thw, thw
        attn = attn + B

        if M is not None:
            """
            attn mask example:
            0 1 1
            0 0 1
            0 0 0
            """
            attn = torch.masked_fill(attn, M.bool(), -1e4)
            attn = torch.softmax(attn, dim=3)  # na, b, thw, thw
            # start_mask = torch.ones(1, 1, 1, attn.size(3), device=attn.device, dtype=attn.dtype)
            # start_mask[0, 0, 0, 0] = 0
            # attn = torch.softmax(attn, dim=3) * start_mask  # na, b, thw, thw
        else:
            attn = torch.softmax(attn, dim=3)  # na, b, thw, thw
        output = torch.matmul(attn, v)  # na, b, thw, da
        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module
    """

    def __init__(self, na, d, da):
        """
        :param na: number of parallel attention heads
        :param d: hidden size
        :param da: number of hidden units in one head
        """
        super(MultiHeadAttention, self).__init__()
        self.na = na
        self.da = da

        self.layer_norm = nn.LayerNorm(d)
        self.w_q = nn.Parameter(torch.FloatTensor(na, d, da), requires_grad=True)
        self.w_k = nn.Parameter(torch.FloatTensor(na, d, da), requires_grad=True)
        self.w_v = nn.Parameter(torch.FloatTensor(na, d, da), requires_grad=True)
        self.attention = ScaledDotProductAttention(da)
        self.proj = nn.Linear(na * da, d, bias=False)

        self.init_weights()

    def init_weights(self, *args, **kwargs):
        init.xavier_normal_(self.w_q)
        init.xavier_normal_(self.w_k)
        init.xavier_normal_(self.w_v)
        init.xavier_normal_(self.proj.weight)

    def forward(self, x, B, M):
        # x: (b, thw, d)
        # B: (na, 1, thw, thw)
        # M: (1,  1, thw, thw)
        b, thw, d = x.size()
        residual = x
        x = x.view(1, b * thw, d).expand(self.na, b * thw, d)
        x = self.layer_norm(x)
        q = torch.bmm(x, self.w_q).view(self.na, b, thw, self.da)  # na, b, thw, da
        k = torch.bmm(x, self.w_k).view(self.na, b, thw, self.da)  # na, b, thw, da
        v = torch.bmm(x, self.w_v).view(self.na, b, thw, self.da)  # na, b, thw, da
        out = self.attention(q, k, v, B, M).view(self.na * b, thw, self.da)  # na * b, thw, da
        out = torch.cat(torch.split(out, b, dim=0), dim=-1)  # b, thw, na * da
        out = self.proj(out)
        out = out + residual
        return out


class BlockLocalAttention(nn.Module):
    def __init__(self, block_size, da, d, n_head, masked=False):
        super().__init__()
        self.block_size = block_size
        self.n_head = n_head
        self.mha = MultiHeadAttention(n_head, d, da)
        self.ffn = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU(True), nn.Linear(d, d))

        # for matrix B
        t, h, w = self.block_size
        self.dt_bank = nn.Parameter(torch.zeros(n_head, 2 * t - 1), requires_grad=True)
        self.dh_bank = nn.Parameter(torch.zeros(n_head, 2 * h - 1), requires_grad=True)
        self.dw_bank = nn.Parameter(torch.zeros(n_head, 2 * w - 1), requires_grad=True)

        idx2t = torch.arange(0, t).view(t, 1, 1).expand(t, h, w).contiguous().view(t * h * w, 1)
        idx2h = torch.arange(0, h).view(1, h, 1).expand(t, h, w).contiguous().view(t * h * w, 1)
        idx2w = torch.arange(0, w).view(1, 1, w).expand(t, h, w).contiguous().view(t * h * w, 1)
        dt = idx2t - idx2t.transpose(0, 1)
        dt += torch.abs(dt.min())
        dt = dt.view(-1)
        self.register_buffer('dt', dt)
        dh = idx2h - idx2h.transpose(0, 1)
        dh += torch.abs(dh.min())
        dh = dh.view(-1)
        self.register_buffer('dh', dh)
        dw = idx2w - idx2w.transpose(0, 1)
        dw += torch.abs(dw.min())
        dw = dw.view(-1)
        self.register_buffer('dw', dw)

        if masked:
            np = t * h * w
            mask = torch.triu(torch.ones(1, 1, np, np), diagonal=1)  # na, b, thw, thw
            self.register_buffer('mask', mask)
        else:
            self.register_buffer('mask', None)

    def get_B(self):
        t, h, w = self.block_size
        Bt = torch.index_select(self.dt_bank, 1, self.dt).view(self.n_head, 1, t * h * w, t * h * w)
        Bh = torch.index_select(self.dh_bank, 1, self.dh).view(self.n_head, 1, t * h * w, t * h * w)
        Bw = torch.index_select(self.dw_bank, 1, self.dw).view(self.n_head, 1, t * h * w, t * h * w)
        return Bt + Bh + Bw

    def forward(self, x):

        B, C, T, H, W = x.size()
        t, h, w = self.block_size
        st, sh, sw = T // t, H // h, W // w

        if t == T and h == H and w == W:
            x = x.view(B, C, T * H * W)
            x = x.transpose(1, 2).contiguous()
            x = self.mha(x, B=self.get_B(), M=self.mask)
            x = self.ffn(x) + x
            x = x.transpose(1, 2).contiguous()
            x = x.view(B, C, T, H, W)
        else:
            x = torch.stack(x.split(w, dim=4), dim=1)  # B, sw, C, T, H, w
            x = torch.stack(x.split(h, dim=4), dim=1)  # B, sh, sw, C, T, h, w
            x = torch.stack(x.split(t, dim=4), dim=1)  # B, st, sh, sw, C, t h, w
            x = x.view(B * st * sh * sw, C, t * h * w)
            x = x.transpose(1, 2).contiguous()  # B * st * sh * sw, t * h * w, C
            x = self.mha(x, B=self.get_B(), M=self.mask)  # B * st * sh * sw, t * h * w, C
            x = self.ffn(x) + x
            x = x.transpose(1, 2)  # B * st * sh * sw, C, t * h * w
            x = x.view(B, st, sh, sw, C, t, h, w)  # B(0), st(1), sh(2), sw(3), C(4), t(5), h(6), w(7)
            x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
            x = x.view(B, C, T, H, W)

        return x
