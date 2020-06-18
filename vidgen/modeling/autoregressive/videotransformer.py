import torch
import torch.nn as nn
import torch.nn.functional as F

from vidgen.config import CfgNode
from vidgen.modeling.autoregressive import Autoregressive, AUTOREGRESSIVE_REGISTRY
from vidgen.modeling.autoregressive.vt_attention import PositionalEncoding, BlockLocalAttention
from vidgen.modeling.autoregressive.vt_utils import MaskedConv3d


class VTEncoder(nn.Module):
    def __init__(self, nc, nv, da, de, d, blocks, n_heads, kernel_size, stride, pad_value=-1, class_num=0):
        super().__init__()
        self.nv = nv
        self.stride = stride
        self.pad_value = pad_value
        self.conv = nn.Conv3d(nc * nv, de, kernel_size, stride, bias=True)
        self.positional_encoder = PositionalEncoding(de)

        self.block_local_attention = []
        for block, n_head in zip(blocks, n_heads):
            self.block_local_attention.append(
                BlockLocalAttention(block, da, d, n_head, masked=False),
            )
        self.block_local_attention = nn.Sequential(*self.block_local_attention)
        st, sh, sw = stride
        self.slice_embedding = nn.Embedding(st * sh * sw, de)
        self.class_num = class_num
        if self.class_num > 0:
            self.class_embedding = nn.Embedding(class_num, de)
            self.linear_projector = nn.Conv3d(2 * de, d, 1, bias=False)
        else:
            self.linear_projector = nn.Conv3d(de, d, 1, bias=False)

    def forward(self, x, slice_idx, class_idx=None):
        """
        x: (b, nc, T, H, W)
        slice_idx: (b, 1)
        class_idx: (b, 1)
        """
        mask = (x == self.pad_value)  # (b, nc, T, H, W)
        x = x.masked_fill(mask, 0)  # fill with any dummy value
        x = F.one_hot(x, num_classes=self.nv)  # (b, nc, T, H, W, nv)

        # we padded video with value 0, but this value becomes meaningful if one-hot is applied on it
        # to avoid this use masking
        mask = mask.unsqueeze(-1)  # (b, nc, T, H, W, 1)
        x = x.masked_fill(mask, 0)  # (b, nc, T, H, W, nv)
        x = x.permute(0, 1, 5, 2, 3, 4).contiguous()  # (b, nc, T, H, W, nv)
        b, nc, nv, T, H, W = x.size()  # torch.Size([8, 1, 256, 7, 64, 64])
        x = x.view(b, nc * nv, T, H, W).float()
        x = self.conv(x)  # (b, de, t, h, w)
        x = x + self.slice_embedding(slice_idx)[:, :, None, None, None]  # (b, de, t, h, w)
        if self.class_num > 0 and class_idx is not None:
            class_emb = self.class_embedding(class_idx)[:, :, None, None, None].expand_as(x)
            x = torch.cat([x, class_emb], dim=1)
        x = self.linear_projector(x)  # (b, d, t, h, w)
        x = self.block_local_attention(x)  # (b, d, t, h, w)
        return x


class VTDecoder(nn.Module):
    def __init__(self, nc, nv, da, de, d, blocks, n_heads):
        super().__init__()
        self.ch_embedder = []
        for i in range(nc):
            self.ch_embedder.append(nn.Embedding(nv, de))
        self.ch_embedder = nn.ModuleList(self.ch_embedder)
        self.de = de
        self.conv = MaskedConv3d(de, d, (3, 3, 3))
        self.positional_encoder = PositionalEncoding(d)
        self.linear_projector = nn.Conv3d(d, d, 1, bias=False)
        self.block_local_attention = []
        for block, n_head in zip(blocks, n_heads):
            self.block_local_attention.append(
                BlockLocalAttention(block, da, d, n_head, masked=True),
            )
        self.block_local_attention = nn.Sequential(*self.block_local_attention)

    def embed_sum(self, slice):
        """
        slice: (b, nc, t, h, w)
        """
        b, nc, t, h, w = slice.size()
        emb = torch.zeros(b, t, h, w, self.de, device=slice.device, dtype=torch.float)
        for k in range(nc):
            emb += self.ch_embedder[k](slice[:, k, ...])  # b, t, h, w, de
        emb = emb.permute(0, 4, 1, 2, 3)
        return emb

    def forward(self, slice, zl):
        """
        slice: (b, nc, t, h, w)
        zl: (b, d, t, h, w)
        """
        slice = self.embed_sum(slice)  # (b, de, t, h, w)
        slice = self.conv(slice)  # (b, d, t, h, w)
        slice = self.positional_encoder(slice)  # (b, d, t, h, w)
        slice = slice + self.linear_projector(zl)  # (b, d, t, h, w)
        slice = self.block_local_attention(slice)  # (b, d, t, h, w)
        return slice


class ChannelPredictor(nn.Module):
    def __init__(self, d, nc, nv, de, share_p=True, share_embeddings=False):
        super().__init__()
        self.nc = nc
        self.nv = nv
        self.share_p = share_p
        self.share_embeddings = share_embeddings

        self.layer_norm = nn.LayerNorm(d)
        self.U = []
        for k in range(1, nc + 1):
            self.U.append(
                nn.Linear(d + (k - 1) * nv, d, bias=True)
            )
        self.U = nn.ModuleList(self.U)
        self.relu = nn.ReLU(inplace=True)

        if self.share_p:
            assert not self.share_embeddings, "does not make sense"
            self.P = nn.Linear(d, nv, bias=True)
        elif self.share_embeddings:
            self.P = nn.Linear(d, de, bias=True)
        else:
            self.P = []
            for k in range(nc):
                self.P.append(nn.Linear(d, nv, bias=True))
            self.P = nn.ModuleList(self.P)

    def forward(self, slice, yl, mode="logits", pixel=None, temp=1.0, ch_embedder=None, target=None):
        """
        slice: (b, nc, t, h, w)
        yl: (b, d, t, h, w)
        pixel: tuple of 3 coordinates: ti, hi, wi
        """
        if mode == "logits":
            b, d, t, h, w = yl.size()
            yl = yl.view(b, d, t * h * w)
            yl = yl.transpose(1, 2)  # b, thw, d
            yl = self.layer_norm(yl)  # b, thw, d
            slice = slice.view(b, self.nc, t * h * w)  # b, nc, thw
            slice = slice.transpose(1, 2)  # b, thw, nc
            slice = F.one_hot(slice, num_classes=self.nv)  # b, thw, nc, nv
            slice = slice.view(b, t * h * w, self.nc * self.nv).float()  # b, thw, nc * nv
            output = []
            for k in range(self.nc):
                u = self.U[k](yl if k == 0 else torch.cat((yl, slice[:, :, :k * self.nv]), dim=2))  # b, thw, d
                if self.share_p:
                    out = self.P(self.relu(u))  # b, thw, nv
                elif self.share_embeddings:
                    out = self.P(self.relu(u))  # b, thw, de
                    out = F.linear(out, weight=ch_embedder[k].weight, bias=None)  # b, thw, nv
                else:
                    out = self.P[k](self.relu(u))  # b, thw, nv
                out = out.transpose(1, 2).contiguous()  # b, nv, thw
                out = out.view(b, self.nv, t, h, w)
                output.append(out)
            return output
        elif mode == "sample_pixel":
            ti, hi, wi = pixel
            yl = yl[:, :, ti, hi, wi]  # b, d
            yl = self.layer_norm(yl)
            b = yl.size(0)
            output = torch.zeros(b, self.nc, self.nv, dtype=yl.dtype, device=yl.device)
            b_list = list(range(b))
            for k in range(self.nc):
                u = self.U[k](
                    yl if k == 0 else torch.cat((yl, output[:, :k, :].view(b, k * self.nv)), dim=1))  # b, d

                if self.share_p:
                    out = self.P(self.relu(u))  # b, nv
                elif self.share_embeddings:
                    out = self.P(self.relu(u))  # b, ne
                    # TODO in future better to use TiedLinear, instead of this
                    out = F.linear(out, weight=ch_embedder[k].weight, bias=None)  # b, nv
                else:
                    out = self.P[k](self.relu(u))  # b, nv
                prob = torch.softmax(out / temp, 1)  # b, nv

                sample = torch.multinomial(prob, 1).squeeze(-1)  # b
                output[b_list, k, sample] = 1
            output = torch.argmax(output, dim=2)  # b, nc
            return output
        else:
            raise ValueError


@AUTOREGRESSIVE_REGISTRY.register()
class VideoTransformer(Autoregressive):

    @classmethod
    def from_config(cls, cfg: CfgNode, **kwargs):
        return cls(
            nc=cfg.MODEL.AUTOREGRESSIVE.VT.NC,
            nv=cfg.MODEL.AUTOREGRESSIVE.VT.NV,
            kernel_size=cfg.MODEL.AUTOREGRESSIVE.VT.KERNEL,
            stride=cfg.MODEL.AUTOREGRESSIVE.VT.STRIDE,
            d=cfg.MODEL.AUTOREGRESSIVE.VT.D,
            da=cfg.MODEL.AUTOREGRESSIVE.VT.DA,
            de=cfg.MODEL.AUTOREGRESSIVE.VT.DE,
            blocks_e=cfg.MODEL.AUTOREGRESSIVE.VT.BLOCKS_E,
            n_head_e=cfg.MODEL.AUTOREGRESSIVE.VT.N_HEAD_E,
            blocks_d=cfg.MODEL.AUTOREGRESSIVE.VT.BLOCKS_D,
            n_head_d=cfg.MODEL.AUTOREGRESSIVE.VT.N_HEAD_D,
            pad_value=cfg.MODEL.AUTOREGRESSIVE.VT.PAD_VALUE,
            share_p=cfg.MODEL.AUTOREGRESSIVE.VT.SHARE_P,
            share_embeddings=cfg.MODEL.AUTOREGRESSIVE.VT.SHARE_EMBEDDINGS,
            class_num=cfg.MODEL.AUTOREGRESSIVE.VT.CLASS_NUM
        )

    def __init__(self, nc, nv, da, de, d, blocks_e, n_head_e, kernel_size, stride, blocks_d, n_head_d, pad_value,
                 share_p, share_embeddings, class_num):
        super().__init__()
        self.nv = nv
        self.encoder = VTEncoder(nc, nv, da, de, d, blocks_e, n_head_e, kernel_size, stride, pad_value, class_num)
        self.decoder = VTDecoder(nc,
                                 nv,
                                 da, de, d, blocks_d, n_head_d)
        self.ch_predictor = ChannelPredictor(d, nc, nv, de,
                                             share_p=share_p,
                                             share_embeddings=share_embeddings)

    def forward(self, context, slice, slice_idx, mode="logits", pixel=None, zl=None, temp=1.0, drop_mask=None,
                class_idx=None):
        """
        context: (b, nc, T, H, W)
        slice: (b, nc, t, h, w)
        slices_idxs: (b,)
        """
        if mode == "logits":
            target = slice.clone()
            zl = self.encoder(context, slice_idx, class_idx=class_idx)  # b, d, t, h, w

            yl = self.decoder(slice, zl)  # b, d, t, h, w
            pred = self.ch_predictor(slice, yl, mode=mode, ch_embedder=self.decoder.ch_embedder,
                                     target=target)  # list of size nc: b, nv, t, h, w
            return pred
        elif mode == "sample_pixel":
            if zl is None:
                zl = self.encoder(context, slice_idx, class_idx=class_idx)  # b, d, t, h, w
            yl = self.decoder(slice, zl)  # b, d, t, h, w
            pred = self.ch_predictor(slice, yl, mode="sample_pixel", pixel=pixel, temp=temp,
                                     ch_embedder=self.decoder.ch_embedder)  # b, nc
            return pred, zl
        else:
            raise ValueError
