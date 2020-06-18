import torch
import torch.nn as nn
import torch.nn.functional as F


def subscale_order(st, sh, sw):
    idx2abc = []
    abc2idx = {}
    for a in range(st):
        for b in range(sh):
            for c in range(sw):
                idx2abc.append((a, b, c))
                abc2idx[(a, b, c)] = len(idx2abc) - 1
    return idx2abc, abc2idx


def test_subscale_order():
    idx2abc, abc2idx = subscale_order(4, 2, 2)
    assert len(idx2abc) == len(abc2idx)
    assert min(abc2idx.values()) >= 0
    assert max(abc2idx.values()) < len(idx2abc)


def slice_mask(a, b, c, st, sh, sw, T, H, W, device=torch.device('cpu'), dtype=torch.float):
    """
    mask with elements equal 1 at slice (a, b, c)
    """
    x = torch.zeros(1, 1, T, H, W, device=device, dtype=dtype)
    for ai in range(a, T, st):
        for bi in range(b, H, sh):
            for ci in range(c, W, sw):
                x[0, 0, ai, bi, ci] = 1
    return x


def test_slice_mask():
    T, H, W = 4, 4, 4
    st, sh, sw = 1, 2, 2
    t, h, w = T // st, H // sh, W // sw
    a, b, c = 0, 1, 1
    assert 0 <= a < st
    assert 0 <= b < sh
    assert 0 <= c < sw
    slice = slice_mask(a, b, c, st, sh, sw, T, H, W)
    assert slice.sum().item() == t * h * w


def visible_abc_mask(a, b, c, st, sh, sw, T, H, W, device=torch.device('cpu'), dtype=torch.float):
    """
    mask with elements equal 1 at slices <(a, b, c)
    """
    idx2abc, abc2idx = subscale_order(st, sh, sw)
    vmask = torch.zeros(1, 1, T, H, W, device=device, dtype=dtype)
    idx = abc2idx[(a, b, c)]
    for (ai, bi, ci) in idx2abc[:idx]:
        vmask += slice_mask(ai, bi, ci, st, sh, sw, T, H, W, device=device, dtype=dtype)
    return vmask


def test_video_ss_mask():
    T, H, W = 4, 4, 4
    st, sh, sw = 2, 2, 1
    t, h, w = T // st, H // sh, W // sw
    a, b, c = 1, 0, 0
    assert 0 <= a < st
    assert 0 <= b < sh
    assert 0 <= c < sw
    vmask = visible_abc_mask(a, b, c, st, sh, sw, T, H, W)

    idx2abc, abc2idx = subscale_order(st, sh, sw)
    idx = abc2idx[(a, b, c)]
    assert vmask.sum().item() == t * h * w * idx


def kernel_ss_mask(a, b, c, st, sh, sw, kt, kh, kw, idx2abc=None, abc2idx=None,
                   device=torch.device('cpu'), dtype=torch.float):
    """
    create mask of size (kt, kh, kw)
    where mask centered at element (no difference which one) from (a, b, c)
    with non zero elements at <(a, b, c)
    """
    if idx2abc is None or abc2idx is None:
        idx2abc, abc2idx = subscale_order(st, sh, sw)
    center = torch.ones(st, sh, sw, device=device, dtype=dtype)
    idx = abc2idx[(a, b, c)]
    for (ai, bi, ci) in idx2abc[idx:]:
        center[ai, bi, ci] = 0

    kmask = torch.ones(1, 1, kt, kh, kw, device=device, dtype=dtype)
    ct, ch, cw = kt // 2, kh // 2, kw // 2

    for kti in range(kt):
        for khi in range(kh):
            for kwi in range(kw):
                ot, oh, ow = kti - ct, khi - ch, kwi - cw
                ai, bi, ci = a + ot, b + oh, c + ow
                aii = ai % st if ai >= 0 else -(abs(ai) % st)
                bii = bi % sh if bi >= 0 else -(abs(bi) % sh)
                cii = ci % sw if ci >= 0 else -(abs(ci) % sw)
                kmask[0, 0, kti, khi, kwi] = center[aii, bii, cii]
    return kmask


def ss_shift(x, a, b, c, st, sh, sw, T, H, W, kt, kh, kw, pad_value=0):
    """
    shift video s.t. the first position we apply convolution on is centered on element from (a, b, c)
    """
    t, h, w, = T // st, H // sh, W // sw

    tl, tr = a, a + (t - 1) * st
    front, back = kt // 2 - tl, kt // 2 - (T - tr - 1)
    o_front, o_back = -min(0, front), -min(0, back)
    p_front, p_back = max(0, front), max(0, back)

    hl, hr = b, b + (h - 1) * sh
    top, bottom = kh // 2 - hl, kh // 2 - (H - hr - 1)
    o_top, o_bottom = -min(0, top), -min(0, bottom)
    p_top, p_bottom = max(0, top), max(0, bottom)

    wl, wr = c, c + (w - 1) * sw
    left, right = kw // 2 - wl, kw // 2 - (W - wr - 1)
    o_left, o_right = -min(0, left), -min(0, right)
    p_left, p_right = max(0, left), max(0, right)

    x = x[:, :, o_front:T - o_back, o_top:H - o_bottom, o_left:W - o_right]
    pad = [p_left, p_right, p_top, p_bottom, p_front, p_back]

    return F.pad(x, pad=pad, mode='constant', value=pad_value)


def test_ss_conv():
    """
    Check that:
    1. video + masking + conv
    2. video + masked conv
    produce the same results
    """
    T, H, W = 20, 20, 20
    st, sh, sw = 5, 4, 10
    a, b, c = 2, 2, 5
    assert 0 <= a < st
    assert 0 <= b < sh
    assert 0 <= c < sw
    x = torch.rand(2, 3, T, H, W)
    kt, kh, kw = 3, 3, 3
    weight = torch.rand(5, 3, kt, kh, kw)

    # Variant 1
    vmask = visible_abc_mask(a, b, c, st, sh, sw, T, H, W).bool()
    x1 = ss_shift(x * vmask, a, b, c, st, sh, sw, T, H, W, kt, kh, kw)
    y1 = F.conv3d(x1, weight, bias=None, stride=(st, sh, sw))

    # Variant 2
    kmask = kernel_ss_mask(a, b, c, st, sh, sw, kt, kh, kw)
    x2 = ss_shift(x, a, b, c, st, sh, sw, T, H, W, kt, kh, kw)
    y2 = F.conv3d(x2, weight * kmask, bias=None, stride=(st, sh, sw))

    assert torch.allclose(y1, y2)


class SSConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True):
        super().__init__()
        assert len(kernel_size) == 3
        assert len(stride) == 3
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.idx2abc, self.abc2idx = subscale_order(*stride)

    def forward(self, x, slice):
        kmask = kernel_ss_mask(*slice, *self.stride, *self.kernel_size, idx2abc=self.idx2abc, abc2idx=self.abc2idx,
                               device=x.device)
        return F.conv3d(ss_shift(x, *slice, *self.stride, *self.kernel_size),
                        self.weight * kmask, bias=self.bias, stride=self.stride)


class MaskedConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        kt, kh, kw = kernel_size
        for k in kernel_size:
            assert k % 2 == 1

        self.pad = [kw // 2, kw // 2, kh - 1, 0, kt - 1, 0]
        self.causal = kw // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=0, bias=bias)
        self.conv.weight.data = torch.ones(out_channels, in_channels, kt, kh, kw)

    def forward(self, x):
        x = F.pad(x, self.pad)
        if self.causal > 0:
            self.conv.weight.data[:, :, -1, -1, self.causal:].zero_()
        return self.conv(x)


def test_masked_conv3d():
    x = torch.rand(2, 3, 10, 30, 40)
    mconv = MaskedConv3d(3, 3, (3, 3, 3))
    assert mconv(x).size() == x.size()
