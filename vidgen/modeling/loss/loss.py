from torch import nn
from torch.nn import functional as F


class PixelLoss(nn.Module):

    def __init__(self, cfg):
        super(PixelLoss, self).__init__()

        mode = cfg.LOSS.PIXEL.MODE
        if mode == "l1":
            self.criterion = F.l1_loss
        elif mode == "l2":
            self.criterion = F.mse_loss
        else:
            raise NotImplementedError
        self._lambda = cfg.LOSS.PIXEL.LAMBDA

    def forward(self, input, target, **kwargs):
        return self._lambda * self.criterion(input, target, **kwargs)
