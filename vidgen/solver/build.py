
from typing import Any, Dict, List, Set, Union

import torch
import torch.optim
from torch.optim.lr_scheduler import LambdaLR

from .lr_scheduler import WarmupCosineLR, WarmupMultiStepLR
from ..config import CfgNode


def _get_params_for_optimizer(model: torch.nn.Module,
                              lr: float, weight_decay: float, weight_decay_norm: float, weight_decay_bias: float):
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module in model.modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            _weight_decay = weight_decay
            if isinstance(module, norm_module_types):
                _weight_decay = weight_decay_norm
            elif key == "bias":
                _weight_decay = weight_decay_bias
            params += [{"params": [value], "lr": lr, "weight_decay": _weight_decay}]
    return params


def build_optimizer(model: Union[List[torch.nn.Module], torch.nn.Module], cfg, suffix="") -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    lr = cfg.SOLVER.__getattr__("LR" + suffix)
    weight_decay = cfg.SOLVER.WEIGHT_DECAY.__getattr__("BASE" + suffix)
    weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY.__getattr__("NORM" + suffix)
    weight_decay_bias = cfg.SOLVER.WEIGHT_DECAY.__getattr__("BIAS" + suffix)

    if isinstance(model, list):
        params = []
        for m in model:
            params.extend(_get_params_for_optimizer(m, lr, weight_decay, weight_decay_norm, weight_decay_bias))
    else:
        params = _get_params_for_optimizer(model, lr, weight_decay, weight_decay_norm, weight_decay_bias)

    name = cfg.SOLVER.OPTIMIZER_NAME
    if name == "adam":
        beta1 = cfg.SOLVER.ADAM.__getattr__("BETA1" + suffix)
        beta2 = cfg.SOLVER.ADAM.__getattr__("BETA2" + suffix)
        optimizer = torch.optim.Adam(params, lr, betas=(beta1, beta2), weight_decay=weight_decay)
    elif name == "rmsprop":
        alpha = cfg.SOLVER.RMSPROP.__getattr__("ALPHA" + suffix)
        momentum = cfg.SOLVER.RMSPROP.__getattr__("MOMENTUM" + suffix)
        optimizer = torch.optim.RMSprop(params, lr, alpha=alpha, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError("Unknown optimizer: {}".format(name))

    return optimizer


def build_lr_scheduler(cfg: CfgNode, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME
    if name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif name == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            cfg.SOLVER.MAX_ITER,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif name == "Identity":
        def compute_lambda(iter):
            return 1.0

        return LambdaLR(optimizer, lr_lambda=[compute_lambda] * len(optimizer.param_groups))
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))
